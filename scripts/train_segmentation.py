#!/usr/bin/env python3
"""第7月 W2：完整训练流水线 —— 合成预训练 + 真实微调（迁移学习）

学习点
------
1. 迁移学习（Transfer Learning）：合成数据预训练 → 真实数据微调
2. 伪标签（Pseudo-label）：光谱指数 NDWI（水体）+ NDVI（植被）自动生成弱监督标签
3. 灾难性遗忘（Catastrophic Forgetting）：微调时监控合成 val mIoU
4. 混合训练：合成 + 真实 patch 联合微调（既学新域又保旧能力）
5. 训练曲线可视化：loss 与 mIoU 随 epoch 变化

数据
----
① 复用 W1 syn_scene（程序化合成 + 真值）做预训练；
② 真实 Sentinel-2 mosaic 用 NDWI/NDVI + 阈值生成伪标签（光谱指数法）。

诚实红旗：伪标签 ≠ 真值。光谱指数对阴影/混合像元/云边界敏感，本节评估的「真实
mIoU」是用伪标签作代理的真实性能估计 —— 只能说明「模型学到了真实分布特征」，
不能直接等同「在人工标注上的精度」。W3 引入 SAM 重新生成更高质量伪标签。

运行（项目 .venv）：
    .venv/bin/python scripts/train_segmentation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rich.console import Console
from rich.table import Table

from unet_segmentation import (
    UNet, syn_scene, build_dataset, PatchDataset, evaluate, infer_full,
    rgb_preview, render_mask, plot_full_scene, CMAP,
    N_CLASSES, CLASSES, PATCH, PROJECT_ROOT,
)

con = Console()


def print_concepts() -> None:
    con.rule("[bold cyan]第7月 W2：合成预训练 + 真实微调（迁移学习）[/]")
    con.print(
        "\n[bold yellow]核心概念速览[/]\n"
        "1. [bold]迁移学习[/]：在源域（大标注/低成本）预训练，在目标域（小标注/高成本）微调。"
        "   本节：源域=合成数据（程序化真值无限量），目标域=真实 Sentinel-2 mosaic（无标注）。\n"
        "2. [bold]伪标签[/]：用领域知识（这里是光谱指数）自动给真实影像生成初版 mask，"
        "   作为弱监督信号驱动微调。W3 引入 SAM 会替换为更高质量伪标签。\n"
        "   [bold]NDWI[/] = (B3 - B8) / (B3 + B8)：绿波段高、近红外低 → 水体（深圳湾 NDWI ≈ 0.10）。\n"
        "   [bold]NDVI[/] = (B8 - B4) / (B8 + B4)：近红外高、红波段低 → 植被（NDVI > 0.3 视为植被）。\n"
        "3. [bold]灾难性遗忘[/]：微调真实分布时，模型可能「忘了」合成上学到的特征，合成 val mIoU 暴跌。"
        "   解法：低学习率 + 合成+真实混合微调，让旧能力保留。\n"
        "4. [bold]混合训练[/]：每个 batch 既包含合成 patch（保住旧能力）也包含真实 patch（学新分布）。\n"
        "5. [bold]训练曲线[/]：loss（下降 = 学到特征）/ val mIoU（上升 = 泛化提升），双 y 轴同图便于发现过拟合。\n"
    )


def make_pseudo_label(bands: np.ndarray,
                      ndwi_thr: float = 0.05,
                      ndvi_thr: float = 0.3,
                      cloud_thr: float = 0.25) -> np.ndarray:
    """NDWI/NDVI 阈值法生成伪标签。bands: (4,H,W) float32 反射率。

    类别规则（优先级：水 > 植 > 城）
        0 水：  NDWI > ndwi_thr
        2 植：  NDWI ≤ ndwi_thr 且 NDVI > ndvi_thr
        1 城：  其余有效像素
       -1 ignore：NoData 或 4 波段都极亮（云/高亮异常）
    """
    B2, B3, B4, B8 = bands
    ndwi = (B3 - B8) / (B3 + B8 + 1e-6)
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    invalid = bands.sum(0) < 0.01
    cloud = bands.min(0) > cloud_thr
    mask = np.ones(bands.shape[1:], dtype=np.int64)
    mask[ndwi > ndwi_thr] = 0
    mask[(ndwi <= ndwi_thr) & (ndvi > ndvi_thr)] = 2
    mask[invalid | cloud] = -1
    return mask


def build_real_items(bands: np.ndarray, mask: np.ndarray, patch: int = 128,
                     stride: int = 64, min_valid: float = 0.5) -> list:
    items = []
    H, W = mask.shape
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            m = mask[y:y + patch, x:x + patch]
            if (m != -1).mean() >= min_valid:
                items.append((bands[:, y:y + patch, x:x + patch], m))
    return items


class CombinedDataset(Dataset):
    def __init__(self, syn_dataset: PatchDataset, real_items: list):
        self.syn = syn_dataset
        self.real = real_items

    def __len__(self) -> int:
        return len(self.syn) + len(self.real)

    def __getitem__(self, i: int):
        if i < len(self.syn):
            return self.syn[i]
        b, m = self.real[i - len(self.syn)]
        if np.random.rand() < 0.5:
            b, m = b[:, :, ::-1].copy(), m[:, ::-1].copy()
        if np.random.rand() < 0.5:
            b, m = b[:, ::-1, :].copy(), m[:: -1, :].copy()
        return (torch.tensor(b, dtype=torch.float32),
                torch.tensor(m, dtype=torch.long))


def train_phase(model, train_dl, val_dl, epochs, lr, device, save_path: Path
                ) -> tuple[float, dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    best, hist = 0.0, {"loss": [], "val_miou": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        per, miou = evaluate(model, val_dl, device)
        hist["loss"].append(tot / max(n, 1))
        hist["val_miou"].append(miou)
        tag = " *" if miou > best else ""
        if miou > best:
            best = miou
            torch.save(model.state_dict(), save_path)
        con.print(f"  ep {ep:>2}/{epochs}  loss {tot/max(n,1):.4f}  "
                  f"val mIoU {miou:.3f}  ({per[0]:.2f}/{per[1]:.2f}/{per[2]:.2f}){tag}")
    return best, hist


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _fp in ("/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def plot_curves(pretrain_hist: dict, finetune_hist: dict, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ep_pt = np.arange(1, len(pretrain_hist["loss"]) + 1)
    ep_ft = ep_pt[-1] + np.arange(1, len(finetune_hist["loss"]) + 1)
    ax1.plot(ep_pt, pretrain_hist["loss"], "o-", color="#1f77b4", label="pretrain loss")
    ax1.plot(ep_ft, finetune_hist["loss"], "s-", color="#ff7f0e", label="finetune loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(ep_pt, pretrain_hist["val_miou"], "o--", color="#2ca02c", label="pretrain val mIoU")
    ax2.plot(ep_ft, finetune_hist["val_miou"], "s--", color="#d62728", label="finetune val mIoU")
    ax2.set_ylabel("val mIoU", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, 1)
    ax1.axvline(ep_pt[-1] + 0.5, color="gray", linestyle=":", alpha=0.6)
    ax1.text(ep_pt[-1] + 0.6, ax1.get_ylim()[1] * 0.95, "→ 微调开始",
             fontsize=9, color="gray", va="top")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
    fig.suptitle("训练曲线：合成预训练 → 真实微调（垂直虚线 = 微调切换点）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    con.print(f"[green]训练曲线 → {out_path}[/]")


@torch.no_grad()
def evaluate_real(model, bands: np.ndarray, mask: np.ndarray, device) -> tuple[list[float], float]:
    model.eval()
    pred = infer_full(model, torch.tensor(bands[None]).to(device), device)
    p, g = torch.tensor(pred), torch.tensor(mask)
    valid = g != -1
    sums, counts = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        sums[c] = ((p == c) & (g == c) & valid).sum().item()
        counts[c] = (((p == c) | (g == c)) & valid).sum().item()
    per = [s / counts[c] if counts[c] > 0 else 0.0 for c, s in enumerate(sums)]
    return per, float(np.mean(per))


def render_3row(rgb: np.ndarray, pred_pre: np.ndarray, pred_ft: np.ndarray,
                pseudo: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, name in zip(axes,
                             [rgb, pred_pre, pred_ft, pseudo],
                             ["RGB 真彩色", "微调前预测", "微调后预测", "伪标签参考"]):
        if name == "RGB 真彩色":
            ax.imshow(img)
        else:
            ax.imshow(render_mask(img))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    con.print(f"[green]微调对比图 → {out_path}[/]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--lr-pretrain", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=3e-4)
    ap.add_argument("--fast", action="store_true", help="快速验证模式")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    con.print(f"[dim]device = {device}（{'MPS 加速' if device == 'mps' else 'CPU'}）[/]\n")
    print_concepts()

    n_tr, n_va, crops = (10, 4, 4) if args.fast else (80, 20, 8)
    pre_ep, ft_ep = (3, 3) if args.fast else (args.pretrain_epochs, args.finetune_epochs)

    syn_train = build_dataset(n_tr, seed=100 + n_tr)
    syn_val = build_dataset(n_va, seed=999)
    syn_train_dl = DataLoader(PatchDataset(syn_train, crops, train=True), batch_size=16, shuffle=True)
    syn_val_dl = DataLoader(PatchDataset(syn_val, crops, train=False), batch_size=16)

    import rasterio
    with rasterio.open(PROJECT_ROOT / "data" / "cogs" / "szbay_real_mosaic.tif") as ds:
        real_bands = ds.read().astype(np.float32) / 10000.0
    real_mask = make_pseudo_label(real_bands)
    n_valid = (real_mask != -1).sum()
    pct = {c: float((real_mask == c).sum() / n_valid) for c in range(N_CLASSES)}
    real_items = build_real_items(real_bands, real_mask, patch=PATCH, stride=64)
    t = Table(title="真实 mosaic 伪标签统计")
    t.add_column("类别"); t.add_column("有效像素占比", justify="right")
    for c, name in enumerate(CLASSES):
        t.add_row(name, f"{pct[c]*100:.1f}%")
    t.add_row("ignore(云/边界)", f"{(real_mask == -1).sum() / real_mask.size * 100:.1f}%")
    t.add_row("有效 patch（>50% valid）", f"{len(real_items)} 个")
    con.print(t)

    combined = CombinedDataset(PatchDataset(syn_train, crops, train=True), real_items)
    mix_dl = DataLoader(combined, batch_size=16, shuffle=True)
    con.print(f"[bold]训练[/]：合成 {len(syn_train_dl.dataset)} patches + "
              f"真实 {len(real_items)} patches → 微调 mix {len(combined)} patches/批 16")
    con.print(f"[dim]阶段 1：合成预训练 {pre_ep} epochs @ lr={args.lr_pretrain}；"
              f"阶段 2：真实微调 {ft_ep} epochs @ lr={args.lr_finetune}[/]\n")

    model = UNet().to(device)
    pre_path = PROJECT_ROOT / "data" / "models" / "unet_pretrain.pt"
    ft_path = PROJECT_ROOT / "data" / "models" / "unet_finetuned.pt"
    pre_path.parent.mkdir(parents=True, exist_ok=True)

    con.rule("[bold]阶段 1：合成预训练[/]")
    pre_best, pre_hist = train_phase(model, syn_train_dl, syn_val_dl, pre_ep,
                                     args.lr_pretrain, device, pre_path)
    model.load_state_dict(torch.load(pre_path, map_location=device))
    pre_per, pre_miou = evaluate(model, syn_val_dl, device)
    pre_real_per, pre_real_miou = evaluate_real(model, real_bands, real_mask, device)
    con.print(f"[cyan]预训练后[/]：合成 val mIoU = {pre_miou:.3f} | 真实伪标签代理 mIoU = {pre_real_miou:.3f}\n")

    pred_pre = infer_full(model, torch.tensor(real_bands[None]).to(device), device)

    con.rule("[bold]阶段 2：真实微调（迁移学习）[/]")
    _, ft_hist = train_phase(model, mix_dl, syn_val_dl, ft_ep,
                             args.lr_finetune, device, ft_path)
    model.load_state_dict(torch.load(ft_path, map_location=device))
    ft_per, ft_miou = evaluate(model, syn_val_dl, device)
    ft_real_per, ft_real_miou = evaluate_real(model, real_bands, real_mask, device)
    con.print(f"[cyan]微调后[/]：合成 val mIoU = {ft_miou:.3f} | 真实伪标签代理 mIoU = {ft_real_miou:.3f}\n")

    t = Table(title="迁移学习效果对比")
    t.add_column("指标", style="bold"); t.add_column("微调前", justify="right")
    t.add_column("微调后", justify="right"); t.add_column("变化", justify="right")
    t.add_row("合成 val mIoU（能力保持）", f"{pre_miou:.3f}", f"{ft_miou:.3f}",
              f"{ft_miou-pre_miou:+.3f}")
    t.add_row("真实伪标签 mIoU（域适应）", f"{pre_real_miou:.3f}", f"{ft_real_miou:.3f}",
              f"{ft_real_miou-pre_real_miou:+.3f}")
    for c, name in enumerate(CLASSES):
        t.add_row(f"  {name}（真实伪标签）", f"{pre_real_per[c]:.3f}", f"{ft_real_per[c]:.3f}",
                  f"{ft_real_per[c]-pre_real_per[c]:+.3f}")
    con.print(t)

    pred_ft = infer_full(model, torch.tensor(real_bands[None]).to(device), device)

    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(exist_ok=True)
    plot_curves(pre_hist, ft_hist, out_dir / "seg_train_curve.png")
    render_3row(rgb_preview(real_bands), pred_pre, pred_ft, real_mask,
                out_dir / "seg_real_before_after.png",
                "真实 Sentinel-2 mosaic：合成预训练 vs 真实微调（蓝=水 红=城 绿=植 灰=云/ignore）")

    con.print(
        "\n[bold yellow]诚实红旗（两层）[/]：\n"
        "[bold]① 伪标签 ≠ 真值[/]：NDWI/NDVI 阈值法对阴影、混合像元、云边界、屋顶/裸地混淆敏感，"
        "本节「真实 mIoU」是代理值（与伪标签的吻合度），不是人工标注上的精度。\n"
        "   后续：W3 引入 SAM 生成更高质量伪标签重做微调。\n"
        "[bold]② 真实 mosaic 范围有限[/]：训练 patch 只来自 1 景真实影像（1400×2200），"
        "分布单一；推广到其他区域/时相仍会失真 —— 需要更多真实数据或数据增强（旋转/色彩抖动等）\n"
        "   已在微调里做水平/垂直翻转，但缺失色彩/几何/季节性增强。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
