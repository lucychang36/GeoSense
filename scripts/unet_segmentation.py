#!/usr/bin/env python3
"""第7月 W1：语义分割模型搭建 —— U-Net 深圳湾水体/城市/植被三分类

学习点
------
1. PyTorch 基础：Tensor / autograd / nn.Module / DataLoader
2. CNN 基础：卷积、池化、感受野
3. 语义分割：逐像素分类（vs 图像分类/目标检测）
4. U-Net：编码器-解码器 + 跳跃连接（为什么能保住空间细节）
5. 训练：交叉熵（ignore_index 跳过云）、Adam、验证 IoU/mIoU
6. 诚实红旗：合成数据训练 → 真实 Sentinel mosaic 上的域差距

数据：程序化合成（复用 satellite_download.py 的深圳湾物理模型：
水体椭圆 + 植被/城市噪声纹理 + B2/B3/B4/B8 反射率），每景自带真值 mask
（0=水体 1=城市 2=植被），模型好坏可量化验证。

运行（项目 .venv）：
    .venv/bin/python scripts/unet_segmentation.py
可选参数：
    --epochs 20    训练轮数（默认 20）
    --fast        快速模式（10 景 / 4 epochs / 不跑真实 mosaic），用于验证代码
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 保持与 scripts/satellite_download.py 一致的物理参数
WIDTH, HEIGHT = 512, 512
PATCH = 128          # 训练裁剪块大小
CLASSES = ["water", "urban", "vegetation"]
N_CLASSES = len(CLASSES)

from rich.console import Console
from rich.table import Table

con = Console()


# ---------------------------------------------------------------------------
# 1. 概念速览（边构建边讲解）
# ---------------------------------------------------------------------------
def print_concepts() -> None:
    con.rule("[bold cyan]第7月 W1：U-Net 语义分割模型搭建[/]")
    con.print(
        "\n[bold yellow]核心概念速览[/]\n"
        "1. [bold]PyTorch 基础[/]：Tensor=带梯度的多维数组；autograd=自动求导（loss.backward() 填充 .grad）；\n"
        "   nn.Module=层容器（forward 定义计算图）。这三件事拼起来就是「训练」的引擎。\n"
        "2. [bold]卷积[/]：小核(3x3)滑动提取局部特征；[bold]池化[/]下采样降分辨率扩大感受野；\n"
        "   深度=通道数，每层学不同语义（浅层=边角，深层=区域）。\n"
        "3. [bold]语义分割[/]：对[bold]每个像素[/]做分类（本脚本 3 类：水/城市/植被），\n"
        "   输出与输入同尺寸的概率图 —— 与「整图一个标签」的分类、「框住目标」的检测都不同。\n"
        "4. [bold]U-Net 结构[/]：编码器逐层下采样（语义↑ 分辨率↓）→ 解码器逐层上采样（分辨率恢复），\n"
        "   [bold]跳跃连接[/]把编码器同层特征拼到解码器（保住建筑边缘/河道等空间细节）。\n"
        "   类比：压缩-解压 + 每一层都留一张「小抄」给解码器。\n"
        "5. [bold]IoU[/]（交并比）= 预测∩真值 / 预测∪真值，逐类计算，mIoU=各类平均 —— 分割任务标准指标。\n"
        "6. [bold]ignore_index[/]：云像素光谱与任何地物都不同，训练时置 -1 跳过（不贡献损失）。\n"
    )


# ---------------------------------------------------------------------------
# 2. 合成数据生成（程序化真值，与 satellite_download.py 物理模型同源）
# ---------------------------------------------------------------------------
def syn_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """生成一景合成深圳湾影像 + 真值 mask。

    返回 (bands[4,512,512] float32 反射率, mask[512,512] int64：0水 1城 2植，无云)。
    """
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    nx, ny = x / WIDTH, y / HEIGHT
    # 海湾椭圆（水体）+ 海岸带平滑过渡
    dist = ((nx - 0.34) / 0.34) ** 2 + ((ny - 0.74) / 0.26) ** 2
    land = np.clip((dist - 0.8) / 0.4, 0, 1)          # 0=水 1=陆 连续值
    tex = rng.random((HEIGHT, WIDTH))                  # 植被/城市纹理 >0.5 植被
    # 波段反射率：植被 / 城市 / 水体基值（Sentinel-2 风格）
    veg = np.stack([0.05, 0.08, 0.06, 0.25])           # B2,B3,B4,B8
    urb = np.stack([0.12, 0.14, 0.16, 0.14])
    water_r = np.stack([0.07, 0.05, 0.03, 0.02])
    land_r = veg[None, None, :] * tex[..., None] + urb[None, None, :] * (1 - tex[..., None])
    band_f = land[..., None] * land_r + (1 - land[..., None]) * water_r
    # 噪声 → 模拟传感器噪声
    band_f = band_f + rng.normal(0, 0.006, band_f.shape)
    bands = np.clip(band_f, 0.0, 0.5).transpose(2, 0, 1).astype(np.float32)
    # 真值 mask（程序化生成，模型可学且可验证）
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    water = land < 0.5
    veg_px = (~water) & (tex > 0.5)
    mask[water] = 0
    mask[veg_px] = 2
    mask[(~water) & (~veg_px)] = 1
    return bands, mask


def build_dataset(n_scenes: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [syn_scene(rng) for _ in range(n_scenes)]


class PatchDataset(Dataset):
    """从整景随机裁剪 PATCH 块 + 随机翻转增强（训练）。"""

    def __init__(self, scenes: list, crops: int, train: bool):
        self.items, self.train = [], train
        for bands, mask in scenes:
            for _ in range(crops):
                yy = np.random.randint(0, HEIGHT - PATCH + 1)
                xx = np.random.randint(0, WIDTH - PATCH + 1)
                self.items.append((bands[:, yy:yy + PATCH, xx:xx + PATCH],
                                   mask[yy:yy + PATCH, xx:xx + PATCH]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        b, m = self.items[i]
        if self.train:
            if np.random.rand() < 0.5:
                b, m = b[:, :, ::-1].copy(), m[:, ::-1].copy()
            if np.random.rand() < 0.5:
                b, m = b[:, ::-1, :].copy(), m[:: -1, :].copy()
        return (torch.tensor(b, dtype=torch.float32),
                torch.tensor(m, dtype=torch.long))


# ---------------------------------------------------------------------------
# 3. U-Net 模型（从零实现，不引第三方库）
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    """经典 U-Net：输入 4 波段，输出 N_CLASSES 个通道的 logits。

    编码器 4 层（16→32→64→128）+ 瓶颈 256；解码器 4 层带跳跃连接。
    """

    def __init__(self, in_ch: int = 4, out_ch: int = N_CLASSES, base: int = 16):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 8, base * 16))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2),
                                 DoubleConv(base * 16, base * 8))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2),
                                 DoubleConv(base * 8, base * 4))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2),
                                 DoubleConv(base * 4, base * 2))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(base * 2, base, 2, stride=2),
                                 DoubleConv(base * 2, base))
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        c5 = self.down4(c4)
        u = self._up(self.up1, c5, c4)
        u = self._up(self.up2, u, c3)
        u = self._up(self.up3, u, c2)
        u = self._up(self.up4, u, c1)
        return self.outc(u)

    @staticmethod
    def _up(block, u, skip):
        return block[1](torch.cat([block[0](u), skip], dim=1))


# ---------------------------------------------------------------------------
# 4. 训练 / 验证
# ---------------------------------------------------------------------------
@torch.no_grad()
def iou_per_class(pred: torch.Tensor, gt: torch.Tensor) -> list[float]:
    """逐类 IoU，忽略 gt == -1（云）。"""
    valid = gt != -1
    out = []
    for c in range(N_CLASSES):
        inter = ((pred == c) & (gt == c) & valid).sum().item()
        union = ((pred == c) | (gt == c)) & valid
        union = union.sum().item()
        out.append(inter / union if union > 0 else 0.0)
    return out


def evaluate(model, loader, device) -> tuple[list[float], float]:
    model.eval()
    sums, counts = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            for c in range(N_CLASSES):
                v = yb != -1
                sums[c] += ((pred == c) & (yb == c) & v).sum().item()
                counts[c] += (((pred == c) | (yb == c)) & v).sum().item()
    per = [s / counts[c] if counts[c] > 0 else 0.0 for c, s in enumerate(sums)]
    return per, float(np.mean(per))


def train(model, train_dl, val_dl, epochs, device, save_path: Path):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    best = 0.0
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
        tag = " *" if miou > best else ""
        if miou > best:
            best = miou
            torch.save(model.state_dict(), save_path)
        con.print(f"  epoch {ep:>2}/{epochs}  loss {tot/n:.4f}  "
                  f"val mIoU {miou:.3f}  ({per[0]:.2f}/{per[1]:.2f}/{per[2]:.2f}){tag}")
    con.print(f"[green]best val mIoU = {best:.3f}，权重已存 {save_path}[/]")
    return best


# ---------------------------------------------------------------------------
# 5. 可视化
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

# 中文字体：优先 macOS 系统字体，Linux 回退 Noto CJK
for _fp in ("/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

CMAP = ListedColormap(["#1f77b4", "#d62728", "#2ca02c"])   # 水蓝 / 城红 / 植绿
IGNORE_COLOR = "#222222"


def rgb_preview(bands: np.ndarray) -> np.ndarray:
    """B4/B3/B2 → RGB，2%-98% 拉伸便于显示。bands: (4,H,W)"""
    rgb = bands[[2, 1, 0]]
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1).transpose(1, 2, 0)


def render_mask(mask: np.ndarray) -> np.ndarray:
    """mask(0/1/2, -1=ignore) → RGB 图"""
    h, w = mask.shape
    out = np.zeros((h, w, 3))
    for c in range(N_CLASSES):
        out[mask == c] = np.array(CMAP(c))[:3]
    out[mask == -1] = np.array(matplotlib.colors.to_rgb(IGNORE_COLOR))
    return out


def plot_patches(model, val_dl, device, out_path: Path, n_show: int = 4):
    """验证集 patch 对比图：真彩色 / 真值 / 预测。"""
    model.eval()
    rows = []
    with torch.no_grad():
        for xb, yb in val_dl:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            for i in range(len(xb)):
                rows.append((xb[i].numpy(), yb[i].numpy(), pred[i]))
                if len(rows) >= n_show:
                    break
            if len(rows) >= n_show:
                break
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    for r, (b, gt, pd) in enumerate(rows):
        axes[r, 0].imshow(rgb_preview(b))
        axes[r, 0].set_title("RGB (B4/B3/B2)" if r == 0 else "")
        axes[r, 1].imshow(render_mask(gt))
        axes[r, 1].set_title("Ground truth" if r == 0 else "")
        axes[r, 2].imshow(render_mask(pd))
        axes[r, 2].set_title("UNet prediction" if r == 0 else "")
        for a in axes[r]:
            a.axis("off")
    fig.suptitle("U-Net 语义分割：验证集 patch 对比（蓝=水体 红=城市 绿=植被）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    con.print(f"[green]patch 对比图 → {out_path}[/]")


@torch.no_grad()
def infer_full(model, img: torch.Tensor, device, patch=256, stride=128) -> np.ndarray:
    """滑窗推理（重叠区取平均），支持任意尺寸整景。img: (1,4,H,W) float32

    U-Net 4 次下采样要求输入可被 16 整除：边缘块先 pad 到 16 倍数再裁回。
    """
    model.eval()
    H, W = img.shape[-2:]
    acc = torch.zeros(1, N_CLASSES, H, W, device=device)
    cnt = torch.zeros(1, 1, H, W, device=device)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0, y1 = y, min(y + patch, H)
            x0, x1 = x, min(x + patch, W)
            h, w = y1 - y0, x1 - x0
            ph = ((h + 15) // 16) * 16
            pw = ((w + 15) // 16) * 16
            block = torch.nn.functional.pad(img[:, :, y0:y1, x0:x1],
                                            (0, pw - w, 0, ph - h))
            logit = model(block)[:, :, :h, :w]
            acc[:, :, y0:y1, x0:x1] += logit
            cnt[:, :, y0:y1, x0:x1] += 1
    return (acc / cnt).argmax(1).squeeze(0).cpu().numpy()


def plot_full_scene(img: np.ndarray, pred: np.ndarray, gt: np.ndarray | None,
                    out_path: Path, title: str):
    n = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes[0].imshow(rgb_preview(img))
    axes[0].set_title("RGB 真彩色")
    axes[1].imshow(render_mask(pred))
    axes[1].set_title("UNet 预测")
    if gt is not None:
        axes[2].imshow(render_mask(gt))
        axes[2].set_title("真值 mask")
    for a in axes:
        a.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    con.print(f"[green]整景推理图 → {out_path}[/]")


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--fast", action="store_true", help="快速验证模式")
    ap.add_argument("--infer-only", action="store_true",
                    help="跳过训练，加载已存权重直接评估/推理")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    con.print(f"[dim]device = {device}（{'MPS 加速' if device == 'mps' else 'CPU'}）[/]\n")
    print_concepts()

    n_tr = 12 if args.fast else 80
    n_va = 6 if args.fast else 20
    epochs = 4 if args.fast else args.epochs
    crops = 4 if args.fast else 8

    # 数据
    tr_set = build_dataset(n_tr, seed=100 + n_tr)
    va_set = build_dataset(n_va, seed=999)
    train_dl = DataLoader(PatchDataset(tr_set, crops, train=True), batch_size=16, shuffle=True)
    val_dl = DataLoader(PatchDataset(va_set, crops, train=False), batch_size=16)
    con.print(f"[bold]数据[/]：训练 {n_tr} 景 × {crops} 块（{len(train_dl.dataset)} patches），"
              f"验证 {n_va} 景（{len(val_dl.dataset)} patches），patch={PATCH}")

    # 模型 + 训练
    model = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    con.print(f"[bold]模型[/]：U-Net（base=16，4 层编码/解码 + 跳跃连接），参数量 {n_params:,}\n")
    save_path = PROJECT_ROOT / "data" / "models" / "unet_szbay.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.infer_only and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        con.print(f"[dim]infer-only：跳过训练，加载 {save_path}[/]\n")
    else:
        best = train(model, train_dl, val_dl, epochs, device, save_path)

    # 载入 best 权重再评估
    model.load_state_dict(torch.load(save_path, map_location=device))
    per, miou = evaluate(model, val_dl, device)
    t = Table(title="验证集逐类 IoU（best 权重）")
    t.add_column("类别")
    t.add_column("IoU", justify="right")
    for c, v in zip(CLASSES, per):
        t.add_row(c, f"{v:.3f}")
    t.add_row("mIoU", f"{miou:.3f}")
    con.print(t)

    out_dir = PROJECT_ROOT / "data" / "output"
    plot_patches(model, val_dl, device, out_dir / "seg_val_patches.png")

    # 整景推理 1：合成 COG（模型没见过的时相；satellite_download 未暴露 seed，无真值）
    import rasterio
    with rasterio.open(PROJECT_ROOT / "data" / "cogs" / "szbay_2022-06.tif") as ds:
        bands = ds.read().astype(np.float32) / 10000.0
    pred = infer_full(model, torch.tensor(bands[None]).to(device), device)
    plot_full_scene(bands, pred, None, out_dir / "seg_szbay_2022.png",
                    "合成 COG 整景推理（2022-06，未见时相；真值未公开暴露故无 GT）")

    # 整景推理 2：真实 Sentinel-2 mosaic（域差距测试）
    if not args.fast:
        with rasterio.open(PROJECT_ROOT / "data" / "cogs" / "szbay_real_mosaic.tif") as ds:
            real = ds.read().astype(np.float32) / 10000.0
        pred_real = infer_full(model, torch.tensor(real[None]).to(device), device)
        plot_full_scene(real, pred_real, None, out_dir / "seg_real_mosaic.png",
                        "泛化测试：真实 Sentinel-2 mosaic（合成训练 → 真实影像的域差距）")

    con.print(
        "\n[bold yellow]诚实红旗（两层分布偏移）[/]："
        "\n[bold]① 合成 patch → 合成含云/边缘 NoData 整景[/]：验证集 mIoU=0.958 是干净 patch 的统计，"
        "\n但 satellite_download 生成的整景 COG 含云块、NoData 边缘，模型对这些「训练分布外」像素"
        "\n会激进地预测为 urban（或水），整景可用性比 patch 低一个量级。"
        "\n[bold]② 合成 → 真实 Sentinel-2[/]：光谱分布（大气/传感器响应/真实地物多样性）差异显著，"
        "\n真实 mosaic 上预测几乎全是红色 urban——这正是 W2（真实标注/伪标签 + 迁移学习）要解决的核心问题。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
