#!/usr/bin/env python
"""
第8月 W1：变化检测（Change Detection）

目标：用合成时间序列（2019-2023 五年 6 月深圳湾）做变化检测，对比两种方法：
  ① 光谱差分 baseline（|ΔNDVI| + |ΔNDWI| > 阈值）
  ② 分类后比较（post-classification）：复用 W1 微调 U-Net 分别分割两时相 → 类别变化

评估：用程序化真值（重算 syn_scene mask）做 IoU / F1 / Precision / Recall。
真实数据演示：合成 2022 vs 真实 2025-07-27 —— 这不是真实变化，是域差距演示，诚实标红旗。

核心概念：
  - 光谱差分：直接看反射率变化 → 快但无语义、对大气/传感器差异敏感
  - 分类后比较：先分类、再看类别变化 → 有语义、抗部分域差距，但分类错误会传递
  - 变化类型：① 类别变化（语义级）② 物候变化（同物候时相差异）③ 灾害变化（突变）

用法：
  python scripts/change_detection.py                  # 默认 2019 vs 2023
  python scripts/change_detection.py --date-a 0 --date-b 1   # 2019 vs 2020（短间隔）
  python scripts/change_detection.py --date-a 1 --date-b 3   # 2020 vs 2022
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rich.console import Console
from rich.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
for _fp in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"]:
    if Path(_fp).exists():
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=_fp).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

# 复用 W1 全部：UNet 类、N_CLASSES、syn_scene 物理模型、rgb_preview / render_mask
from unet_segmentation import UNet, N_CLASSES, syn_scene, rgb_preview, render_mask

con = Console()
OUT_DIR = PROJECT_ROOT / "data" / "output"
COGS_DIR = PROJECT_ROOT / "data" / "cogs"
DATES = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06"]
SEED = 42
N_DATES = len(DATES)
UNET_PT = PROJECT_ROOT / "data" / "models" / "unet_szbay.pt"   # W1 纯合成训练，匹配本场景分布
SPEC_THRESHOLD = 0.08    # |ΔNDVI| + |ΔNDWI| > 阈值即判变化


# ---------------- 配对数据生成 ----------------
def get_scene(date_idx: int, n_dates: int = N_DATES, seed: int = SEED):
    """按 SEED=42 序贯生成 N_DATES 景，返回 idx=date_idx 的 (bands, mask)。
    与 satellite_download.py 主流程同种子（42），保证 mask 与历史生成一致。"""
    rng = np.random.default_rng(seed)
    for i in range(n_dates):
        bands, mask = syn_scene(rng)
        if i == date_idx:
            return bands.copy(), mask.copy()
    raise ValueError(f"date_idx {date_idx} 超出范围（n_dates={n_dates}）")


# ---------------- 方法 1：光谱差分 baseline ----------------
def spectral_diff_change(bands_a: np.ndarray, bands_b: np.ndarray, threshold: float = SPEC_THRESHOLD) -> np.ndarray:
    """|ΔNDVI| + |ΔNDWI| > 阈值 → 变化（uint8: 0=不变 1=变）。"""
    b8a, b4a, b3a = bands_a[3], bands_a[2], bands_a[1]
    b8b, b4b, b3b = bands_b[3], bands_b[2], bands_b[1]
    ndvi_a = (b8a - b4a) / (b8a + b4a + 1e-6)
    ndvi_b = (b8b - b4b) / (b8b + b4b + 1e-6)
    ndwi_a = (b3a - b8a) / (b3a + b8a + 1e-6)
    ndwi_b = (b3b - b8b) / (b3b + b8b + 1e-6)
    change = (np.abs(ndvi_a - ndvi_b) + np.abs(ndwi_a - ndwi_b)) > threshold
    return change.astype(np.uint8)


# ---------------- 方法 2：分类后比较 ----------------
def load_unet(device: str) -> UNet:
    model = UNet(in_ch=4, out_ch=N_CLASSES, base=16).to(device)
    model.load_state_dict(torch.load(UNET_PT, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def unet_predict(model: UNet, bands: np.ndarray, device: str) -> np.ndarray:
    """bands: (4,H,W) float32 反射率 → mask: (H,W) int64 类别标签。"""
    x = torch.from_numpy(bands[None]).to(device)        # (1,4,H,W)
    return model(x).argmax(1).squeeze(0).cpu().numpy()


def post_class_change(model: UNet, bands_a: np.ndarray, bands_b: np.ndarray, device: str) -> np.ndarray:
    pa = unet_predict(model, bands_a, device)
    pb = unet_predict(model, bands_b, device)
    return (pa != pb).astype(np.uint8)


# ---------------- 评估 ----------------
def evaluate_change(gt: np.ndarray, pred: np.ndarray) -> dict:
    """二值变化 mask 评估：IoU / Precision / Recall / F1 + TP/FP/FN/TN。"""
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    iou = tp / max(tp + fp + fn, 1)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"iou": iou, "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def transition_matrix(mask_a: np.ndarray, mask_b: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    """n×n 转换计数：行=date_a 类别，列=date_b 类别。"""
    mat = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            mat[i, j] = int(((mask_a == i) & (mask_b == j)).sum())
    return mat


# ---------------- 可视化 ----------------
def plot_pair_results(bands_a, mask_a, bands_b, mask_b, gt_change, pred_change, out_path, title):
    """2×3 面板：A 真彩 | B 真彩 | A GT mask | B GT mask | GT 变化 | 预测变化。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(rgb_preview(bands_a))
    axes[0, 1].imshow(rgb_preview(bands_b))
    axes[0, 2].imshow(render_mask(mask_a))
    axes[1, 0].imshow(render_mask(mask_b))
    axes[1, 1].imshow(gt_change, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].imshow(pred_change, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("A 真彩色")
    axes[0, 1].set_title("B 真彩色")
    axes[0, 2].set_title("A mask（GT）")
    axes[1, 0].set_title("B mask（GT）")
    axes[1, 1].set_title("GT 变化（白=变）")
    axes[1, 2].set_title("预测变化（白=变）")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_transition_matrix(mat: np.ndarray, date_a: int, date_b: int, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="hot", aspect="auto")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(["水", "城", "植"])
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(["水", "城", "植"])
    ax.set_xlabel(f"{DATES[date_b]} 类别")
    ax.set_ylabel(f"{DATES[date_a]} 类别")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="white" if mat[i, j] < mat.max() * 0.6 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, label="像素数")
    fig.suptitle(f"类别转换矩阵 · {DATES[date_a]} → {DATES[date_b]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-a", type=int, default=0, help="起始日期索引（0=2019, 4=2023）")
    ap.add_argument("--date-b", type=int, default=4, help="结束日期索引")
    ap.add_argument("--threshold", type=float, default=SPEC_THRESHOLD, help="光谱差分阈值")
    args = ap.parse_args()

    date_a, date_b = args.date_a, args.date_b
    if date_a == date_b:
        con.print("[bold red]date_a 与 date_b 相同，无变化可检测[/]")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con.print(f"[bold]第8月 W1：变化检测[/] | {DATES[date_a]} → {DATES[date_b]} | device={device}")

    # 1) 配对数据
    bands_a, mask_a = get_scene(date_a)
    bands_b, mask_b = get_scene(date_b)
    gt_change = (mask_a != mask_b).astype(np.uint8)
    con.print(f"配对尺寸 [bold]{bands_a.shape[1]}x{bands_a.shape[2]}[/] | "
              f"GT 变化像素 [bold]{int(gt_change.sum()):,}[/] / {gt_change.size:,}"
              f"（[bold]{gt_change.mean():.1%}[/]）")

    # 2) 方法 1：光谱差分
    spec_change = spectral_diff_change(bands_a, bands_b, threshold=args.threshold)
    m_spec = evaluate_change(gt_change, spec_change)

    # 3) 方法 2：分类后比较（W1 U-Net）
    model = load_unet(device)
    pcc_change = post_class_change(model, bands_a, bands_b, device)
    m_pcc = evaluate_change(gt_change, pcc_change)

    # 4) 类别转换矩阵（GT）
    trans = transition_matrix(mask_a, mask_b)

    # 5) 多间隔扫描：2019 → 2020/2021/2022/2023 变化像素数
    con.print("\n[bold cyan]多间隔变化面积扫描（2019 起始 → 2020/2021/2022/2023）[/]")
    multi_table = Table(title="多间隔变化趋势")
    multi_table.add_column("终点", style="cyan")
    multi_table.add_column("间隔（年）")
    multi_table.add_column("GT 变化像素", justify="right")
    multi_table.add_column("GT 变化占比", justify="right")
    mask_2019 = mask_a
    for d in range(1, N_DATES):
        _, mb = get_scene(d)
        gt = (mask_2019 != mb).astype(np.uint8)
        multi_table.add_row(DATES[d], f"{d}", f"{int(gt.sum()):,}", f"{gt.mean():.2%}")
    con.print(multi_table)

    # 6) 真实数据演示：合成 2022 vs 真实 2025-07-27（域差距红旗）
    con.print("\n[bold cyan]真实数据演示：合成 2022-06 vs 真实 2025-07-27（域差距红旗）[/]")
    real_path = COGS_DIR / "szbay_real_20250727.tif"
    real_demo_done = False
    if real_path.exists():
        with rasterio.open(real_path) as ds:
            real_bands = ds.read().astype(np.float32) / 10000.0
        H, W = real_bands.shape[1:]
        cy, cx = H // 2, W // 2
        half = 256
        real_crop = real_bands[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        if real_crop.shape[1] < 512:
            pad = 512 - real_crop.shape[1]
            real_crop = np.pad(real_crop, ((0, 0), (0, pad), (0, pad)), mode="edge")
        bands_2022, _ = get_scene(3)   # 2022-06 合成
        spec_real = spectral_diff_change(bands_2022, real_crop, threshold=args.threshold)
        pcc_real = post_class_change(model, bands_2022, real_crop, device)
        con.print(f"  真实 mosaic 中心裁剪 {real_crop.shape[1]}x{real_crop.shape[2]}")
        con.print(f"  光谱差分：{int(spec_real.sum()):,} 像素判变化 | 分类后：{int(pcc_real.sum()):,}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(rgb_preview(bands_2022)); axes[0, 0].set_title("合成 2022-06（真彩色）")
        axes[0, 1].imshow(rgb_preview(real_crop)); axes[0, 1].set_title("真实 2025-07-27（真彩色）")
        axes[1, 0].imshow(spec_real, cmap="gray"); axes[1, 0].set_title("方法1：光谱差分（域差距主导）")
        axes[1, 1].imshow(pcc_real, cmap="gray"); axes[1, 1].set_title("方法2：分类后比较（域差距主导）")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle("真实数据演示：合成 2022 vs 真实 2025 —— 这不是真实变化，是域差距")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "chg_real_domain_gap.png", dpi=130)
        plt.close(fig)
        real_demo_done = True

    # 7) 主结果图
    plot_pair_results(bands_a, mask_a, bands_b, mask_b, gt_change, spec_change,
                      OUT_DIR / "chg_spectral.png",
                      f"变化检测 · {DATES[date_a]} → {DATES[date_b]} · 方法1：光谱差分 baseline（thr={args.threshold}）")
    plot_pair_results(bands_a, mask_a, bands_b, mask_b, gt_change, pcc_change,
                      OUT_DIR / "chg_postclass.png",
                      f"变化检测 · {DATES[date_a]} → {DATES[date_b]} · 方法2：分类后比较（W1 U-Net）")
    plot_transition_matrix(trans, date_a, date_b, OUT_DIR / "chg_transition_matrix.png")

    # 8) 汇总表
    tbl = Table(title=f"变化检测结果 · {DATES[date_a]} → {DATES[date_b]}")
    tbl.add_column("方法", style="cyan")
    tbl.add_column("IoU", justify="right")
    tbl.add_column("Precision", justify="right")
    tbl.add_column("Recall", justify="right")
    tbl.add_column("F1", justify="right")
    tbl.add_column("预测变化像素", justify="right")
    for name, m, p in [("光谱差分（baseline）", m_spec, spec_change),
                       ("分类后比较（W1 U-Net）", m_pcc, pcc_change)]:
        tbl.add_row(name, f"{m['iou']:.3f}", f"{m['precision']:.3f}",
                    f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{int(p.sum()):,}")
    con.print(tbl)

    # 9) 诚实红旗
    con.print("""
[bold yellow]诚实红旗（变化检测的三层约束）[/]
① 域差距 vs 真实变化：当前真实数据仅有 1 景（2025-07-27），无法构成真实时相对。
   "合成 2022 vs 真实 2025" 的差异主要来自域差距（光谱/传感器/大气），
   几乎不含真实地表变化 → 这是 demo 图，不是真实验证。
② 合成纹理"随机变化" != 真实地物变化：2019-2023 合成场景只随机切换植被/城市纹理，
   海岸线固定无变化。算法在真实场景还需应对"新建/拆除/淹没"等结构化变化。
③ 合成 → 真实光谱差异：方法 1（光谱差分）严重依赖光谱稳定，跨域/跨季会失效；
   方法 2（分类后比较）有部分抗性，但分类错误会同时影响两时相，
   当两时相都错判为同一错误类时会产生"双错抵消"假象（看起来好于实际）。""" +
              ("\n[yellow]注[/]：真实数据演示图未生成（szbay_real_20250727.tif 不存在）"
               if not real_demo_done else ""))


if __name__ == "__main__":
    main()
