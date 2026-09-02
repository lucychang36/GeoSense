#!/usr/bin/env python
"""
第8月 W2：真实时相对变化检测（补 W1 红旗① —— 真实数据无时相对）

数据：真实 Sentinel-2 深圳湾同 tile（49QGE）双时相
  - szbay_real_20230708.tif（2023-07-08，S2B，云 27.5%）
  - szbay_real_20250727.tif（2025-07-27，S2B，云 15.1%）
  同 CRS（EPSG:4326）同 bounds 同 transform → 像素级天然对齐，无需配准。

方法与评估（真实数据无人工 GT，评估策略与合成版不同）：
  ① 光谱差分 baseline：|ΔNDVI| + |ΔNDWI| > 阈值
  ② 分类后比较：U-Net（真实微调权重 unet_finetuned.pt，伪标签 mIoU 0.927）
  ③ 弱参考：NDWI/NDVI 阈值伪标签（与微调同规则）生成两时相伪 mask → 伪变化
  无 GT 评估 = 报告两方法一致率（非准确率）+ 水域稳定性自检 + 变化面积。

诚实红旗（无 GT 场景下必须声明）：
  ① 伪标签是代理不是真值，两方法一致 ≠ 正确
  ② 2023 景云量 27.5%，云/NoData 区需排除（min 波段 > 0.25 判云）
  ③ 潮汐/物候/大气差异会制造假变化（如滩涂水位、季相植被）
  ④ 水域稳定性自检：深水区两年内不该大面积变化，若水→城翻转多 = 分类错误红旗

用法：
  python scripts/real_change_detection.py                  # 默认 2023-07 vs 2025-07
  python scripts/real_change_detection.py --threshold 0.10  # 调光谱差分阈值
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

from unet_segmentation import UNet, N_CLASSES, rgb_preview, render_mask, infer_full

con = Console()
OUT_DIR = PROJECT_ROOT / "data" / "output"
COGS_DIR = PROJECT_ROOT / "data" / "cogs"
UNET_PT = PROJECT_ROOT / "data" / "models" / "unet_finetuned.pt"   # W2 真实微调权重（真实域 mIoU 0.927）
PAIR = [("2023-07-08", COGS_DIR / "szbay_real_20230708.tif"),
        ("2025-07-27", COGS_DIR / "szbay_real_20250727.tif")]
SPEC_THRESHOLD = 0.08
NDWI_TH, NDVI_TH = 0.05, 0.3     # 与 train_segmentation.py 伪标签规则一致
CLOUD_MIN = 0.25                 # 4 波段 min > 0.25 → 高亮（云/异常）


# ---------------- 数据 ----------------
def load_pair() -> tuple[np.ndarray, np.ndarray]:
    """读两时相 (4,H,W) float32 反射率；两景同网格已由下载流程保证。"""
    outs = []
    for _, p in PAIR:
        with rasterio.open(p) as ds:
            outs.append(ds.read().astype(np.float32) / 10000.0)
    return outs[0], outs[1]


def cloud_mask(bands: np.ndarray) -> np.ndarray:
    """4 波段逐像素 min > CLOUD_MIN → 云/高亮（与 W2 云规则一致）。"""
    return bands.min(0) > CLOUD_MIN


def valid_mask(bands: np.ndarray) -> np.ndarray:
    """有效像素：任一波段和 > 0.02（排除 NoData=0）。"""
    return bands.sum(0) > 0.02


# ---------------- 方法 1：光谱差分 ----------------
def spectral_diff_change(bands_a: np.ndarray, bands_b: np.ndarray, threshold: float) -> np.ndarray:
    b8a, b4a, b3a = bands_a[3], bands_a[2], bands_a[1]
    b8b, b4b, b3b = bands_b[3], bands_b[2], bands_b[1]
    ndvi_a = (b8a - b4a) / (b8a + b4a + 1e-6)
    ndvi_b = (b8b - b4b) / (b8b + b4b + 1e-6)
    ndwi_a = (b3a - b8a) / (b3a + b8a + 1e-6)
    ndwi_b = (b3b - b8b) / (b3b + b8b + 1e-6)
    return (np.abs(ndvi_a - ndvi_b) + np.abs(ndwi_a - ndwi_b) > threshold).astype(np.uint8)


# ---------------- 方法 2：U-Net 分类后比较 ----------------
def load_unet(device: str) -> UNet:
    model = UNet(in_ch=4, out_ch=N_CLASSES, base=16).to(device)
    model.load_state_dict(torch.load(UNET_PT, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def unet_predict_full(model: UNet, bands: np.ndarray, device: str) -> np.ndarray:
    """整景滑窗推理（复用 W1 infer_full：边缘块自动 pad 到 16 倍数）。
    bands: (4,H,W) float32 → mask: (H,W) int64。"""
    x = torch.from_numpy(bands[None]).to(device)      # (1,4,H,W)
    return infer_full(model, x, device)


# ---------------- 方法 3：伪标签弱参考 ----------------
def pseudo_mask(bands: np.ndarray) -> np.ndarray:
    """NDWI/NDVI 阈值伪标签（与 train_segmentation.py 同规则）：0=水 1=城 2=植，云=-1。"""
    b3, b4, b8 = bands[1], bands[2], bands[3]
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    m = np.ones(bands.shape[1:], dtype=np.int64)          # 默认城
    m[ndwi > NDWI_TH] = 0                                  # 水
    m[(ndwi <= NDWI_TH) & (ndvi > NDVI_TH)] = 2            # 植
    m[cloud_mask(bands)] = -1                              # 云 ignore
    return m


# ---------------- 一致性 / 稳定性 ----------------
def agreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """两方法在有效像素上的变化判定一致率。"""
    sel = mask
    if sel.sum() == 0:
        return float("nan")
    return float((a[sel] == b[sel]).mean())


def change_stats(change: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    """有效像素内变化数 + 占比。"""
    sel = mask
    n = int(sel.sum())
    c = int((change[sel] == 1).sum())
    return c, c / max(n, 1)


def water_stability(mask_a: np.ndarray, mask_b: np.ndarray, water_prior_a: np.ndarray,
                    mask_valid: np.ndarray) -> dict:
    """水域稳定性自检：时相 A 判为深水的区域，时相 B 若大量变城 = 红旗。

    water_prior_a: A 时相高置信水区（NDWI > 0.15，排除滩涂噪声）。
    """
    deep_a = water_prior_a & mask_valid
    flipped = (mask_b[deep_a] == 1).sum()            # 深水区变城（城市）数量
    n = int(deep_a.sum())
    return {"deep_water_px": n, "flipped_to_urban": int(flipped),
            "flip_ratio": float(flipped) / max(n, 1)}


# ---------------- 可视化 ----------------
def plot_real_change(bands_a, bands_b, spec_c, pcc_c, pseudo_c, mask_valid,
                     unet_a, unet_b, out_path):
    """3×2 面板：两时相真彩 + 分类 + 三种变化判定 + 有效区。"""
    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    axes[0, 0].imshow(rgb_preview(bands_a)); axes[0, 0].set_title("2023-07-08 真彩色")
    axes[0, 1].imshow(rgb_preview(bands_b)); axes[0, 1].set_title("2025-07-27 真彩色")
    axes[1, 0].imshow(render_mask(unet_a)); axes[1, 0].set_title("U-Net 分类 2023")
    axes[1, 1].imshow(render_mask(unet_b)); axes[1, 1].set_title("U-Net 分类 2025")
    axes[2, 0].imshow(spec_c, cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("方法1：光谱差分（白=变）")
    axes[2, 1].imshow(pcc_c, cmap="gray", vmin=0, vmax=1)
    axes[2, 1].set_title("方法2：分类后比较（白=变）")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("真实时相对变化检测 · 同 tile 49QGE · 2023-07-08 → 2025-07-27（无 GT，双方法对比）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pseudo_reference(bands_a, bands_b, pseudo_a, pseudo_b, mask_valid, out_path):
    """伪标签弱参考：两时相伪标签变化（仅参考，明确标注非真值）。"""
    pseudo_c = ((pseudo_a != pseudo_b) & (pseudo_a >= 0) & (pseudo_b >= 0)).astype(np.uint8)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(rgb_preview(bands_a)); axes[0, 0].set_title("2023-07-08 真彩色")
    axes[0, 1].imshow(rgb_preview(bands_b)); axes[0, 1].set_title("2025-07-27 真彩色")
    axes[1, 0].imshow(pseudo_a, cmap="tab10", vmin=-1, vmax=2)
    axes[1, 0].set_title("伪标签 2023（NDWI/NDVI 阈值，紫=云）")
    axes[1, 1].imshow(pseudo_c, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("伪标签变化（弱参考，非真值）")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("伪标签弱参考（NDWI/NDVI 阈值法）—— 代理参考，不代表人工真值")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=SPEC_THRESHOLD, help="光谱差分阈值")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 加载像素级对齐的双时相
    bands_a, bands_b = load_pair()
    H, W = bands_a.shape[1:]
    con.print(f"[bold]第8月 W2：真实时相对变化检测[/] | "
              f"{PAIR[0][0]} → {PAIR[1][0]} | {W}x{H} | device={device}")

    # 2) 有效区 / 云区
    va, vb = valid_mask(bands_a), valid_mask(bands_b)
    ca, cb = cloud_mask(bands_a), cloud_mask(bands_b)
    mask_valid = va & vb & ~ca & ~cb            # 两时相都有效且无云才参与评估
    con.print(f"  有效像素（双时相有效且无云）：{mask_valid.sum():,} / {mask_valid.size:,}"
              f"（[bold]{mask_valid.mean():.1%}[/]）")
    con.print(f"  2023 云区 {ca.mean():.1%} | 2025 云区 {cb.mean():.1%}")

    # 3) 方法 1：光谱差分
    spec_c = spectral_diff_change(bands_a, bands_b, threshold=args.threshold)
    n_spec, p_spec = change_stats(spec_c, mask_valid)

    # 4) 方法 2：U-Net 分类后比较（真实微调权重）
    model = load_unet(device)
    unet_a = unet_predict_full(model, bands_a, device)
    unet_b = unet_predict_full(model, bands_b, device)
    pcc_c = (unet_a != unet_b).astype(np.uint8)
    n_pcc, p_pcc = change_stats(pcc_c, mask_valid)

    # 5) 弱参考：伪标签变化
    pseudo_a = pseudo_mask(bands_a)
    pseudo_b = pseudo_mask(bands_b)
    both_known = (pseudo_a >= 0) & (pseudo_b >= 0) & mask_valid
    pseudo_c = ((pseudo_a != pseudo_b) & both_known).astype(np.uint8)
    n_ps, p_ps = change_stats(pseudo_c, mask_valid)

    # 6) 无 GT 评估：一致率 + 水域稳定性自检
    agree_spec_pcc = agreement(spec_c, pcc_c, mask_valid)
    agree_pcc_ps = agreement(pcc_c, pseudo_c, mask_valid)
    ndwi_a = (bands_a[1] - bands_a[3]) / (bands_a[1] + bands_a[3] + 1e-6)
    water_prior = (ndwi_a > 0.15) & ~ca           # 2023 高置信深水
    stab = water_stability(unet_a, unet_b, water_prior, mask_valid)

    # 7) 类别转换（U-Net 分类）
    trans = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            trans[i, j] = int(((unet_a == i) & (unet_b == j) & mask_valid).sum())

    # ---- 汇总表 ----
    tbl = Table(title=f"真实时相对变化检测 · {PAIR[0][0]} → {PAIR[1][0]}")
    tbl.add_column("方法", style="cyan")
    tbl.add_column("变化像素", justify="right")
    tbl.add_column("变化占比", justify="right")
    tbl.add_row("① 光谱差分 baseline", f"{n_spec:,}", f"{p_spec:.2%}")
    tbl.add_row("② U-Net 分类后比较", f"{n_pcc:,}", f"{p_pcc:.2%}")
    tbl.add_row("③ 伪标签弱参考", f"{n_ps:,}", f"{p_ps:.2%}")
    con.print(tbl)

    agree_tbl = Table(title="无 GT 评估（一致率 ≠ 准确率）")
    agree_tbl.add_column("指标", style="cyan")
    agree_tbl.add_column("值", justify="right")
    agree_tbl.add_row("① vs ② 变化判定一致率", f"{agree_spec_pcc:.1%}")
    agree_tbl.add_row("② vs ③ 变化判定一致率", f"{agree_pcc_ps:.1%}")
    agree_tbl.add_row("深水区像素（2023 NDWI>0.15）", f"{stab['deep_water_px']:,}")
    agree_tbl.add_row("其中被判水→城翻转", f"{stab['flipped_to_urban']:,}（{stab['flip_ratio']:.2%}）")
    con.print(agree_tbl)

    trans_tbl = Table(title="U-Net 类别转换（2023 → 2025，有效像素）")
    trans_tbl.add_column("2023＼2025", style="cyan")
    trans_tbl.add_column("水", justify="right")
    trans_tbl.add_column("城", justify="right")
    trans_tbl.add_column("植", justify="right")
    for i, name in enumerate(["水", "城", "植"]):
        trans_tbl.add_row(name, *(f"{trans[i, j]:,}" for j in range(N_CLASSES)))
    con.print(trans_tbl)

    # 8) 出图
    plot_real_change(bands_a, bands_b, spec_c, pcc_c, pseudo_c, mask_valid,
                     unet_a, unet_b, OUT_DIR / "chg_real_pair.png")
    plot_pseudo_reference(bands_a, bands_b, pseudo_a, pseudo_b, mask_valid,
                          OUT_DIR / "chg_real_pseudo_ref.png")

    # 9) 诚实红旗
    con.print(
        "\n[bold yellow]诚实红旗（真实数据无人工 GT，以下必须声明）[/]"
        "\n① 弱参考是代理不是真值：两方法一致率只说明互相吻合，不说明正确。"
        "\n② 2023 景云量 27.5%：云/NoData 已排除，但云影/薄云可能漏判。"
        "\n③ 潮汐/物候/大气差异制造假变化：深圳湾潮差大，滩涂水位差会判变。"
        "\n④ 水域稳定性自检若深水区翻城比例高 = 分类错误红旗（模型真实域仍有偏差）。"
        "\n   W2 用 NDWI/NDVI 伪标签微调过 unet_finetuned.pt，但 2023 时相是微调未见过的域。"
    )


if __name__ == "__main__":
    main()
