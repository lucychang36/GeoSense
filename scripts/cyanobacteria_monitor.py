#!/usr/bin/env python3
"""第9月 W4：蓝藻监测系统原型 —— 阶段3 集成（CyanobacteriaMonitor）

把第7-9月的能力收口成一个真实场景原型：输入两期深圳湾 Sentinel-2 真实影像，
输出「水体范围 + 疑似藻华（水色异常）风险区 + 两期风险变化」。

为什么是它
----------
阶段3 学了三件事：U-Net 分割（水体/城市/植被）、YOLO 检测、模型服务化 + 异步推理。
蓝藻监测天然串起这几样 —— 这也是「集成项目」的意义：不是再写一个模型，
而是把已有模型组装成能回答业务问题的系统。

方法（诚实声明在前）
--------------------
⚠ 本原型**不是**标准蓝藻浓度产品。标准算法（CI-cyano / NDCI / FAI）都需要
   Sentinel-2 的**红边（B5/B6）或 SWIR 波段**做基线，而本项目真实 COG 只保留了
   B2/B3/B4/B8 四波段（下载时的取舍）。因此这里用的是文献支持的**代理启发式**：
   高浓度蓝藻使水面在近红外（B8）反射率显著抬升（细胞散射），同时叶绿素吸收红光（B4）：
   ① Level1「水色异常抬升」：水体内 B8 相对稳健背景（median+MAD）Z>2.5
   ② Level2「疑似藻华」：Level1 且水面 NDVI>0（红光吸收佐证，帮助排除纯悬浮泥沙）
   ③ 结论是「疑似/初筛」，不是浓度定量 —— 要真定量需红边波段 + 实地叶绿素验证。

学习点
------
1. 集成模式：遥感业务系统 = 数据获取 → 物理掩膜（云/无效）→ 模型推理 → 光谱指标 → 统计出图
2. 稳健统计：水体背景用 median/MAD（不是 mean/std）—— 藻华/泥沙本来就是离群点，
   用会被离群点污染的均值估计背景是自欺欺人
3. 交叉验证思维：U-Net 水体与 NDWI 是两条独立证据链，报告一致率做质量控制（而非硬交集）
4. 两期对比要在「双时相都有效且无云」的交集上做 —— 否则云影伪变化会淹没真信号

用法（项目 .venv，WorkBuddy 会话内加 env -u PYTHONPATH）：
  .venv/bin/python scripts/cyanobacteria_monitor.py            # 两期全流程（~40s）
  .venv/bin/python scripts/cyanobacteria_monitor.py --z 3.0     # 收紧 NIR 异常阈值
可选：
  --z FLOAT   NIR 抬升稳健 Z 阈值（默认 2.5，越大越保守）
  --ndvi F    水面 NDVI 佐证阈值（默认 0.0，藻华水面 NDVI 转正）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from unet_segmentation import UNet, N_CLASSES, rgb_preview  # noqa: E402

con = Console()
OUT_DIR = PROJECT_ROOT / "data" / "output"
COGS_DIR = PROJECT_ROOT / "data" / "cogs"
UNET_PT = PROJECT_ROOT / "data" / "models" / "unet_finetuned.pt"   # W2 真实微调权重
PAIR = [("2023-07-08", COGS_DIR / "szbay_real_20230708.tif"),
        ("2025-07-27", COGS_DIR / "szbay_real_20250727.tif")]

CLOUD_MIN = 0.25      # 与 W8 一致的云/高亮规则：4 波段 min > 0.25
NDWI_TH = 0.05        # McFeeters NDWI 水体阈值（与 W2 伪标签规则一致）


def print_concepts() -> None:
    con.rule("[bold cyan]第9月 W4：蓝藻监测系统原型（阶段3 集成）[/]")
    con.print(
        "\n[bold yellow]核心概念速览[/]\n"
        "1. [bold]集成（Integration）[/]：不是新模型，是把已有件拼成业务系统 —— "
        "掩膜 → U-Net 水体 → 光谱异常 → 两期对比 → 出图报告。\n"
        "2. [bold]稳健背景估计[/]：median/MAD 抗离群。藻华/泥沙本身就是异常像素，"
        "用 mean/std 估计背景会被污染（均值被拉高 → 异常被吞掉）。\n"
        "3. [bold]双证据交叉[/]：U-Net（监督学习）vs NDWI（物理阈值）独立判水体，"
        "一致率是质量计 —— 两条链同时错的可能性远低于一条错。\n"
        "4. [bold]代理指标（Proxy）[/]：缺红边波段时用 NIR 抬升 + 红光吸收近似藻华信号，"
        "是「先圈风险区再验证」的初筛哲学，不是定量。\n"
        "5. [bold]时相对比陷阱[/]：两期对比必须限定在双时相都无云的像素上，"
        "否则云影造成的伪差异会把结论带偏。\n"
    )


# ---------------- 数据与掩膜（复用 W8 约定） ----------------
def load_bands(path: Path) -> tuple[np.ndarray, float]:
    """读 COG → (4,H,W) float32 反射率 + 单像素面积 m²（从 transform 算）。"""
    with rasterio.open(path) as ds:
        bands = ds.read().astype(np.float32) / 10000.0
        res = abs(ds.transform.a) * abs(ds.transform.e)   # 通常 10×10=100 m²
    return bands, res


def cloud_mask(bands: np.ndarray) -> np.ndarray:
    """4 波段逐像素 min > 0.25 → 云/高亮（与 W8 一致）。"""
    return bands.min(0) > CLOUD_MIN


def valid_mask(bands: np.ndarray) -> np.ndarray:
    """有效像素：任一波段和 > 0.02（排除 NoData=0）。"""
    return bands.sum(0) > 0.02


# ---------------- U-Net 水体（复用 W2 真实微调权重） ----------------
def load_unet(device: str) -> UNet:
    model = UNet(in_ch=4, out_ch=N_CLASSES, base=16).to(device)
    model.load_state_dict(torch.load(UNET_PT, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def unet_predict_full(model: UNet, bands: np.ndarray, device: str) -> np.ndarray:
    """整景滑窗推理（infer_full：边缘自动 pad 16 倍数）→ (H,W) int64 mask。"""
    from unet_segmentation import infer_full

    x = torch.from_numpy(bands[None]).to(device)
    return infer_full(model, x, device)


def extract_water(bands: np.ndarray, unet_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """水体 = U-Net 水类（0）；NDWI 作独立光谱证据，报告两者一致率。"""
    water = unet_mask == 0
    ndwi = (bands[1] - bands[3]) / (bands[1] + bands[3] + 1e-6)   # B3−B8 / B3+B8
    agree = (water & (ndwi > NDWI_TH)).sum() / max(water.sum(), 1)
    return water, float(agree)


# ---------------- 疑似藻华代理检测（NIR 抬升 + 红光吸收） ----------------
def algae_risk(bands: np.ndarray, water: np.ndarray, valid: np.ndarray,
               z_thr: float = 2.5, ndvi_thr: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """水体内 NIR 异常检测 → (level1 水色异常, level2 疑似藻华, 背景诊断)。

    背景估计只用水体像素（陆/云不参与）；median/MAD 抗藻华离群点污染。
    z = (b8 − med) / (1.4826·MAD) —— 1.4826 使 MAD 近似 σ（正态假设下）。
    level2 附加 NDVI>ndvi_thr：叶绿素吸收红光 → NDVI 转正，帮助区分藻类与纯泥沙。
    """
    b8, b4 = bands[3], bands[2]
    b8w = b8[water]
    med = float(np.median(b8w))
    mad = float(np.median(np.abs(b8w - med))) + 1e-9
    z = (b8 - med) / (1.4826 * mad)
    level1 = water & valid & (z > z_thr)
    ndvi_w = (b8 - b4) / (b8 + b4 + 1e-6)
    level2 = level1 & (ndvi_w > ndvi_thr)
    return level1, level2, {"b8_median": med, "b8_mad_adj": mad, "z_threshold": z_thr}


# ---------------- 统计 + 出图 ----------------
def to_km2(n_px: int, px_area: float) -> float:
    return n_px * px_area / 1e6


def stats_table(name_a: str, res_a: dict, name_b: str, res_b: dict,
                px_area: float) -> None:
    t = Table(title=f"蓝藻（水色异常）监测 —— {name_a} vs {name_b}")
    t.add_column("指标")
    t.add_column(name_a, justify="right")
    t.add_column(name_b, justify="right")
    for row, f in [
        ("水体面积 km²", lambda d: f"{to_km2(int(d['water_px']), px_area):.2f}"),
        ("U-Net 水体 ∩ NDWI 一致率", lambda d: f"{d['ndwi_agree']:.1%}"),
        ("Level1 水色异常 km²", lambda d: f"{to_km2(int(d['l1_px']), px_area):.3f}"),
        ("Level2 疑似藻华 km²", lambda d: f"{to_km2(int(d['l2_px']), px_area):.3f}"),
        ("L2 占水体比例", lambda d: f"{d['l2_px'] / max(d['water_px'], 1):.2%}"),
        ("水体 B8 背景中位数", lambda d: f"{d['b8_median']:.4f}"),
    ]:
        t.add_row(row, f(res_a), f(res_b))
    con.print(t)


def render_overlay(rgb: np.ndarray, water: np.ndarray, l1: np.ndarray, l2: np.ndarray,
                   valid: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """风险叠加图：底=真彩，水体淡蓝、Level1 橙、Level2 红、云白、无效灰。"""
    out = rgb * 0.55
    out[water] = np.array([0.35, 0.55, 0.95]) * 0.6 + out[water] * 0.4
    out[l1] = np.array([1.0, 0.62, 0.1])
    out[l2] = np.array([0.95, 0.15, 0.15])
    out[cloud] = np.array([1.0, 1.0, 1.0])
    out[~valid] = np.array([0.55, 0.55, 0.55])
    return np.clip(out, 0, 1)


def plot_cyano(name_a: str, rgb_a: np.ndarray, ov_a: np.ndarray,
               name_b: str, rgb_b: np.ndarray, ov_b: np.ndarray,
               l2_a: np.ndarray, l2_b: np.ndarray, out_path: Path) -> None:
    """2×3 总览面板：真彩 / 风险叠加 / 两期疑似藻华变化（新增红 消退蓝 持续黄）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    added = l2_b & ~l2_a
    gone = l2_a & ~l2_b
    kept = l2_a & l2_b
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, img, title in [
        (axes[0, 0], rgb_a, f"{name_a} 真彩"),
        (axes[0, 1], ov_a, f"{name_a} 疑似藻华风险"),
        (axes[1, 0], rgb_b, f"{name_b} 真彩"),
        (axes[1, 1], ov_b, f"{name_b} 疑似藻华风险"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    # 变化面板（第三列跨两行）
    chg = np.zeros((*l2_a.shape, 3))
    base = np.clip(rgb_b * 0.35 + 0.15, 0, 1)
    chg[:] = base
    chg[kept] = np.array([0.95, 0.85, 0.2])    # 持续 = 黄
    chg[added] = np.array([0.9, 0.15, 0.15])   # 新增 = 红
    chg[gone] = np.array([0.2, 0.4, 0.95])     # 消退 = 蓝
    axes[0, 2].imshow(chg)
    axes[0, 2].set_title(f"疑似藻华变化  {name_a} → {name_b}", fontsize=11)
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")
    fig.suptitle("深圳湾蓝藻监测原型（Sentinel-2 真实 49QGE · 启发式代理指标，非标准 CI）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------- 主流程 ----------------
def detect_date(model: UNet, device: str, label: str, path: Path,
                z_thr: float, ndvi_thr: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """单期全流程：读 → 掩膜 → U-Net 水体 → 藻华代理 → 统计。"""
    bands, _ = load_bands(path)
    valid = valid_mask(bands)
    cloud = cloud_mask(bands)
    unet_mask = unet_predict_full(model, bands, device)
    water, ndwi_agree = extract_water(bands, unet_mask)
    l1, l2, diag = algae_risk(bands, water, valid, z_thr, ndvi_thr)
    stats = {
        "water_px": int(water.sum()),
        "ndwi_agree": ndwi_agree,
        "l1_px": int(l1.sum()),
        "l2_px": int(l2.sum()),
        "cloud_px": int(cloud.sum()),
        "b8_median": diag["b8_median"],
        "water": water, "l1": l1, "l2": l2,
        "cloud": cloud, "valid": valid, "bands": bands,
    }
    return water, l2, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=2.5, help="NIR 抬升稳健 Z 阈值")
    ap.add_argument("--ndvi", type=float, default=0.0, help="水面 NDVI 佐证阈值")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    con.print(f"[dim]device = {device}[/]\n")
    print_concepts()

    for _, p in PAIR:
        if not p.exists():
            con.print(f"[red]COG 不存在：{p}[/]")
            return 1

    model = load_unet(device)
    t0 = time.time()
    results = []
    for label, p in PAIR:
        con.print(f"[bold]处理 {label}（{p.name}）…[/]")
        _, _, st = detect_date(model, device, label, p, args.z, args.ndvi)
        results.append((label, st))
        con.print(f"  水体 {st['water_px']:,} px（一致率 {st['ndwi_agree']:.1%}）"
                  f"｜ L1 异常 {st['l1_px']:,}｜ L2 疑似 {st['l2_px']:,}｜ 云 {st['cloud_px']:,}\n")

    (name_a, sta), (name_b, stb) = results
    px_area = 100.0   # 10m 分辨率，见 load_bands 的 res 计算；此处两景同网格
    stats_table(name_a, sta, name_b, stb, px_area)

    # 两期对比（限制在双时相有效且无云交集 —— 见学习点 5）
    common = sta["valid"] & stb["valid"] & ~sta["cloud"] & ~stb["cloud"]
    l2a, l2b = sta["l2"] & common, stb["l2"] & common
    added, gone, kept = (l2b & ~l2a).sum(), (l2a & ~l2b).sum(), (l2b & l2a).sum()
    t = Table(title="疑似藻华两期变化（双时相有效且无云交集内）")
    t.add_column("状态")
    t.add_column("像素", justify="right")
    t.add_column("面积 km²", justify="right")
    for name, n in [("新增", added), ("消退", gone), ("持续", kept)]:
        t.add_row(name, f"{n:,}", f"{to_km2(int(n), px_area):.3f}")
    t.add_row("变化净额", f"{int(added - gone):+,}", f"{to_km2(int(added - gone), px_area):+.3f}")
    con.print(t)

    # 出图
    ov_a = render_overlay(rgb_preview(sta["bands"]), sta["water"], sta["l1"], sta["l2"],
                          sta["valid"], sta["cloud"])
    ov_b = render_overlay(rgb_preview(stb["bands"]), stb["water"], stb["l1"], stb["l2"],
                          stb["valid"], stb["cloud"])
    out_png = OUT_DIR / "cyano_monitor.png"
    OUT_DIR.mkdir(exist_ok=True)
    plot_cyano(name_a, rgb_preview(sta["bands"]), ov_a,
               name_b, rgb_preview(stb["bands"]), ov_b,
               l2a, l2b, out_png)
    con.print(f"\n[green]总览图已保存：{out_png}[/]  （耗时 {time.time() - t0:.1f}s）")

    con.print(
        "\n[bold yellow]诚实红旗[/]："
        "\n① 无红边/无 SWIR 波段（COG 仅 B2/B3/B4/B8）→ 无法用标准 CI-cyano/NDCI/FAI，"
        "本结果是『水色异常初筛』非藻华浓度定量"
        "\n② Level2 只佐证『NIR 抬升 + 红光吸收』，悬浮泥沙等其它水色异常无法排除"
        "（深圳湾为浑浊湾，需红边数据 + 实地叶绿素验证才能确证）"
        "\n③ 无地面真值：指标是启发式，未经实验室/原位数据标定，阈值 z/NDVI 需按水域调"
        "\n④ 两期日期相差两年，潮汐/季相/大气差异会制造水色差异（对比在无云交集上已部分缓解）"
        "\n⑤ 2023 景云覆盖 27.5%，云下水面不可见（mask 已剔除，但漏检不可避免）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
