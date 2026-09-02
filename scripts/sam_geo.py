#!/usr/bin/env python3
"""第7月 W3：SAM 遥感零样本分割 —— 对象级伪标签

学习点
------
1. SAM 原理：Image Encoder (ViT) + Prompt Encoder（点/框/掩码）+ Mask Decoder
   —— 输入影像 + 提示点 → 输出对象级 mask，零训练、零标注
2. 网格点提示（grid prompt）：在影像上均匀撒点，每个点让 SAM 分割"该点所在的物体"
3. 伪标签升级：SAM 提供**对象级空间边界**（比像素级阈值平滑、边界准），
   光谱指数（NDWI/NDVI）提供**语义类别**（每个对象平均反射率决定水/城/植）
   —— 空间 × 语义 互补，这是遥感弱监督标注的标准套路
4. 与 W2 对比：对象级 vs 像素级伪标签的差异（块边界断裂、小对象噪声等红旗）

数据
----
真实 Sentinel-2 mosaic（1400×2200 × 4 波段）+ SAM vit-base（9370 万参数，
transformers SamModel，首次运行自动下载 ~375MB）。

诚实红旗：
① SAM 在 512×512 块上独立推理，块边界处对象会断裂（缺跨块融合）；
② SAM 训练于自然影像，对遥感 4 波段/假彩色不友好 —— 必须用真彩色 RGB 输入；
③ SAM 对象无语义，类别完全依赖光谱平均 —— 阴影/混合像元仍会误分类。

运行（项目 .venv）：
    .venv/bin/python scripts/sam_geo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch

from rich.console import Console
from rich.table import Table

from unet_segmentation import (
    rgb_preview, render_mask, CLASSES, N_CLASSES, PROJECT_ROOT,
)
from train_segmentation import make_pseudo_label

con = Console()

BLOCK = 512
AREA_MIN = 150   # SAM 小对象面积阈值（像素），更小视为噪声


def print_concepts() -> None:
    con.rule("[bold cyan]第7月 W3：SAM 遥感零样本分割（对象级伪标签）[/]")
    con.print(
        "\n[bold yellow]核心概念速览[/]\n"
        "1. [bold]SAM 三段式架构[/]：Image Encoder（ViT，把图变成特征）→ Prompt Encoder（点/框/掩码转 token）"
        "→ Mask Decoder（融合特征 + 提示输出 mask）。训练于 1100 万张图 + 10 亿 mask，零样本分割一切。\n"
        "2. [bold]网格点提示[/]：整张影像均匀撒点，SAM 为每个点分割「这个点所在的物体」——"
        "不依赖人工提示的自动分割方案。\n"
        "3. [bold]空间 × 语义互补[/]：SAM 只给空间边界（无类别），光谱指数只给语义（无边界）。"
        "组合 = 每个对象取其光谱平均的 NDWI/NDVI 决定类别 → 得到平滑、对象级、带语义的伪标签。\n"
        "4. [bold]像素级 vs 对象级[/]：W2 阈值法是逐像素判定（碎片多、边界毛刺）；"
        "SAM 伪标签按对象整体判定（边界贴合真实地物轮廓，但块边界会断裂）。\n"
        "5. [bold]弱监督链[/]：程序化真值（W1）→ 光谱指数伪标签（W2）→ SAM 对象级伪标签（W3）——"
        "逐步逼近「真标注」的三种策略，量化对比是本周验收点。\n"
    )


def load_sam(device: str):
    """加载 SAM vit-base（transformers SamModel）。已缓存时秒级加载。"""
    from transformers import SamModel, SamProcessor
    con.print("[dim]加载 SAM vit-base（facebook/sam-vit-base，首次下载 ~375MB）...[/]")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = model.to(device)
    n = sum(p.numel() for p in model.parameters())
    con.print(f"[green]SAM 加载完成 | 参数量 {n:,} | device {device}[/]")
    return model, processor


@torch.no_grad()
def sam_block(model, processor, rgb: np.ndarray, device, n_grid: int) -> np.ndarray:
    """对单块 RGB (H,W,3) uint8 生成对象级 mask（int32：0=背景，>0=对象 id）。

    网格点提示 → 每点取 IoU 最高候选 mask → 按面积降序覆盖合并。
    """
    from PIL import Image
    H, W = rgb.shape[:2]
    ny = max(n_grid, 2)
    nx = max(int(round(n_grid * W / H)), 2)
    ys = np.linspace(H // (ny * 2), H - H // (ny * 2), ny).astype(int)
    xs = np.linspace(W // (nx * 2), W - W // (nx * 2), nx).astype(int)
    # 每个点独立 query → (1, P, 1, 2)。注意：嵌套 [[[x,y]]] 是 num_queries=P, num_points_per_query=1；
    # 若写成 [[(x1,y1), (x2,y2), ...]] 会被 SAM 当作「同一目标的多个点」合并为一个 mask（已踩坑）
    points = [[[[int(x), int(y)]] for y in ys for x in xs]]
    inputs = processor(Image.fromarray(rgb), input_points=points, return_tensors="pt")
    # MPS 不支持 float64，SAM processor 默认 input_points 是 float64 → 显式转 float32
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and v.dtype == torch.float64:
            inputs[k] = v.float()
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    model.eval()
    out = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0]                       # (P, M, H, W) bool
    iou = out.iou_scores[0]    # (P, M)
    best = masks[torch.arange(masks.shape[0]), iou.argmax(1)]   # (P, H, W)
    areas = best.sum(dim=(1, 2))
    merged = torch.zeros((H, W), dtype=torch.long, device=device)
    for idx in torch.argsort(areas, descending=True).tolist():
        if areas[idx].item() < AREA_MIN:
            break
        merged[best[idx]] = idx + 1
    return merged.cpu().numpy()


def segment_mosaic(model, processor, rgb: np.ndarray, device, n_grid: int) -> np.ndarray:
    """分块 SAM 推理，块内对象 id 全局累加，返回全景对象图（0=背景）。"""
    H, W = rgb.shape[:2]
    obj_map = np.zeros((H, W), dtype=np.int64)
    gid = 1
    n_blocks = 0
    for y0 in range(0, H, BLOCK):
        for x0 in range(0, W, BLOCK):
            y1, x1 = min(y0 + BLOCK, H), min(x0 + BLOCK, W)
            block = sam_block(model, processor, rgb[y0:y1, x0:x1], device, n_grid)
            bm = block > 0
            obj_map[y0:y1, x0:x1][bm] = block[bm] + gid - 1
            gid += int(block.max())
            n_blocks += 1
    con.print(f"[dim]分块 SAM：{n_blocks} 块，对象总数 {gid - 1}[/]")
    return obj_map


def classify_objects(obj_map: np.ndarray, ndwi: np.ndarray, ndvi: np.ndarray,
                     ndwi_thr: float = 0.05, ndvi_thr: float = 0.3) -> np.ndarray:
    """每个对象按光谱平均贴类别：0 水 / 2 植 / 1 城；背景(0) 置 -1 待填充。"""
    out = np.full(obj_map.shape, -1, dtype=np.int64)
    ids = np.unique(obj_map)
    for oid in ids:
        if oid == 0:
            continue
        m = obj_map == oid
        nw, nv = ndwi[m].mean(), ndvi[m].mean()
        out[m] = 0 if nw > ndwi_thr else (2 if nv > ndvi_thr else 1)
    return out


def fill_background(sam_pseudo: np.ndarray, ndwi: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """SAM 未覆盖像素（-1）用像素级阈值法兜底填充。"""
    bg = sam_pseudo == -1
    mask = np.ones_like(ndwi, dtype=np.int64)
    mask[ndwi > 0.05] = 0
    mask[(ndwi <= 0.05) & (ndvi > 0.3)] = 2
    sam_pseudo[bg] = mask[bg]
    return sam_pseudo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=4, help="每块网格点数（4=4x4=16 点）")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 推理（MPS 失败时用）")
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
    con.print(f"[dim]device = {device}[/]\n")
    print_concepts()

    # 数据
    import rasterio
    with rasterio.open(PROJECT_ROOT / "data" / "cogs" / "szbay_real_mosaic.tif") as ds:
        bands = ds.read().astype(np.float32) / 10000.0
    ndwi = (bands[1] - bands[3]) / (bands[1] + bands[3] + 1e-6)
    ndvi = (bands[3] - bands[2]) / (bands[3] + bands[2] + 1e-6)
    rgb8 = (rgb_preview(bands) * 255).astype("uint8")   # 真彩色 uint8，SAM 友好

    # W2 阈值伪标签（对比基线）
    thr_pseudo = make_pseudo_label(bands)

    model, processor = load_sam(device)

    # SAM 对象分割 → 光谱分类
    obj_map = segment_mosaic(model, processor, rgb8, device, args.n_grid)
    sam_pseudo = classify_objects(obj_map, ndwi, ndvi)
    sam_pseudo = fill_background(sam_pseudo, ndwi, ndvi)

    # 对比评估（有效像素内）
    valid = (thr_pseudo != -1) & (sam_pseudo != -1)
    agree = float((sam_pseudo[valid] == thr_pseudo[valid]).mean())
    t = Table(title="伪标签对比（真实 mosaic）")
    t.add_column("类别")
    t.add_column("W2 阈值伪标签占比", justify="right")
    t.add_column("W3 SAM 伪标签占比", justify="right")
    t.add_column("一致率", justify="right")
    for c, name in enumerate(CLASSES):
        w2 = float((thr_pseudo == c).sum() / thr_pseudo.size * 100)
        w3 = float((sam_pseudo == c).sum() / sam_pseudo.size * 100)
        agree_c = float(((sam_pseudo == c) & (thr_pseudo == c) & valid).sum()
                        / max(((thr_pseudo == c) & valid).sum(), 1))
        t.add_row(name, f"{w2:.1f}%", f"{w3:.1f}%", f"{agree_c*100:.1f}%")
    t.add_row("总一致率", "", "", f"{agree*100:.1f}%")
    con.print(t)

    # 可视化
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

    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb_preview(bands)); axes[0].set_title("RGB 真彩色")
    axes[1].imshow(obj_map, cmap="tab20"); axes[1].set_title(f"SAM 对象分割（{int(obj_map.max())} 个对象）")
    axes[2].imshow(render_mask(sam_pseudo)); axes[2].set_title("W3 SAM 伪标签")
    axes[3].imshow(render_mask(thr_pseudo)); axes[3].set_title("W2 阈值伪标签")
    for a in axes:
        a.axis("off")
    fig.suptitle("W2 vs W3 伪标签对比（蓝=水 红=城 绿=植 灰=ignore）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "seg_sam_pseudo.png", dpi=110)
    plt.close(fig)
    con.print(f"[green]对比图 → {out_dir / 'seg_sam_pseudo.png'}[/]")

    con.print(
        "\n[bold yellow]诚实红旗（三层）[/]：\n"
        "[bold]① SAM 块边界断裂[/]：512×512 块独立推理，跨块对象被切成两半（对象图可见块缝）；"
        "升级方案：带 overlap 的块融合或滑窗平均。\n"
        "[bold]② SAM 面向自然影像[/]：遥感 4 波段必须转真彩色 RGB 输入，丢失 NIR 信息；"
        "假彩色（NIR 合成）会让 SAM 分割质量骤降。\n"
        "[bold]③ 语义靠光谱平均[/]：对象内阴影/混合像元会被平均掉或误分类；"
        "类别标签质量上限 = NDWI/NDVI 阈值法的上限。\n"
        "一致性统计说明：两种伪标签都是「代理」，一致率高只代表两者吻合，不代表更接近人工标注。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
