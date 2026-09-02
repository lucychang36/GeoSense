#!/usr/bin/env python
"""
第7月 W4：YOLOv8 目标检测（船舶检测）

目标：对比语义分割（"哪里是水"）与目标检测（"船在哪里"）的输出差异，
并诚实暴露遥感目标检测的两个核心约束：域差距 + 分辨率极限。

流程：
  1. 程序化生成"水面 + 船舶"合成数据集（带 YOLO 标注框，无人工标注成本）
     - 水体反射率复用 unet_segmentation 物理模型（water_r）
     - 船舶 = 高反射旋转椭圆（10m/像素 → 船长 10-45px ≈ 100-450m 中大型船舶）+ 尾迹
  2. 零样本测试：COCO 预训练 yolov8n 直接在真实 Sentinel-2 mosaic 上检测
     （预期红旗①：COCO 自然影像域 → 卫星影像域，几乎检不到船）
  3. 合成数据微调 yolov8n（COCO 权重做迁移学习）
  4. 评估：合成 val mAP50（合成域内能力）+ 真实 mosaic 推理
     （预期红旗②：10m 分辨率下渔船仅 1-3px，必然漏检）

用法：
  python scripts/yolo_detection.py            # 完整模式（120 训练 / 30 验证 / 40 epochs）
  python scripts/yolo_detection.py --fast     # 快速验证模式（20/8 / 3 epochs）

核心概念速览：
  - 目标检测 vs 语义分割：分割回答"每个像素是什么类"（稠密预测），
    检测回答"物体在哪、多大"（稀疏预测：框 + 类别 + 置信度），二者互补。
  - YOLO = You Only Look Once：一次前向直接回归出所有框
    （单阶段检测器），相对两阶段（R-CNN 系列：候选框 → 分类回归）更快。
  - YOLO 标签格式：每行 `class x_center y_center w h`（归一化 0-1）。
  - mAP50 = IoU 阈值 0.5 下的平均精度（AP 再按类别平均）。
  - 迁移学习：COCO 预训练权重 → 船舶微调，小数据集也能收敛。

输出：
  data/output/yolo_zero_shot.png    COCO 预训练在真实 mosaic 上的零样本检测
  data/output/yolo_synth_val.png    合成验证集 GT vs 预测
  data/output/yolo_real_mosaic.png  微调后在真实 mosaic 上的船舶检测
  data/models/yolov8n_ship.pt       微调权重
  data/ships_synth/                 程序化合成数据集（YOLO 格式）
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rich.console import Console
from rich.table import Table

con = Console()

# ---------------- matplotlib 中文字体（macOS 系统字体） ----------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

for _fp in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"]:
    if Path(_fp).exists():
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=_fp).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 常量 ----------------
SIZE = 640                      # 合成 patch 尺寸
COCO_PT = PROJECT_ROOT / "data" / "models" / "yolov8n_coco.pt"
SHIP_PT = PROJECT_ROOT / "data" / "models" / "yolov8n_ship.pt"
SYNTH_DIR = PROJECT_ROOT / "data" / "ships_synth"
YAML_PATH = SYNTH_DIR / "dataset.yaml"
REAL_MOSAIC = PROJECT_ROOT / "data" / "cogs" / "szbay_real_mosaic.tif"
OUT_DIR = PROJECT_ROOT / "data" / "output"

WATER_R = np.array([0.07, 0.05, 0.03, 0.02], dtype=np.float32)  # B2 B3 B4 B8


# ---------------- 合成船舶数据生成 ----------------
def load_mosaic_rgb_stretch() -> tuple[np.ndarray, float, float]:
    """读真实 mosaic → (uint8 RGB, 拉伸下界, 拉伸上界)。
    用真实 mosaic 的全局 2%-98% 分位作为合成数据的固定拉伸参数，
    保证合成 patch 与真实 mosaic 的明暗口径一致（域一致性）。"""
    import rasterio
    with rasterio.open(REAL_MOSAIC) as ds:
        bands = ds.read().astype(np.float32) / 10000.0
    rgb = bands[[2, 1, 0]]                      # B4 B3 B2 → R G B
    lo, hi = np.percentile(rgb, [2, 98])
    rgb8 = (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")
    return np.transpose(rgb8, (1, 2, 0)), lo, hi


def make_ship_scene(rng: np.random.Generator, lo: float, hi: float,
                    n_ships=(3, 8)) -> tuple[np.ndarray, list, np.ndarray]:
    """生成一景"水面 + 船舶"合成场景。

    返回 (bands[4,SIZE,SIZE] float32 反射率, boxes[(x1,y1,x2,y2)...], rgb8 uint8)。
    船长为 10-45px（10m/像素 ≈ 100-450m 中大型船舶），渔船级（2-4px）有意不生成，
    作为"分辨率极限"红旗的对照。"""
    size = SIZE
    bands = np.zeros((4, size, size), dtype=np.float32)
    for c in range(4):
        bands[c] = WATER_R[c] + rng.normal(0, 0.006, (size, size))
    # 水面波纹（低频正弦叠加，模拟真实水纹）
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    wave = 0.015 * np.sin(2 * np.pi * (xx + yy) / 50.0) * np.sin(2 * np.pi * xx / 31.0)
    for c in range(4):
        bands[c] += wave * (1 + 0.4 * c)

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    boxes = []
    n = int(rng.integers(*n_ships))
    for _ in range(n):
        length = rng.uniform(10, 45)            # 船长 px
        width = length / rng.uniform(3.0, 5.0)  # 长宽比 3:1 ~ 5:1
        a, b = length / 2, width / 2
        cx, cy = rng.uniform(35, size - 35), rng.uniform(35, size - 35)
        theta = rng.uniform(0, np.pi)           # 船头朝向
        hull_r = np.array([0.35, 0.38, 0.40, 0.30]) + rng.uniform(-0.06, 0.06, 4)

        dx, dy = x - cx, y - cy
        ax = dx * np.cos(theta) + dy * np.sin(theta)     # 沿船轴距离
        tr = -dx * np.sin(theta) + dy * np.cos(theta)    # 横向距离
        in_hull = (ax / a) ** 2 + (tr / b) ** 2 <= 1.0
        # 尾迹：船尾后方 1.4 倍船长，亮度随距离衰减，模拟航迹
        in_wake = (ax < -a) & (ax > -a - 1.4 * length) & (np.abs(tr) < 1.6 * b)
        wake_taper = np.clip((-ax - a) / (1.4 * length), 0, 1)
        wake_r = WATER_R[:, None, None] + 0.06 * (1 - wake_taper)[None] * (1 + rng.normal(0, 0.3, (1, size, size)))
        for c in range(4):
            bands[c][in_hull] = hull_r[c] + rng.normal(0, 0.01, in_hull.sum())
            bands[c][in_wake] = wake_r[c][in_wake]

        # 轴对齐包围盒（旋转椭圆的 AABB），裁剪到图内
        hw = a * abs(np.cos(theta)) + b * abs(np.sin(theta))
        hh = a * abs(np.sin(theta)) + b * abs(np.cos(theta))
        x1, y1 = max(0, cx - hw), max(0, cy - hh)
        x2, y2 = min(size - 1, cx + hw), min(size - 1, cy + hh)
        if x2 - x1 >= 5 and y2 - y1 >= 3:       # 过滤太小/出界的船
            boxes.append((x1, y1, x2, y2))

    bands = np.clip(bands, 0.0, 0.6)
    rgb8 = (np.clip((bands[[2, 1, 0]] - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
    rgb8 = np.transpose(rgb8, (1, 2, 0)).astype("uint8")
    return bands, boxes, rgb8


def write_yolo_dataset(n_train: int, n_val: int) -> None:
    """生成合成数据集到 data/ships_synth/（YOLO 格式：images + labels + yaml）。"""
    from PIL import Image
    rng = np.random.default_rng(42)
    _, lo, hi = load_mosaic_rgb_stretch()
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = SYNTH_DIR / "images" / split
        lbl_dir = SYNTH_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _, boxes, rgb8 = make_ship_scene(rng, lo, hi, n_ships=(3, 8) if split == "train" else (2, 5))
            name = f"{split}_{i:04d}"
            Image.fromarray(rgb8).save(img_dir / f"{name}.png")
            with open(lbl_dir / f"{name}.txt", "w") as f:
                for (x1, y1, x2, y2) in boxes:
                    xc, yc = (x1 + x2) / 2 / SIZE, (y1 + y2) / 2 / SIZE
                    w, h = (x2 - x1) / SIZE, (y2 - y1) / SIZE
                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    YAML_PATH.write_text(
        f"path: {SYNTH_DIR}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: ship\n"
    )
    con.print(f"[bold green]合成数据集[/] {n_train} train + {n_val} val → [bold]{SYNTH_DIR}[/]")


def load_coco_model():
    """加载 COCO 预训练 yolov8n；权重不在 data/models 时从 cwd 挪入。"""
    if not COCO_PT.exists():
        tmp = PROJECT_ROOT / "yolov8n.pt"
        if not tmp.exists():
            YOLO("yolov8n.pt")          # 触发自动下载到 cwd
        shutil.copy(tmp, COCO_PT)
    return YOLO(str(COCO_PT))


# ---------------- 推理辅助 ----------------
def predict_rgb(model, rgb8: np.ndarray, imgsz: int, conf: float = 0.25):
    """rgb8: (H,W,3) uint8 → (boxes xyxy, confs, clss, names)"""
    r = model.predict(rgb8, imgsz=imgsz, conf=conf, device="mps", verbose=False)[0]
    names = r.names
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), names
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int), names)


def plot_detections(rgb8, boxes, confs, clss, names, out_path, title, gt_boxes=None):
    """画检测框：GT 绿色（如有），预测红色 + 类别/置信度。"""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(rgb8)
    if gt_boxes is not None:
        for (x1, y1, x2, y2) in gt_boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="lime", lw=1.4))
    for (x1, y1, x2, y2), cf, cl in zip(boxes, confs, clss):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor="red", lw=1.6))
        label = f"{names[cl]} {cf:.2f}"
        ax.text(x1, max(0, y1 - 4), label, color="red", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------- 主流程 ----------------
def analyze_real_detections(boxes: np.ndarray, confs: np.ndarray) -> tuple[int, int, int, int, float, float, float]:
    """对真实 mosaic 检测结果做后置分析：水上/陆/云占比 + 框型统计。
    用 NDWI>0.05 划水域（复用 W2 阈值）；真船应细长，长宽比中位 > 2.5。
    返回 (n_water, n_land, n_cloud, n_ship_like, med_w, med_h, med_ratio)。"""
    import rasterio
    with rasterio.open(REAL_MOSAIC) as ds:
        bands = ds.read().astype(np.float32) / 10000.0
    ndwi = (bands[1] - bands[3]) / (bands[1] + bands[3] + 1e-6)
    water = ndwi > 0.05
    valid = bands.sum(0) > 0.01
    n_w = n_l = n_c = 0
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        r = water[y1:y2, x1:x2]
        v = valid[y1:y2, x1:x2]
        if r.size == 0 or v.sum() == 0:
            n_c += 1
            continue
        wr = (r & v).sum() / v.sum()
        if wr > 0.5:
            n_w += 1
        elif wr < 0.2:
            n_l += 1
        else:
            n_c += 1
    if len(boxes) == 0:
        return n_w, n_l, n_c, 0, 0.0, 0.0, 0.0
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    ratios = np.maximum(ws / np.maximum(hs, 1), hs / np.maximum(ws, 1))
    return n_w, n_l, n_c, int((ratios > 2.5).sum()), float(np.median(ws)), float(np.median(hs)), float(np.median(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="快速验证模式")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640, help="训练输入尺寸")
    args = ap.parse_args()

    fast = args.fast
    n_train, n_val = (20, 8) if fast else (120, 30)
    epochs = args.epochs or (3 if fast else 40)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con.print(f"[bold]第7月 W4：YOLOv8 船舶检测[/] | device={device} | "
              f"train={n_train} val={n_val} epochs={epochs}")

    # 1) 合成数据集
    write_yolo_dataset(n_train, n_val)

    # 2) 零样本测试：COCO 预训练 → 真实 mosaic（域差距红旗）
    con.print("\n[bold cyan]步骤 2：COCO 预训练零样本检测真实 mosaic（预期域差距红旗）[/]")
    rgb_real, _, _ = load_mosaic_rgb_stretch()
    coco = load_coco_model()
    boxes0, confs0, clss0, names0 = predict_rgb(coco, rgb_real, imgsz=1280, conf=0.25)
    n_boat = int((clss0 == 8).sum()) if len(clss0) else 0
    plot_detections(rgb_real, boxes0, confs0, clss0, names0,
                    OUT_DIR / "yolo_zero_shot.png",
                    "零样本：COCO 预训练 yolov8n 直接检测真实 Sentinel-2 mosaic（域差距测试）")
    con.print(f"  COCO 检测总数 [bold]{len(clss0)}[/]，其中 boat(COCO-8) [bold]{n_boat}[/] 个"
              f"（类别分布：{[names0[c] for c in set(clss0.tolist())] if len(clss0) else '无'}）")

    # 3) 微调
    con.print(f"\n[bold cyan]步骤 3：合成数据微调 yolov8n（{epochs} epochs）[/]")
    model = load_coco_model()
    model.train(data=str(YAML_PATH), epochs=epochs, imgsz=args.imgsz, device=device,
                batch=8, workers=0, amp=False, seed=0, verbose=False,
                project=str(PROJECT_ROOT / "data" / "models" / "yolo_runs"),
                name="ship", exist_ok=True, plots=False, val=True)
    best_pt = PROJECT_ROOT / "data" / "models" / "yolo_runs" / "ship" / "weights" / "best.pt"
    shutil.copy(best_pt, SHIP_PT)
    trained = YOLO(str(best_pt))

    # 4) 合成 val 评估 + 可视化
    con.print("\n[bold cyan]步骤 4：合成验证集评估[/]")
    metrics = trained.val(data=str(YAML_PATH), imgsz=args.imgsz, device=device,
                          verbose=False, plots=False)
    con.print(f"  [bold]mAP50 = {metrics.box.map50:.3f}[/] | mAP50:95 = {metrics.box.map:.3f} "
              f"| precision = {metrics.box.mp:.3f} | recall = {metrics.box.mr:.3f}")

    # 可视化：3 张 val 图 GT vs 预测
    rng = np.random.default_rng(7)
    _, lo, hi = load_mosaic_rgb_stretch()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k in range(3):
        _, boxes, rgb8 = make_ship_scene(rng, lo, hi, n_ships=(2, 5))
        pb, pc, pcl, _ = predict_rgb(trained, rgb8, imgsz=args.imgsz, conf=0.25)
        ax = axes[k]
        ax.imshow(rgb8)
        for (x1, y1, x2, y2) in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="lime", lw=1.3))
        for (x1, y1, x2, y2), cf in zip(pb, pc):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="red", lw=1.5))
            ax.text(x1, max(0, y1 - 4), f"{cf:.2f}", color="red", fontsize=8)
        ax.set_title(f"val 样例 {k + 1}：绿=GT 红=预测")
        ax.axis("off")
    fig.suptitle("合成验证集：YOLOv8 船舶检测（绿色 GT / 红色预测）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yolo_synth_val.png", dpi=130)
    plt.close(fig)

    # 5) 真实 mosaic 推理（分辨率极限红旗）
    con.print("\n[bold cyan]步骤 5：微调模型检测真实 mosaic[/]")
    boxes1, confs1, clss1, names1 = predict_rgb(trained, rgb_real, imgsz=1280, conf=0.25)
    plot_detections(rgb_real, boxes1, confs1, clss1, names1,
                    OUT_DIR / "yolo_real_mosaic.png",
                    "微调后：yolov8n 检测真实 Sentinel-2 mosaic 船舶")
    con.print(f"  真实 mosaic 检出船舶 [bold]{len(boxes1)}[/] 个"
              f"（conf 均值 {confs1.mean():.3f}）" if len(boxes1) else "  真实 mosaic 未检出船舶（conf>0.25）")

    # 后置分析：水上/陆/云占比 + 框型（真船应细长）
    n_w, n_l, n_c, n_sl, mw, mh, mr = analyze_real_detections(boxes1, confs1)
    con.print(f"  位置 [bold]水上 {n_w}[/] / 陆 {n_l} / 云或混合 {n_c}"
              f"（水上 {n_w / max(1, len(boxes1)):.1%}）")
    con.print(f"  框型 [bold]中位 {mw:.0f}x{mh:.0f} px[/]，长宽比中位 {mr:.2f}"
              f"，细长（>2.5）{n_sl} 个" if len(boxes1) else "")

    # 6) 诚实红旗
    con.print("""
[bold yellow]诚实红旗（遥感目标检测的两个核心约束）[/]
① 域差距：COCO 预训练（自然影像）在卫星影像上几乎检不到船 —— 目标检测同样
   逃不过域差距，必须像 W2 一样用目标域数据（此处为合成+后续真实伪标签）微调。
② 分辨率极限：Sentinel-2 是 10m/像素，合成数据只生成了 10-45px（100-450m）
   的中大型船舶；渔船级（20-40m = 2-4px）在 10m 影像上低于一个像元对，
   任何检测器都无法可靠识别 —— 这是传感器分辨率的硬边界，不是模型问题。
③ 合成→真实：合成 mAP50 高只说明合成域内学得好；真实 mosaic 上模型常把
   亮云边缘、白屋顶识别为"船"（亮色 + 暗背景的捷径），后置分析见结果表。
   下一步（第8月 W1）可用真实船舶标注（如 SpaceNet 航运数据集）做外部验证。""")

    # 汇总表
    tbl = Table(title="W4 结果汇总")
    tbl.add_column("阶段", style="cyan")
    tbl.add_column("指标", style="bold")
    tbl.add_column("数值")
    tbl.add_row("COCO 零样本（真实）", "检出船数", f"{n_boat}（COCO-8）")
    tbl.add_row("合成微调（val）", "mAP50", f"{metrics.box.map50:.3f}")
    tbl.add_row("合成微调（val）", "mAP50:95", f"{metrics.box.map:.3f}")
    tbl.add_row("微调后（真实）", "检出船数", f"{len(boxes1)}")
    tbl.add_row("微调后（真实）", "水上/陆/云", f"{n_w}/{n_l}/{n_c}（水上 {n_w / max(1, len(boxes1)):.1%}）")
    tbl.add_row("微调后（真实）", "细长框（>2.5）", f"{n_sl} / {len(boxes1)}")
    tbl.add_row("全流程耗时", "—", f"见运行时长")
    con.print(tbl)


if __name__ == "__main__":
    main()
