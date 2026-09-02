"""YOLOv8 目标检测服务：COG 路径 → 检测框 JSON（conf/cls/bbox）+ 域差距后置分析。

复用 W4 yolo_detection.predict_rgb / analyze_real_detections（已含水域稳定性自检）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from yolo_detection import predict_rgb, analyze_real_detections, load_mosaic_rgb_stretch  # noqa: E402

from .loader import get_yolo, safe_cog_path  # noqa: E402


def detect_cog(cog_path: str, weight: str = "yolov8n_ship.pt",
               conf: float = 0.25, imgsz: int = 1280) -> dict:
    """对单景 COG 做船舶检测。

    返回：
      - detections: [{cls, conf, bbox:[x1,y1,x2,y2]}, ...]（conf >= conf_thr）
      - n:          检测框数
      - n_water:    落在水上的框数（NDWI > 0.05 的区域，>50% 像素算水上）
      - n_land:     落在陆地的框数
      - n_cloud:    落在云/边界/混合的框数
      - flip_ratio: 深水区翻成"船"的比例（域差距指标，越高越差）
      - median_ar:  检测框长宽比中位数（真船应 > 2.5；< 1.5 大量是误检）
    """
    p = safe_cog_path(cog_path)
    model = get_yolo(weight)
    rgb8 = load_mosaic_rgb_stretch_for(p)
    boxes, confs, clss, names = predict_rgb(model, rgb8, imgsz=imgsz, conf=conf)

    detections = []
    for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
        detections.append({
            "cls": names.get(int(k), str(int(k))),
            "conf": round(float(c), 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    n_water, n_land, n_cloud, n_slim, mw, mh, mr = analyze_real_detections(boxes, confs)
    return {
        "cog": str(p.relative_to(p.parents[1])),
        "weight": weight,
        "conf_thr": conf,
        "n": len(detections),
        "n_water": n_water,
        "n_land": n_land,
        "n_cloud": n_cloud,
        "slim_boxes": n_slim,
        "median_aspect": round(float(mr), 2),
        "median_size_px": [int(mw), int(mh)],
        "detections": detections,
    }


def load_mosaic_rgb_stretch_for(p: Path):
    """复用 yolo_detection 的 load_mosaic_rgb_stretch 思路，但接受任意 COG 路径。

    原函数只对 szbay_real_mosaic.tif 硬编码，本函数对任意 COG 都做 2%-98% 拉伸 → uint8 RGB。
    """
    with rasterio.open(p) as ds:
        b = ds.read().astype(np.float32) / 10000.0
    b = b[:4]                       # 只取前 4 波段（蓝/绿/红/近红）
    rgb = b[[2, 1, 0]]              # 选 R/G/B → (3,H,W)
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    H, W = rgb.shape[1:]
    return (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
