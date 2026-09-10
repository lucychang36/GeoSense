"""U-Net 语义分割推理服务：COG 路径 → 分割 PNG（base64） + 类别面积统计。

复用 W1 unet_segmentation.infer_full / rgb_preview / render_mask，不重造轮子。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from unet_segmentation import infer_full, render_mask, rgb_preview   # noqa: E402
import torch  # noqa: E402

from .loader import get_unet, get_device, safe_cog_path  # noqa: E402


def _png_b64(arr: np.ndarray) -> str:
    """numpy 数组（H,W,3 uint8）→ base64 PNG 字符串。"""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_bands(cog_path: Path) -> np.ndarray:
    """读 COG → (4, H, W) float32 反射率（与训练输入一致）。"""
    with rasterio.open(cog_path) as ds:
        return ds.read().astype(np.float32) / 10000.0


def segment_cog(cog_path: str, weight: str = "unet_finetuned.pt",
                save_path: Path | None = None) -> dict:
    """对单景 COG 做三分类（水/城/植）分割。

    返回：
      - mask_png:  渲染后的 mask PNG（base64），前端可直接 <img src=...>
      - rgb_png:   真彩色 PNG（base64），便于对照
      - stats:     {class: {name, pixels, ratio}}  类别面积
      - shape:     (H, W)
      - weight:    使用的权重文件名
    """
    p = safe_cog_path(cog_path)
    bands = _load_bands(p)
    H, W = bands.shape[1:]

    model = get_unet(weight)
    x = torch.from_numpy(bands[None]).to(get_device())  # (1,4,H,W)
    with torch.inference_mode():   # 第12月W1：no_grad 的严格超集升级（省 version counter/view tracking）
        pred = infer_full(model, x, get_device())        # (H,W) int64

    # 类别面积统计（0=水 1=城 2=植）
    total = pred.size
    stats = {}
    for cls, name in enumerate(["水", "城", "植"]):
        n = int((pred == cls).sum())
        stats[name] = {"pixels": n, "ratio": round(n / total, 4)}

    rgb = rgb_preview(bands)
    rgb_png = _png_b64((rgb * 255).astype(np.uint8)) if rgb.max() <= 1.0 else _png_b64(rgb)
    mask_png = _png_b64((render_mask(pred) * 255).astype(np.uint8)) \
        if render_mask(pred).max() <= 1.0 else _png_b64(render_mask(pred))

    result = {
        "cog": str(p.relative_to(p.parents[1])),
        "shape": [H, W],
        "weight": weight,
        "stats": stats,
        "mask_png": mask_png,
        "rgb_png": rgb_png,
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(str(result["stats"]), encoding="utf-8")
    return result
