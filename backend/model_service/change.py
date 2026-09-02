"""变化检测服务：两景 COG 路径 → 变化 PNG（base64）+ 类别转换矩阵 + 变化占比。

支持两种方法：
  ① 光谱差分（baseline，无 GT 适用）
  ② U-Net 分类后比较（复用 W2 真实微调权重）

复用 real_change_detection 的工具函数（valid_mask / cloud_mask / pseudo_mask 等）。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

from unet_segmentation import infer_full, rgb_preview, render_mask  # noqa: E402
from real_change_detection import (  # noqa: E402
    cloud_mask, load_pair as _load_pair_real,    # noqa: F401 (保留下游可拓展)
    pseudo_mask, spectral_diff_change, valid_mask,
)

from .loader import get_unet, get_device, safe_cog_path  # noqa: E402


def _png_b64(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load(cog_path: str) -> tuple[np.ndarray, Path]:
    p = safe_cog_path(cog_path)
    with rasterio.open(p) as ds:
        bands = ds.read().astype(np.float32) / 10000.0
    return bands, p


def change_cog(cog_a: str, cog_b: str, method: str = "postclass",
               weight: str = "unet_finetuned.pt",
               threshold: float = 0.08) -> dict:
    """两景 COG 变化检测。

    method: "postclass"（U-Net 分类后比较，默认） | "spectral"（光谱差分 baseline）
    """
    bands_a, pa = _load(cog_a)
    bands_b, pb = _load(cog_b)
    if bands_a.shape != bands_b.shape:
        raise ValueError(
            f"两景尺寸不一致：{bands_a.shape} vs {bands_b.shape}。"
            " 变化检测需要像素级对齐（同 tile + 同 W/H + 同 bbox）。"
        )

    # 有效区 mask
    va, vb = valid_mask(bands_a), valid_mask(bands_b)
    ca, cb = cloud_mask(bands_a), cloud_mask(bands_b)
    mask_valid = va & vb & ~ca & ~cb

    change_mask = np.zeros(bands_a.shape[1:], dtype=np.uint8)
    pred_a = pred_b = None
    transition = None

    if method == "spectral":
        change_mask = spectral_diff_change(bands_a, bands_b, threshold=threshold)
    else:  # postclass
        model = get_unet(weight)
        with torch.no_grad():
            xa = torch.from_numpy(bands_a[None]).to(get_device())
            xb = torch.from_numpy(bands_b[None]).to(get_device())
            pred_a = infer_full(model, xa, get_device())
            pred_b = infer_full(model, xb, get_device())
        change_mask = (pred_a != pred_b).astype(np.uint8)
        # 类别转换矩阵（3×3）
        n_cls = 3
        transition = np.zeros((n_cls, n_cls), dtype=np.int64)
        for i in range(n_cls):
            for j in range(n_cls):
                transition[i, j] = int(((pred_a == i) & (pred_b == j) & mask_valid).sum())
        transition = transition.tolist()

    n_change = int((change_mask & mask_valid).sum())
    n_valid = int(mask_valid.sum())
    # 弱参考：伪标签变化（两时相 NDWI/NDVI 阈值法）
    pseudo_a = pseudo_mask(bands_a)
    pseudo_b = pseudo_mask(bands_b)
    both_known = (pseudo_a >= 0) & (pseudo_b >= 0) & mask_valid
    pseudo_change = ((pseudo_a != pseudo_b) & both_known).astype(np.uint8)

    # 变化图 base64（叠加到 RGB 上：变化处标红半透明）
    rgb_a = rgb_preview(bands_a)
    rgb_b = rgb_preview(bands_b)
    overlay = np.concatenate([rgb_a, rgb_b], axis=1)        # 左右并排
    vis = (overlay * 255).astype(np.uint8) if overlay.max() <= 1.0 \
        else overlay.astype(np.uint8)
    change_vis = (change_mask * 255).astype(np.uint8)

    return {
        "cog_a": str(pa.relative_to(pa.parents[1])),
        "cog_b": str(pb.relative_to(pb.parents[1])),
        "method": method,
        "weight": weight if method == "postclass" else None,
        "threshold": threshold if method == "spectral" else None,
        "shape": list(bands_a.shape[1:]),
        "n_valid": n_valid,
        "n_change": n_change,
        "change_ratio": round(n_change / max(n_valid, 1), 4),
        "n_change_pseudo": int((pseudo_change & mask_valid).sum()),
        "change_mask_png": _png_b64(change_vis),
        "rgb_png": _png_b64(vis),
        "transition": transition,           # 3x3 matrix [[w2w,w2u,w2v],[u2w,...],...]，仅 postclass
        "class_names": ["水", "城", "植"],
    }
