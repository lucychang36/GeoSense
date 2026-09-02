"""第9月 W4：蓝藻监测任务的服务层封装（AsyncQueue task_executor 注入）。

设计要点
--------
- scripts/cyanobacteria_monitor.py 是独立可跑演示（顶部 print_concepts + main）；
  不能直接 import 它的 main（会触发 print 副作用）。这里只复用其纯函数：
  load_unet / detect_date / rgb_preview / render_overlay / plot_cyano。
- executor 签名 `(job, model_factory) -> dict`：从 services/job 拿 args + 模型工厂，
  把结果整理成 JSON 可序列化 dict（numpy 字段剥离、PIL/PNG 转 base64、stats 标量化）。
- 落盘 + base64 双轨：PNG 写到 data/output/cyano_jobs/ 留证据，base64 给前端直显。

注入位置：backend/model_service/jobs.py:get_queue() → AsyncQueue(task_executors={...})
"""
from __future__ import annotations

import base64
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from .loader import get_device  # noqa: E402

_JOBS_OUT = PROJECT_ROOT / "data" / "output" / "cyano_jobs"
_JOBS_OUT.mkdir(parents=True, exist_ok=True)


def _serializable_stats(d: dict) -> dict:
    """从 detect_date 返回的 stats 里剥掉 ndarray，只留标量字段。"""
    out: dict = {}
    for k, v in d.items():
        if k in ("bands", "water", "l1", "l2", "cloud", "valid"):
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def cyano_executor(job, model_factory) -> dict:
    """AsyncQueue task_executors 注入的函数：跑两期蓝藻检测 + 出图 + 序列化。

    job.args 期望字段：
      cog_a / cog_b  —— data/cogs/ 下的文件名（safe_cog_path 白名单在 service 校验）
      z (可选, 默认 2.5)   —— NIR 抬升稳健 Z 阈值
      ndvi (可选, 默认 0.0) —— 水面 NDVI 佐证阈值
    """
    from scripts import cyanobacteria_monitor as cm
    from .loader import safe_cog_path

    # ① 安全边界：cog 路径白名单（FileNotFoundError/ValueError 上抛，jobs 转 4xx）
    cog_a_path = safe_cog_path(job.args["cog_a"])
    cog_b_path = safe_cog_path(job.args["cog_b"])
    z_thr = float(job.args.get("z", 2.5))
    ndvi_thr = float(job.args.get("ndvi", 0.0))

    # ② 模型：服务注入 → 进程单例；脚本 fallback 自 load
    if model_factory is not None:
        model, device = model_factory()
    else:
        device = get_device()
        model = cm.load_unet(device)

    # ③ 两期检测（进度 0.5 / 1.0 单步推进；滑窗推理本身够重，不再细分）
    # 像素面积硬编码 100 m²（10m 分辨率，对应 Sentinel-2 L2A 默认）——
    # W4 演示场景固定 10m；如换高分数据或 UTM 重投影需按 transform 自适应计算
    # （不要照搬 WGS84 地理坐标的 ds.transform.a 直接相乘，那是度²量级）
    import rasterio  # 局部 import：避免启动时连累
    with rasterio.open(cog_a_path) as ds:
        bands_a = ds.read().astype(np.float32) / 10000.0
    _, _, st_a = cm.detect_date(model, device, cog_a_path.name, cog_a_path, z_thr, ndvi_thr)
    job.progress = 0.5
    with rasterio.open(cog_b_path) as ds:
        bands_b = ds.read().astype(np.float32) / 10000.0
    _, _, st_b = cm.detect_date(model, device, cog_b_path.name, cog_b_path, z_thr, ndvi_thr)
    job.progress = 1.0
    px_area = 100.0

    # ④ 双时相有效交集上算两期变化（学习点：避免云影伪差异淹没真信号）
    common = st_a["valid"] & st_b["valid"] & ~st_a["cloud"] & ~st_b["cloud"]
    l2a = st_a["l2"] & common
    l2b = st_b["l2"] & common
    added, gone, kept = int((l2b & ~l2a).sum()), int((l2a & ~l2b).sum()), int((l2a & l2b).sum())

    # ⑤ 出图落盘（留证据）+ base64（前端直显）
    rgb_a, rgb_b = cm.rgb_preview(bands_a), cm.rgb_preview(bands_b)
    ov_a = cm.render_overlay(rgb_a, st_a["water"], st_a["l1"], st_a["l2"],
                             st_a["valid"], st_a["cloud"])
    ov_b = cm.render_overlay(rgb_b, st_b["water"], st_b["l1"], st_b["l2"],
                             st_b["valid"], st_b["cloud"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = _JOBS_OUT / f"cyano__{cog_a_path.stem}__{cog_b_path.stem}__{ts}.png"
    cm.plot_cyano(cog_a_path.name, rgb_a, ov_a, cog_b_path.name, rgb_b, ov_b,
                  l2a, l2b, out_png)

    return {
        "device": device,
        "z_threshold": z_thr, "ndvi_threshold": ndvi_thr,
        "px_area_m2": round(px_area, 2),
        "cog_a": cog_a_path.name, "cog_b": cog_b_path.name,
        "stats_a": _serializable_stats(st_a),
        "stats_b": _serializable_stats(st_b),
        "delta": {
            "added_px": added, "gone_px": gone, "kept_px": kept,
            "net_px": added - gone,
            "added_km2": round(added * px_area / 1e6, 4),
            "gone_km2": round(gone * px_area / 1e6, 4),
            "net_km2": round((added - gone) * px_area / 1e6, 4),
        },
        "png_path": str(out_png.relative_to(PROJECT_ROOT)),
        "png_b64": base64.b64encode(out_png.read_bytes()).decode("ascii"),
        "png_bytes": out_png.stat().st_size,
    }
