"""第9月 W3：异步推理任务的 HTTP 服务层（进程级任务队列 + job 轮询）。

设计决策
--------
- 复用 scripts/async_inference.py 的 AsyncQueue（学习脚本已实现：生产者-消费者、
  job 状态机、分块推理 + 进度回调），本模块只做两件事：
    1. 依赖注入：把队列里的模型加载换成 model_service.get_unet() 进程单例。
       （scripts 里每个任务自 load 一次权重是「独立演示」；服务里权重必须
       只 load 一次跨任务复用 —— 这就是依赖注入的价值：队列逻辑零改动。）
    2. 安全边界：cog 文件名走 loader.safe_cog_path 白名单（防 ../ 越权）；
       big_image 合成尺寸设上限（防误传超大值把进程内存打爆）。

生产升级路径（scripts docstring 已述）：AsyncQueue → Celery + Redis broker，
对外 API（submit / poll）不变 —— 接口屏蔽实现，与 W2 的 VectorStore 同一哲学。
"""
from __future__ import annotations

from threading import Lock
from typing import Any

# scripts/async_inference 顶层只 import 库（torch/rasterio/rich，不 load 权重）——
# 真实模型加载发生在 worker 执行任务那一刻（lazy），与 model_service 哲学一致。
from scripts.async_inference import AsyncQueue  # noqa: E402

from .loader import get_device, get_unet, safe_cog_path  # noqa: E402

_queue: AsyncQueue | None = None
_lock = Lock()

# big_image（合成大影像演示）防滥用上限
_MAX_SIZE = 8192
_ALLOWED_TILES = (256, 512, 1024)


def _svc_model_factory() -> tuple:
    """FastAPI 服务的模型工厂：返回进程单例 U-Net + 设备（跨任务复用，不重复 load）。"""
    device = get_device()
    return get_unet(), device


def get_queue() -> AsyncQueue:
    """进程级单例队列：首个 worker 线程在首次提交时创建并常驻（daemon 后台跑）。"""
    global _queue
    with _lock:
        if _queue is None:
            _queue = AsyncQueue(max_workers=1, model_factory=_svc_model_factory)
    return _queue


def submit(task_type: str, args: dict) -> dict:
    """提交异步任务 → 立即返回轻量 job 快照（status=queued），不阻塞等待结果。

    任务在后台 worker 线程分块推理，调用方用 poll(job_id) 轮询进度。
    """
    args = dict(args or {})
    if task_type == "segment":
        # 安全边界放在服务入口：只允许 data/cogs/ 下的现存文件
        cog = safe_cog_path(args.get("cog", ""))  # FileNotFoundError / ValueError 上抛
        args["cog"] = cog.name
    elif task_type == "big_image":
        size = int(args.get("size", 2048))
        tile = int(args.get("tile", 512))
        if size > _MAX_SIZE or tile not in _ALLOWED_TILES:
            raise ValueError(f"size ≤ {_MAX_SIZE}，tile ∈ {_ALLOWED_TILES}，收到 size={size} tile={tile}")
        args["size"], args["tile"] = size, tile
    else:
        raise ValueError(f"task_type 仅支持 segment | big_image，收到：{task_type!r}")

    queue = get_queue()
    job_id = queue.submit(task_type, args)
    return payload(queue.get(job_id), full=False)


def poll(job_id: str) -> dict | None:
    """轮询任务状态；job 不存在返回 None（路由层转 404）。

    带宽策略：任务 done/error 前只回轻量字段（status/progress/error）；
    完成后才附完整 result —— 因为 result 里含 base64 PNG（~100KB），
    高频轮询期反复传大包是浪费。
    """
    queue = get_queue()
    job = queue.get(job_id)
    if job is None:
        return None
    return payload(job, full=job.status in ("done", "error"))


def payload(job: Any, full: bool = False) -> dict:
    """Job dataclass → JSON 字典（不直接 asdict：result 可能含大 base64）。"""
    d: dict[str, Any] = {
        "job_id": job.id,
        "task_type": job.task_type,
        "status": job.status,                  # queued → running → done / error
        "progress": round(job.progress, 4),    # 0~1，由已完成块/总块数推进
        "created": round(job.created, 2),
        "finished": round(job.finished, 2) if job.finished else None,
    }
    if job.error:
        d["error"] = job.error
    if full and job.result:
        d["result"] = job.result
    return d
