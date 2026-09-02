"""单例模型加载器：进程内只 load 一次，跨请求复用（MPS 显存友好）。

使用：
  from backend.model_service import get_unet, get_yolo, get_device
  model = get_unet()   # 首次调用加载，后续直接返回缓存
"""
from __future__ import annotations

import sys
from pathlib import Path

# model_service 在 backend/ 下，scripts/ 是兄弟目录 → 加进 sys.path 让推理函数可 import
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402
from unet_segmentation import UNet, N_CLASSES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
COGS_DIR = PROJECT_ROOT / "data" / "cogs"

_loaded: dict[str, object] = {}


def get_device() -> str:
    """统一设备选择：MPS > CUDA > CPU。"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_unet(weight: str = "unet_finetuned.pt") -> UNet:
    """单例加载 U-Net（默认用 W2 真实微调权重，真实域 mIoU 0.927）。"""
    key = f"unet::{weight}"
    if key not in _loaded:
        ckpt = MODELS_DIR / weight
        model = UNet(in_ch=4, out_ch=N_CLASSES, base=16).to(get_device())
        model.load_state_dict(torch.load(ckpt, map_location=get_device()))
        model.eval()
        _loaded[key] = model
    return _loaded[key]  # type: ignore[return-value]


def get_yolo(weight: str = "yolov8n_ship.pt"):
    """单例加载 YOLOv8（ultralytics 包）。"""
    key = f"yolo::{weight}"
    if key not in _loaded:
        from ultralytics import YOLO
        ckpt = MODELS_DIR / weight
        _loaded[key] = YOLO(str(ckpt))
    return _loaded[key]


def safe_cog_path(rel_or_name: str) -> Path:
    """把用户传入的 COG 路径规范化为 data/cogs/ 下的绝对路径。

    - 接受纯文件名（如 "szbay_real_20250727.tif"）→ 直接拼到 COGS_DIR
    - 接受 "data/cogs/foo.tif" → 取文件名拼到 COGS_DIR（防 ../ 越权）
    - 任何解析出 COGS_DIR 之外的路径 → 抛 ValueError
    - 文件不存在、是目录、空文件名 → 抛 FileNotFoundError / ValueError
      （目录不能当 COG 读，缺字段会原样溜进队列后才在 worker 报错，违反"边界在服务入口"）
    """
    name = Path(rel_or_name).name  # 剥掉目录前缀，只取文件名
    if not name:
        raise ValueError("cog 文件名不能为空")
    target = (COGS_DIR / name).resolve()
    if not str(target).startswith(str(COGS_DIR.resolve())):
        raise ValueError(f"路径越权：{rel_or_name} 不在 data/cogs/ 下")
    if not target.exists():
        raise FileNotFoundError(f"COG 不存在：{target}")
    if not target.is_file():
        raise FileNotFoundError(f"不是有效 COG 文件（是目录）：{target}")
    return target


def list_loaded() -> dict:
    """健康检查端点用：列出已加载模型。"""
    return {
        "device": get_device(),
        "loaded": {k: str(type(v).__name__) for k, v in _loaded.items()},
    }
