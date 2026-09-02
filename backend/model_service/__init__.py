"""模型服务：把第 7-8 月训练的 U-Net / YOLOv8 封装成可复用的 Python API。

设计原则：
  - 单例（_loaded 字典）→ 进程启动后只 load 一次，多请求共享
  - 复用 scripts/ 下已实现的推理函数（不重造轮子）
  - 输入统一接受"已入库 COG 路径"（避免大文件传输，data/cogs/ 已就绪）
  - 路径安全：只允许 data/cogs/ 下文件（防任意文件读取）
  - 输出：base64 PNG + JSON 统计 → 前端直接 <img src="data:image/png;base64,...">

依赖：scripts/ 需在 sys.path（__init__.py 启动时注入）。
"""
import sys
from pathlib import Path

# model_service 在 backend/ 下，scripts/ 是兄弟目录 → 加进 sys.path 让推理函数可 import
# 必须在任何子模块 import 之前注入；本 __init__.py 优于子模块先执行。
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# 把子模块的常用符号提到包顶层，方便 from backend.model_service import ...
from .loader import get_unet, get_yolo, get_device, list_loaded  # noqa: E402, F401
from . import segment, change, detect, jobs  # noqa: E402, F401
