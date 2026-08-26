"""GeoSense 全局配置。

核心概念：配置与代码分离
- API Key 属于"机密 + 环境相关"的信息，绝不能写死在代码里。
- 业界标准做法：代码读环境变量；本地开发用 .env 文件（python-dotenv 自动加载），
  生产环境用容器/K8s 注入环境变量。这就是著名的 "12-Factor App" 第三条原则。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = 本文件的上上级目录（backend/core/config.py -> 根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 加载根目录下的 .env（若存在），不会覆盖已存在的系统环境变量
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class LLMConfig:
    """LLM 接入配置（OpenAI 兼容协议三要素 + 生成参数）。

    base_url : API 端点。换厂商只需改这一个值（DeepSeek/通义/本地 Ollama 均兼容）。
    api_key  : 鉴权密钥。
    model    : 模型名。deepseek-chat = DeepSeek-V3 系列。
    temperature : 采样温度，0=确定性输出（适合工具调用/SQL生成），越大越发散（适合创意）。
    max_tokens  : 单次响应的最大输出长度（单位是 token，不是字符）。
    """

    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    @property
    def available(self) -> bool:
        """是否已配置可用的 API Key。"""
        return bool(self.api_key) and not self.api_key.startswith("sk-xxxx")


llm_config = LLMConfig()


@dataclass(frozen=True)
class DBConfig:
    """PostGIS 数据库连接配置。

    url : 连接串（DSN，Data Source Name）。一条 URL 描述数据库位置/账号/库名，
         代码里不散落主机、密码。读环境变量 DATABASE_URL，默认值与 docker-compose
         里的 postgis 服务一致（用户 geosense / 密码 geosense / 库 geosense）。
    """

    url: str = os.getenv(
        "DATABASE_URL", "postgresql://geosense:geosense@localhost:5432/geosense"
    )


db_config = DBConfig()
