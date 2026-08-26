"""第3月 W1 交付物：basic_agent.py —— ReAct 模式 GIS 问答 Agent

核心概念：ReAct 轨迹（trace）
- 每轮问题，Agent 的完整决策过程是一串消息：
  AIMessage（思考+决定调工具）→ ToolMessage（工具结果）→ ... → AIMessage（最终答案）
- 本脚本用流式（stream）逐条打印这条轨迹，让你"看见" Agent 怎么想、怎么动手。

运行方式：
  .venv/bin/python scripts/basic_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from backend.agent.graph import get_agent
from backend.core.config import llm_config

console = Console()

DEMO_QUERIES = [
    "深圳市民中心 (114.0579, 22.5431) 到腾讯滨海大厦 (113.9305, 22.5148) 的直线距离是多少？",
    "以 (114.0579, 22.5431) 为中心画 2 公里缓冲区，覆盖多大面积？",
    "把 WGS84 坐标 (114.0579, 22.5431) 转成 EPSG:3857，结果是多少？",
]


def text_of(content) -> str:
    """兼容 langchain 1.x：content 可能是 str 或 content-block 列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def run_query(agent, query: str) -> None:
    console.print(Panel(query, title="用户提问", border_style="blue"))
    prev = 0
    # stream_mode="values"：每步返回累积的完整 state，我们只打印新增的消息
    for step in agent.stream({"messages": [("user", query)]}, stream_mode="values"):
        msgs = step["messages"]
        for m in msgs[prev:]:
            _print_message(m)
        prev = len(msgs)


def _print_message(m) -> None:
    from langchain_core.messages import AIMessage, ToolMessage

    if isinstance(m, AIMessage):
        content = text_of(m.content).strip()
        # 有 tool_calls = 这是"思考+行动"步（可能附带一句开场白）
        if m.tool_calls:
            for call in m.tool_calls:
                console.print(f"  [yellow]🤔 思考 → 决定调用[/] {call['name']}({call['args']})")
            if content:
                console.print(f"  [dim]💭 {content[:80]}[/]")
        # 无 tool_calls 且是最后一条 = 最终回答
        elif content:
            console.print(Panel(content, title="最终回答", border_style="green"))
    elif isinstance(m, ToolMessage):
        console.print(f"  [dim]👁 观察（工具结果）：{m.content[:120]}[/]")


def main() -> None:
    if not llm_config.available:
        console.print("[yellow]请先在 .env 配置 DEEPSEEK_API_KEY[/]")
        return
    agent = get_agent()
    for q in DEMO_QUERIES:
        console.rule(f"[bold cyan]{q[:20]}…")
        run_query(agent, q)
    console.rule("[bold green]W1 完成：ReAct Agent 跑通 —— 从工具循环到正式 Agent 框架")


if __name__ == "__main__":
    main()
