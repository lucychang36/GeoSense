"""第3月 W2 交付物：gis_agent.py —— 多工具空间分析 Agent

核心概念：多工具 Agent 的组合调用
- 工具多了，Agent 需要"编排"：先查数据 → 再查询/分析 → 最后出图。
- 一个自然语言任务（"查询深圳市民中心周边3公里的学校"）会被拆成一串工具调用，
  本脚本流式展示这条调用链 —— 这是"多步推理"的雏形，为 W3 的工作流铺路。

运行方式：
  .venv/bin/python scripts/gis_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from backend.agent.graph import get_spatial_agent
from backend.core.config import llm_config

console = Console()

DEMO_QUERIES = [
    # 组合查询：数据检索 + 空间查询（对应学习计划里程碑"查询A市B区周边3公里学校"）
    "查询深圳市民中心 (114.0579, 22.5431) 周边 3 公里内的所有学校，按距离排序列出",
    # 缓冲区 + 统计
    "以深圳市民中心 (114.0579, 22.5431) 为中心画 5 公里缓冲区，覆盖面积多大？",
]


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def run_query(agent, query: str) -> None:
    console.print(Panel(query, title="用户提问", border_style="blue"))
    prev = 0
    for step in agent.stream({"messages": [("user", query)]}, stream_mode="values"):
        msgs = step["messages"]
        for m in msgs[prev:]:
            _print_message(m)
        prev = len(msgs)


def _print_message(m) -> None:
    from langchain_core.messages import AIMessage, ToolMessage

    if isinstance(m, AIMessage):
        content = text_of(m.content).strip()
        if m.tool_calls:
            for call in m.tool_calls:
                args = str(call["args"])
                console.print(f"  [yellow]🤔 调用[/] {call['name']}({args[:80]})")
            if content:
                console.print(f"  [dim]💭 {content[:70]}[/]")
        elif content:
            console.print(Panel(content, title="最终回答", border_style="green"))
    elif isinstance(m, ToolMessage):
        console.print(f"  [cyan]👁 结果：{m.content[:100]}[/]")


def main() -> None:
    if not llm_config.available:
        console.print("[yellow]请先在 .env 配置 DEEPSEEK_API_KEY[/]")
        return
    agent = get_spatial_agent()
    for q in DEMO_QUERIES:
        console.rule(f"[bold cyan]{q[:22]}…")
        run_query(agent, q)
    console.rule("[bold green]W2 完成：多工具空间分析 Agent 跑通")


if __name__ == "__main__":
    main()
