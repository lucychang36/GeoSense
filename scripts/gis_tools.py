"""W4 交付物：gis_tools.py —— Function Calling 完整闭环

用户提问 → LLM 决策调工具 → 本地执行 GIS 计算 → 结果回喂 → LLM 自然语言作答

核心概念：Agent 主循环（Agent Loop）
  Function Calling 不是一次请求，而是一个"循环"：
    1. 把 问题 + 工具说明书(tools) 发给模型
    2. 模型若返回 tool_calls → 我们执行 → 把结果作为 role="tool" 消息追加进历史
    3. 再次请求模型 → 它基于工具结果生成最终回答（也可能继续调下一个工具）
  循环直到模型返回普通文本。这就是第3个月 ReAct Agent 的最小原型。

运行方式：
  .venv/bin/python scripts/gis_tools.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from backend.agent.tools import TOOL_SCHEMAS, execute_tool
from backend.core.config import llm_config
from backend.core.llm import LLMClient

console = Console()

SYSTEM = (
    "你是 GeoSense，空间智能分析助手。涉及距离、范围、坐标转换的计算，"
    "必须调用工具获取精确结果，禁止凭记忆估算数字。回答时附上关键数值和单位。"
)

# 三个演示问题，分别触发三个工具（坐标写在问题里，W4 先不接地理编码）
DEMO_QUERIES = [
    "深圳市民中心 (114.0579, 22.5431) 到腾讯滨海大厦 (113.9305, 22.5148) 的直线距离是多少？",
    "以深圳市民中心 (114.0579, 22.5431) 为中心画 3 公里缓冲区，覆盖面积多大？用的是哪个投影带？",
    "把 WGS84 坐标 (114.0579, 22.5431) 转成 EPSG:4547（CGCS2000 三度带第38带），结果是多少？",
]

MAX_ROUNDS = 5  # 防止模型陷入无限工具调用的保险丝


def run_agent_turn(client: LLMClient, query: str) -> None:
    """单轮提问的完整 Agent 循环。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]
    console.print(Panel(query, title="用户提问", border_style="blue"))

    for round_no in range(1, MAX_ROUNDS + 1):
        choice = client.chat_with_tools(messages, TOOL_SCHEMAS)

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # 模型要求调工具：先把它的"调用意图"原样存入历史（协议要求）
            messages.append(choice.message)
            for call in choice.message.tool_calls:
                name, args = call.function.name, call.function.arguments
                console.print(f"  [yellow]⚙ 第{round_no}轮 · 模型决策调用[/] {name}({args})")
                result = execute_tool(name, args)
                preview = result if len(result) <= 200 else result[:200] + "…(GeoJSON已截断)"
                console.print(f"  [dim]→ 本地执行结果：{preview}[/]")
                # 工具结果以 role="tool" 回喂，tool_call_id 关联调用
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
        else:
            # 模型给出最终回答，循环结束
            console.print(Panel(choice.message.content or "(空)",
                                title="GeoSense 最终回答", border_style="green"))
            return

    console.print("[red]达到最大轮数，强制终止（保险丝触发）")


def main() -> None:
    if not llm_config.available:
        console.print("[yellow]未配置 API Key，请先在 .env 填入 DEEPSEEK_API_KEY[/]")
        return
    client = LLMClient()
    for query in DEMO_QUERIES:
        console.rule(f"[bold cyan]{query[:24]}…")
        run_agent_turn(client, query)
    console.rule("[bold green]W4 完成：Function Calling 闭环跑通 —— 第1个月里程碑达成")


if __name__ == "__main__":
    main()
