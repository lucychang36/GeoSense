"""第3月 W3 交付物：workflow_agent.py —— 多步推理工作流 + 错误处理

三个演示（覆盖正常 + 错误两类路径）：
  ① 周边查询：parse → query_node → summarize（正常）
  ② 缓冲区：  parse → buffer_node → summarize（正常）
  ③ 越界任务：parse → handle_error → summarize（错误降级，不崩溃）

运行方式：
  .venv/bin/python scripts/workflow_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from backend.agent.workflow import get_workflow
from backend.core.config import llm_config

console = Console()

DEMO_QUERIES = [
    "查询深圳市民中心 (114.0579, 22.5431) 周边 3 公里内的学校",
    "以深圳市民中心 (114.0579, 22.5431) 为中心画 2 公里缓冲区，面积多大",
    "帮我生成深圳的热力图",  # 越界任务 → 触发错误降级
]


def run(workflow, query: str) -> None:
    console.print(Panel(query, title="用户提问", border_style="blue"))
    # stream_mode="updates"：每步只返回"哪个节点、更新了什么"，便于看清流程流转
    for step in workflow.stream(
        {"messages": [("user", query)], "intent": {}, "results": {}, "errors": []},
        stream_mode="updates",
    ):
        for node, update in step.items():
            _show(node, update)
    console.print()


def _show(node: str, update: dict) -> None:
    if node == "parse":
        console.print(f"  [cyan]① parse → 意图：{update.get('intent', {})}[/]")
    elif node in ("query_node", "buffer_node"):
        if update.get("results"):
            key = list(update["results"].keys())[0]
            console.print(f"  [yellow]② {node} → 执行成功（{key}）[/]")
        else:
            console.print(f"  [red]② {node} → 出错：{update.get('errors', [])}[/]")
    elif node == "handle_error":
        console.print(f"  [red]② handle_error → 降级：{update.get('errors', [])}[/]")
    elif node == "summarize":
        msg = update["messages"][-1].content
        console.print(Panel(msg, title="③ summarize 最终报告", border_style="green"))


def main() -> None:
    if not llm_config.available:
        console.print("[yellow]请先在 .env 配置 DEEPSEEK_API_KEY[/]")
        return
    wf = get_workflow()
    for q in DEMO_QUERIES:
        console.rule(f"[bold cyan]{q[:18]}…")
        run(wf, q)
    console.rule("[bold green]W3 完成：多步工作流 + 错误处理跑通")


if __name__ == "__main__":
    main()
