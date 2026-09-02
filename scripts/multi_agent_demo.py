#!/usr/bin/env python3
"""第10月 W1：多 Agent 协作 —— 端到端 demo。

场景：用户问"对比深圳湾 2023-07 vs 2025-07 水域变化并出图"
链路：Planner (LLM 拆解) → Data (选 COG) → Analysis (光谐差分) → Cartography (出图) → Supervisor (汇总)

学习点
------
1. 4 个 Agent 各司其职，**不通过 messages 隐式传数据**，而是写/读共享 State
   （plan / cog_a / analysis_result / map_path）—— 这是 LangGraph Multi-Agent 与
   AutoGen/CrewAI 的关键区别：状态显式 = 可观测/可检查点/可回滚
2. Supervisor 节点在末尾集中汇总，避免每个 worker 各自"给用户回话"造成的混乱
3. 失败有兜底：Planner LLM 失败 → 规则匹配；任意 worker 抛错 → 错误传给 supervisor 输出
4. W1 简化：worker 内部是确定性函数（不调 LLM 选工具），W2 起会换成 ReAct sub-agent

用法（项目 .venv，WorkBuddy 会话内加 env -u PYTHONPATH）：
  .venv/bin/python scripts/multi_agent_demo.py
  .venv/bin/python scripts/multi_agent_demo.py "深圳湾 2023-07 vs 2025-07 变化"
  .venv/bin/python scripts/multi_agent_demo.py --trace   # 打印每步 state 变化
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from backend.agent.multi_agent import get_multi_agent  # noqa: E402

con = Console()


def print_concepts() -> None:
    con.rule("[bold cyan]第10月 W1：多 Agent 协作（4 Agent + Supervisor）[/]")
    con.print("""
[bold yellow]核心概念速览[/]
1. [bold]共享 State vs 消息总线[/]：LangGraph Multi-Agent 用 TypedDict 共享状态
   （plan / cog_a / analysis_result / map_path），节点间通过读写 state 协作，
   不靠隐式 messages。优势：可观测、可检查点、可回滚、生产可调试。
2. [bold]Supervisor 模式[/]：一个集中节点负责最终汇总，避免 worker 各自回话导致上下文碎片化。
   W1 简化为线性顺序 + 末尾 supervisor；W2 起会引入 conditional_edges 做动态路由。
3. [bold]分工哲学[/]：按"角色"切，不是按"步骤"切。Planner 拆解、Data 取数、
   Analysis 计算、Cartography 出图 —— 每个 Agent 单一职责，
   可独立替换/升级（W2 换 ReAct sub-agent 时不影响其它节点）。
4. [bold]失败兜底[/]：Planner 调 LLM 失败 → 规则回退；任意 worker 抛错 → 状态字段
   `analysis_error` 携带异常信息，supervisor 输出对用户可见的失败原因。
5. [bold]何时不要多 Agent[/]：单步任务（如「查一下深圳 POI」）用单 Agent 就够，
   多 Agent 增加 LLM 调用、状态字段、调试复杂度 —— W1 末尾会明确标这条红旗。
""")


def run(query: str, trace: bool = False) -> dict:
    app = get_multi_agent()
    state_in = {"user_query": query, "step_log": []}
    if trace:
        # stream 模式逐节点打印 state 变化
        for event in app.stream(state_in):
            for node_name, update in event.items():
                con.print(f"[dim]── {node_name} ──[/]")
                for k, v in (update or {}).items():
                    val = repr(v)
                    if len(val) > 120:
                        val = val[:117] + "..."
                    con.print(f"    [yellow]{k}[/] = {val}")
        # 重新跑一遍拿最终 state（stream 已消费图）
        final = app.invoke(state_in)
    else:
        final = app.invoke(state_in)
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?",
                    default="对比深圳湾 2023-07 和 2025-07 的水域变化并出图",
                    help="用户问题")
    ap.add_argument("--trace", action="store_true", help="打印每步 state 变化")
    args = ap.parse_args()

    print_concepts()
    t0 = time.time()
    final = run(args.query, args.trace)
    dt = time.time() - t0

    # 总结
    con.rule("[bold green]最终回复[/]")
    con.print(Panel(final.get("final_answer", "（无）"), title="Supervisor", border_style="green"))

    # 关键指标表
    t = Table(title="多 Agent 协作指标")
    t.add_column("指标", justify="left")
    t.add_column("值", justify="right")
    plan = final.get("plan", {})
    res = final.get("analysis_result") or {}
    t.add_row("用户问题", final.get("user_query", "?")[:40])
    t.add_row("Planner 任务类型", plan.get("task_type", "—"))
    t.add_row("cog_a", final.get("cog_a", "—"))
    t.add_row("cog_b", final.get("cog_b", "—"))
    t.add_row("变化占比", f"{res.get('change_ratio', 0):.2%}" if res else "—")
    t.add_row("专题图", final.get("map_path", "—").split("/")[-1] if final.get("map_path") else "—")
    t.add_row("步骤数", str(len(final.get("step_log", []))))
    t.add_row("耗时", f"{dt:.2f}s")
    con.print(t)

    if final.get("analysis_error"):
        con.print(f"\n[red]错误：{final['analysis_error']}[/]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
