"""第10月 W1：多 Agent 协作图 —— 组合 4 Agent + Supervisor。

W1 架构：线性顺序图（planner → data → analysis → cartography → supervisor → END）。
"supervisor 模式"的精神体现在：
  1. supervisor 节点集中汇总各 agent 的产出（不是分散在 worker 里写答案）
  2. 各 agent 共享 State（plan / cog_a / analysis_result / map_path），而不是用 messages 隐式传递
  3. W2 起会加 conditional_edges 让 supervisor 动态路由（W1 简化：任务固定 4 步）

生产升级路径：把节点内 deterministic 函数替换为 ReAct sub-agent（每个 worker 内部
也用 LLM 选工具），state 共享保持不变。这是 LangGraph Multi-Agent Collaboration
模板的标准做法。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import (
    analysis_node,
    cartography_node,
    data_node,
    planner_node,
    report_worker,
    supervisor_node,
)
from .state import MultiAgentState


def build_multi_agent_graph():
    """编译并返回可调用的多 Agent 图。"""
    g = StateGraph(MultiAgentState)
    g.add_node("planner", planner_node)
    g.add_node("data", data_node)
    g.add_node("analysis", analysis_node)
    g.add_node("cartography", cartography_node)
    g.add_node("report", report_worker)          # 第11月 W1：管道第 5 个 worker
    g.add_node("supervisor", supervisor_node)

    # 顺序图：planner→data→analysis→cartography→report→supervisor→END
    g.add_edge(START, "planner")
    g.add_edge("planner", "data")
    g.add_edge("data", "analysis")
    g.add_edge("analysis", "cartography")
    g.add_edge("cartography", "report")
    g.add_edge("report", "supervisor")
    g.add_edge("supervisor", END)
    return g.compile()


_multi_agent_app = None


def get_multi_agent():
    """进程级单例多 Agent 图（首次调用编译，后续直接复用）。"""
    global _multi_agent_app
    if _multi_agent_app is None:
        _multi_agent_app = build_multi_agent_graph()
    return _multi_agent_app
