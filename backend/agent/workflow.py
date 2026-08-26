"""多步推理工作流 —— 手写 LangGraph StateGraph。

核心概念：从"自动 ReAct"到"显式工作流"
- W1/W2 的 create_react_agent 是官方预置的黑盒：LLM 自己决定每一步调什么。
- 本周手写 StateGraph：由我们定义"有哪些节点、按什么顺序、什么条件走哪条边"。
  好处：流程可控、可预测、可审计 —— 生产系统里复杂分析需要的是确定性，不是自由发挥。

核心概念：State（状态）
- 图在节点间传递一个共享状态。这里状态有 4 个字段：
  messages（对话，用 add_messages 累加）、intent（解析出的意图）、
  results（各步骤结果）、errors（错误列表）。

核心概念：条件边（conditional edge）
- 一个节点执行完，按函数返回值决定下一个节点：
  意图是"周边查询"→ query_node；是"缓冲区"→ buffer_node；否则→ handle_error。

核心概念：错误处理（两层）
1. parse 节点：LLM 返回的 JSON 解析失败 → 兜底为 error 意图，不崩溃；
2. 工具节点：工具抛异常 → try/except 捕获，记入 errors，流程继续走到 summarize。
   —— "优雅降级"：宁可输出带错误说明的报告，也不让 Agent 中途崩掉。
"""
from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..core.config import llm_config
from .spatial_tools import buffer_analysis, data_retrieval, spatial_query


class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: dict
    results: dict
    errors: list


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=llm_config.base_url, api_key=llm_config.api_key,
        model=llm_config.model, temperature=0,
    )


# ---------------------------------------------------------------------------
# 节点 1：parse —— LLM 把自然语言解析成结构化意图
# ---------------------------------------------------------------------------
PARSE_PROMPT = (
    "你是意图解析器。把用户的空间分析请求解析成 JSON，只输出 JSON，不要其他文字。\n"
    "字段：action（取值 \"poi_query\" 或 \"buffer\"，其他任务填 \"other\"）、"
    "lon、lat、radius_m、poi_type。\n"
    "示例：\"查询市民中心周边3公里的学校\" → "
    '{"action":"poi_query","lon":114.0579,"lat":22.5431,"radius_m":3000,"poi_type":"学校"}'
)


def parse_node(state: WorkflowState) -> dict:
    """解析意图，JSON 解析失败时优雅降级为 error 意图。"""
    user = state["messages"][-1].content if state["messages"] else ""
    resp = _model().invoke([SystemMessage(content=PARSE_PROMPT), HumanMessage(content=user)])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    # 剥掉可能的 markdown 代码块围栏
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        intent = {"action": "error", "reason": f"意图解析失败: {raw[:60]}"}
    return {"intent": intent}


# ---------------------------------------------------------------------------
# 条件路由：按意图 action 决定下一步
# ---------------------------------------------------------------------------
def route(state: WorkflowState) -> str:
    action = state["intent"].get("action", "error")
    if action == "poi_query":
        return "query_node"
    if action == "buffer":
        return "buffer_node"
    return "handle_error"


# ---------------------------------------------------------------------------
# 节点 2a：周边查询（含工具 try/except）
# ---------------------------------------------------------------------------
def query_node(state: WorkflowState) -> dict:
    intent = state["intent"]
    try:
        overview = data_retrieval()
        hits = spatial_query(intent["lon"], intent["lat"],
                             intent.get("radius_m", 3000),
                             poi_type=intent.get("poi_type", ""))
        return {"results": {"overview": overview, "hits": hits}}
    except Exception as exc:  # noqa: BLE001 —— 工具边界兜底
        return {"errors": [f"查询失败: {exc}"]}


# ---------------------------------------------------------------------------
# 节点 2b：缓冲区分析（含工具 try/except）
# ---------------------------------------------------------------------------
def buffer_node(state: WorkflowState) -> dict:
    intent = state["intent"]
    try:
        result = buffer_analysis(intent["lon"], intent["lat"],
                                 intent.get("radius_m", 1000))
        return {"results": {"buffer": result}}
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"缓冲区分析失败: {exc}"]}


# ---------------------------------------------------------------------------
# 节点 2c：错误降级
# ---------------------------------------------------------------------------
def handle_error(state: WorkflowState) -> dict:
    intent = state["intent"]
    if intent.get("action") == "other":
        reason = "该任务超出当前工作流支持的能力范围"
    else:
        reason = intent.get("reason", "未知")
    return {"errors": [f"无法处理该请求（{reason}）"]}


# ---------------------------------------------------------------------------
# 节点 3：summarize —— LLM 汇总成最终报告（有错误也照常汇报）
# ---------------------------------------------------------------------------
SUMMARIZE_PROMPT = (
    "你是 GeoSense。根据【分析结果】和【错误】向用户汇报。\n"
    "1. 有结果就给出关键数值和结论；2. 有错误要如实说明哪里出了问题；"
    "3. 简洁、专业。"
)


def summarize_node(state: WorkflowState) -> dict:
    payload = json.dumps({
        "结果": state.get("results", {}),
        "错误": state.get("errors", []),
    }, ensure_ascii=False)
    resp = _model().invoke([
        SystemMessage(content=SUMMARIZE_PROMPT),
        HumanMessage(content=payload),
    ])
    return {"messages": [AIMessage(content=resp.content)]}


# ---------------------------------------------------------------------------
# 组装图
# ---------------------------------------------------------------------------
def build_workflow():
    g = StateGraph(WorkflowState)
    g.add_node("parse", parse_node)
    g.add_node("query_node", query_node)
    g.add_node("buffer_node", buffer_node)
    g.add_node("handle_error", handle_error)
    g.add_node("summarize", summarize_node)

    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", route, {
        "query_node": "query_node",
        "buffer_node": "buffer_node",
        "handle_error": "handle_error",
    })
    # 三条支线最终都汇入 summarize
    g.add_edge("query_node", "summarize")
    g.add_edge("buffer_node", "summarize")
    g.add_edge("handle_error", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow
