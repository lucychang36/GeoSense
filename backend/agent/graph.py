"""LangGraph Agent 构建 —— GeoSense 的"大脑"。

核心概念：ReAct（Reasoning + Acting）
- 传统 LLM 调用是一问一答；ReAct 让模型在"思考(Thought) → 行动(Action) →
  观察(Observation)"之间循环，直到得到最终答案。
- 这正是 W4 手写的那个 while 循环，只不过现在交给 LangGraph 用状态机管理。

核心概念：LangGraph 是什么？
- 把 Agent 建模成"图"：节点(node)= 处理步骤，边(edge)= 流转方向。
- create_react_agent 是官方预置的 ReAct 图：一个 agent 节点（LLM 决策）+
  一个 tools 节点（执行工具）+ 条件边（要不要调工具决定下一步）。
- 用图的好处：状态显式、可检查点/回滚、可扩展成多 Agent —— 为第10个月铺路。
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..core.config import llm_config
from .langchain_tools import GIS_TOOLS, SPATIAL_TOOLS

SYSTEM_PROMPT = (
    "你是 GeoSense，一名空间智能分析助手。\n"
    "涉及距离、缓冲区、坐标转换的计算，必须调用工具获取精确结果，"
    "禁止凭记忆估算数字。回答时附上关键数值和单位。"
)

SPATIAL_SYSTEM_PROMPT = (
    "你是 GeoSense，一名空间智能分析助手，擅长用空间工具完成分析任务。\n"
    "规则：\n"
    "1. 涉及地点周边查询（学校/医院等）、缓冲区、坐标转换、叠加分析、空间统计，"
    "必须调用对应工具获取真实结果，禁止编造地名、坐标或数量；\n"
    "2. 复杂任务先查数据（data_retrieval），需要查数据库真实数据或做空间关系分析时"
    "用 spatial_sql 让模型生成并执行 PostGIS 查询（其结果含 GeoJSON 会自动上图），"
    "最后用 map_generation 输出结果；\n"
    "3. 用户问两期遥感影像的对比变化（如\"深圳湾 2023 和 2025 变化\"\"两期影像水域变化\"）时，"
    "用 temporal_change_tool 计算真实变化，参数是 data/cogs/ 下的文件名"
    "（如 szbay_real_20230708.tif、szbay_real_20250727.tif），不要编造文件名；\n"
    "4. 用户想要制图/地图样式（\"做一张...图\"\"把...叠加到地图上\"\"给...上色\"\"标注...\"）时，"
    "用 text_to_map_tool，参数 query 传用户原话；其返回的 map_style 会在前端地图自动渲染，"
    "回答时只需解释样式决策，不要复述 JSON；\n"
    "5. 用户要求「生成分析报告」「出一份报告」「写报告」等报告类请求时，"
    "用 report_tool，参数 query 传用户原话（工具内部会跑完整分析管道，耗时较长属正常）；\n"
    "6. 回答时给出关键数值、单位，并说明用了哪些工具。"
)


def _make_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model=llm_config.model,
        temperature=0,  # 工具调用要确定性
    )


def build_agent():
    """构建 ReAct Agent（3 工具集，第3月 W1）。"""
    return create_react_agent(_make_model(), GIS_TOOLS, prompt=SYSTEM_PROMPT)


def build_spatial_agent():
    """构建多工具空间分析 Agent（7 工具集，第3月 W2）。"""
    return create_react_agent(_make_model(), SPATIAL_TOOLS, prompt=SPATIAL_SYSTEM_PROMPT)


_agent = None
_spatial_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def get_spatial_agent():
    global _spatial_agent
    if _spatial_agent is None:
        _spatial_agent = build_spatial_agent()
    return _spatial_agent
