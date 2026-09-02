"""第10月 W1：多 Agent 协作 —— 共享状态 TypedDict。

设计要点
--------
- total=False：所有字段都是 NotRequired，节点只更新自己关心的字段（LangGraph 会合并）
- messages 用 LangChain 消息列表不是我们这次 W1 的主通信通道 —— 节点用专用字段
  （plan / cog_a / analysis_result）直传结果，messages 仅作为可读审计流
- current_step + step_log 让 supervisor/外部观测当前进度
"""
from __future__ import annotations

from typing import NotRequired

from typing_extensions import TypedDict


class MultiAgentState(TypedDict, total=False):
    # ---- 输入 ----
    user_query: str                                # 用户原始问题

    # ---- Planner 产出 ----
    plan: NotRequired[dict]                        # {task_type, cog_a, cog_b, reason}
    planner_raw: NotRequired[str]                  # LLM 原始输出（调试用）

    # ---- Data Agent 产出 ----
    cog_a: NotRequired[str]                        # 较早 COG 文件名
    cog_b: NotRequired[str]                        # 较晚 COG 文件名
    cogs_listed: NotRequired[list[str]]            # 候选 COG 列表（供审计）

    # ---- Analysis Agent 产出 ----
    analysis_result: NotRequired[dict]             # {method, change_ratio, change_px, valid_px, ...}
    analysis_error: NotRequired[str]

    # ---- Cartography Agent 产出 ----
    map_path: NotRequired[str]                     # 生成的图绝对路径
    map_bytes: NotRequired[int]                    # PNG 字节数（自检）

    # ---- Supervisor 产出 ----
    final_answer: NotRequired[str]                 # 给用户的最终回复

    # ---- 观测 ----
    current_step: NotRequired[str]                 # 当前节点名
    step_log: NotRequired[list[str]]               # 节点执行流水（拼装给用户看）
