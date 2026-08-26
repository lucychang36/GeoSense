"""SQL 生成 + 执行工具 —— 把自然语言变成 PostGIS 查询结果。

核心概念：工具内部的"二次 LLM 调用"
- 外层 ReAct Agent 负责"决定要不要查数据"；真正"写 SQL"由本工具内的
  一次专用 LLM 调用完成（temperature=0 保证确定性，prompt 里塞入 schema 提示）。
- 这种分层：外层管意图，内层管精确的代码生成，各自用各自最合适的 prompt，
  比让主 Agent 一把梭更可靠、更易调试。

核心概念：生成 → 清洗 → 校验 → 执行的流水线
- LLM 输出不可信：可能带 ```sql 代码块围栏、可能有多余解释、可能越权。
- 所以：清洗输出 → sql_validator 安全校验 → 参数化执行，三道关缺一不可。
"""
from __future__ import annotations

import json
import re

from ..core.llm import LLMClient
from ..db.connection import query, rows_to_geojson
from ..db.schema import SCHEMA_HINT
from ..db.sql_validator import SQLValidationError, validate

SQL_SYSTEM_PROMPT = (
    "你是 PostGIS 专家。根据用户问题与给定的表结构，只输出一条 SELECT 语句。\n"
    "规则：\n"
    "1. 只输出 SQL 本身：不要解释、不要代码块围栏、不要分号结尾；\n"
    "2. 只使用提供的表；几何输出必须用 ST_AsGeoJSON(geom) AS geom；\n"
    "3. 距离用 geography（单位米），空间关系用 ST_Within / ST_Intersects / ST_DWithin；\n"
    "4. 单条查询、只读。"
)


def _extract_sql(text: str) -> str:
    """清洗模型输出：去掉 ```sql 代码块围栏、结尾分号与首尾空白。"""
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text.rstrip(";").strip()


def spatial_sql(question: str) -> str:
    """自然语言 → 生成 PostGIS SQL → 安全校验 → 执行 → 返回结果。

    参数 question : 要回答的空间问题，如"南山区内有多少所学校"。
    返回 JSON 字符串：{sql, count, geojson | rows}。
    - 结果含 geom 列 → 附带 geojson（前端据此自动画图）；
    - 校验/执行失败 → 返回 error 字段（Agent 会看到并转述）。
    """
    llm = LLMClient()
    content, _ = llm.chat(
        [
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": f"表结构：\n{SCHEMA_HINT}\n\n问题：{question}"},
        ],
        temperature=0,
    )
    sql = _extract_sql(content)

    try:
        sql = validate(sql)          # 安全校验（缺 LIMIT 会自动补）
    except SQLValidationError as exc:
        return json.dumps({"sql": sql, "error": str(exc)}, ensure_ascii=False)

    try:
        rows = query(sql)
    except Exception as exc:  # noqa: BLE001 —— DB 执行错误也要变成可回喂的结果
        return json.dumps({"sql": sql, "error": f"{type(exc).__name__}: {exc}"},
                          ensure_ascii=False)

    # 有几何列 → 拼 GeoJSON 供前端画图；否则返回普通行
    if rows and "geom" in rows[0]:
        return json.dumps(
            {"sql": sql, "count": len(rows), "geojson": rows_to_geojson(rows)},
            ensure_ascii=False,
        )
    return json.dumps({"sql": sql, "count": len(rows), "rows": rows},
                      ensure_ascii=False)
