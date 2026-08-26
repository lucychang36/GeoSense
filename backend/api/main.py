"""FastAPI 后端 —— GeoSense 的 Web 接口层。

核心概念：SSE（Server-Sent Events，服务器推送事件）
- 普通 HTTP 是"一问一答"；SSE 让服务器把结果"一段一段"推给浏览器，
  浏览器边收边渲染 —— 这就是聊天界面"打字机效果"的实现。
- 与 W1 的 chat_stream 一脉相承，只是从"命令行打印"变成了"HTTP 流式推送"。

核心概念：接口分层
- /api/chat   —— 对话接口（调用 LangGraph Agent，流式返回工具调用链 + 最终答案）
- /api/poi    —— POI 数据接口（PostGIS 真实数据，前端底图聚合渲染用）
- /            —— 静态前端（frontend/index.html）
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agent.graph import get_spatial_agent
from ..core.config import PROJECT_ROOT, llm_config

app = FastAPI(title="GeoSense API", version="0.1.0")

FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _sse(event: str, data: dict) -> str:
    """把一条事件编码成 SSE 格式：`event: xxx\\ndata: {...}\\n\\n`。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


# 深圳 10 个区县名：用户问题中明确提到时，前端要显示该区边界
DISTRICT_NAMES = (
    "南山区", "福田区", "罗湖区", "宝安区", "龙岗区",
    "龙华区", "光明区", "坪山区", "盐田区", "大鹏新区",
)


def _mentioned_districts(text: str) -> list[str]:
    """找出问题里明确提到的区名（按完整区名子串匹配）。"""
    return [d for d in DISTRICT_NAMES if d in text]


def _district_geojson(names: list[str]) -> dict | None:
    """从 PostGIS 取这些区的边界，拼成 GeoJSON FeatureCollection。"""
    from ..db.connection import query, rows_to_geojson

    rows = []
    for name in names:
        rows.extend(
            query(
                "SELECT name, level, ST_AsGeoJSON(geom) AS geom "
                "FROM admin_boundary WHERE name = %s",
                (name,),
            )
        )
    if not rows:
        return None
    return rows_to_geojson(rows)


@app.get("/api/config")
def app_config():
    """下发前端运行配置（Mapbox token 等）。

    设计原则：前端不写死密钥 —— token 放在 .env（已 gitignore），
    由后端统一读取并下发，避免密钥随前端代码提交仓库（会被 GitHub 密钥扫描拦截）。
    """
    import os

    return {"mapbox_token": os.getenv("MAPBOX_TOKEN", "")}


@app.get("/api/poi")
def list_poi():
    """返回全部 POI（GeoJSON FeatureCollection，来自 PostGIS）。

    前端把结果作为聚合（cluster）数据源渲染：3175 个点不逐个建 DOM 节点，
    而是交给 Mapbox GL 做空间聚合，缩放级别越高显示越细。
    """
    from ..db.connection import query, rows_to_geojson

    rows = query(
        "SELECT name, type, ST_AsGeoJSON(geom) AS geom FROM poi ORDER BY type, name"
    )
    return rows_to_geojson(rows)


@app.post("/api/chat")
async def chat(request: Request):
    """对话接口：接收 {query}，用多工具 Agent 流式处理并返回 SSE。"""
    body = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return StreamingResponse(_empty(), media_type="text/event-stream")
    return StreamingResponse(_stream(query), media_type="text/event-stream")


async def _empty():
    yield _sse("error", {"message": "问题不能为空"})


async def _stream(query: str):
    """核心流式逻辑：逐节点推送 Agent 的决策过程。

    事件协议（前端据此渲染）：
      status   → 状态提示（"正在分析"）
      district → 问题里提到区名时，推送该区边界 GeoJSON（前端高亮显示）
      tool     → Agent 决定调用某个工具（含参数）
      result   → 工具执行结果（含 GeoJSON 时前端画图）
      answer   → 最终回答
      done     → 结束
    """
    from langchain_core.messages import AIMessage, ToolMessage

    yield _sse("status", {"message": "正在分析…"})

    # 确定性规则：问题里明确提到区名 → 先推该区边界，前端立刻高亮查询范围
    districts = _mentioned_districts(query)
    if districts:
        try:
            boundary = _district_geojson(districts)
            if boundary:
                yield _sse("district", {"name": "、".join(districts), "geojson": boundary})
        except Exception as exc:  # noqa: BLE001 —— 边界取不到不应中断对话
            yield _sse("district", {"name": "、".join(districts), "error": str(exc)})

    agent = get_spatial_agent()
    try:
        async for step in agent.astream(
            {"messages": [("user", query)]}, stream_mode="updates"
        ):
            for node, update in step.items():
                for m in update.get("messages", []):
                    if isinstance(m, AIMessage):
                        if m.tool_calls:
                            for tc in m.tool_calls:
                                yield _sse("tool", {"name": tc["name"], "args": tc["args"]})
                        content = _text_of(m.content).strip()
                        if content and not m.tool_calls:
                            yield _sse("answer", {"delta": content})
                    elif isinstance(m, ToolMessage):
                        # 结果里若含 GeoJSON，推一条 map 事件供前端绘图
                        payload = {"tool": m.name, "summary": _text_of(m.content)[:200]}
                        try:
                            data = json.loads(_text_of(m.content))
                            if isinstance(data, dict) and "geojson" in data and data["geojson"]:
                                payload["geojson"] = data["geojson"]
                        except (json.JSONDecodeError, TypeError):
                            pass
                        yield _sse("result", payload)
        yield _sse("done", {"message": "完成"})
    except Exception as exc:  # noqa: BLE001 —— 接口层兜底，不让流崩溃
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})


# 前端静态文件
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
