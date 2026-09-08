"""FastAPI 后端 —— GeoSense 的 Web 接口层。

核心概念：SSE（Server-Sent Events，服务器推送事件）
- 普通 HTTP 是"一问一答"；SSE 让服务器把结果"一段一段"推给浏览器，
  浏览器边收边渲染 —— 这就是聊天界面"打字机效果"的实现。
- 与 W1 的 chat_stream 一脉相承，只是从"命令行打印"变成了"HTTP 流式推送"。

核心概念：接口分层
- /api/chat         —— 对话接口（调用 LangGraph Agent，流式返回工具调用链 + 最终答案）
- /api/poi          —— POI 数据接口（PostGIS 真实数据，前端底图聚合渲染用）
- /api/model/*      —— 第9月 W2：模型推理服务（U-Net 分割 / 变化检测 / YOLO 检测）
- /                 —— 静态前端（frontend/index.html）
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agent.graph import get_spatial_agent
from ..core.config import PROJECT_ROOT, llm_config
from ..model_service import (  # noqa: E402  （启动时不强 load，首次请求 lazy 加载）
    change as msvc_change,
    detect as msvc_detect,
    jobs as msvc_jobs,
    list_loaded as msvc_list_loaded,
    segment as msvc_segment,
)

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
                            # 第10月 W4 Text-to-Map：工具返回含 map_style → 新事件类型 style
                            # 全量下发（不经 result 的 [:200] 摘要路径），前端 addLayer 注入渲染
                            if isinstance(data, dict) and data.get("map_style"):
                                yield _sse("style", data["map_style"])
                            # 第11月 W1 报告生成：report_tool 结果 → 新事件类型 report
                            # URL 型载荷（前端可直接下载），不经 result 的 [:200] 摘要路径
                            if isinstance(data, dict) and data.get("report_path"):
                                from pathlib import Path as _P
                                fname = _P(data["report_path"]).name
                                yield _sse("report", {
                                    "title": data.get("title", "GeoSense 分析报告"),
                                    "change_ratio": data.get("change_ratio"),
                                    "md_url": f"/api/reports/{fname}" if fname else "",
                                    "narrative_skipped": data.get("narrative_skipped", False),
                                })
                        except (json.JSONDecodeError, TypeError):
                            pass
                        yield _sse("result", payload)
        yield _sse("done", {"message": "完成"})
    except Exception as exc:  # noqa: BLE001 —— 接口层兜底，不让流崩溃
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})


# ============================================================
# 第9月 W2：模型推理服务（U-Net 分割 / 变化检测 / YOLO 检测）
# 设计：单例加载、路径安全、输出 base64 PNG → 前端可直接 <img src=...>
# 注意：这些路由必须定义在 app.mount("/") 之前 —— Starlette 按注册顺序匹配，
#       mount("/") 是 catch-all，若先注册会拦截所有 /api/* 请求（返回 404）。
# ============================================================

@app.get("/api/model/health")
def model_health():
    """健康检查：列出已加载模型 + 设备。模型按需 lazy load（首次请求时）。"""
    return msvc_list_loaded()


@app.post("/api/model/segment")
async def model_segment(request: Request):
    """U-Net 三分类（水/城/植）分割。Body: {"cog": "szbay_real_20250727.tif"}"""
    body = await request.json()
    cog = body.get("cog", "").strip()
    weight = body.get("weight", "unet_finetuned.pt")
    if not cog:
        raise HTTPException(400, "cog 不能为空（应传入 data/cogs/ 下的文件名）")
    try:
        return msvc_segment.segment_cog(cog, weight=weight)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/model/change")
async def model_change(request: Request):
    """变化检测。Body: {"cog_a": "...", "cog_b": "...", "method": "postclass|spectral"}"""
    body = await request.json()
    cog_a = (body.get("cog_a") or "").strip()
    cog_b = (body.get("cog_b") or "").strip()
    method = body.get("method", "postclass")
    weight = body.get("weight", "unet_finetuned.pt")
    threshold = float(body.get("threshold", 0.08))
    if not cog_a or not cog_b:
        raise HTTPException(400, "cog_a / cog_b 都不能为空")
    try:
        return msvc_change.change_cog(cog_a, cog_b, method=method, weight=weight,
                                      threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/model/detect")
async def model_detect(request: Request):
    """YOLOv8 船舶检测。Body: {"cog": "...", "conf": 0.25, "imgsz": 1280}"""
    body = await request.json()
    cog = (body.get("cog") or "").strip()
    weight = body.get("weight", "yolov8n_ship.pt")
    conf = float(body.get("conf", 0.25))
    imgsz = int(body.get("imgsz", 1280))
    if not cog:
        raise HTTPException(400, "cog 不能为空")
    try:
        return msvc_detect.detect_cog(cog, weight=weight, conf=conf, imgsz=imgsz)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------
# 第9月 W3：异步推理任务队列 —— 提交即返回 job_id，后台线程分块跑，轮询查进度
# 设计：POST /api/model/jobs 秒回 {job_id, status: queued}（不阻塞），
#       前端轮询 GET /api/model/jobs/{id} 拿 progress（0~1）与最终 result。
# 对比 W2 同步端点：影像小（700×1100）同步够用；整景 Sentinel-2（万级×万级）
#       同步推理是分钟级 —— HTTP 请求会干等到超时，异步把「重活」挪到后台。
# ------------------------------------------------------------

@app.post("/api/model/jobs")
async def model_submit_job(request: Request):
    """提交异步任务。Body:
      {"task_type": "segment", "cog": "szbay_real_20250727.tif", "tile": 512}
      {"task_type": "big_image", "size": 2048, "tile": 512}   # 合成大影像内存演示
    立即返回轻量 job 快照 —— 用 GET /api/model/jobs/{job_id} 轮询。
    """
    body = await request.json()
    task_type = (body.get("task_type") or "").strip()
    try:
        return msvc_jobs.submit(task_type, body)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/model/jobs/{job_id}")
def model_poll_job(job_id: str):
    """轮询任务状态：queued → running → done / error。
    done/error 前只回轻量字段；完成后附完整 result（含 stats + base64 PNG）。
    """
    job = msvc_jobs.poll(job_id)
    if job is None:
        raise HTTPException(404, f"job 不存在：{job_id}")
    return job


# 前端静态文件（必须最后注册 —— catch-all，放在最后才不会拦截 /api/* 路由）
@app.get("/api/reports/{name}")
def download_report(name: str):
    """第11月 W1：报告交付物下载（md/html/png）。

    路径净化同第9月 safe_cog_path 边界教训：剥目录只留文件名防 ../、
    后缀白名单、is_file() 显式检查（Path("").name == "" 会拼出父目录本身）。
    """
    from fastapi.responses import FileResponse, JSONResponse
    reports_dir = PROJECT_ROOT / "data" / "output" / "reports"
    safe = Path(name).name
    if not safe or safe != name or Path(safe).suffix.lower() not in {".md", ".html", ".png", ".jpg"}:
        return JSONResponse({"error": "非法文件名"}, status_code=400)
    f = reports_dir / safe
    if not f.is_file():
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    return FileResponse(f)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
