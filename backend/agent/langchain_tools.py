"""LangChain 工具适配层 —— 把 W4 的 GIS 工具暴露给 LangGraph。

核心概念：LangChain 的 @tool 装饰器
- 把"函数 + 类型标注 + docstring"自动转换成工具定义（name + description + 参数 schema）。
- docstring 第一行就是工具的 description —— 和 W4 一样，描述质量决定模型调用准确率。
- 返回值必须是 str：工具结果会作为字符串塞回对话历史（role="tool" 消息）。

为什么单独一层？W4 的工具实现（pyproj/shapely 计算）是纯 GIS 逻辑、与框架无关；
这一层只做"框架适配"，将来换 LangGraph / LlamaIndex 只改这里，不动核心计算。
"""
from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from .tools import calc_distance, create_buffer, transform_coord
from .sql_tools import spatial_sql
from .spatial_tools import (
    buffer_analysis,
    coordinate_transform,
    data_retrieval,
    map_generation,
    overlay_analysis,
    spatial_query,
    statistics_calc,
)


@tool
def calc_distance_tool(lon1: float, lat1: float, lon2: float, lat2: float) -> str:
    """计算两个 WGS84 经纬度坐标点之间的真实地表距离（测地线距离），单位米。当用户问两地距离、多远、直线距离时使用。"""
    return json.dumps(calc_distance(lon1, lat1, lon2, lat2), ensure_ascii=False)


@tool
def create_buffer_tool(lon: float, lat: float, radius_m: float) -> str:
    """以一个经纬度点为中心生成指定半径（米）的圆形缓冲区，返回覆盖面积。当用户问周边X公里范围、影响范围时使用。"""
    return json.dumps(create_buffer(lon, lat, radius_m), ensure_ascii=False)


@tool
def transform_coord_tool(lon: float, lat: float, to_epsg: int, from_epsg: int = 4326) -> str:
    """把坐标从一个坐标参考系统（EPSG 代码）转换到另一个，例如 WGS84(4326) 转 CGCS2000 投影(4547) 或 Web 墨卡托(3857)。"""
    return json.dumps(transform_coord(lon, lat, to_epsg, from_epsg), ensure_ascii=False)


# ---- 第3月 W2 扩充的 7 个空间领域工具 ----

@tool
def data_retrieval_tool(dataset_name: str = "sz_poi") -> str:
    """查看数据集中有哪些 POI 及类型分布。当用户问"有哪些数据/学校/医院/POI"时先调用它了解可用数据。"""
    return data_retrieval(dataset_name)


@tool
def spatial_query_tool(lon: float, lat: float, radius_m: float, poi_type: str = "") -> str:
    """查询某经纬度点周边 radius_m 米范围内的 POI（学校/医院等），poi_type 可按类型过滤。当用户问"周边X公里有哪些学校/医院"时使用。"""
    return spatial_query(lon, lat, radius_m, poi_type=poi_type)


@tool
def buffer_analysis_tool(lon: float, lat: float, radius_m: float) -> str:
    """以经纬度点为中心生成半径（米）缓冲区，返回 GeoJSON 与面积。"""
    return buffer_analysis(lon, lat, radius_m)


@tool
def overlay_analysis_tool(geojson_a: str, geojson_b: str) -> str:
    """计算两个 GeoJSON 几何的交集（叠加分析），返回交集 GeoJSON。"""
    return overlay_analysis(geojson_a, geojson_b)


@tool
def coordinate_transform_tool(lon: float, lat: float, to_epsg: int, from_epsg: int = 4326) -> str:
    """坐标参考系统转换（EPSG 代码），如 WGS84(4326) → Web 墨卡托(3857)。"""
    return coordinate_transform(lon, lat, to_epsg, from_epsg)


@tool
def statistics_calc_tool(geojson: str) -> str:
    """对 GeoJSON FeatureCollection 做空间统计：要素数、类型分布、经纬度范围。"""
    return statistics_calc(geojson)


@tool
def map_generation_tool(geojson: str, title: str = "analysis_result") -> str:
    """把 GeoJSON 存盘生成地图数据文件，返回文件路径与要素数。分析结果需要输出/展示时使用。"""
    return map_generation(geojson, title)


# ---- 阶段1 补课：自然语言 → PostGIS SQL 生成 + 执行 ----

@tool
def spatial_sql_tool(question: str) -> str:
    """把自然语言空间问题转成 PostGIS SQL 并执行，返回查询结果（含 GeoJSON 可画图）。
    当需要查数据库里的真实数据（学校/医院/POI/行政区边界）、做空间过滤/统计/空间关系分析时使用，
    例如"南山区内有多少所学校""某个点周边3公里有哪些医院"。"""
    return spatial_sql(question)


# ---- 第9月 W2 模型能力接入：遥感影像时相对比（2026-09-04 OpenSpec 变更，最小工具） ----
# 为什么函数内 import：model_service 依赖 torch/U-Net（重），lazy import 让 Agent 构建秒级完成；
# 工具首次被调用时才触发模型加载 —— 且与 /api/model 服务共享同进程 MPS 单例（零网络，不打 HTTP 环路）。

@tool
def temporal_change_tool(cog_a: str, cog_b: str, method: str = "postclass") -> str:
    """对 data/cogs/ 下两景同区域遥感影像（文件名）做变化检测，返回变化占比与类别转换统计。
    当用户问"两期/两个年份的影像对比变化""深圳湾 2023 和 2025 相比变了多少"时使用。
    例：cog_a=szbay_real_20230708.tif, cog_b=szbay_real_20250727.tif；
    method="postclass"（U-Net 分类后比较，默认，推荐）或 "spectral"（光谱差分 baseline）。"""
    try:
        # lazy import：保持 Agent 构建轻量；运行时与 model_service 共享 MPS 单例
        from ..model_service.change import change_cog
        r = change_cog(cog_a, cog_b, method=method)
        # 摘要化返回：只取数值 + 转换矩阵，丢弃 change_cog 里的 base64 PNG（文本 LLM 不可消费）
        # 注意字段顺序：关键数值必须前置 —— main.py 的 SSE result 事件对 summary 做 [:200] 截断，
        # 前置保证前端事件流里能看到核心数字（LLM 拿完整 ToolMessage 不受影响）。
        return json.dumps({
            "change_ratio": r["change_ratio"],   # 变化像素占有效像素比例（0~1）
            "n_change": r["n_change"],
            "n_valid": r["n_valid"],
            "method": r["method"],
            "cog_a": r["cog_a"],
            "cog_b": r["cog_b"],
            "class_names": r["class_names"],     # 类别顺序说明（transition 行列含义）
            "transition": r["transition"],       # 3×3 转换矩阵，仅 postclass 有值；[i][j]=类 i→类 j 像素数
        }, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 —— 错误转文本不炸 ReAct（2026-09-24：RuntimeError
        # 穿透曾炸穿循环，与 text_to_map_tool 的边界策略对齐；文件名类错误仍给候选兜底
        from ..model_service.loader import COGS_DIR
        candidates = sorted(
            p.name for p in COGS_DIR.glob("*.tif")
            if re.search(r"\d{8}", p.name)      # 只列带日期的真实 COG（剔除合成 mosaic）
        )
        hint = "、".join(candidates) if candidates else "（data/cogs/ 下暂无带日期 COG）"
        return f"[工具错误] {type(e).__name__}: {e}。可用带日期的 COG 文件：{hint}"


# ---- 第10月 W4 Text-to-Map：自然语言 → Mapbox 样式（2026-09-07 OpenSpec 变更） ----
# 为什么函数内 import：text_to_map 依赖 rasterio/W2/W3 引擎，lazy import 保持 Agent 构建轻量；
# 意图-渲染分层（explore 方案 B）：planner 只负责把制图请求路由到本工具，query 传用户原话。

@tool
def text_to_map_tool(query: str) -> str:
    """把用户的自然语言制图需求转成 Mapbox 地图样式并在前端地图上渲染。
    当用户想要「做一张...图」「把...叠加到地图上」「给...上色」「标注...」等制图/地图样式请求时使用，
    query 传用户原话（不要改写）。支持三类主题：
    poi=兴趣点分类图（地铁站/公园/学校，含标注避让）、
    ndvi=植被指数网格专题图、
    cog=卫星影像叠加（透明度可调，文件名来自 data/cogs/）。
    返回 JSON 含 map_style（系统会自动上图），summary 为一句话结果说明。"""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _root = str(_Path(__file__).resolve().parents[2])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from scripts.text_to_map import text_to_map
        return json.dumps(text_to_map(query), ensure_ascii=False)
    except Exception as e:                                   # noqa: BLE001 —— 错误转文本不炸 ReAct
        return (f"[工具错误] {type(e).__name__}: {e}。可用的制图主题："
                f"poi（兴趣点分类图）/ ndvi（NDVI 网格图）/ cog（卫星影像叠加）")


# ---- 第11月 W1 报告生成：multi_agent 管道（规划→取数→分析→制图→报告）的薄封装 ----
# 为什么封装整个图：报告需要"先分析再写"，让 ReAct planner 只做一次路由决策，
# 子管道的步骤编排交给 multi_agent 图（职责分层，与 temporal_change_tool 直调单函数不同）。

@tool
def report_tool(query: str) -> str:
    """端到端自动分析报告：输入自然语言问题（含时相对比/区域），内部运行多 Agent 管道
    （规划→数据→变化检测→制图→报告），生成 Markdown + HTML 分析报告文件。
    当用户要求「生成分析报告」「出一份报告」「写报告」等报告类请求时使用，
    query 传用户原话（不要改写）。
    返回 JSON 字段：title（报告标题）、change_ratio（变化占比）、report_path（md 路径）。"""
    try:
        from backend.agent.multi_agent import get_multi_agent
        final = get_multi_agent().invoke({"user_query": query, "step_log": []})
        if final.get("analysis_error"):
            return json.dumps({
                "error": f"[报告未生成] {final['analysis_error']}",
                "change_ratio": None, "report_path": None,
            }, ensure_ascii=False)
        res = final.get("analysis_result") or {}
        out = {
            "title": final.get("report_title", "GeoSense 分析报告"),
            "change_ratio": res.get("change_ratio"),
            "change_px": res.get("change_px"),
            "report_path": final.get("report_path", ""),
            "narrative_skipped": final.get("narrative_skipped", False),
        }
        # 地图叠加（map-result-linkage D5）：URL 型载荷（通用协议 D1），经 overlay 事件上图
        ov = final.get("overlay_meta") or {}
        if ov.get("path"):
            from pathlib import Path as _P
            out["overlay"] = {
                "url": f"/api/overlays/{_P(ov['path']).name}",
                "fit_bounds": ov.get("bbox") or res.get("bbox"),
                "legend": ov.get("legend", []),
                "render_hint": "fill",
                "basemap": ov.get("basemap"),
            }
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:                               # noqa: BLE001 —— 错误转文本不炸 ReAct
        return f"[工具错误] {type(e).__name__}: {e}。报告依赖 data/cogs/ 下的两期带日期 COG。"


# ---- 第12月 W2 region-data-inventory：区域数据清单（空间求交，开放语义） ----
# 为什么函数内 import：inventory 依赖 data/admin 缓存与 shapely，lazy import 保持构建轻量；
# manifest 是唯一真相（D2），本工具是 agent 看见"我手里有哪些区域数据"的唯一窗口。

@tool
def data_inventory_tool(region: str = "", bbox: str = "") -> str:
    """查询某区域有哪些可用遥感影像数据（区域数据清单）。任意地名（金水区/杭州/任意城市）
    或任意 bbox（"west,south,east,north"，支持全球）都可查，返回：
    coverage（full/partial/none）、各期影像的日期/云量/波段，以及无数据时的最近可用区域推荐。
    当用户问「XX区/XX市有没有数据」「对比 XX 的影像变化」但不确定数据可用性时，先调用它再下结论；
    查不到数据时必须按 recommendation 诚实告知并推荐替代区域，禁止编造影像或数字。"""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _root = str(_Path(__file__).resolve().parents[2])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from scripts.data_inventory import inventory_query
        result = inventory_query(region, bbox=bbox or None)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:                               # noqa: BLE001 —— 错误转文本不炸 ReAct
        return (f"[工具错误] {type(e).__name__}: {e}。"
                f"可改用 bbox 参数直给（如 \"113.5,34.7,113.7,34.9\"）。")


# LangGraph 使用的工具列表（W1 的 3 个原子工具 + W2 的 7 个领域工具 + SQL 工具）
GIS_TOOLS = [calc_distance_tool, create_buffer_tool, transform_coord_tool]
SPATIAL_TOOLS = [
    data_retrieval_tool,
    spatial_query_tool,
    buffer_analysis_tool,
    overlay_analysis_tool,
    coordinate_transform_tool,
    statistics_calc_tool,
    map_generation_tool,
    spatial_sql_tool,
    temporal_change_tool,
    text_to_map_tool,
    report_tool,
    data_inventory_tool,
]

