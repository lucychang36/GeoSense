"""GIS 工具集 —— Agent 的"手"。

核心概念：Function Calling 的两个半边
1. TOOL_SCHEMAS（给模型看的"使用说明书"）：name + description + JSON Schema 参数。
   模型不执行任何代码，它只输出"我想调哪个工具、传什么参数"的 JSON。
   description 写得越清楚，模型调得越准 —— 这本质上是 Prompt Engineering 的延伸。
2. 实现函数（真正干活的代码）：模型输出参数 → 我们的分发器执行 → 结果回喂模型。

工具实现的 GIS 知识点：
- 距离用 WGS84 椭球测地线（pyproj.Geod），不是平面欧氏距离 —— 经纬度下
  直接用勾股定理是 GIS 新手最常见的错误（1 度纬度 ≈ 111km，1 度经度随纬度变化）。
- 缓冲区必须转到投影坐标系（这里按经度自动选 UTM 带）再算半径，算完转回 WGS84。
"""
from __future__ import annotations

import json
from typing import Any

from pyproj import Geod, Transformer
from shapely.geometry import Point, mapping
from shapely.ops import transform as shp_transform

# WGS84 椭球（测地线计算标准）
_GEOD = Geod(ellps="WGS84")


def calc_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> dict:
    """两点间测地线距离（椭球面上沿最短路径的真实距离，单位：米）。"""
    _, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    return {"distance_m": round(dist, 2), "distance_km": round(dist / 1000, 3)}


def _utm_epsg(lon: float, lat: float) -> int:
    """按经度自动选择 UTM 投影带 EPSG 代码（北半球 326xx，南半球 327xx）。"""
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def create_buffer(lon: float, lat: float, radius_m: float) -> dict:
    """以点为中心生成圆形缓冲区，返回 GeoJSON 多边形与面积。

    流程：WGS84 → UTM（米制）→ buffer → 转回 WGS84。这样半径单位才是真正的"米"。
    """
    epsg = _utm_epsg(lon, lat)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    to_wgs = Transformer.from_crs(epsg, 4326, always_xy=True)
    pt_utm = Point(*to_utm.transform(lon, lat))
    buf_utm = pt_utm.buffer(radius_m, resolution=64)
    buf_wgs = mapping(shp_transform(to_wgs.transform, buf_utm))
    return {
        "geojson": buf_wgs,
        "area_km2": round(buf_utm.area / 1e6, 3),
        "utm_epsg": epsg,
    }


def transform_coord(lon: float, lat: float, to_epsg: int, from_epsg: int = 4326) -> dict:
    """坐标参考系统转换（如 WGS84 → CGCS2000 投影 / Web 墨卡托）。"""
    t = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    x, y = t.transform(lon, lat)
    return {"x": round(x, 3), "y": round(y, 3), "epsg": to_epsg}


# ---------------------------------------------------------------------------
# 工具注册表：schema（模型看的） ↔ 实现（我们跑的）
# ---------------------------------------------------------------------------
TOOL_IMPLEMENTATIONS = {
    "calc_distance": calc_distance,
    "create_buffer": create_buffer,
    "transform_coord": transform_coord,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calc_distance",
            "description": "计算两个 WGS84 经纬度坐标点之间的真实地表距离（测地线距离）。当用户问两地距离、多远、直线距离时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lon1": {"type": "number", "description": "起点经度"},
                    "lat1": {"type": "number", "description": "起点纬度"},
                    "lon2": {"type": "number", "description": "终点经度"},
                    "lat2": {"type": "number", "description": "终点纬度"},
                },
                "required": ["lon1", "lat1", "lon2", "lat2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_buffer",
            "description": "以一个经纬度点为中心生成指定半径（米）的圆形缓冲区，返回 GeoJSON 边界和面积。当用户问周边X公里范围、影响范围时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lon": {"type": "number", "description": "中心点经度"},
                    "lat": {"type": "number", "description": "中心点纬度"},
                    "radius_m": {"type": "number", "description": "缓冲半径，单位米"},
                },
                "required": ["lon", "lat", "radius_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transform_coord",
            "description": "把坐标从一个坐标参考系统（EPSG 代码）转换到另一个，例如 WGS84(4326) 转 CGCS2000 投影(4547) 或 Web 墨卡托(3857)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lon": {"type": "number", "description": "源坐标 X（经度或东向坐标）"},
                    "lat": {"type": "number", "description": "源坐标 Y（纬度或北向坐标）"},
                    "to_epsg": {"type": "integer", "description": "目标坐标系的 EPSG 代码"},
                    "from_epsg": {"type": "integer", "description": "源坐标系的 EPSG 代码，默认 4326"},
                },
                "required": ["lon", "lat", "to_epsg"],
            },
        },
    },
]


def execute_tool(name: str, arguments_json: str) -> str:
    """分发器：按工具名执行实现函数，统一返回 JSON 字符串。

    核心概念：工具执行是"不可信输入"处理 —— 参数来自模型，必须容错：
    未知工具、JSON 解析失败、运行时报错，都要变成可回喂的错误信息，
    而不是让 Agent 循环崩溃（鲁棒性是 Agent 工程的核心）。
    """
    func = TOOL_IMPLEMENTATIONS.get(name)
    if func is None:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        args: dict[str, Any] = json.loads(arguments_json)
        result = func(**args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 —— 工具边界必须兜住一切
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
