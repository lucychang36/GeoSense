"""空间分析工具集（第3月 W2 扩充版）—— Agent 的"专业工具箱"。

核心概念：工具集的领域化
- W4 的 3 个工具是"原子能力"（距离/缓冲/坐标转换）；
- 本周扩到 7 个，其中 spatial_query / overlay_analysis / statistics_calc /
  map_generation 是"领域工具"——面向 GIS 业务语义，Agent 才能理解
  "查询周边3公里学校"这种组合任务。

核心概念：数据源抽象
- data_retrieval + spatial_query 现在读 PostGIS 数据库（poi 表），
  此前读内存 JSON 数据集（深圳 POI）。切换只改了函数内部实现，
  Agent 和工具签名完全没变 —— "接口稳定、实现可替换"的实践。

工具清单（对应学习计划第3月的 TOOLS 列表）：
  data_retrieval       数据检索（POI 概览）
  spatial_query        空间查询（点周边半径内、按类型过滤）
  buffer_analysis      缓冲区分析
  overlay_analysis     叠加分析（两个几何求交）
  coordinate_transform 坐标转换
  statistics_calc      空间统计（计数 / 类型分布 / 范围）
  map_generation       制图（GeoJSON 落盘 + 摘要）
"""
from __future__ import annotations

import json

from shapely.geometry import shape, mapping

from ..core.config import PROJECT_ROOT
from ..db.connection import query
from .tools import create_buffer, transform_coord

# 数据集名 → 表名 的映射（白名单：只允许这几个固定值，杜绝把模型输入拼进 SQL）
_DATASET_TABLES = {
    "sz_poi": "poi",
    "poi": "poi",
    "schools": "schools",
}


def _db_error(exc: Exception) -> str:
    """数据库异常统一转成 JSON 错误串（Agent 会看到并转述给用户）。"""
    return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 数据检索
# ---------------------------------------------------------------------------
def data_retrieval(dataset_name: str = "sz_poi") -> str:
    """返回指定数据集（表）的概览：总数 + 类型分布。数据来自 PostGIS poi 表。

    dataset_name 必须是 _DATASET_TABLES 白名单里的值（模型只能选，不能传任意表名）。
    """
    table = _DATASET_TABLES.get(dataset_name)
    if table is None:
        return json.dumps({"error": f"未知数据集: {dataset_name}"}, ensure_ascii=False)
    try:
        # table 来自固定映射表（非用户输入），f-string 拼接是安全的
        rows = query(
            f"SELECT type, count(*) AS cnt FROM {table} GROUP BY type ORDER BY cnt DESC"
        )
    except Exception as exc:  # noqa: BLE001 —— 数据库没起/连接失败时给模型一个干净的报错
        return _db_error(exc)
    types = {r["type"]: r["cnt"] for r in rows}
    return json.dumps(
        {"dataset": dataset_name, "total": sum(types.values()), "types": types},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 空间查询：点周边半径内、按类型过滤（PostGIS geography 距离，单位米）
# ---------------------------------------------------------------------------
def spatial_query(lon: float, lat: float, radius_m: float, poi_type: str = "") -> str:
    """查询某点周边 radius_m 米范围内的 POI（PostGIS poi 表），可按类型过滤，按距离升序。"""
    try:
        rows = query(
            """
            SELECT name, type,
                   ST_Distance(geom::geography,
                               ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) AS distance_m,
                   ST_X(geom) AS lon, ST_Y(geom) AS lat
            FROM poi
            WHERE ST_DWithin(geom::geography,
                             ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
              AND (%s = '' OR type = %s)      -- 可选过滤用参数化实现，不拼 SQL
            ORDER BY distance_m
            LIMIT 500
            """,
            (lon, lat, lon, lat, radius_m, poi_type, poi_type),
        )
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    hits = [
        {
            "name": r["name"],
            "type": r["type"],
            "distance_m": round(r["distance_m"], 1),
            "lon": r["lon"],
            "lat": r["lat"],
        }
        for r in rows
    ]
    return json.dumps({"count": len(hits), "pois": hits}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 叠加分析：两个 GeoJSON 几何求交
# ---------------------------------------------------------------------------
def overlay_analysis(geojson_a: str, geojson_b: str) -> str:
    """计算两个 GeoJSON 几何的交集，返回交集 GeoJSON 与面积（平方度）。"""
    a = shape(json.loads(geojson_a))
    b = shape(json.loads(geojson_b))
    inter = a.intersection(b)
    if inter.is_empty:
        return json.dumps({"geojson": None, "area_deg2": 0.0,
                           "note": "两几何不相交"}, ensure_ascii=False)
    return json.dumps({"geojson": mapping(inter), "area_deg2": round(inter.area, 8)},
                      ensure_ascii=False)


# ---------------------------------------------------------------------------
# 空间统计：计数 / 类型分布 / 范围
# ---------------------------------------------------------------------------
def statistics_calc(geojson: str) -> str:
    """对 GeoJSON FeatureCollection 做统计：要素数、按 type 分布、经纬度范围。"""
    data = json.loads(geojson)
    feats = data.get("features", [])
    by_type: dict[str, int] = {}
    lons, lats = [], []
    for f in feats:
        t = f["properties"].get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        g = shape(f["geometry"])
        lons.append(g.centroid.x)
        lats.append(g.centroid.y)
    bbox = {"min_lon": min(lons), "max_lon": max(lons),
            "min_lat": min(lats), "max_lat": max(lats)} if lons else None
    return json.dumps({"feature_count": len(feats), "types": by_type, "bbox": bbox},
                      ensure_ascii=False)


# ---------------------------------------------------------------------------
# 制图：GeoJSON 落盘
# ---------------------------------------------------------------------------
def map_generation(geojson: str, title: str = "analysis_result") -> str:
    """把 GeoJSON 存盘（供前端/下一步展示），返回文件路径与要素数。"""
    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{title}.geojson"
    data = json.loads(geojson)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.dumps({"path": str(path.relative_to(PROJECT_ROOT)),
                       "feature_count": len(data.get("features", []))}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 复用 W4 的两个原子工具（统一返回 JSON 字符串）
# ---------------------------------------------------------------------------
def buffer_analysis(lon: float, lat: float, radius_m: float) -> str:
    """以经纬度点为中心生成半径（米）缓冲区，返回 GeoJSON 与面积。"""
    return json.dumps(create_buffer(lon, lat, radius_m), ensure_ascii=False)


def coordinate_transform(lon: float, lat: float, to_epsg: int, from_epsg: int = 4326) -> str:
    """坐标参考系统转换（EPSG 代码），如 WGS84(4326) → Web 墨卡托(3857)。"""
    return json.dumps(transform_coord(lon, lat, to_epsg, from_epsg), ensure_ascii=False)
