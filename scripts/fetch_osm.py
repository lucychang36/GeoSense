"""scripts/fetch_osm.py —— 从 OpenStreetMap（Overpass API）拉取深圳真实数据并入库

为什么用 Overpass API？
- OSM 全量数据太大（中国区 pbf 数十 GB）；Overpass 支持按标签/区域实时查询，
  只下载我们需要的：深圳 10 个区的真实行政边界 + 学校/医院/地铁站等真实 POI。
- 公开实例有频率限制，脚本把原始响应缓存到 data/osm/，重复运行不重复请求。

数据流水线（核心概念：外部数据的清洗链）：
  下载（不可信原始数据）→ 解析（relation 成员组装多边形 / 标签映射类型）
  → 清洗（同名去重、过滤无名要素、坐标去重、几何简化）→ 入库（重建表 + 批量插入）

用法：
  .venv/bin/python scripts/fetch_osm.py            # 优先用本地缓存（data/osm/）
  .venv/bin/python scripts/fetch_osm.py --refresh  # 强制重新从 Overpass 下载
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console

from backend.db.connection import execute, executemany
from backend.db.schema import DDL_STATEMENTS

console = Console()

OVERPASS_URL = "https://overpass-api.de/api/interpreter"   # 国内可直连的公共实例
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "osm"

# 深圳 10 个区县（含功能区）。按全名精确匹配，避免误抓周边城市同名区。
DISTRICT_QUERY = """
[out:json][timeout:180];
rel["boundary"="administrative"]["name"~"^(南山区|福田区|罗湖区|宝安区|龙岗区|龙华区|光明区|坪山区|盐田区|大鹏新区)$"];
out geom;
"""

# POI 按类型拆成独立小查询：单类响应小、不易超时，且每类独立缓存、部分成功可复用。
# out center：节点直接给坐标；way/relation 给中心点（POI 展示用中心点足够）。
POI_TAG_QUERIES = {
    "school":       '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="school"](area.c);out center tags;',
    "kindergarten": '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="kindergarten"](area.c);out center tags;',
    "university":   '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="university"](area.c);out center tags;',
    "hospital":     '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="hospital"](area.c);out center tags;',
    "clinic":       '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="clinic"](area.c);out center tags;',
    "police":       '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="police"](area.c);out center tags;',
    "fire_station": '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="fire_station"](area.c);out center tags;',
    "library":      '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["amenity"="library"](area.c);out center tags;',
    "park":         '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["leisure"="park"](area.c);out center tags;',
    "subway":       '[out:json][timeout:120];area["name"="深圳市"]["boundary"="administrative"]->.c;nwr["railway"="station"]["station"="subway"](area.c);out center tags;',
}

# OSM amenity 标签 → 中文类型
AMENITY_TYPE = {
    "school": "学校",
    "kindergarten": "幼儿园",
    "university": "大学",
    "hospital": "医院",
    "clinic": "诊所",
    "police": "警察局",
    "fire_station": "消防站",
    "library": "图书馆",
}


# ---------------------------------------------------------------------------
# 1. 下载（带缓存 + 重试）
# ---------------------------------------------------------------------------
def _post_json(ql: str) -> dict:
    """POST 一次 Overpass 查询，失败自动重试（最多 3 次，15s 退避）。"""
    body = urllib.parse.urlencode({"data": ql}).encode()
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(
                OVERPASS_URL, data=body,
                headers={"User-Agent": "GeoSense-Learning/1.0 (contact: local)"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310 —— 教学项目固定公网地址
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            if "remark" in data or "error" in data:   # Overpass 的软错误（如限流）
                raise RuntimeError(data.get("remark") or str(data.get("error")))
            return data
        except Exception as exc:  # noqa: BLE001 —— 网络抖动/504/限流都走重试
            last_err = exc
            console.print(f"[yellow]  第 {attempt} 次请求失败：{type(exc).__name__}，15s 后重试…[/]")
            time.sleep(15)
    raise RuntimeError(f"Overpass 请求多次失败：{last_err}")


def cached_query(ql: str, cache_path: Path, refresh: bool) -> dict:
    """执行查询；命中缓存且未要求刷新时直接读缓存。"""
    if cache_path.exists() and not refresh:
        console.print(f"[dim]  使用缓存 {cache_path.name}[/]")
        return json.loads(cache_path.read_text(encoding="utf-8"))
    data = _post_json(ql)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def download_pois(refresh: bool) -> list[dict]:
    """逐类下载 POI，合并所有元素；每类独立缓存 + 类间停顿防限流。"""
    elements: list[dict] = []
    for tag, ql in POI_TAG_QUERIES.items():
        console.print(f"[cyan]查询 Overpass：{tag}…[/]")
        data = cached_query(ql, CACHE_DIR / f"poi_{tag}.json", refresh)
        elements.extend(data.get("elements", []))
        time.sleep(3)
    return elements


# ---------------------------------------------------------------------------
# 2. 解析行政区边界：relation 成员 → 多边形
# ---------------------------------------------------------------------------
# 深圳大致范围：用于排除"同名不同地"的污染（如海南龙华区、黑龙江南山区）
SZ_BBOX = (113.70, 22.40, 114.70, 22.90)   # (minx, miny, maxx, maxy)


def _assemble_polygon(members: list[dict]):
    """把 Overpass relation 的成员（线几何）组装成多边形/多多边形。

    为什么用 linemerge 而不是 polygonize？
    - 实测：OSM 边界 way 在接点处常有微小不闭合，polygonize（精确拓扑）
      只能拼出沿边界的碎片（福田区 4km²）；linemerge 能容忍并正确合并
      首尾相连的线段，得到完整闭合环（福田区 85km²）。
    - role=outer 的线 → 外环；role=inner 的线 → 洞（subtract）。
    """
    from shapely.geometry import LineString, MultiPolygon, Polygon
    from shapely.ops import linemerge, unary_union

    outer, inner = [], []
    for m in members:
        geom = m.get("geometry")
        if not geom:                       # 成员是子 relation（out geom 不带其几何），跳过
            continue
        pts = [(p["lon"], p["lat"]) for p in geom]
        if len(pts) < 2:
            continue
        line = LineString(pts)
        (inner if m.get("role") == "inner" else outer).append(line)

    if not outer:
        return None
    merged = linemerge(unary_union(outer))
    rings = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    shells = [Polygon(r) for r in rings if r.is_closed and len(r.coords) >= 4]
    shells = [s for s in shells if s.area > 1e-6]     # 丢掉微小的杂环
    if not shells:
        return None

    # inner 环 → 洞
    holes: list[Polygon] = []
    if inner:
        im = linemerge(unary_union(inner))
        irings = list(im.geoms) if im.geom_type == "MultiLineString" else [im]
        holes = [Polygon(r) for r in irings if r.is_closed and r.area > 1e-6]

    polys = []
    for s in shells:
        shell_holes = [h.exterior for h in holes
                       if s.covers(h.representative_point())]
        poly = Polygon(s.exterior, shell_holes)
        if not poly.is_valid:
            poly = poly.buffer(0)          # 自相交等无效几何的兜底修复
        if poly.area > 1e-9:
            polys.append(poly)
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def parse_districts(elements: list[dict]) -> list[tuple[str, str, str]]:
    """返回 [(区名, admin_level, WKT)]。

    过滤与去重（都是"真实数据不完美"的应对）：
    - 只保留与深圳 bbox 相交的 relation（排除海南/黑龙江等地的同名区）；
    - 同一 relation 可能有多个不相连的外环（如龙岗区 relation 里残留了大鹏区域），
      用 relation 自带的 label 节点（区中心）选出真正属于本区的那个环；
    - 多边形再裁剪到深圳 bbox，去掉多余的近海海域；
    - 同名区保留组装面积最大的（边界最完整）。
    """
    from shapely.geometry import Point, box

    sz_box = box(*SZ_BBOX)
    best: dict[str, tuple[str, object]] = {}
    for e in elements:
        tags = e.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        # relation 自带的区中心参考点（label / admin_centre 节点）
        label_pt = None
        for m in e.get("members", []):
            if m.get("type") == "node" and m.get("role") in ("label", "admin_centre") \
                    and "lat" in m and "lon" in m:
                label_pt = Point(m["lon"], m["lat"])
                break

        poly = _assemble_polygon(e.get("members", []))
        if poly is None or poly.is_empty:
            continue
        if not poly.intersects(sz_box):
            console.print(f"[dim]  跳过同名异地区：{name}（不在深圳范围）[/]")
            continue
        # 多环时选包含区中心的环（兜底：取最大环）
        if poly.geom_type == "MultiPolygon":
            if label_pt is not None:
                contained = [p for p in poly.geoms if p.contains(label_pt)]
                if contained:
                    poly = contained[0]
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
        poly = poly.intersection(sz_box)      # 裁剪多余海域
        if poly.is_empty:
            continue

        cur = best.get(name)
        if cur is None or poly.area > cur[1].area:
            best[name] = (tags.get("admin_level", "?"), poly)

    result: list[tuple[str, str, str]] = []
    for name, (level, poly) in best.items():
        poly = poly.simplify(0.0005, preserve_topology=True)   # ~55m 精度的适度简化
        result.append((name, level, poly.wkt))
    return result


# ---------------------------------------------------------------------------
# 3. 解析 POI：标签映射 + 去重 + 过滤无名
# ---------------------------------------------------------------------------
def parse_pois(elements: list[dict]) -> list[dict]:
    pois: list[dict] = []
    seen: set[tuple] = set()
    for e in elements:
        tags = e.get("tags", {})
        name = (tags.get("name") or "").strip()
        if not name:
            continue
        # 类型映射：地铁站 > 公园 > amenity
        if tags.get("railway") == "station" and tags.get("station") == "subway":
            ptype = "地铁站"
        elif tags.get("leisure") == "park":
            ptype = "公园"
        else:
            ptype = AMENITY_TYPE.get(tags.get("amenity"))
        if not ptype:
            continue
        # 坐标：节点直接有 lat/lon；way/relation 用 out center 的中心点
        if "lat" in e and "lon" in e:
            lon, lat = e["lon"], e["lat"]
        elif e.get("center"):
            lon, lat = e["center"]["lon"], e["center"]["lat"]
        else:
            continue
        key = (name, round(lon, 5), round(lat, 5))   # 5 位小数 ≈ 1m，坐标去重
        if key in seen:
            continue
        seen.add(key)
        pois.append({"name": name, "type": ptype, "lon": lon, "lat": lat})
    return pois


# ---------------------------------------------------------------------------
# 4. 入库：重建表 + 批量插入
# ---------------------------------------------------------------------------
def load_to_postgis(districts: list[tuple], pois: list[dict]) -> None:
    # 重建三张表（DROP 后按 DDL 重建，确保列类型与最新 schema 一致）
    for table in ("poi", "schools", "admin_boundary"):
        execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    for ddl in DDL_STATEMENTS:
        execute(ddl)
    console.print("[green]✅ 表已重建[/]")

    # 行政区边界（数量少，逐条插入；WKT → 几何）
    for name, level, wkt in districts:
        execute(
            "INSERT INTO admin_boundary (name, level, geom) "
            "VALUES (%s, %s, ST_GeomFromText(%s, 4326))",
            (name, level, wkt),
        )
    console.print(f"[green]✅ admin_boundary 入库 {len(districts)} 条[/]")

    # POI（几千条，批量 executemany）
    executemany(
        "INSERT INTO poi (name, type, geom) "
        "VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
        [(p["name"], p["type"], p["lon"], p["lat"]) for p in pois],
    )
    console.print(f"[green]✅ poi 入库 {len(pois)} 条[/]")

    # 学校表：从 poi 派生教育设施
    execute(
        "INSERT INTO schools (name, type, geom) "
        "SELECT name, type, geom FROM poi WHERE type IN ('学校', '幼儿园', '大学')"
    )
    console.print("[green]✅ schools 已从 poi 派生[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 OSM 拉取深圳边界 + POI 并入库")
    parser.add_argument("--refresh", action="store_true", help="强制重新下载（忽略缓存）")
    args = parser.parse_args()

    console.rule("[bold]1/3 下载并解析行政区边界")
    raw_districts = cached_query(DISTRICT_QUERY, CACHE_DIR / "districts_raw.json", args.refresh)
    districts = parse_districts(raw_districts.get("elements", []))
    console.print(f"  解析出 {len(districts)} 个区："
                  + "、".join(f"{n}({l}级)" for n, l, _ in districts))

    console.rule("[bold]2/3 下载并解析 POI")
    pois = parse_pois(download_pois(args.refresh))
    console.print("  类型分布：" + ", ".join(f"{k} {v}" for k, v in Counter(p["type"] for p in pois).items()))

    console.rule("[bold]3/3 入库")
    load_to_postgis(districts, pois)
    console.print("🎉 完成。验证：python scripts/postgis_sql_agent.py --check")


if __name__ == "__main__":
    main()
