#!/usr/bin/env python3
"""auto_cartography.py — 第10月 W3：自动标注 + 地图综合（完整制图流水线的第 2、3 环）。

学习计划（第10月 AI Cartography）：
  自动标注 = 标注冲突检测与避让（MapboxGL 标注策略的手写实现）
  地图综合 = 缩放级别自适应简化（Douglas-Peucker + 要素级 min-area 过滤）

核心思想：
  1) 一张"成品图"= W2 符号化（选色）+ W3 标注（标谁/标哪）+ W3 综合（缩放自适应简化）。
     本脚本全部纯函数、零 LLM——LLM 进场是 W4（Text-to-Map）。
  2) 标注避让是 GIS 制图经典 NP-hard 问题的工程近似：每个要素生成 8 方位候选锚点
     （右上优先，MapboxGL text-anchor 惯例），按优先级贪心放置；候选的标签包围盒
     与已放置标签、所有要素点位碰撞则换下一候选，全撞则丢弃——宁可不标不能叠。
     MapboxGL 的 symbol 引擎在客户端做的就是这件事，手写一遍才能懂"它为什么看起来自动避开了"。
  3) 地图综合 = Douglas-Peucker 线/面简化 + min-area 要素过滤，tolerance 随 zoom 换算：
     m_per_px = 156543.03392 * cos(lat) / 2**zoom，tol = px_tolerance * m_per_px。
     zoom 小 → 容忍大偏移 + 小多边形直接隐藏（不是"变小"而是"消失"）。

坐标系决策（design D1）：内部全程 Web Mercator 米制（EPSG:3857 的度量本质），
碰撞/简化/面积/容差全部在米制做，与 PostGIS ST_Simplify 的对照也在同一坐标系下公平比较。

用法：
  python scripts/auto_cartography.py --selftest   # 9 断言自测（秒级，零数据依赖）
  python scripts/auto_cartography.py              # 真实 OSM 数据三联 demo + PostGIS 对照
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shapely.geometry import LineString, Point, Polygon  # noqa: E402
from shapely import wkt as shp_wkt  # noqa: E402

from scripts.auto_symbology import choose_symbology, profile_data  # noqa: E402  （W2 引擎，验收④零改动）

# ---------------------------------------------------------------- 常量
R_MERC = 6378137.0                       # Web Mercator 球半径（米）
M_PER_PX_Z0 = 156543.03392               # zoom 0 赤道处 米/像素（256px 瓦片）
FONT_PX = 11                             # 标注字号（px）
CHAR_W = FONT_PX * 1.05                  # CJK 全宽字符宽（px，保守估计）
LINE_H = FONT_PX * 1.2                   # 行高
GAP_PX = 3                               # 标签与锚点间距（px）> MARKER_PX 保证不压自己
MARKER_PX = 2                            # 要素点位避让半径（px）
PX_TOL = 1.0                             # 综合：像素容差
MIN_AREA_PX = 3.0                        # 综合：多边形最小可视边长（px）
CLASS_PRIORITY = {"subway": 0, "park": 1, "school": 2, "district": 3}   # 数字小者优先标注
CLASS_LABEL = {"subway": "地铁", "park": "公园", "school": "学校", "district": "区界"}

# 8 方位候选，右上优先（design D4）：(text-anchor, dx, dy)
ANCHORS = [
    ("top-right", 1, 1), ("left", 1, 0), ("bottom-right", 1, -1),
    ("top", 0, 1), ("top-left", -1, 1), ("right", -1, 0),
    ("bottom-left", -1, -1), ("bottom", 0, -1),
]

OSM_DIR = PROJECT_ROOT / "data" / "osm"
OUT_DIR = PROJECT_ROOT / "data" / "output"
DB_KW = dict(host="127.0.0.1", port=5432, dbname="geosense",
             user="geosense", password="geosense", connect_timeout=3)

# ------------------------------------------------------------- 墨卡托工具
def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 经纬度（度）→ Web Mercator（米）。"""
    x = R_MERC * math.radians(lon)
    y = R_MERC * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def merc_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Web Mercator（米）→ WGS84 经纬度（度）。"""
    lon = math.degrees(x / R_MERC)
    lat = math.degrees(2 * math.atan(math.exp(y / R_MERC)) - math.pi / 2)
    return lon, lat


def m_per_px(lat_ref: float, zoom: float) -> float:
    """zoom 级别下 lat_ref 纬度处的地面分辨率（米/像素）。"""
    return M_PER_PX_Z0 * math.cos(math.radians(lat_ref)) / 2 ** zoom


# ---------------------------------------------------------------- 数据模型
@dataclass
class Feature:
    name: str
    fclass: str
    geom: object                      # shapely 几何（Web Mercator 米制）
    priority: int = field(default_factory=int)

    def __post_init__(self):
        self.priority = CLASS_PRIORITY[self.fclass]


# ---------------------------------------------------------------- 数据加载
def load_osm_layer(path: Path, fclass: str) -> list[Feature]:
    """Overpass JSON（node/way/relation）→ Feature 列表（design D3）。

    node → Point；way.geometry 闭合 → Polygon、不闭合 → LineString；
    无 geometry 的 way/relation → center → Point。全部转 Web Mercator。
    """
    elements = json.loads(path.read_text(encoding="utf-8")).get("elements", [])
    feats: list[Feature] = []
    for e in elements:
        tags = e.get("tags", {})
        name = tags.get("name", "")
        if "lat" in e:                                   # node
            x, y = lonlat_to_merc(e["lon"], e["lat"])
            geom = Point(x, y)
        elif "geometry" in e:                            # way 带完整几何
            pts = [lonlat_to_merc(p["lon"], p["lat"]) for p in e["geometry"]]
            if len(pts) >= 4 and pts[0] == pts[-1]:
                geom = Polygon(pts)
                if not geom.is_valid:
                    geom = geom.buffer(0)                # 自相交兜底（诚实红旗：合法性未深校验）
                if geom.is_empty:
                    continue
            else:
                geom = LineString(pts)
        elif "center" in e:                              # way/relation 只回中心
            c = e["center"]
            x, y = lonlat_to_merc(c["lon"], c["lat"])
            geom = Point(x, y)
        else:
            continue
        feats.append(Feature(name=name, fclass=fclass, geom=geom))
    return feats


def load_districts() -> list[Feature]:
    """从 PostGIS admin_boundary 读行政区边界（SRID 4326 → 3857 米制）。

    数据源切换说明：data/osm/ 的 way/relation 当年用 `out center` 下载，只有中心点
    无多边形几何——综合实验改用库里的真实行政区边界（更曲折、顶点更多，效果更好）。
    """
    import psycopg
    with psycopg.connect(**DB_KW) as conn, conn.cursor() as cur:
        cur.execute("SELECT name, ST_AsText(ST_Transform(geom, 3857)) "
                    "FROM admin_boundary WHERE name IS NOT NULL;")
        return [Feature(name=n, fclass="district", geom=shp_wkt.loads(w))
                for n, w in cur.fetchall()]


# ---------------------------------------------------------------- 地图综合
def generalize(feats: list[Feature], zoom: float, lat_ref: float,
               px_tol: float = PX_TOL, min_px: float = MIN_AREA_PX):
    """缩放自适应综合（design D5）：DP 简化 + min-area 过滤。

    返回 (kept, n_hidden)：Point 原样保留；Polygon/LineString 按容差简化；
    Polygon 面积 < (min_px*mpp)^2 的在低 zoom 直接隐藏。
    """
    tol = px_tol * m_per_px(lat_ref, zoom)
    min_area = (min_px * m_per_px(lat_ref, zoom)) ** 2
    kept, n_hidden = [], 0
    for f in feats:
        g = f.geom
        if isinstance(g, Point):
            kept.append(f)
        elif isinstance(g, Polygon):
            if g.area < min_area:
                n_hidden += 1
                continue
            kept.append(Feature(f.name, f.fclass, g.simplify(tol, preserve_topology=True), f.priority))
        else:                                            # LineString
            kept.append(Feature(f.name, f.fclass, g.simplify(tol, preserve_topology=True), f.priority))
    return kept, n_hidden


# ---------------------------------------------------------------- 标注避让
def _label_bbox(x: float, y: float, name: str, dx: int, dy: int, mpp: float):
    """锚点 (x,y) + 方位 (dx,dy) → 标签包围盒 (x0,y0,x1,y1)。CJK 全宽近似。"""
    w = max(len(name), 1) * CHAR_W * mpp
    h = LINE_H * mpp
    g = GAP_PX * mpp
    if dx > 0:
        x0, x1 = x + g, x + g + w
    elif dx < 0:
        x0, x1 = x - g - w, x - g
    else:
        x0, x1 = x - w / 2, x + w / 2
    if dy > 0:
        y0, y1 = y + g, y + g + h
    elif dy < 0:
        y0, y1 = y - g - h, y - g
    else:
        y0, y1 = y - h / 2, y + h / 2
    return x0, y0, x1, y1


def _overlap(a, b) -> bool:
    """轴对齐包围盒相交（触碰不算）。"""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_labels(feats: list[Feature], zoom: float, lat_ref: float):
    """标注冲突检测与避让（design D4）：8 方位候选 + 包围盒碰撞 + 优先级贪心。

    碰撞两类：① 已放置标签盒 ② 所有要素点位小盒（标签不许压点）。
    返回 (placed, dropped)：placed 元素为 dict(feature/anchor/dx/dy/bbox/x/y)。
    """
    mpp = m_per_px(lat_ref, zoom)

    # 优先级贪心：class 优先级 → 面积大优先 → 名称稳定排序
    ordered = sorted(feats, key=lambda f: (f.priority,
                                           -getattr(f.geom, "area", 0.0), f.name))
    # 预置所有要素点位避让盒（含自己的锚点，GAP_PX > MARKER_PX 保证不压自己）
    occupied = []
    for f in feats:
        if isinstance(f.geom, Point):
            px, py = f.geom.x, f.geom.y
        else:
            px, py = f.geom.representative_point().x, f.geom.representative_point().y
        r = MARKER_PX * mpp
        occupied.append((px - r, py - r, px + r, py + r))

    placed, dropped = [], []
    for f in ordered:
        if not f.name:
            dropped.append((f, "无名"))
            continue
        if isinstance(f.geom, Point):
            ax_, ay_ = f.geom.x, f.geom.y
        else:
            rp = f.geom.representative_point()
            ax_, ay_ = rp.x, rp.y
        got = None
        for anchor, dx, dy in ANCHORS:                   # 右上优先依次尝试
            bb = _label_bbox(ax_, ay_, f.name, dx, dy, mpp)
            if not any(_overlap(bb, ob) for ob in occupied):
                got = (anchor, dx, dy, bb)
                break
        if got is None:
            dropped.append((f, "8 方位全碰撞"))
        else:
            anchor, dx, dy, bb = got
            placed.append(dict(feature=f, anchor=anchor, dx=dx, dy=dy,
                               bbox=bb, x=ax_, y=ay_))
            occupied.append(bb)
    return placed, dropped


def to_mapbox_labels(placed: list[dict]) -> dict:
    """placed → MapboxGL symbol layer 骨架（design D6）。

    symbol-sort-key = 优先级：客户端引擎做最终避让时，优先级高者先放置——
    我们的贪心排序就是它的输入质量。W4 Text-to-Map 直接组合 W2+W3 两份 JSON。
    """
    features = []
    for p in placed:
        f = p["feature"]
        lon, lat = merc_to_lonlat(p["x"], p["y"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            "properties": {
                "name": f.name,
                "class": f.fclass,
                "priority": f.priority,
                "anchor": p["anchor"],
            },
        })
    return {
        "id": "poi-labels",
        "type": "symbol",
        "source": {"type": "geojson", "data": {"type": "FeatureCollection", "features": features}},
        "layout": {
            "text-field": ["get", "name"],
            "text-size": FONT_PX,
            "text-anchor": ["get", "anchor"],
            "symbol-sort-key": ["get", "priority"],
            "text-allow-overlap": False,     # 客户端引擎继续避让
        },
    }


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    """9 断言（design D7 清单），零数据依赖。"""
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # 1. 墨卡托往返
    lon, lat = 114.0579, 22.5431        # 深圳市中心附近
    x, y = lonlat_to_merc(lon, lat)
    lo, la = merc_to_lonlat(x, y)
    check("墨卡托往返误差<1e-6度", abs(lo - lon) < 1e-6 and abs(la - lat) < 1e-6,
          f"d_lon={abs(lo - lon):.2e} d_lat={abs(la - lat):.2e}")

    # 2. m_per_px：z0≈156543*cos(lat)，z+1 减半
    m0, m1, m2 = m_per_px(22.65, 0), m_per_px(22.65, 1), m_per_px(22.65, 2)
    check("m_per_px z0≈156543*cos(lat)", abs(m0 - M_PER_PX_Z0 * math.cos(math.radians(22.65))) < 1e-6)
    check("m_per_px 每+1级减半", abs(m1 - m0 / 2) < 1e-6 and abs(m2 - m1 / 2) < 1e-6)

    # 3. 候选顺序：首候选 = 右上（NE）
    check("首候选锚点为右上(NE)", ANCHORS[0][0] == "top-right" and ANCHORS[0][1] > 0 and ANCHORS[0][2] > 0)

    # 4. 碰撞正确性：9 个共点要素 → 8 方位放满后溢出者被丢弃，且已放置标签两两不重叠
    zoom, lat_ref = 14, 22.65           # z14: ~8.8 m/px
    coloc = lonlat_to_merc(114.0, 22.55)
    cofeat = [Feature(f"同点站{i}", "subway", Point(*coloc)) for i in range(9)]
    pc, dc = place_labels(cofeat, zoom, lat_ref)
    bbsc = [p["bbox"] for p in pc]
    check("共点9要素溢出丢弃且标签两两不重叠",
          len(dc) >= 1 and len(pc) + len(dc) == 9
          and not any(_overlap(bbsc[i], bbsc[j])
                      for i in range(len(bbsc)) for j in range(i + 1, len(bbsc))),
          f"placed={len(pc)} dropped={len(dc)}")
    f3 = Feature("测试点丙", "subway", Point(*lonlat_to_merc(114.01, 22.555)))        # ~1km：远离
    p3, _ = place_labels([cofeat[0], f3], zoom, lat_ref)
    check("远离两点放2个标签", len(p3) == 2, f"placed={len(p3)}")

    # 6. DP 顶点数单调不增
    line = LineString([(0, 0), (1, 0.1), (2, -0.1), (3, 0.05), (4, 0), (5, 0.2), (6, 0)])
    counts = [len(line.simplify(t).coords) for t in (0.01, 0.05, 0.1, 0.5)]
    check("DP 顶点数随容差单调不增", counts == sorted(counts, reverse=True), f"counts={counts}")

    # 7. DP 面积守恒（容差 << 多边形尺度；~100m 多边形 vs z14 1px 容差 8.8m）
    poly = Polygon(lonlat_to_merc(114.03 + dx * 1e-3, 22.55 + dy * 1e-3)
                   for dx, dy in [(0, 0), (1, 0), (1.2, 1), (0.5, 1.5), (0, 1)])
    simp = poly.simplify(1.0 * m_per_px(22.65, 14), preserve_topology=True)
    check("DP 面积守恒(容差内≥0.95)", simp.area / poly.area >= 0.95,
          f"ratio={simp.area / poly.area:.4f}")

    # 8. min-area 过滤：低 zoom 小多边形隐藏
    small = Feature("小微公园", "park", Polygon(lonlat_to_merc(114.0 + dx * 3e-5, 22.55 + dy * 3e-5)
                                               for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1)]))
    big = Feature("大公园", "park", Polygon(lonlat_to_merc(114.1 + dx * 3e-2, 22.6 + dy * 3e-2)
                                           for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1)]))
    kept8, hidden8 = generalize([small, big], zoom=10, lat_ref=22.65)
    names8 = [f.name for f in kept8]
    check("min-area 低zoom隐藏小多边形", "小微公园" not in names8 and hidden8 >= 1
          and "大公园" in names8, f"kept={names8} hidden={hidden8}")

    # 9. to_mapbox_labels 结构
    p9, _ = place_labels([cofeat[0], f3], zoom, lat_ref)
    st9 = to_mapbox_labels(p9)
    feats9 = st9["source"]["data"]["features"]
    check("mapbox symbol 骨架结构", len(feats9) == len(p9)
          and st9["layout"]["text-field"] == ["get", "name"]
          and st9["layout"]["symbol-sort-key"] == ["get", "priority"]
          and all(ft["properties"]["anchor"] in
                  {"top-right", "left", "bottom-right", "top", "top-left",
                   "right", "bottom-left", "bottom"} for ft in feats9),
          f"features={len(feats9)}")

    n_pass = 0
    print("\n=== auto_cartography selftest ===")
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        n_pass += ok
    print(f"=== 汇总: {n_pass}/{len(results)} PASS ===")
    return 0 if n_pass == len(results) else 1


# ---------------------------------------------------------------- PostGIS 对照
def compare_with_postgis(poly_merc: Polygon, tolerances: list[float]) -> list[dict] | None:
    """自写 DP（shapely/GEOS）vs PostGIS ST_Simplify（GEOS）对照（design D7 / 验收③）。

    两侧同在 EPSG:3857 米制、同 GEOS 内核 → 结果应几乎逐点一致（Hausdorff≈0）。
    daemon 不可用 / 连接失败返回 None（诚实降级，不造假对照）。
    """
    try:
        import psycopg
        with psycopg.connect(**DB_KW) as conn, conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE t_poly (g geometry(Polygon, 3857));")
            cur.execute("INSERT INTO t_poly VALUES (ST_GeomFromText(%s, 3857));",
                        (poly_merc.wkt,))
            rows = []
            for tol in tolerances:
                cur.execute(
                    "SELECT ST_NPoints(ST_Simplify(g, %s)), ST_AsText(ST_Simplify(g, %s)), "
                    "ST_NPoints(ST_SimplifyPreserveTopology(g, %s)), "
                    "ST_AsText(ST_SimplifyPreserveTopology(g, %s)) "
                    "FROM t_poly;", (tol, tol, tol, tol))
                n1, w1, n2, w2 = cur.fetchone()
                rows.append((tol, n1, shp_wkt.loads(w1), n2, shp_wkt.loads(w2)))
    except Exception as e:                                   # noqa: BLE001 —— 诚实降级
        print(f"[降级] PostGIS 不可用（{type(e).__name__}: {e}），对照实验跳过")
        return None
    out = []
    for tol, n1, pg1, n2, pg2 in rows:
        mine = poly_merc.simplify(tol, preserve_topology=True)
        out.append(dict(tol_m=tol, my_n=len(mine.exterior.coords),
                        pg_n=n1, hausdorff_m=mine.hausdorff_distance(pg1),
                        pgt_n=n2, hausdorff_t_m=mine.hausdorff_distance(pg2)))
    return out


# ---------------------------------------------------------------- demo 渲染
def _set_window(ax, x0, y0, x1, y1):
    """设置显示窗口 + 等比坐标（墨卡托保形，等比显示即正确形状）。"""
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_labels(ax, placed, mpp):
    """按 anchor 绘制标签（与 bbox 计算同一几何）。"""
    ha_map = {1: "left", 0: "center", -1: "right"}
    va_map = {1: "bottom", 0: "center", -1: "top"}
    for p in placed:
        dx, dy = p["dx"], p["dy"]
        g = GAP_PX * mpp
        tx = p["x"] + (g if dx > 0 else (-g if dx < 0 else 0))
        ty = p["y"] + (g if dy > 0 else (-g if dy < 0 else 0))
        ax.text(tx, ty, p["feature"].name, fontsize=FONT_PX * 0.72,
                ha=ha_map[dx], va=va_map[dy], color="#222222", zorder=5)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]

    # ---------- 数据加载（真实 OSM）----------
    subway = load_osm_layer(OSM_DIR / "poi_subway.json", "subway")
    park = load_osm_layer(OSM_DIR / "poi_park.json", "park")
    school = load_osm_layer(OSM_DIR / "poi_school.json", "school")
    print(f"数据：subway={len(subway)}（Polygon {sum(isinstance(f.geom, Polygon) for f in subway)}）"
          f" park={len(park)}（Polygon {sum(isinstance(f.geom, Polygon) for f in park)}）"
          f" school={len(school)}")

    # ---------- 配色：复用 W2 引擎（语义表无 地铁/公园/学校 → Set2 机械回落，实证 W2 红旗②）----------
    values = ([0] * len(subway)) + ([1] * len(park)) + ([2] * len(school))
    labels = ([CLASS_LABEL["subway"]] * len(subway)
              + [CLASS_LABEL["park"]] * len(park)
              + [CLASS_LABEL["school"]] * len(school))
    plan = choose_symbology(profile_data(values, labels=labels))
    cmap = dict(zip(plan.class_values, plan.colors))
    print(f"W2 符号化：{plan.palette_name} — {plan.reason}")

    LAT_REF = 22.65
    # 底图上下文：行政区境界线（联1/联2 底图 + 联3 综合实验原料）
    districts = load_districts()
    fig = matplotlib.figure.Figure(figsize=(15.6, 4.6), dpi=100)
    ax1 = fig.add_axes([0.02, 0.10, 0.30, 0.74])
    ax2 = fig.add_axes([0.36, 0.10, 0.28, 0.74])
    ax3 = fig.add_axes([0.68, 0.10, 0.30, 0.74])

    # ---------- 联1：z10 全市 subway 标注（丢弃率教学）----------
    all_x = [f.geom.x for f in subway] + [f.geom.x for f in park] + [f.geom.x for f in school]
    all_y = [f.geom.y for f in subway] + [f.geom.y for f in park] + [f.geom.y for f in school]
    x0, x1 = min(all_x) * 0.999, max(all_x) * 1.001
    y0, y1 = min(all_y) * 0.999, max(all_y) * 1.001
    mpp10 = m_per_px(LAT_REF, 10)
    placed10, dropped10 = place_labels(subway, 10, LAT_REF)
    # 硬断言：已放置标签两两包围盒不重叠（验收②）
    bbs = [p["bbox"] for p in placed10]
    assert not any(_overlap(bbs[i], bbs[j])
                   for i in range(len(bbs)) for j in range(i + 1, len(bbs))), "联1 标注重叠！"
    _set_window(ax1, x0, y0, x1, y1)
    for f in subway:
        c = cmap[0]
        if isinstance(f.geom, Point):
            ax1.plot(f.geom.x, f.geom.y, "o", ms=2.2, color=c, zorder=3)
        else:
            geoms = getattr(f.geom, "geoms", [f.geom])
            for g in geoms:
                xs_, ys_ = g.exterior.xy
                ax1.fill(xs_, ys_, color=c, alpha=0.45, lw=0.3, zorder=2)
    _draw_labels(ax1, placed10, mpp10)
    drop10 = len(dropped10) / (len(placed10) + len(dropped10))
    ax1.set_title(f"联1 z10 全市·地铁 {len(subway)} 要素\n"
                  f"放置 {len(placed10)} / 丢弃 {len(dropped10)}（丢弃率 {drop10:.0%}）",
                  fontsize=9)

    # ---------- 联2：z12 中心城区避让（福田-罗湖窗口）----------
    cx, cy = lonlat_to_merc(114.085, 22.56)
    half_w, half_h = 8000.0, 4000.0                       # 16km × 8km
    win = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    in_win = [f for f in subway + park + school
              if win[0] <= (f.geom.x if isinstance(f.geom, Point) else f.geom.representative_point().x) <= win[2]
              and win[1] <= (f.geom.y if isinstance(f.geom, Point) else f.geom.representative_point().y) <= win[3]]
    mpp12 = m_per_px(LAT_REF, 12)
    placed12, dropped12 = place_labels(in_win, 12, LAT_REF)
    bbs = [p["bbox"] for p in placed12]
    assert not any(_overlap(bbs[i], bbs[j])
                   for i in range(len(bbs)) for j in range(i + 1, len(bbs))), "联2 标注重叠！"
    _set_window(ax2, *win)
    for d in districts:                                   # 境界底图
        xs_, ys_ = d.geom.exterior.xy
        ax2.plot(xs_, ys_, color="#999999", lw=0.6, zorder=1)
    for f in in_win:
        c = cmap[f.priority]
        if isinstance(f.geom, Point):
            ax2.plot(f.geom.x, f.geom.y, "o", ms=3, color=c, zorder=3)
        elif isinstance(f.geom, Polygon):
            xs_, ys_ = f.geom.exterior.xy
            ax2.fill(xs_, ys_, color=c, alpha=0.35, lw=0.4, zorder=2)
        else:
            xs_, ys_ = f.geom.xy
            ax2.plot(xs_, ys_, color=c, lw=0.8, zorder=2)
    _draw_labels(ax2, placed12, mpp12)
    drop12 = len(dropped12) / (len(placed12) + len(dropped12))
    ax2.set_title(f"联2 z12 中心城区（福田-罗湖 16×8km）\n"
                  f"三层 {len(in_win)} 要素：放置 {len(placed12)} / 丢弃 {len(dropped12)}（{drop12:.0%}）",
                  fontsize=9)

    # ---------- 联3：行政区边界综合（z10/z12/z14 容差叠加 + PostGIS 对照）----------
    # 数据源切换：OSM way 当年 `out center` 下载无多边形 → 改用 PostGIS 真实区界（顶点更多更曲折）
    districts = load_districts()
    big_border = max(districts, key=lambda f: len(getattr(f.geom, "exterior", []).coords))
    tols = {z: PX_TOL * m_per_px(LAT_REF, z) for z in (10, 12, 14)}
    simps = {z: big_border.geom.simplify(t, preserve_topology=True) for z, t in tols.items()}
    bx0, by0, bx1, by1 = big_border.geom.bounds
    pad = (bx1 - bx0) * 0.12
    ax3.set_xlim(bx0 - pad, bx1 + pad)
    ax3.set_ylim(by0 - pad, by1 + pad)
    ax3.set_aspect("equal")
    ax3.set_xticks([])
    ax3.set_yticks([])
    xs_, ys_ = big_border.geom.exterior.xy
    ax3.fill(xs_, ys_, color="#bbbbbb", alpha=0.30, lw=0.8,
             label=f"原始（{len(big_border.geom.exterior.coords)} 顶点）")
    zoom_colors = {10: "#1f77b4", 12: "#ff7f0e", 14: "#2ca02c"}
    for z in (10, 12, 14):
        g = simps[z]
        xs_, ys_ = g.exterior.xy
        ax3.plot(xs_, ys_, color=zoom_colors[z], lw=1.2,
                 label=f"z{z} 简化（{len(g.exterior.coords)} 顶点, tol={tols[z]:.0f}m）")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_title(f"联3 地图综合：{big_border.name}边界（PostGIS admin_boundary）\n"
                  f"Douglas-Peucker 容差随 zoom 换算（原始 {big_border.geom.area / 1e6:.0f} km²）",
                  fontsize=9)

    fig.suptitle("GeoSense 第10月 W3 自动标注 + 地图综合（数据：OSM 深圳；配色：W2 引擎 Set2 回落）",
                 fontsize=11)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "auto_cartography_demo.png"
    fig.savefig(out_png, dpi=100)
    print(f"\n出图：{out_png}（{out_png.stat().st_size:,} B）")
    print(f"标注小结：z10 全市地铁丢弃率 {drop10:.0%}（城市概览只标重点）；"
          f"z12 中心窗口丢弃率 {drop12:.0%}；两联已放置标签两两不重叠（硬断言通过）")

    # ---------- PostGIS ST_Simplify 对照（验收③）----------
    print("\n=== 自写 DP vs PostGIS ST_Simplify（同 EPSG:3857 米制、同 GEOS 内核）===")
    rows = compare_with_postgis(big_border.geom, [tols[z] for z in (10, 12, 14)])
    if rows:
        for z, r in zip((10, 12, 14), rows):
            print(f"  z{z}: tol={r['tol_m']:.1f}m  自写(保拓扑) {r['my_n']} 顶点")
            print(f"       vs ST_Simplify(无拓扑保持)      {r['pg_n']} 顶点 | Hausdorff={r['hausdorff_m']:.3f} m")
            print(f"       vs ST_SimplifyPreserveTopology   {r['pgt_n']} 顶点 | Hausdorff={r['hausdorff_t_m']:.3f} m")

    mapbox = to_mapbox_labels(placed12)
    n_mb = len(mapbox["source"]["data"]["features"])
    print(f"\nMapbox symbol layer 骨架：{n_mb} 个标注 Feature（text-field/symbol-sort-key 就绪，W4 直接组合）")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
