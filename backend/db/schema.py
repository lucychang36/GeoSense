"""数据库表结构定义 —— Agent 与种子脚本的"单一事实来源"。

核心概念：schema-as-code（把表结构当代码管理）
- 建表 DDL、列说明、SQL 校验白名单都集中在这里，而不是散落在
  SQL 文件 / prompt 字符串 / 校验器各处 —— 改表只改这一个文件。
- LLM 生成 SQL 前需要知道"有哪些表、哪些列"，我们把 SCHEMA_HINT
  拼进 spatial_sql 工具的描述/prompt，模型才知道能查什么（也约束它不瞎编表名）。
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 建表 DDL：docker 首次启动的 init 脚本只建扩展，建表统一走 scripts/seed_postgis.py
# ---------------------------------------------------------------------------
DDL_STATEMENTS = [
    # POI 表：通用兴趣点（点，WGS84）
    """
    CREATE TABLE IF NOT EXISTS poi (
        id   SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT,                    -- 类型：学校/医院/地标…
        geom GEOMETRY(Point, 4326)    -- WGS84 经纬度坐标
    );
    """,
    # 空间索引：ST_Within/ST_Intersects 等空间谓词的加速关键（第6月 W4 压测发现缺失）
    """
    CREATE INDEX IF NOT EXISTS poi_geom_gix ON poi USING gist(geom);
    """,
    # 学校表：独立维护（示例数据从 POI 中 type='学校' 派生）
    """
    CREATE TABLE IF NOT EXISTS schools (
        id   SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT,
        geom GEOMETRY(Point, 4326)
    );
    """,
    # 行政区边界表：真实边界（scripts/fetch_osm.py 从 OSM 拉取，含 MultiPolygon）
    """
    CREATE TABLE IF NOT EXISTS admin_boundary (
        id    SERIAL PRIMARY KEY,
        name  TEXT NOT NULL,
        level TEXT,                     -- 级别：区（6）/功能区（7）…
        geom  GEOMETRY(Geometry, 4326)  -- 用通用 Geometry 类型，兼容 Polygon/MultiPolygon
    );
    """,
]

# 表白名单：SQL 校验器只放行这些表（见 sql_validator.py）
ALLOWED_TABLES = ("poi", "schools", "admin_boundary")

# 给 LLM 看的 schema 提示（拼进 spatial_sql 工具描述，帮助模型写出正确 SQL）
SCHEMA_HINT = """
可用表（PostgreSQL/PostGIS，几何 SRID=4326，数据来自 OpenStreetMap，深圳市）：
1. poi(id, name, type, geom<Point>) —— 真实 POI（类型含：学校/幼儿园/大学/医院/诊所/警察局/消防站/图书馆/公园/地铁站）
2. schools(id, name, type, geom<Point>) —— 教育设施（学校/幼儿园/大学，由 poi 派生）
3. admin_boundary(id, name, level, geom<Polygon|MultiPolygon>) —— 深圳 10 个区县的真实行政边界（name 如 '南山区'/'福田区'/'宝安区'）

常用 PostGIS 写法：
- 距离（米）：ST_Distance(a.geom::geography, b.geom::geography)
- 半径过滤（米）：ST_DWithin(a.geom::geography, b.geom::geography, radius_m)
- 空间关系：ST_Within / ST_Intersects / ST_Contains
- 输出几何：ST_AsGeoJSON(geom) AS geom   ← 查询结果必须用这个别名，前端才能画图
- 注意：geography 的距离计算以"点"为准；面到面的距离请用 ST_Centroid 取中心点再算。
""".strip()
