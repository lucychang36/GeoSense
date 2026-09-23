"""阶段1 补课交付物：seed_postgis.py —— 建表 + 灌样例数据

核心概念：种子数据（seed data）
- 让 PostGIS 表里有可查的真实数据：POI 点来自 sz_poi.json，
  学校从 POI 派生，行政区边界用"演示用简化矩形"（非真实行政边界）。
- 脚本幂等：先 TRUNCATE 再 INSERT，可反复重跑重建数据。

运行前提：
  1. docker compose up -d              # 起 PostGIS（含 PostGIS + pgvector 扩展）
  2. .venv/bin/python scripts/seed_postgis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db.connection import execute, executemany
from backend.db.schema import DDL_STATEMENTS

POI_JSON = Path(__file__).resolve().parents[1] / "backend" / "agent" / "data" / "sz_poi.json"

# 演示用简化行政区边界（矩形 bbox，非真实行政边界，仅用于空间关系教学）
ADMIN_BOUNDARIES = [
    ("南山区", "区", "POLYGON((113.85 22.48, 114.05 22.48, 114.05 22.60, 113.85 22.60, 113.85 22.48))"),
    ("福田区", "区", "POLYGON((114.02 22.51, 114.13 22.51, 114.13 22.58, 114.02 22.58, 114.02 22.51))"),
    ("宝安区", "区", "POLYGON((113.76 22.55, 113.93 22.55, 113.93 22.85, 113.76 22.85, 113.76 22.55))"),
]


def import_admin_geojson(paths: list[Path]) -> int:
    """region-data-inventory（design D6）：把 fetch_admin_boundaries.py 产出的全国
    省/市/县 GeoJSON upsert 进 admin_boundary。

    - 级别枚举：district / city / province（country 预留）；OSM 深圳 10 区（level='6'）
      命名空间不同、互不覆盖，保留共存。
    - 幂等：UNIQUE(name, level) + ON CONFLICT DO UPDATE，可反复重跑。
    -     不 TRUNCATE：与默认种子模式（重建全表）不同，导入模式只增量更新。
    """
    from backend.db.connection import executemany

    # 幂等前提：唯一索引（老库无此约束，IF NOT EXISTS 兜底）
    execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS admin_boundary_name_level_uidx "
        "ON admin_boundary (name, level)"
    )

    total = 0
    for path in paths:
        gj = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for feat in gj.get("features", []):
            props = feat.get("properties") or {}
            name, level = props.get("name"), props.get("level")
            if not name or not level:
                continue
            # 同名不同级的历史数据让位于真实边界（如旧演示矩形 level='区'）
            execute("DELETE FROM admin_boundary WHERE name = %s AND level <> %s",
                    (name, level))
            rows.append((name, level, json.dumps(feat["geometry"], ensure_ascii=False)))
        executemany(
            "INSERT INTO admin_boundary (name, level, geom) "
            "VALUES (%s, %s, ST_GeomFromGeoJSON(%s)) "
            "ON CONFLICT (name, level) DO UPDATE "
            "SET geom = EXCLUDED.geom",
            rows,
        )
        print(f"✅ 导入 {path.name}：{len(rows)} 条（level={rows[0][1] if rows else '-'}）")
        total += len(rows)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-geojson", nargs="+", metavar="GEOJSON",
                    help="导入模式：把 fetch_admin_boundaries.py 产出的边界 GeoJSON "
                         "upsert 进 admin_boundary（不重建表、不动 poi/schools）")
    args = ap.parse_args()
    if args.admin_geojson:
        n = import_admin_geojson([Path(p) for p in args.admin_geojson])
        print(f"🎉 全国边界入库完成（{n} 条）。验证："
              f".venv/bin/python scripts/postgis_sql_agent.py \"查询 admin_boundary 中 name='金水区'\"")
        return

    _seed_default()


def _seed_default() -> None:
    # 1. 建表（IF NOT EXISTS，可安全重复执行）
    for ddl in DDL_STATEMENTS:
        execute(ddl)
    print(f"✅ 建表完成（{len(DDL_STATEMENTS)} 张）")

    # 2. 清空旧数据（幂等重跑；表名来自硬编码元组，非用户输入，拼接安全）
    for table in ("poi", "schools", "admin_boundary"):
        execute(f"TRUNCATE {table} RESTART IDENTITY CASCADE")

    # 3. 灌 POI（从 sz_poi.json；值全部走 %s 参数化）
    pois = json.loads(POI_JSON.read_text(encoding="utf-8"))
    for p in pois:
        execute(
            "INSERT INTO poi (name, type, geom) "
            "VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
            (p["name"], p["type"], p["lon"], p["lat"]),
        )
    print(f"✅ 导入 POI {len(pois)} 条")

    # 4. 学校（从 POI 派生 type='学校'）
    schools = [p for p in pois if p["type"] == "学校"]
    for s in schools:
        execute(
            "INSERT INTO schools (name, type, geom) "
            "VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
            (s["name"], s["type"], s["lon"], s["lat"]),
        )
    print(f"✅ 导入学校 {len(schools)} 条")

    # 5. 行政区边界（ST_GeomFromText 把 WKT 文本转成几何）
    for name, level, wkt in ADMIN_BOUNDARIES:
        execute(
            "INSERT INTO admin_boundary (name, level, geom) "
            "VALUES (%s, %s, ST_GeomFromText(%s, 4326))",
            (name, level, wkt),
        )
    print(f"✅ 导入行政区边界 {len(ADMIN_BOUNDARIES)} 条")

    print("🎉 种子数据就绪。验证：python scripts/postgis_sql_agent.py --check")


if __name__ == "__main__":
    main()
