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

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db.connection import execute
from backend.db.schema import DDL_STATEMENTS

POI_JSON = Path(__file__).resolve().parents[1] / "backend" / "agent" / "data" / "sz_poi.json"

# 演示用简化行政区边界（矩形 bbox，非真实行政边界，仅用于空间关系教学）
ADMIN_BOUNDARIES = [
    ("南山区", "区", "POLYGON((113.85 22.48, 114.05 22.48, 114.05 22.60, 113.85 22.60, 113.85 22.48))"),
    ("福田区", "区", "POLYGON((114.02 22.51, 114.13 22.51, 114.13 22.58, 114.02 22.58, 114.02 22.51))"),
    ("宝安区", "区", "POLYGON((113.76 22.55, 113.93 22.55, 113.93 22.85, 113.76 22.85, 113.76 22.55))"),
]


def main() -> None:
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
