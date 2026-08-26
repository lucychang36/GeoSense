"""第5月 W3：export_geoparquet.py —— PostGIS → GeoParquet 导出

核心概念：GeoParquet（地理 Parquet）
- Parquet 是"列式存储"：按列压缩存放，查询只读需要的列 —— 与 GeoJSON（行式、
  整文件解析）相比，大数据量下速度可差几个数量级。
- GeoParquet = Parquet + 标准化的 geometry 列（GeoArrow 编码）：
  任何支持该规范的引擎（DuckDB/GeoPandas/Arrow）都能直接读，不再各自为政。

核心概念：谁负责"格式转换"？
- 本脚本用 DuckDB 自己完成 GeoJSON → GeoParquet：
  ST_GeomFromGeoJSON(列) 造出几何列 → COPY ... (FORMAT PARQUET, GEOMETRY_ENCODING GEOARROW)
- 数据源是 PostGIS（阶段1 补课入库的 OSM 真实数据），通过 psycopg 拉取。

运行方式：
  .venv/bin/python scripts/export_geoparquet.py   （需 PostGIS 容器已启动）
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
from rich.console import Console
from rich.table import Table

from backend.core.config import PROJECT_ROOT, db_config
from backend.db.connection import get_conn

console = Console()
OUT_DIR = PROJECT_ROOT / "data" / "gpq"
TABLES = [
    ("admin_boundary", "深圳 10 区县行政边界"),
    ("poi", "深圳 POI（学校/医院/地铁站等）"),
]


def export_table(con: duckdb.DuckDBPyConnection, table: str, label: str) -> dict:
    """拉取一张表 → 临时 JSON → DuckDB 转 GeoParquet。"""
    # 1. 从 PostGIS 拉取：几何转成 GeoJSON 文本，交给 DuckDB 重建
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, name, ST_AsGeoJSON(geom)::text AS g FROM {table}"
        ).fetchall()
    console.print(f"[dim]  {table}: 拉取 {len(rows)} 行[/]")

    # 2. 写临时 JSON（DuckDB 可直接读）
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([{"id": r["id"], "name": r["name"], "g": r["g"]} for r in rows],
                  f, ensure_ascii=False)
        tmp = f.name

    # 3. DuckDB：GeoJSON → GeoParquet（geometry 列自动按 GeoArrow 编码）
    out = OUT_DIR / f"{table}.gpq"
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT id, name, ST_GeomFromGeoJSON(g) AS geometry
            FROM read_json_auto('{tmp}')
        ) TO '{out}' (FORMAT PARQUET)
    """)
    Path(tmp).unlink()
    return {"table": table, "label": label, "rows": len(rows),
            "size_kb": round(out.stat().st_size / 1024, 1),
            "seconds": round(time.time() - t0, 2)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("LOAD spatial")

    console.rule("[bold cyan]PostGIS → GeoParquet 导出")
    stats = [export_table(con, t, label) for t, label in TABLES]

    table = Table(title="导出结果（data/gpq/）")
    table.add_column("表")
    table.add_column("说明")
    table.add_column("行数", justify="right")
    table.add_column("文件大小", justify="right")
    table.add_column("耗时", justify="right")
    for s in stats:
        table.add_row(s["table"], s["label"], str(s["rows"]),
                      f"{s['size_kb']} KB", f"{s['seconds']}s")
    console.print(table)

    # 校验：DuckDB 直接读回 GeoParquet 并统计
    n = con.execute("SELECT count(*) FROM read_parquet('data/gpq/poi.gpq')").fetchone()[0]
    console.print(f"[green]读回校验：poi.gpq 共 {n} 行，GeoParquet 可用[/]")


if __name__ == "__main__":
    main()
