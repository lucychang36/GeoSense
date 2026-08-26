"""第5月 W3 交付物：duckdb_spatial.py —— GeoParquet + DuckDB Spatial 查询与基准

核心概念：为什么 DuckDB 快？
- 列式存储：只读查询涉及的列（如 count 只扫元数据，比行式整读快几个数量级）；
- 向量化执行 + 多核并行：单机就能榨干 CPU 吞吐 —— 无需分布式即可百万级秒查。

四个实验：
  实验1  真实数据空间查询：POI 按区县统计（ST_Within 点在区内）
  实验2  真实数据 bbox 过滤（ST_Intersects + 信封）
  实验3  百万级基准：合成 500 万点（含 geometry 列），count / bbox / 空间连接计时
  实验4  列式威力：只读 count（不扫 geometry）vs 全列读取，对比耗时

运行方式：
  .venv/bin/python scripts/duckdb_spatial.py   （需先跑 export_geoparquet.py）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
from rich.console import Console
from rich.table import Table

from backend.core.config import PROJECT_ROOT

console = Console()
GPQ = PROJECT_ROOT / "data" / "gpq"
POI = "read_parquet('data/gpq/poi.gpq')"
BOUND = "read_parquet('data/gpq/admin_boundary.gpq')"
SYNTH = "read_parquet('data/gpq/synth_5m.gpq')"
N_SYNTH = 5_000_000


def timed(con: duckdb.DuckDBPyConnection, sql: str, label: str) -> float:
    t0 = time.time()
    result = con.execute(sql).fetchall()
    dt = (time.time() - t0) * 1000
    console.print(f"  [green]{label:<22}[/] {dt:>8.1f} ms  → {result}")
    return dt


def main() -> None:
    con = duckdb.connect()
    con.execute("LOAD spatial")
    # 让 DuckDB 用满本机核心（列式 + 并行）
    con.execute("SET threads TO 8")

    # 实验1：真实数据 —— 每个区县有多少 POI（ST_Within 点在区内）
    console.rule("[bold cyan]实验1：POI 按区县统计（ST_Within）")
    timed(con, f"""
        SELECT d.name, count(p.id) AS n
        FROM {POI} p JOIN {BOUND} d ON ST_Within(p.geometry, d.geometry)
        GROUP BY d.name ORDER BY n DESC
    """, "按区统计 POI")

    # 实验2：真实数据 —— bbox 过滤（福田中心 3km 见方的 POI）
    console.rule("[bold cyan]实验2：bbox 空间过滤（ST_Intersects）")
    timed(con, f"""
        SELECT count(*) FROM {POI} p
        WHERE ST_Intersects(p.geometry,
              ST_Envelope(ST_GeomFromText('LINESTRING(113.95 22.5, 114.05 22.56)')))
    """, "福田中心区域 POI 数")

    # 实验3：百万级基准 —— 合成 500 万点
    console.rule(f"[bold cyan]实验3：百万级基准（合成 {N_SYNTH // 10000} 万点，含 geometry 列）")
    synth_path = GPQ / "synth_5m.gpq"
    if not synth_path.exists():
        console.print("[dim]生成合成数据（一次约 20~40s）…[/]")
        con.execute(f"""
            COPY (
                SELECT lon, lat, ST_Point(lon, lat) AS geometry
                FROM (SELECT 113.7 + random() * 0.6 AS lon, 22.4 + random() * 0.4 AS lat
                      FROM range({N_SYNTH}))
            ) TO 'data/gpq/synth_5m.gpq' (FORMAT PARQUET)
        """)
        console.print(f"[dim]已生成 {synth_path.name}（{synth_path.stat().st_size / 1e6:.0f} MB）[/]")

    timed(con, f"SELECT count(*) FROM {SYNTH}", "count(*)")
    timed(con, f"""
        SELECT count(*) FROM {SYNTH}
        WHERE lon BETWEEN 113.9 AND 114.1 AND lat BETWEEN 22.5 AND 22.6
    """, "bbox 过滤（纯列）")
    timed(con, f"""
        SELECT count(*) FROM {SYNTH} p, {BOUND} d
        WHERE d.name = '南山区' AND ST_Within(p.geometry, d.geometry)
    """, "ST_Within 空间连接")

    # 实验4：列式威力 —— count 只扫统计信息，不读 geometry 列
    console.rule("[bold cyan]实验4：列式威力（count vs 全列）")
    timed(con, f"SELECT count(*) FROM {SYNTH}", "只读元数据 count")
    timed(con, f"SELECT count(*), sum(lon), sum(lat) FROM {SYNTH}", "扫 3 列聚合")

    console.rule("[bold green]W3 完成：GeoParquet + DuckDB 百万级秒查跑通")


if __name__ == "__main__":
    main()
