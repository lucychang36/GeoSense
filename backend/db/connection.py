"""PostGIS 数据库连接层 —— 代码访问数据库的唯一入口。

核心概念：连接串（DSN，Data Source Name）
- 一条 URL 描述数据库位置/账号/库名，代码里不散落主机、密码：
    postgresql://user:password@host:5432/dbname
- 读环境变量 DATABASE_URL（12-Factor 原则），与 config.py 的 LLM 配置一脉相承。

核心概念：为什么每个查询"开新连接、用完即关"？
- 简单可靠：with 语法保证异常时也关闭，不担心连接泄漏、事务悬挂；
- 代价是每次有 TCP 握手开销。生产环境用连接池（psycopg_pool / PgBouncer）
  复用连接。学习项目数据量小，先求正确，阶段2 再优化。

核心概念：参数化查询（防 SQL 注入的执行层）
- 值一律用 %s 占位 + params 传入，绝不 f-string 拼接用户输入；
- 驱动把参数与 SQL 分开传输，值里就算带单引号也只会被当作"值"，
  无法改变 SQL 结构 —— 这是注入防护的最后一道（也是最可靠的一道）闸。
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from ..core.config import db_config

# 约定：查询结果中几何列的别名必须叫 geom（前端据此画图）
GEOM_FIELD = "geom"


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """打开一个连接，退出 with 时自动关闭（即使中途抛异常）。"""
    conn = psycopg.connect(db_config.url, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def query(sql: str, params: tuple | list | None = None) -> list[dict[str, Any]]:
    """执行一条 SELECT，返回行列表（每行是 dict）。

    row_factory=dict_row 让每行变成 {列名: 值}，方便后续拼 GeoJSON。
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:      # 非 SELECT（正常情况下不会发生）
                return []
            return cur.fetchall()


def execute(sql: str, params: tuple | list | None = None) -> None:
    """执行一条 DDL / DML（建表、插入等），并提交事务。

    与 query 的区别：query 只读返回行；execute 写库并 commit。
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def executemany(sql: str, params_seq: Iterable[tuple | list]) -> None:
    """用同一条 SQL 批量插入多行（如几千条 POI）。

    比逐条 execute 快很多（少开/少提交连接）。psycopg3 的 executemany
    在底层会尽量复用一条语句；数据量再大一个量级时，生产环境应换 COPY。
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params_seq)
        conn.commit()


def rows_to_geojson(rows: list[dict[str, Any]], geom_field: str = GEOM_FIELD) -> dict:
    """把查询结果拼成 GeoJSON FeatureCollection。

    约定：SQL 里用 `ST_AsGeoJSON(geom) AS geom` 输出几何列；
    这里把每条记录包成一个 Feature：geometry=几何，properties=其余字段。
    """
    features = []
    for row in rows:
        geom_str = row.get(geom_field)
        props = {k: v for k, v in row.items() if k != geom_field}
        features.append({
            "type": "Feature",
            "geometry": json.loads(geom_str) if geom_str else None,
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}
