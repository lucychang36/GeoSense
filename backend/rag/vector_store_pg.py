"""向量存储层（生产版）—— 基于 PostgreSQL + pgvector 的实现。

核心概念：pgvector 是什么
- PostgreSQL 的一个扩展：给表加一列 vector(n)，就能用 SQL 做向量相似度检索
  （<=> 余弦距离、<-> L2、<#> 内积），还能和普通 SQL 条件/空间查询混合。
- 相比开发期 Chroma（独立进程、独立存储），pgvector 让"向量 + 业务数据 +
  空间数据"共用同一套 PostgreSQL —— 一条 SQL 同时过滤元数据、算相似度、JOIN 业务表。

核心概念：为什么叫"迁移"而不是"重写"
- VectorStore 是统一接口（add/search/count/list_documents/reset），
  Chroma 与 pgvector 各自实现，业务代码（RAG/Agent）无感知 —— 这正是 W2 就定下的
  "接口稳定、实现可替换"。本文件是第二套实现，切换只改一行实例化代码。

索引：HNSW（近似最近邻）
- 与 Chroma 同源的 ANN 索引；vector_cosine_ops 表示按余弦距离建图，
  和 Chroma 的 cosine 空间口径一致（归一化后 cosine = 点积）。

连接池（第6月 W4 优化）：psycopg_pool
- W2 实测：每次查询新建连接 ≈100ms 建连开销 → 语义检索 144ms。
- 改为模块级连接池（min 1 / max 5，configure 回调注册 vector 适配器），
  检索延迟应回到个位数毫秒 —— 压测见 scripts/benchmark_w4.py。
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ..core.config import PROJECT_ROOT, db_config
from ..core.embedding import EmbeddingModel

# 模块级连接池（单例）：多实例/多线程共享，避免每次查询新建 TCP 连接
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            db_config.url,
            min_size=1, max_size=5,
            open=False,
            # 连接创建时注册 pgvector 适配器（psycopg3 适配器按连接隔离）
            configure=lambda conn: register_vector(conn),
            kwargs={"autocommit": True},
        )
        _pool.open()
        # 解释器退出时优雅关闭后台线程（避免压测脚本结尾报 thread 警告）
        atexit.register(_pool.close)
    return _pool


class PgVectorStore:
    """pgvector 实现的向量库（接口与 Chroma 版 VectorStore 完全一致）。"""

    def __init__(self, collection: str = "gis_knowledge", path: Path | None = None):
        # 参数对齐 Chroma 版（path 无意义，pgvector 数据在数据库里，忽略）
        self._collection = collection
        self._embed = EmbeddingModel.get()
        self._dim = self._embed.encode(["探"]).shape[1]  # bge-small-zh = 512
        with _get_pool().connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    collection TEXT NOT NULL,
                    id         TEXT NOT NULL,
                    text       TEXT NOT NULL,
                    metadata   JSONB NOT NULL DEFAULT '{{}}',
                    embedding  vector({self._dim}),
                    PRIMARY KEY (collection, id)
                )
            """)
            # HNSW 索引：余弦距离；ef_construction=64 / m=16 是教学友好的默认值
            conn.execute("CREATE INDEX IF NOT EXISTS documents_hnsw_idx "
                         "ON documents USING hnsw (embedding vector_cosine_ops)")

    # ------------------------------------------------------------------
    def add(self, ids: list[str], documents: list[str],
            metadatas: list[dict] | None = None) -> None:
        """写入一批 chunk：文本 → 向量，连同元数据入库。"""
        vectors = self._embed.encode(documents).tolist()
        metadatas = metadatas or [{}] * len(documents)
        with _get_pool().connection() as conn:
            for i, doc_id in enumerate(ids):
                conn.execute(
                    "INSERT INTO documents (collection, id, text, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (collection, id) DO UPDATE SET "
                    "  text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                    "  embedding = EXCLUDED.embedding",
                    (self._collection, doc_id, documents[i],
                     json.dumps(metadatas[i], ensure_ascii=False), vectors[i]),
                )

    def search(self, query: str, top_k: int = 3,
               where: dict[str, Any] | None = None) -> list[dict]:
        """语义搜索：返回 [{id, text, metadata, score}]，score 为余弦相似度。

        where 过滤：metadata jsonb 的等值过滤（如 {"category": "遥感"}），
        一条 SQL 里同时完成"过滤 + 相似度排序" —— pgvector 相比 Chroma 的优势。
        """
        qvec = self._embed.encode([query]).tolist()[0]
        sql = ("SELECT id, text, metadata, 1 - (embedding <=> %s::vector) AS score "
               "FROM documents WHERE collection = %s")
        params: list[Any] = [qvec, self._collection]
        if where:
            for k, v in where.items():
                sql += f" AND metadata->>'{k}' = %s"
                params.append(v)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params += [qvec, top_k]

        with _get_pool().connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {"id": r[0], "text": r[1],
             "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
             "score": round(float(r[3]), 4)}
            for r in rows
        ]

    def count(self) -> int:
        with _get_pool().connection() as conn:
            return conn.execute(
                "SELECT count(*) FROM documents WHERE collection = %s",
                (self._collection,),
            ).fetchone()[0]

    def list_documents(self) -> list[dict]:
        with _get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT id, text, metadata FROM documents WHERE collection = %s "
                "ORDER BY id", (self._collection,),
            ).fetchall()
        return [
            {"id": r[0], "text": r[1],
             "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2])}
            for r in rows
        ]

    def reset(self) -> None:
        with _get_pool().connection() as conn:
            conn.execute("DELETE FROM documents WHERE collection = %s",
                         (self._collection,))
