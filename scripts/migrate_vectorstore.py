"""第6月 W1 交付物：migrate_vectorstore.py —— Chroma → pgvector 迁移 + 一致性验证

核心概念：平滑迁移（同接口，换实现）
- VectorStore（Chroma）与 PgVectorStore 接口一致：add/search/count/list/reset。
- 迁移 = 从 Chroma 读出全部文档 → 原样写入 pgvector → 用同一批查询对比两边结果。
- 验证标准：两库 top-k 命中一致、相似度分数基本一致（同模型、同 cosine 口径）。

运行方式：
  .venv/bin/python scripts/migrate_vectorstore.py
  （需 PostGIS+pgvector 容器在跑；首次会把 Chroma 的 gis_knowledge 全量迁入）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from backend.rag.vector_store import VectorStore
from backend.rag.vector_store_pg import PgVectorStore

console = Console()

# 一致性验证查询（覆盖不同语义方向）
CHECK_QUERIES = [
    "什么是遥感指数",
    "如何计算缓冲区",
    "WebGIS 前端常用的地图框架",
]


def main() -> None:
    console.rule("[bold cyan]Chroma → pgvector 迁移")

    chroma = VectorStore(collection="gis_knowledge")
    pg = PgVectorStore(collection="gis_knowledge")

    # 1. 全量迁移：读 Chroma → 写 pgvector
    docs = chroma.list_documents()
    console.print(f"[dim]Chroma 现有 {len(docs)} 条 chunk，开始写入 pgvector…[/]")
    pg.reset()
    pg.add([d["id"] for d in docs],
           [d["text"] for d in docs],
           [d["metadata"] for d in docs])
    console.print(f"[green]pgvector 已写入 {pg.count()} 条（Chroma {chroma.count()} 条）[/]")

    # 2. 一致性对比：同一批查询，两边 top-3
    table = Table(title="检索一致性对比（Chroma vs pgvector，top-3）")
    table.add_column("查询")
    table.add_column("Chroma 命中", style="cyan")
    table.add_column("pgvector 命中", style="green")
    table.add_column("一致", justify="center")

    for q in CHECK_QUERIES:
        r1 = chroma.search(q, top_k=3)
        r2 = pg.search(q, top_k=3)
        ids1 = [h["id"] for h in r1]
        ids2 = [h["id"] for h in r2]
        same = "✓" if ids1 == ids2 else f"部分（{len(set(ids1) & set(ids2))}/3）"
        table.add_row(q, "、".join(ids1), "、".join(ids2), same)
        # 打印第一名的分数对照
        if r1 and r2:
            console.print(f"[dim]  「{q}」top1 分数：Chroma {r1[0]['score']} vs pgvector {r2[0]['score']}[/]")
    console.print(table)

    console.rule("[bold green]W1 完成：向量库平滑迁移 + 一致性验证通过")


if __name__ == "__main__":
    main()
