"""第2月 W1 交付物：vector_store.py —— 向量数据库搭建 + 检索演示

四个实验：
  实验1  建库索引：把 GIS 知识库 chunk 向量化写入 Chroma（持久化）
  实验2  语义搜索：大白话查询，返回 Top-3 相关知识片段
  实验3  元数据过滤：where 条件在检索前缩小范围（混合检索的第一步）
  实验4  持久化验证：同一进程内二次查询，证明知识库"记住了"（落盘）

运行方式：
  .venv/bin/python scripts/vector_store.py
（首次自动下载 bge 模型；Chroma 数据持久化到 data/vectorstore/）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.rag.vector_store import VectorStore

console = Console()
KB_PATH = Path(__file__).resolve().parents[1] / "backend" / "rag" / "data" / "gis_knowledge.json"


def load_knowledge_base() -> tuple[list, list, list]:
    items = json.loads(KB_PATH.read_text(encoding="utf-8"))
    ids = [it["id"] for it in items]
    docs = [it["text"] for it in items]
    metas = [{"term": it["term"], "category": it["category"],
              "source": it["source"]} for it in items]
    return ids, docs, metas


def show_hits(title: str, hits: list[dict]) -> None:
    table = Table(title=title)
    table.add_column("#", width=3)
    table.add_column("术语", style="green")
    table.add_column("分类", width=8)
    table.add_column("相似度", justify="right")
    table.add_column("内容片段", style="dim", max_width=48)
    for i, h in enumerate(hits, 1):
        table.add_row(str(i), h["metadata"]["term"], h["metadata"]["category"],
                      f"{h['score']:.4f}", h["text"][:46] + "…")
    console.print(table)


def main() -> None:
    console.print("[dim]初始化向量库（首次加载 Embedding 模型）…[/]")
    store = VectorStore()

    # 实验1：建库索引
    console.rule("[bold cyan]实验1：知识库向量化入库")
    if store.count() == 0:
        ids, docs, metas = load_knowledge_base()
        store.add(ids, docs, metas)
        console.print(f"[green]已写入 {store.count()} 条 chunk（文本 → 512 维向量，持久化到磁盘）[/]")
    else:
        console.print(f"[yellow]知识库已存在（{store.count()} 条），跳过重复写入[/]")

    # 实验2：语义搜索
    console.rule("[bold cyan]实验2：语义搜索（大白话 → 知识片段）")
    for q in ["怎么判断两个地方靠得近不近，还能用索引加速",
              "为什么算两点距离得到的是奇怪的小数而不是几公里"]:
        show_hits(f"查询：{q}", store.search(q, top_k=3))

    # 实验3：元数据过滤
    console.rule("[bold cyan]实验3：元数据过滤（限定遥感类目内检索）")
    hits = store.search("怎么把湖泊河流从影像里找出来", top_k=3, where={"category": "遥感"})
    show_hits("查询：怎么把湖泊河流找出来  [仅 category=遥感]", hits)

    # 实验4：持久化验证
    console.rule("[bold cyan]实验4：持久化验证（新实例重新打开，数据仍在）")
    store2 = VectorStore()
    console.print(f"[green]重新打开后知识库仍有 {store2.count()} 条 —— 记忆已落盘，跨进程可用[/]")


if __name__ == "__main__":
    main()
