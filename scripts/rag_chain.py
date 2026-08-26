"""第2月 W3 交付物：rag_chain.py —— 混合检索 + RAG 问答端到端

三个实验：
  实验1  单路检索对照：同一个查询，语义 / 关键词 / 混合 三种结果对比
          （演示"缓冲区分析只排第2"的语义失手如何被混合检索救回）
  实验2  RRF 融合过程：看两路排序如何融合成最终排名
  实验3  RAG 问答链路：检索 → 拼带引用的 Prompt → LLM 生成带出处答案

运行方式：
  .venv/bin/python scripts/rag_chain.py    # 实验3 需要 DEEPSEEK_API_KEY
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.core.config import llm_config
from backend.core.llm import LLMClient
from backend.rag.retriever import HybridRetriever
from backend.rag.vector_store import VectorStore

console = Console()
KB_PATH = Path(__file__).resolve().parents[1] / "backend" / "rag" / "data" / "gis_knowledge.json"

RAG_SYSTEM = (
    "你是 GeoSense，空间智能分析助手。请根据下方【参考资料】回答问题。\n"
    "规则：\n"
    "1. 每条资料前有编号 [1][2]...，回答中引用资料处必须标注对应编号；\n"
    "2. 资料中没有的信息不得编造，直接说明'知识库中未找到相关资料'；\n"
    "3. 回答简洁、准确。"
)


def build_store() -> HybridRetriever:
    """初始化语料库：把 gis_knowledge.json 索引进向量库，返回混合检索器。"""
    items = json.loads(KB_PATH.read_text(encoding="utf-8"))
    store = VectorStore(collection="rag_corpus")
    if store.count() == 0:
        store.add(
            ids=[it["id"] for it in items],
            documents=[it["text"] for it in items],
            metadatas=[{"term": it["term"], "category": it["category"],
                        "source": it["source"]} for it in items],
        )
    console.print(f"[dim]语料库就绪：{store.count()} 条 chunk[/]")
    return HybridRetriever(store)


def show_rank_table(title: str, hits: list[dict]) -> None:
    table = Table(title=title)
    table.add_column("#", width=3)
    table.add_column("术语", style="green")
    table.add_column("分类", width=8)
    table.add_column("分数", justify="right", width=8)
    for i, h in enumerate(hits, 1):
        score = f"{h['score']:.4f}" if h["score"] is not None else "融合"
        table.add_row(str(i), h["metadata"]["term"], h["metadata"]["category"], score)
    console.print(table)


def demo_1_compare(retriever: HybridRetriever) -> None:
    console.rule("[bold cyan]实验1：单路检索对照（W3 的失手案例）")
    query = "工厂周边5公里影响哪些地方"
    console.print(f"查询：[bold]{query}[/]（正确答案：缓冲区分析）")
    show_rank_table("语义检索（向量）", retriever.semantic(query, top_k=3))
    show_rank_table("关键词检索（BM25）", retriever.keyword(query, top_k=3))
    show_rank_table("混合检索（RRF）", retriever.hybrid(query, top_k=3))


def demo_2_rrf(retriever: HybridRetriever) -> None:
    console.rule("[bold cyan]实验2：RRF 融合过程")
    query = "ST_DWithin 是什么"
    sem = [h["id"] for h in retriever.semantic(query, top_k=3)]
    kw = [h["id"] for h in retriever.keyword(query, top_k=3)]
    console.print(f"语义排名：{sem}")
    console.print(f"关键词排名：{kw}")
    from backend.rag.retriever import reciprocal_rank_fusion
    console.print(f"[green]RRF 融合后：{reciprocal_rank_fusion(sem, kw)}[/]")


def demo_3_rag_answer(retriever: HybridRetriever, client: LLMClient) -> None:
    console.rule("[bold cyan]实验3：RAG 问答链路（检索 + 引用 + LLM）")
    query = "在 PostGIS 里计算两点距离，为什么有时候得到的数字很小？该怎么做？"
    hits = retriever.hybrid(query, top_k=3)

    # 拼带引用的上下文
    context = "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    console.print(Panel(context, title="检索到的资料（喂给 LLM）", border_style="dim"))
    console.print(f"[dim]→ 检索到的来源：{[h['metadata']['term'] for h in hits]}[/]")

    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"【参考资料】\n{context}\n\n问题：{query}"},
    ]
    answer, usage = client.chat(messages)
    console.print(Panel(answer, title="GeoSense（带引用）", border_style="green"))
    console.print(f"[yellow]Token 用量：{usage}")


def main() -> None:
    retriever = build_store()
    demo_1_compare(retriever)
    demo_2_rrf(retriever)

    if not llm_config.available:
        console.print("[yellow]实验3 需配置 DEEPSEEK_API_KEY（当前跳过）[/]")
        return
    demo_3_rag_answer(retriever, LLMClient())
    console.rule("[bold green]W3 完成：混合检索 + RAG 问答跑通")


if __name__ == "__main__":
    main()
