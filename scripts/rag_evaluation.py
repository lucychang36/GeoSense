"""第2月 W4 交付物：rag_evaluation.py —— RAG 检索评估与优化

三个实验：
  实验1  三路检索对照评估：语义 / 关键词 / 混合，看 HitRate / Recall / MRR
  实验2  top_k 优化：不同 top_k 对召回率的影响（找甜点）
  实验3  LLM 忠实度裁判：检索 + 生成后，让 LLM 评判答案是否忠实于资料（可选）

运行方式：
  .venv/bin/python scripts/rag_evaluation.py    # 实验3 需要 DEEPSEEK_API_KEY
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
from backend.rag.evaluation import evaluate
from backend.rag.retriever import HybridRetriever
from backend.rag.vector_store import VectorStore

console = Console()
ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "backend" / "rag" / "data" / "gis_knowledge.json"
EVAL_PATH = ROOT / "backend" / "rag" / "data" / "eval_dataset.json"

JUDGE_SYSTEM = (
    "你是一名严格的 RAG 答案评审。根据【问题】【资料】【候选答案】判断：\n"
    "1. 忠实度(faithfulness)：答案是否只基于资料、没有编造？0~5 分，编造扣分；\n"
    "2. 相关度(relevance)：答案是否直接回答了问题？0~5 分；\n"
    "3. 是否有未标注的引用（用了资料却没标编号）？\n"
    "只输出 JSON：{\"faithfulness\": 数字, \"relevance\": 数字, \"comment\": \"一句点评\"}"
)


def build_retriever() -> HybridRetriever:
    items = json.loads(KB_PATH.read_text(encoding="utf-8"))
    store = VectorStore(collection="rag_corpus")
    if store.count() == 0:
        store.add(ids=[it["id"] for it in items], documents=[it["text"] for it in items],
                  metadatas=[{"term": it["term"], "category": it["category"],
                              "source": it["source"]} for it in items])
    return HybridRetriever(store)


def report(name: str, metrics: dict) -> None:
    table = Table(title=name)
    table.add_column("指标", style="cyan")
    table.add_column("数值", justify="right", style="green")
    table.add_row("HitRate@3", f"{metrics['hit_rate']:.2%}")
    table.add_row("Recall@3", f"{metrics['recall']:.2%}")
    table.add_row("MRR", f"{metrics['mrr']:.4f}")
    console.print(table)


def demo_1_compare(retriever: HybridRetriever, dataset: list[dict]) -> None:
    console.rule("[bold cyan]实验1：三路检索对照评估")
    report("语义检索（向量）", evaluate(lambda q, k: [h["id"] for h in retriever.semantic(q, k)], dataset))
    report("关键词检索（BM25）", evaluate(lambda q, k: [h["id"] for h in retriever.keyword(q, k)], dataset))
    report("混合检索（RRF）", evaluate(lambda q, k: [h["id"] for h in retriever.hybrid(q, k)], dataset))


def demo_2_topk(retriever: HybridRetriever, dataset: list[dict]) -> None:
    console.rule("[bold cyan]实验2：top_k 对召回率的影响（找甜点）")
    table = Table(title="混合检索：top_k 扫描")
    table.add_column("top_k", justify="right")
    table.add_column("Recall", justify="right", style="green")
    table.add_column("MRR", justify="right")
    for k in (1, 2, 3, 5, 8):
        m = evaluate(lambda q, kk: [h["id"] for h in retriever.hybrid(q, kk)], dataset, top_k=k)
        table.add_row(str(k), f"{m['recall']:.2%}", f"{m['mrr']:.4f}")
    console.print(table)
    console.print("[dim]解读：召回率随 top_k 上升而提高，但 k 越大给 LLM 的上下文越长（更贵、更易稀释）。"
                  "工程上取召回率不再明显增长的最小 k 即可。[/]")


def demo_3_judge(retriever: HybridRetriever, client: LLMClient) -> None:
    console.rule("[bold cyan]实验3：LLM 忠实度裁判")
    q = "为什么在 PostGIS 里算两点距离得到的数字很小"
    hits = retriever.hybrid(q, top_k=3)
    context = "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    # 先生成答案（复用 W3 的 RAG 链路）
    answer, _ = client.chat([
        {"role": "system", "content": "你是 GeoSense，据资料回答，引用处标注 [编号]，资料没有的不要编造。"},
        {"role": "user", "content": f"【资料】\n{context}\n\n问题：{q}"},
    ])
    console.print(Panel(answer, title="候选答案", border_style="green"))
    # 再让 LLM 当裁判打分
    verdict, _ = client.chat([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"【问题】{q}\n【资料】{context}\n【候选答案】{answer}"},
    ])
    console.print(Panel(verdict, title="裁判打分（JSON）", border_style="yellow"))


def main() -> None:
    retriever = build_retriever()
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    console.print(f"[dim]评估集：{len(dataset)} 条查询（含标注答案）[/]")

    demo_1_compare(retriever, dataset)
    demo_2_topk(retriever, dataset)

    if llm_config.available:
        demo_3_judge(retriever, LLMClient())
    else:
        console.print("[yellow]实验3 需配置 DEEPSEEK_API_KEY（当前跳过）[/]")
    console.rule("[bold green]W4 完成：RAG 评估闭环 —— 第2个月里程碑达成")


if __name__ == "__main__":
    main()
