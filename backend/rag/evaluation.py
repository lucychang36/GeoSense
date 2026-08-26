"""RAG 检索评估指标 —— 给检索质量装一把"尺子"。

核心概念：为什么要评估？
- RAG 质量 = 检索质量 × 生成质量。生成好坏肉眼可看，检索好坏却要量化：
  "检索对了没"直接决定 LLM 能不能答对（垃圾进垃圾出）。
- 学习计划验收标准"RAG 问答准确率 > 70%"不是拍脑袋，靠的就是这套指标。

核心概念：三个检索指标（都基于"标注答案" ground truth）
- HitRate@K（命中率）：Top-K 里"至少命中一条正确答案"的问题占比。
  衡量"答不答得上"。
- Recall@K（召回率）：所有正确答案里，被 Top-K 捞回来的比例。
  衡量"漏没漏"。
- MRR（Mean Reciprocal Rank，平均倒数排名）：第一条正确答案排第几的倒数平均。
  衡量"排得靠不靠前"，对排序质量敏感（排第1得1.0，排第10只得0.1）。

三者的关系：命中率看"有没有"，召回率看"全不全"，MRR 看"准不准"。
"""
from __future__ import annotations

from typing import Callable


def hit_rate_at_k(results: list[list[str]], relevant: list[set[str]], k: int) -> float:
    """命中率：Top-K 里命中 ≥1 条正确答案的查询占比。"""
    hits = sum(1 for res, rel in zip(results, relevant) if set(res[:k]) & rel)
    return hits / len(results) if results else 0.0


def recall_at_k(results: list[list[str]], relevant: list[set[str]], k: int) -> float:
    """召回率：每条查询检索出的正确答案 / 该查询全部正确答案，再取平均。"""
    total = 0.0
    for res, rel in zip(results, relevant):
        if not rel:
            continue
        total += len(set(res[:k]) & rel) / len(rel)
    return total / len(results) if results else 0.0


def mrr(results: list[list[str]], relevant: list[set[str]]) -> float:
    """MRR：第一条正确答案的倒数排名，再取平均。"""
    total = 0.0
    for res, rel in zip(results, relevant):
        for rank, doc_id in enumerate(res, start=1):
            if doc_id in rel:
                total += 1.0 / rank
                break
    return total / len(results) if results else 0.0


def evaluate(
    retrieve: Callable[[str, int], list[str]],
    dataset: list[dict],
    top_k: int = 3,
) -> dict:
    """统一评估入口。

    参数 retrieve：检索函数 (query, k) -> [id,...]，这样语义/关键词/混合都能复用。
    返回 {hit_rate, recall, mrr, per_query}。
    """
    results = [retrieve(q["query"], top_k) for q in dataset]
    relevant = [set(q["relevant"]) for q in dataset]
    return {
        "hit_rate": round(hit_rate_at_k(results, relevant, top_k), 4),
        "recall": round(recall_at_k(results, relevant, top_k), 4),
        "mrr": round(mrr(results, relevant), 4),
        "per_query": [
            {"query": q["query"], "retrieved": r, "relevant": q["relevant"]}
            for q, r in zip(dataset, results)
        ],
    }
