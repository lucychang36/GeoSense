"""混合检索器 —— 语义检索 + 关键词检索 + RRF 融合。

核心概念：为什么需要"混合"？
- 语义检索（向量）：理解同义改写（"图层重叠"→"叠加分析"），但面对精确术语、
  缩写、编号时容易失手 —— 还记得 W3 里"缓冲区分析只排第2"吗？
- 关键词检索（BM25）：精确匹配术语/缩写，但不懂同义、不懂上下文。
- 两者互补：一个管"意思像"，一个管"字面像"，融合后召回率和精度都更好。
  这正是学习计划里"混合检索（语义+关键词）"验收标准的来源。

核心概念：BM25（Okapi）
- 关键词检索的经典算法：词频（TF）越高分越高，但用饱和函数避免长文档刷分；
- IDF 压低"常见词"权重（"的"、"是"几乎不加分），抬高"稀有词"权重。

核心概念：RRF（Reciprocal Rank Fusion，倒数排名融合）
- 把多个排序融合成单一排序：每个文档的得分 = Σ 1/(k + rank)（k 通常取 60）。
- 优点：无需归一化、无需调权重，对"两种分数量纲不同"的问题天然免疫。
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .vector_store import VectorStore

# 分词：中文按"字"切（中文无空格），英文/数字/下划线按"词"切（保留 ST_DWithin 这类术语）
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _WORD.finditer(text.lower()):
        tokens.append(m.group())
    tokens.extend(ch for ch in text if _CJK.match(ch))
    return tokens


class BM25:
    """Okapi BM25 关键词检索。"""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d) for d in corpus]
        self.doc_len = [len(t) for t in self.corpus]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        # 文档频率 df：每个词出现在多少篇文档里（用于算 IDF）
        self.df: Counter = Counter()
        for doc in self.corpus:
            self.df.update(set(doc))
        n = len(self.corpus)
        self.idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1) for t, df in self.df.items()}

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        tf = Counter(doc)
        dl = self.doc_len[doc_idx]
        score = 0.0
        for q in query_tokens:
            if q not in self.idf:
                continue
            f = tf.get(q, 0)
            # BM25 核心公式：tf 饱和 + 文档长度归一化
            score += self.idf[q] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def top_indices(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        qt = tokenize(query)
        scored = [(i, self._score(qt, i)) for i in range(len(self.corpus))]
        scored = [x for x in scored if x[1] > 0]  # 只留命中的
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


def reciprocal_rank_fusion(*ranked_id_lists: list[str], k: int = 60) -> list[str]:
    """RRF：把多路排序融合成单一排序，返回按融合分降序的 id 列表。"""
    scores: dict[str, float] = {}
    for lst in ranked_id_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])


class HybridRetriever:
    """混合检索器：语义（向量）+ 关键词（BM25）+ RRF 融合。"""

    def __init__(self, store: VectorStore):
        self.store = store
        self.documents = store.list_documents()
        self.by_id = {d["id"]: d for d in self.documents}
        self.bm25 = BM25([d["text"] for d in self.documents])

    def semantic(self, query: str, top_k: int = 5) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    def keyword(self, query: str, top_k: int = 5) -> list[dict]:
        hits = self.bm25.top_indices(query, top_k)
        return [
            {"id": self.documents[i]["id"],
             "text": self.documents[i]["text"],
             "metadata": self.documents[i]["metadata"],
             "score": round(score, 4)}
            for i, score in hits
        ]

    def hybrid(self, query: str, top_k: int = 5) -> list[dict]:
        """两路召回 → RRF 融合 → 返回 Top-K（附各文档的融合分）。"""
        sem_ids = [h["id"] for h in self.semantic(query, top_k=top_k)]
        kw_ids = [h["id"] for h in self.keyword(query, top_k=top_k)]
        fused_ids = reciprocal_rank_fusion(sem_ids, kw_ids)[:top_k]
        return [
            {"id": did, "text": self.by_id[did]["text"],
             "metadata": self.by_id[did]["metadata"],
             "score": None}  # 融合分非语义相似度，故置空
            for did in fused_ids
        ]
