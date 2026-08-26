"""Embedding 引擎 —— 把文本变成"可计算的语义"。

核心概念：Embedding（向量嵌入）
- 一个大模型训练出来的映射函数：任意文本 → 固定维度的向量（如 512 维）。
- 语义相近的文本，向量在空间中的距离也近。"叠加分析"和"图层重叠"
  没有一个字相同，但向量很接近 —— 这就是语义搜索的全部魔法。

核心概念：为什么不用 DeepSeek API？
- DeepSeek 目前没有 Embedding 接口。开源模型（BGE 系列）中英文效果好、
  可离线运行、零成本 —— RAG 系统里 Embedding 调用量远大于 LLM，
  本地化是标准做法。生产环境也可换 OpenAI text-embedding-3 / 硅基流动 bge-m3。

模型选择：BAAI/bge-small-zh-v1.5（约 100MB，512 维，中文优化，MTEB 中文榜前列）
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


class EmbeddingModel:
    """本地 Embedding 模型封装（单例，模型只加载一次）。"""

    _instance: "EmbeddingModel | None" = None

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        # 优先尝试在线加载（首次会下载）；若网络抖动（代理断连）导致失败，
        # 回退到 local_files_only 直接读本地缓存 —— 避免已缓存模型因 HEAD 请求失败而崩溃。
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        self.dim = self.model.get_embedding_dimension()

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: list[str]) -> np.ndarray:
        """文本列表 → (N, dim) 矩阵，已做 L2 归一化。

        核心概念：归一化后，余弦相似度 = 向量点积。
        这是工程上的关键技巧：相似度计算从 O(dim) 的除法开方变成纯点积，
        百万级向量检索时性能差距巨大。
        """
        return self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, corpus_vecs: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
        """查询向量 vs 语料库矩阵 → [(语料索引, 相似度)] 按相似度降序。

        核心概念：余弦相似度（Cosine Similarity）
        - 度量两个向量的"夹角"：1=完全同向，0=正交无关，-1=完全相反。
        - 不管文本长短（向量模长），只比方向 —— 所以适合语义比较。
        """
        q = self.encode([query])  # (1, dim)
        scores = (corpus_vecs @ q.T).flatten()  # 归一化向量点积 = 余弦相似度
        ranked = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in ranked]
