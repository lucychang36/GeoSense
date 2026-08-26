"""向量存储层 —— RAG 系统的"记忆体"。

核心概念：向量数据库（Vector Database）
- 专为"高维向量相似度检索"设计的数据库。普通数据库用 = 精确匹配，
  向量库问的是"哪条记录和这个向量最像"。
- 关键技术：ANN 近似最近邻索引（HNSW 图导航），在亿级向量里毫秒级返回 Top-K，
  代价是"近似"而非绝对精确 —— 这正是"检索延迟 < 2秒"验收标准的来源。

核心概念：Chunk（检索单元）
- 文档太长不能整篇塞进 Embedding（超长会稀释语义、浪费 token）。
- 知识库的最小单元是"切片 chunk"：一段自包含的文本 + 元数据。
- 本项目的 chunk = 一条术语定义（text 字段）+ 元数据（term/category/source）。

核心概念：为什么用 Chroma、封装一层？
- 开发期用 Chroma（本地持久化、零基础设施）；生产期换 pgvector（直连 PostgreSQL）。
- VectorStore 是统一接口：Agent/RAG 只面对 add()/search()，底层换库不动业务代码。
"""
from __future__ import annotations

from typing import Any

from ..core.config import PROJECT_ROOT
from ..core.embedding import EmbeddingModel

# 向量库持久化目录（Chroma 落盘到这里，跨进程可复用）
# 用 config.PROJECT_ROOT 统一定位项目根，避免手算 parents[N] 层级出错
STORE_DIR = PROJECT_ROOT / "data" / "vectorstore"


class VectorStore:
    """基于 Chroma 的持久化向量库封装。"""

    def __init__(self, collection: str = "gis_knowledge", path: Path = STORE_DIR):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        # cosine 空间：与 W3 的余弦相似度口径一致（归一化向量下 cosine=点积）
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._embed = EmbeddingModel.get()

    def add(self, ids: list[str], documents: list[str],
            metadatas: list[dict] | None = None) -> None:
        """写入一批 chunk：文本 → 向量，连同元数据一起入库。"""
        vectors = self._embed.encode(documents).tolist()
        self._col.add(ids=ids, documents=documents, embeddings=vectors,
                      metadatas=metadatas)

    def search(self, query: str, top_k: int = 3,
               where: dict[str, Any] | None = None) -> list[dict]:
        """语义搜索：返回 [{id, text, metadata, score}]，score 为余弦相似度。

        参数 where：元数据过滤（如 {"category": "遥感"}），这是"混合检索"里
        "过滤后召回"的一步 —— 先用元数据缩小范围，再在范围内做语义相似。
        """
        qvec = self._embed.encode([query]).tolist()
        res = self._col.query(query_embeddings=qvec, n_results=top_k, where=where)
        hits = []
        for i, doc_id in enumerate(res["ids"][0]):
            hits.append({
                "id": doc_id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                # Chroma cosine 空间返回 distance = 1 - 相似度，反推回相似度
                "score": round(1 - res["distances"][0][i], 4),
            })
        return hits

    def count(self) -> int:
        return self._col.count()

    def list_documents(self) -> list[dict]:
        """返回语料库全部文档 [{id, text, metadata}]。

        用途：混合检索的关键词索引（BM25）需要拿到全部原文建倒排索引，
        而向量库只负责存向量 —— 所以这里把原文再取出来。
        """
        res = self._col.get()
        return [
            {"id": i, "text": d, "metadata": m}
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def reset(self) -> None:
        """清空集合（教学演示用，生产环境勿轻易调用）。"""
        try:
            self._client.delete_collection(self._col.name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._col.name, metadata={"hnsw:space": "cosine"}
        )
