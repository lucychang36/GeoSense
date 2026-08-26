"""文档处理流水线 —— 把原始文档变成可入库的 chunk。

核心概念：为什么文档要"切块"？
1. Embedding 模型有输入上限（bge-small-zh 约 512 token，中文约 300~400 字）。
   整篇 PDF 几千字，一次塞不下；硬塞也会把语义"稀释"成一锅粥。
2. 检索精度：知识库返回的是"片段"，用户想看的是和问题最相关的那几行。
   切得越精准，检索越聚焦 —— 这是 RAG 质量的核心变量。

核心概念：chunk_size 与 chunk_overlap
- chunk_size：每块多大。太小 → 语义不完整；太大 → 检索不聚焦。
  bge 系模型经验值 200~500 字。
- chunk_overlap：相邻两块重叠多少。文档在"块边界"处语义会断裂，
  重叠相当于给切缝打补丁，保留跨边界的上下文（尤其切在句子中间时）。

核心概念：按结构递归切分（优于"硬切字数"）
- 优先在标题/段落/句号处切，尽量不切断完整句子；
- 只有实在找不到分隔符时，才退化成按固定字数硬切。
"""
from __future__ import annotations

import re
from pathlib import Path

# 分隔符优先级：从大到小（先整段，再句子，再逗号，最后空格）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]

# Markdown/HTML 标题的简单提取正则（用于生成 chunk 标题元数据）
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


class DocumentProcessor:
    """文档加载 + 递归切片。"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # 第一半：把不同格式的文档抽成纯文本
    # ------------------------------------------------------------------
    def load(self, path: str | Path) -> str:
        """按扩展名分派到对应解析器，返回纯文本。"""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".md", ".markdown", ".txt"):
            return path.read_text(encoding="utf-8")
        if suffix in (".html", ".htm"):
            return self._load_html(path)
        if suffix == ".pdf":
            return self._load_pdf(path)
        raise ValueError(f"不支持的文档格式: {suffix}")

    @staticmethod
    def _load_html(path: Path) -> str:
        """HTML → 纯文本：剔除 script/style 标签，保留结构换行。"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)

    @staticmethod
    def _load_pdf(path: Path) -> str:
        """PDF → 纯文本：逐页抽取并拼接。"""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    # ------------------------------------------------------------------
    # 第二半：把纯文本切成 chunk
    # ------------------------------------------------------------------
    def split(self, text: str, source: str = "") -> list[dict]:
        """主入口：文本 → 带元数据的 chunk 列表。"""
        text = text.strip()
        units = self._recursive_split(text, _SEPARATORS)
        chunks = self._apply_overlap(units)
        return [
            {"text": c, "source": source, "index": i}
            for i, c in enumerate(chunks)
            if c.strip()
        ]

    def _recursive_split(self, text: str, seps: list[str]) -> list[str]:
        """递归切分：从大到小找分隔符，把长文本切成 ≤ chunk_size 的小块。"""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        # 找当前层级能用的第一个分隔符
        sep = next((s for s in seps if s in text), None)
        if sep is None:
            # 实在找不到分隔符 → 退化硬切
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # 按分隔符切开，每段再递归用更小的分隔符切
        pieces: list[str] = []
        for part in text.split(sep):
            pieces.extend(self._recursive_split(part, seps[1:]))
        # 切散的小块用原分隔符尽量拼回，减少碎片化
        return self._merge(pieces, sep)

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        """贪心合并：在不超 chunk_size 的前提下，把小块拼成大块。"""
        merged, cur = [], ""
        for p in pieces:
            cand = p if not cur else cur + sep + p
            if len(cand) <= self.chunk_size:
                cur = cand
            else:
                if cur:
                    merged.append(cur)
                cur = p
        if cur:
            merged.append(cur)
        return merged

    def _apply_overlap(self, units: list[str]) -> list[str]:
        """给相邻 chunk 加重叠：下一块开头带上上一块结尾的 overlap 字数。"""
        if len(units) <= 1 or self.chunk_overlap <= 0:
            return units
        chunks = [units[0]]
        for u in units[1:]:
            chunks.append(chunks[-1][-self.chunk_overlap:] + u)
        return chunks

    def extract_title(self, text: str, fallback: str = "") -> str:
        """抽取文档标题：优先第一个 Markdown 标题，否则退回文件名。"""
        m = _HEADING_RE.search(text)
        return m.group(1).strip() if m else fallback
