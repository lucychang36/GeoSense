"""第2月 W2 交付物：doc_processor.py —— 文档处理流水线端到端演示

四个实验：
  实验1  文本抽取：Markdown / HTML / PDF 三种格式 → 纯文本
  实验2  递归切片：按结构切成 ≤ chunk_size 的 chunk
  实验3  重叠演示：相邻 chunk 的 overlap 打补丁（肉眼可见）
  实验4  切片入库：chunk → Embedding → 向量库 → 语义检索（W1+W2 打通）

运行方式：
  .venv/bin/python scripts/doc_processor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.rag.doc_processor import DocumentProcessor
from backend.rag.vector_store import VectorStore

console = Console()
DOCS_DIR = Path(__file__).resolve().parents[1] / "backend" / "rag" / "data" / "docs"


def _make_sample_pdf() -> Path:
    """用 fpdf2 生成一份英文 GIS 术语 PDF（核心字体无需中文字体文件）。"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 6, "GIS Glossary for Remote Sensing\n\n"
                          "NDVI (Normalized Difference Vegetation Index) is computed as "
                          "(NIR - Red) / (NIR + Red). It quantifies vegetation greenness.\n\n"
                          "NDWI (Normalized Difference Water Index) uses Green and NIR bands "
                          "to enhance water bodies in satellite imagery.\n\n"
                          "DEM (Digital Elevation Model) records terrain elevation in raster "
                          "form and supports slope and watershed analysis.")
    out = DOCS_DIR / "gis_glossary.pdf"
    pdf.output(str(out))
    return out


def show_chunks(title: str, chunks: list[dict]) -> None:
    table = Table(title=title)
    table.add_column("#", width=3)
    table.add_column("字数", justify="right", width=5)
    table.add_column("内容片段", style="dim", max_width=60)
    for c in chunks:
        table.add_row(str(c["index"]), str(len(c["text"])), c["text"][:58] + "…")
    console.print(table)


def main() -> None:
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=60)

    # 实验1：文本抽取
    console.rule("[bold cyan]实验1：三种格式 → 纯文本")
    md_text = processor.load(DOCS_DIR / "postgis_intro.md")
    html_text = processor.load(DOCS_DIR / "ogc_wms.html")
    pdf_text = processor.load(_make_sample_pdf())
    for name, t in [("Markdown", md_text), ("HTML", html_text), ("PDF", pdf_text)]:
        console.print(f"[green]{name}:[/] 抽取 {len(t)} 字  |  标题 = "
                      f"{processor.extract_title(t, '(无)')}")

    # 实验2：递归切片
    console.rule("[bold cyan]实验2：递归切片（chunk_size=300, overlap=60）")
    md_chunks = processor.split(md_text, source="postgis_intro.md")
    show_chunks("postgis_intro.md 切片结果", md_chunks)

    # 实验3：重叠演示 —— 相邻 chunk 尾部/开头共享的文字
    console.rule("[bold cyan]实验3：重叠（overlap）打补丁")
    if len(md_chunks) >= 2:
        a, b = md_chunks[0]["text"], md_chunks[1]["text"]
        overlap = a[-60:]
        console.print(Panel(f"chunk0 结尾…[bold red]{overlap}[/]\n"
                            f"chunk1 开头…[bold red]{b[:60]}[/]",
                            title="重叠文字（两块的公共部分，避免语义断裂）",
                            border_style="yellow"))

    # 实验4：切片入库 + 检索
    console.rule("[bold cyan]实验4：切片入库 → 语义检索（W1+W2 打通）")
    store = VectorStore(collection="gis_docs")
    ids = [f"{Path(c['source']).stem}_{c['index']}" for c in md_chunks]
    store.add(ids=ids, documents=[c["text"] for c in md_chunks],
              metadatas=[{"source": c["source"], "index": c["index"]} for c in md_chunks])
    console.print(f"[green]已入库 {store.count()} 个切片（collection=gis_docs）[/]")
    hits = store.search("为什么算面积距离之前要先投影", top_k=3)
    for i, h in enumerate(hits, 1):
        console.print(f"  {i}. [{h['score']:.4f}] {h['text'][:50]}…")
    console.rule("[bold green]W2 完成：文档处理流水线跑通")


if __name__ == "__main__":
    main()
