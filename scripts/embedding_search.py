"""W3 交付物：embedding_search.py —— GIS 术语语义搜索引擎

三个实验：
  实验1  语义搜索：用"大白话"搜专业术语（核心演示）
  实验2  对照实验：同一个查询，关键词匹配 vs 语义搜索
  实验3  语义地图：PCA 把 512 维压到 2 维，肉眼看到"语义聚类"

运行方式（无需 API Key，模型本地运行，首次自动下载约 100MB）：
  .venv/bin/python scripts/embedding_search.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rich.console import Console
from rich.table import Table

from backend.core.embedding import EmbeddingModel

console = Console()

# ---------------------------------------------------------------------------
# GIS 术语知识库：未来 RAG 知识库的最小原型。
# 核心概念：检索单元（chunk）= 术语名 + 定义拼在一起向量化，
# 命中文本和返回文本可以不同（搜的是语义，返回的是知识）。
# ---------------------------------------------------------------------------
GIS_TERMS = [
    {"term": "缓冲区分析", "category": "空间分析", "desc": "以点线面要素为中心，按指定距离向外扩展生成圆形或带状区域，常用于影响范围评估"},
    {"term": "叠加分析", "category": "空间分析", "desc": "将两个或多个图层按几何位置叠置，计算交集、并集或差集，提取跨图层的空间关系"},
    {"term": "空间插值", "category": "空间分析", "desc": "根据离散采样点估算未知位置的数值，如克里金、反距离权重，生成连续表面"},
    {"term": "网络分析", "category": "空间分析", "desc": "基于道路网计算最短路径、服务区范围、最近设施，常用于物流与应急"},
    {"term": "视域分析", "category": "空间分析", "desc": "基于地形高程计算从观察点可见的区域范围，用于选址与景观评估"},
    {"term": "空间索引", "category": "数据库", "desc": "R-Tree、GiST 等加速结构，让数据库快速定位空间对象，避免全表扫描"},
    {"term": "拓扑关系", "category": "数据模型", "desc": "描述空间对象间相交、包含、相邻等不随几何变形改变的关系"},
    {"term": "坐标参考系统", "category": "基础概念", "desc": "定义坐标如何对应地球表面，含地理坐标系（如WGS84）与投影坐标系（如Web墨卡托）"},
    {"term": "矢量数据", "category": "数据模型", "desc": "用点、线、面几何对象表达离散地物，如道路、行政区边界"},
    {"term": "栅格数据", "category": "数据模型", "desc": "用规则格网像素表达连续现象，如卫星影像、高程、气温"},
    {"term": "NDVI 植被指数", "category": "遥感", "desc": "红光与近红外波段归一化差值，量化植被长势与覆盖度"},
    {"term": "NDWI 水体指数", "category": "遥感", "desc": "绿光与近红外波段组合，增强水体信息，用于提取河湖水面"},
    {"term": "数字高程模型", "category": "遥感", "desc": "以栅格记录地表海拔的数据集，是坡度坡向、流域分析的基础"},
    {"term": "监督分类", "category": "遥感", "desc": "用人工标注样本训练分类器，把影像像元划分为建设用地、耕地等类别"},
    {"term": "变化检测", "category": "遥感", "desc": "对比不同时相的影像或分类结果，识别地物随时间的变化区域"},
    {"term": "瓦片金字塔", "category": "WebGIS", "desc": "把地图按多级缩放预先切分成小图片，前端按需加载实现流畅浏览"},
    {"term": "WMS 服务", "category": "WebGIS", "desc": "OGC 标准地图服务，按请求动态渲染并返回指定范围的地图图片"},
    {"term": "WFS 服务", "category": "WebGIS", "desc": "OGC 标准要素服务，返回矢量数据本身而非图片，支持空间过滤查询"},
    {"term": "空间聚类", "category": "空间分析", "desc": "DBSCAN 等算法发现要素的高密度聚集区域，用于热点识别"},
    {"term": "地理编码", "category": "WebGIS", "desc": "把文字地址转换为经纬度坐标，或反向把坐标解析为地址描述"},
]

# 大白话查询：故意不出现术语原词，考验语义理解
DEMO_QUERIES = [
    "怎么算两个图层的重叠部分",
    "想知道一个工厂周边5公里影响哪些地方",
    "怎么从卫星影像里把湖泊河流提取出来",
    "地址文字怎么变成地图上的坐标点",
]


def demo_1_semantic_search(model: EmbeddingModel, vecs: np.ndarray) -> None:
    console.rule("[bold cyan]实验1：语义搜索（大白话 → 专业术语）")
    for query in DEMO_QUERIES:
        hits = model.search(query, vecs, top_k=3)
        table = Table(title=f"查询：{query}", show_lines=False)
        table.add_column("排名", width=4)
        table.add_column("命中术语", style="green")
        table.add_column("相似度", justify="right")
        table.add_column("定义摘要", style="dim", max_width=40)
        for rank, (idx, score) in enumerate(hits, 1):
            item = GIS_TERMS[idx]
            table.add_row(str(rank), item["term"], f"{score:.4f}", item["desc"][:38] + "…")
        console.print(table)


def demo_2_keyword_vs_semantic(model: EmbeddingModel, vecs: np.ndarray) -> None:
    console.rule("[bold cyan]实验2：对照实验 —— 关键词匹配 vs 语义搜索")
    query = "怎么算两个图层的重叠部分"
    # 关键词匹配：查询中的字是否出现在文本里
    kw_hits = [
        (item["term"], sum(1 for ch in set(query) if ch in item["term"] + item["desc"]))
        for item in GIS_TERMS
    ]
    kw_hits.sort(key=lambda x: -x[1])
    console.print(f"查询：[bold]{query}[/]")
    console.print(f"[red]关键词 Top1：{kw_hits[0][0]}（命中 {kw_hits[0][1]} 个字——只是凑巧字面重叠）")
    top = model.search(query, vecs, top_k=1)[0]
    console.print(f"[green]语义搜索 Top1：{GIS_TERMS[top[0]]['term']}（相似度 {top[1]:.4f}——真正理解意思）")
    console.print("[dim]结论：关键词匹配只看字面，语义搜索理解意图。RAG 检索质量的天壤之别就在这里。[/]")


def demo_3_semantic_map(vecs: np.ndarray) -> None:
    """PCA 降维到 2D 并导出坐标 —— 数据会用于绘制语义地图。"""
    console.rule("[bold cyan]实验3：语义地图（PCA 512维 → 2维）")
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(vecs)
    payload = [
        {"term": item["term"], "category": item["category"],
         "x": round(float(x), 3), "y": round(float(y), 3)}
        for item, (x, y) in zip(GIS_TERMS, coords)
    ]
    out = Path(__file__).resolve().parents[1] / "data" / "semantic_map.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    console.print(f"二维坐标已导出：{out}（可用作语义聚类可视化）")


def main() -> None:
    console.print("[dim]加载 Embedding 模型（首次运行自动下载，约 100MB）…[/]")
    model = EmbeddingModel.get()
    console.print(f"[dim]模型就绪：向量维度 {model.dim}[/]")

    texts = [f"{item['term']}：{item['desc']}" for item in GIS_TERMS]
    vecs = model.encode(texts)
    console.print(f"[dim]知识库向量化完成：{vecs.shape[0]} 条 × {vecs.shape[1]} 维[/]")

    demo_1_semantic_search(model, vecs)
    demo_2_keyword_vs_semantic(model, vecs)
    demo_3_semantic_map(vecs)
    console.rule("[bold green]W3 完成：语义搜索跑通")


if __name__ == "__main__":
    main()
