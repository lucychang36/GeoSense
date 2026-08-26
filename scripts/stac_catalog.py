"""第5月 W1 交付物：stac_catalog.py —— 为 COG 影像创建 STAC 目录

核心概念：STAC（SpatioTemporal Asset Catalog，时空资产目录）
- 遥感数据"没有 STAC 前"：一堆 tif 文件躺在磁盘，外人不知道有什么、
  覆盖哪、何时拍的、怎么用。检索只能靠人肉翻文件名。
- "有了 STAC 后"：每个场景变成一份标准 JSON（Item），组织成 Collection，
  任何人/程序都能用统一协议问"2020 年云量 < 10% 覆盖深圳湾的影像有哪些"。

核心概念：三层结构（学习计划第5月 W1 的核心）
  Catalog（根）→ Collection（数据集）→ Item（单景影像）
  - Item：单景影像的最小描述单元。必含 4 要素：
      geometry（覆盖范围，GeoJSON）、datetime（拍摄时间）、
      assets（关联的数据文件）、properties（扩展属性：云量/浑浊度/波段）
  - Collection：一组同源 Item 的集合，声明时空范围与波段摘要
  - Catalog：最外层容器，组织多个 Collection
- 与数据分离：STAC 只写元数据 JSON，不复制/移动影像本身 —— 资产用相对路径引用。

本脚本：读 data/cogs/manifest.json（第4月 W3 产出的影像清单）→ 生成
data/stac/（catalog.json + collection.json + items/），并用 pystac 校验。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pystac import (Asset, Catalog, CatalogType, Collection, Extent, Item,
                    MediaType, SpatialExtent, Summaries, TemporalExtent)

console = Console()
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "cogs" / "manifest.json"
OUT_DIR = ROOT / "data" / "stac"

# 数据集级元数据
COLLECTION_ID = "sentinel2-szbay"
COLLECTION_TITLE = "Sentinel-2 深圳湾影像（2019-2025）"
COLLECTION_DESCRIPTION = (
    "深圳湾 Sentinel-2 场景库：5 景合成影像（2019-2023 每年 6 月）+ "
    "1 景真实 L2A（2025-07-27）+ 1 景多时相镶嵌。波段 B2/B3/B4/B8，"
    "由 GeoSense 第4月卫星仓库构建，以 COG 格式存储。"
)
LICENSE = "proprietary"  # 合成数据为本项目生成；真实场景来自 Sentinel-2（CC-BY 4.0）


def bbox_to_geometry(bbox: list[float]) -> dict:
    """由 [west, south, east, north] 生成 GeoJSON Polygon（四角闭合）。"""
    w, s, e, n = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


def parse_dt(date_str: str) -> datetime:
    """'2019-06' → 时区感知的 datetime（UTC）；区间 'A~B' 取起始。"""
    date_str = date_str.split("~")[0].strip()
    dt: datetime | None = None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m"):
        try:
            dt = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(f"无法解析日期: {date_str}")
    return dt.replace(tzinfo=timezone.utc)


def build_scene_item(scene: dict) -> Item:
    """把一个场景清单变成 STAC Item（单景影像的元数据卡片）。"""
    item = Item(
        id=scene["id"],
        geometry=bbox_to_geometry(scene["bbox"]),
        bbox=scene["bbox"],
        datetime=parse_dt(scene["date"]),
        properties={
            # 扩展属性（manifest 里的分析元数据）。STAC 要求 properties 值为 JSON 标量；
            # ⚠ 裸键 `bands` 是 STAC 保留键（正确应走 eo 扩展的 eo:bands），故不加在这里，
            #   波段信息统一放在 Collection.summaries。
            "cloud_cover": scene.get("cloud", 0.0),
            "turbidity": scene.get("turbidity"),
        },
        collection=COLLECTION_ID,
    )
    # 资产：影像本身（role=data）+ 预览图（role=overview）
    item.add_asset(
        "cog",
        Asset(href=str(scene["path"]), media_type=MediaType.COG,
              roles=["data"], title="COG 影像"),
    )
    if scene.get("preview"):
        item.add_asset(
            "preview",
            Asset(href=str(scene["preview"]), media_type=MediaType.PNG,
                  roles=["overview"], title="真彩色预览"),
        )
    return item


def build_collection(scenes: list[dict]) -> Collection:
    """创建 Collection：声明时空范围 + 波段/云量摘要。"""
    bboxes = [s["bbox"] for s in scenes]
    dates = sorted(parse_dt(s["date"]) for s in scenes)
    clouds = [s.get("cloud", 0.0) for s in scenes]

    extent = Extent(
        spatial=SpatialExtent(bboxes),
        temporal=TemporalExtent([[dates[0], dates[-1]]]),
    )
    collection = Collection(
        id=COLLECTION_ID,
        title=COLLECTION_TITLE,
        description=COLLECTION_DESCRIPTION,
        license=LICENSE,
        extent=extent,
        summaries=Summaries({
            "cloud_cover": {"minimum": min(clouds), "maximum": max(clouds)},
            "bands": ["B2", "B3", "B4", "B8"],
        }),
    )
    return collection


def main() -> None:
    scenes = json.loads(MANIFEST.read_text(encoding="utf-8"))
    console.print(f"[dim]影像清单：{len(scenes)} 景[/]")

    # 1. 根 Catalog
    catalog = Catalog(
        id="geosense-sz-catalog",
        title="GeoSense 深圳湾遥感数据目录",
        description="深圳湾 Sentinel-2 场景库的 STAC 目录（第5月 W1）",
    )

    # 2. Collection（同一数据源的所有场景）
    collection = build_collection(scenes)

    # 3. 每景一个 Item，挂到 Collection
    items = [build_scene_item(s) for s in scenes]
    for it in items:
        collection.add_item(it)
    catalog.add_child(collection)

    # 4. 落盘（SELF_CONTAINED：catalog.json + collection.json + items/*.json）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog.normalize_and_save(str(OUT_DIR), catalog_type=CatalogType.SELF_CONTAINED)
    console.print(f"[green]STAC 目录已写入：{OUT_DIR}[/]")

    # 5. 重新打开并校验（pystac 官方校验器）
    reopened = Catalog.from_file(str(OUT_DIR / "catalog.json"))
    console.print("[dim]pystac 校验：[/]", end="")
    result = reopened.validate_all()
    console.print("[green]通过 ✓[/]")

    # 6. 目录树概览
    table = Table(title="STAC 目录树")
    table.add_column("层级", width=12)
    table.add_column("ID")
    table.add_column("要素", style="dim")
    table.add_row("Catalog", catalog.id, "1 个 Collection")
    table.add_row("Collection", collection.id, f"{len(items)} 个 Item · 时空范围已声明")
    for it in items:
        roles = "+".join(sorted({a.roles[0] for a in it.assets.values()}))
        table.add_row("Item", it.id, f"assets({roles}) · cloud={it.properties.get('cloud_cover')}%")
    console.print(table)

    # 7. 示例检索：模拟"2020-2021 年 + 云量<5%"的 STAC 查询
    console.rule("[bold cyan]示例检索：2020~2021 年且云量 < 5%")
    hits = [it for it in items
            if 2020 <= it.datetime.year <= 2021 and it.properties["cloud_cover"] < 5]
    console.print(f"命中 {len(hits)} 景：[green]{[it.id for it in hits]}[/]")


if __name__ == "__main__":
    main()
