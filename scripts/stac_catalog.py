"""第5月 W1 交付物：stac_catalog.py —— 为 COG 影像创建 STAC 目录
（2026-09-23 region-data-inventory 变更泛化：region 自动分 collection，去硬编码）

核心概念：STAC（SpatioTemporal Asset Catalog，时空资产目录）
- 遥感数据"没有 STAC 前"：一堆 tif 文件躺在磁盘，外人不知道有什么、
  覆盖哪、何时拍的、怎么用。检索只能靠人肉翻文件名。
- "有了 STAC 后"：每个场景变成一份标准 JSON（Item），组织成 Collection，
  任何人/程序都能用统一协议问"2025 年云量 < 10% 覆盖郑州的影像有哪些"。

核心概念：三层结构（学习计划第5月 W1 的核心）
  Catalog（根）→ Collection（数据集）→ Item（单景影像）
  - Item：单景影像的最小描述单元。必含 4 要素：
      geometry（覆盖范围，GeoJSON）、datetime（拍摄时间）、
      assets（关联的数据文件）、properties（扩展属性：云量/浑浊度/波段）
  - Collection：一组同源 Item 的集合，声明时空范围与波段摘要
  - Catalog：最外层容器，组织多个 Collection
- 与数据分离：STAC 只写元数据 JSON，不复制/移动影像本身 —— 资产用相对路径引用。

⚠ 本脚本的定位（region-data-inventory D2）：STAC 是 manifest.json 的
  「标准化导出视图」，**manifest 是唯一真相**——新影像下载即入 manifest，
  STAC 需重跑本脚本同步；消费方（data_inventory）直读 manifest 而非本目录，
  避免 stale（金水区案例的实证：郑州两景只在 manifest，STAC 漂移）。

本脚本：读 data/cogs/manifest.json → 按 region 字段自动分组生成
data/stac/（catalog.json + 每个 region 一个 collection + items/），pystac 校验。
region 未登记 REGION_META 时自动生成元数据（开放-封闭：新区域加数据即生效）。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table
from pystac import (Asset, Catalog, CatalogType, Collection, Extent, Item,
                    MediaType, SpatialExtent, Summaries, TemporalExtent)

console = Console()
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "cogs" / "manifest.json"
OUT_DIR = ROOT / "data" / "stac"

LICENSE = "proprietary"  # 合成数据为本项目生成；真实场景来自 Sentinel-2（CC-BY 4.0）


def legacy_region(scene: dict) -> str:
    """region 归一：老条目无 region 字段 → 回落 szbay（提案红旗 R2，inventory 同口径标注）。"""
    return scene.get("region") or "szbay"


def collection_meta(region: str, scenes: list[dict]) -> dict:
    """collection 元数据：REGION_META 注册表给显示文案；未注册自动生成（D1 开放语义）。"""
    from scripts.data_inventory import REGION_META
    meta = REGION_META.get(region, {})
    dates = sorted({s["date"] for s in scenes})
    label = meta.get("label", region)
    desc = meta.get("desc") or f"{label} Sentinel-2 场景库（{dates[0]} ~ {dates[-1]}，{len(scenes)} 景）"
    return {"label": label, "title": f"Sentinel-2 {label}影像", "description": desc}


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


def build_scene_item(scene: dict, collection_id: str) -> Item:
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
        collection=collection_id,
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


def build_collection(region: str, scenes: list[dict]) -> Collection:
    """创建 Collection：声明时空范围 + 波段/云量摘要（bands 取该 region 实际波段的并集）。"""
    meta = collection_meta(region, scenes)
    # STAC spatial extent 要求 bbox 数组去重后为 1 个或 ≥3 个（同 bbox 多景会触发重复数组校验失败）
    uniq_bboxes = []
    for b in (s["bbox"] for s in scenes):
        if b not in uniq_bboxes:
            uniq_bboxes.append(b)
    dates = sorted(parse_dt(s["date"]) for s in scenes)
    clouds = [s.get("cloud", 0.0) for s in scenes]
    bands = sorted({b for s in scenes for b in s.get("bands", [])})

    extent = Extent(
        spatial=SpatialExtent(uniq_bboxes),
        temporal=TemporalExtent([[dates[0], dates[-1]]]),
    )
    collection = Collection(
        id=f"sentinel2-{region}",
        title=meta["title"],
        description=meta["description"],
        license=LICENSE,
        extent=extent,
        summaries=Summaries({
            "cloud_cover": {"minimum": min(clouds), "maximum": max(clouds)},
            "bands": bands,
        }),
    )
    return collection


def main() -> None:
    scenes = json.loads(MANIFEST.read_text(encoding="utf-8"))
    console.print(f"[dim]影像清单：{len(scenes)} 景[/]")

    # 0. 按 region 分组（legacy 无 region → szbay；开放语义：任意 region 名都成 collection）
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scenes:
        groups[legacy_region(s)].append(s)
    console.print(f"[dim]区域分组：{ {k: len(v) for k, v in sorted(groups.items())} }[/]")

    # 1. 根 Catalog
    catalog = Catalog(
        id="geosense-catalog",
        title="GeoSense 遥感数据目录",
        description="GeoSense Sentinel-2 场景库的 STAC 目录（按 region 自动分组，manifest 唯一真相的导出视图）",
    )

    # 2~3. 每个 region 一个 Collection + 其下 Items
    n_items = 0
    for region in sorted(groups):
        region_scenes = groups[region]
        collection = build_collection(region, region_scenes)
        items = [build_scene_item(s, f"sentinel2-{region}") for s in region_scenes]
        for it in items:
            collection.add_item(it)
        catalog.add_child(collection)
        n_items += len(items)

    # 4. 落盘（SELF_CONTAINED：catalog.json + <collection>/collection.json + items/*.json）
    # 重建前清旧目录（旧版单 collection 结构残留会混入）
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog.normalize_and_save(str(OUT_DIR), catalog_type=CatalogType.SELF_CONTAINED)
    console.print(f"[green]STAC 目录已写入：{OUT_DIR}（{len(groups)} collections / {n_items} items）[/]")

    # 5. 重新打开并校验（pystac 官方校验器）
    reopened = Catalog.from_file(str(OUT_DIR / "catalog.json"))
    console.print("[dim]pystac 校验：[/]", end="")
    reopened.validate_all()
    console.print("[green]通过 ✓[/]")

    # 6. 目录树概览
    table = Table(title="STAC 目录树（region 自动分组）")
    table.add_column("Collection")
    table.add_column("Items", justify="right")
    table.add_column("波段")
    for region in sorted(groups):
        cols = sorted({b for s in groups[region] for b in s.get("bands", [])})
        table.add_row(f"sentinel2-{region}", str(len(groups[region])), ",".join(cols))
    console.print(table)

    # 7. manifest 一致性自检：item 总数 == manifest 条目数（防重建漏景）
    assert n_items == len(scenes), f"STAC items {n_items} != manifest {len(scenes)}"
    console.print(f"[green]manifest 一致性 ✓（{n_items}/{len(scenes)}）[/]")


if __name__ == "__main__":
    main()
