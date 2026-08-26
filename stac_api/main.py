"""stac_api —— 轻量 STAC API 服务（第5月 W2 交付物）

核心概念：STAC API 是什么
- W1 的 STAC 目录是"静态 JSON 文件"；STAC API 是把这些文件变成"REST 服务"，
  客户端可以用 HTTP 问：有哪些数据集？某集合有什么影像？按时间/空间过滤出哪些？
- 本实现覆盖 STAC API 规范的核心端点（子集），可被任何 STAC 客户端调用：

    GET  /                              → 根目录（catalog）
    GET  /collections                   → 全部数据集
    GET  /collections/{id}              → 单个数据集
    GET  /collections/{id}/items        → 数据集内影像（支持过滤）
    GET  /collections/{id}/items/{id2}  → 单景影像
    GET/POST /search                    → 跨集合检索（STAC API 的灵魂端点）

核心概念：检索三要素
- datetime：时间过滤，支持区间（如 2020-01-01T00:00:00Z/2022-12-31T23:59:59Z）
- bbox：空间过滤，[west, south, east, north] 四角
- limit：返回条数上限（STAC API 默认 10，这里默认 100）
- 组合使用 = "2020~2022 年、覆盖某区域、云量 < 5%"这类查询的基础

生产环境：正式部署用 stac-fastapi（pgstac/sqlalchemy 后端）+ PostgreSQL；
本实现语义一致、零依赖，直接读 W1 生成的 data/stac/ 目录。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from pystac import Catalog

ROOT = Path(__file__).resolve().parents[1]
STAC_DIR = ROOT / "data" / "stac"

app = FastAPI(title="GeoSense STAC API", version="0.1.0",
              description="深圳湾 Sentinel-2 影像目录（STAC API 子集实现）")

_catalog: Catalog | None = None


def _get_catalog() -> Catalog:
    """惰性加载 STAC 目录（读 W1 生成的 catalog.json）。"""
    global _catalog
    if _catalog is None:
        _catalog = Catalog.from_file(str(STAC_DIR / "catalog.json"))
    return _catalog


# ---------------------------------------------------------------------------
# 过滤逻辑
# ---------------------------------------------------------------------------
def _parse_datetime_range(dt_str: str | None) -> tuple[datetime | None, datetime | None]:
    """'2020-01-01T00:00:00Z/2022-12-31T23:59:59Z' → (start, end)；'..' 表示开放边界。"""
    if not dt_str:
        return None, None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start_s, _, end_s = dt_str.partition("/")
    start = datetime.strptime(start_s, fmt).replace(tzinfo=timezone.utc) if start_s not in ("", "..") else None
    end = datetime.strptime(end_s, fmt).replace(tzinfo=timezone.utc) if end_s not in ("", "..") else None
    return start, end


def _in_bbox(item, bbox: list[float] | None) -> bool:
    if not bbox:
        return True
    w, s, e, n = bbox
    ib = item.bbox
    return not (ib[2] < w or ib[0] > e or ib[3] < s or ib[1] > n)  # 相交即命中


def _filter_items(items, datetime_range, bbox, limit) -> list[dict]:
    start, end = _parse_datetime_range(datetime_range)
    out = []
    for it in items:
        if start and it.datetime < start:
            continue
        if end and it.datetime > end:
            continue
        if not _in_bbox(it, bbox):
            continue
        out.append(it.to_dict())
        if len(out) >= limit:
            break
    return out


def _search(datetime_range: str | None, bbox: list[float] | None, limit: int, collection_id: str | None = None):
    catalog = _get_catalog()
    items = list(catalog.get_all_items())
    if collection_id:
        items = [it for it in items if it.collection_id == collection_id]
    hits = _filter_items(items, datetime_range, bbox, limit)
    return {
        "type": "FeatureCollection",
        "features": hits,
        "links": [],
        "numberMatched": len(hits),
        "numberReturned": len(hits),
    }


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
@app.get("/")
def root(request: Request):
    """根目录：返回 API 视角的 catalog（链接指向 API 端点，而非磁盘文件）。

    核心概念：STAC API 的"可发现性"
    - 静态 catalog.json 里的链接指向磁盘文件；API 必须改写成指向自己端点的链接
      （self/root/data/search/child），客户端才能顺着链接"爬"到 collections/items。
    - conformsTo 声明本服务实现了哪些规范（core/collections/item-search），
      标准客户端据此决定是否走 API 协议。
    """
    base = str(request.base_url).rstrip("/")
    catalog = _get_catalog()
    d = catalog.to_dict()
    d["conformsTo"] = [
        "https://api.stacspec.org/v1.0.0/core",
        "https://api.stacspec.org/v1.0.0/collections",
        "https://api.stacspec.org/v1.0.0/item-search",
    ]
    d["links"] = [
        {"rel": "self", "href": base + "/", "type": "application/json"},
        {"rel": "root", "href": base + "/", "type": "application/json"},
        {"rel": "data", "href": base + "/collections", "type": "application/json"},
        {"rel": "search", "href": base + "/search", "type": "application/json",
         "method": "GET"},
    ]
    for child in catalog.get_children():
        d["links"].append({"rel": "child", "href": f"{base}/collections/{child.id}",
                           "type": "application/json", "title": child.title})
    return d


@app.get("/collections")
def collections():
    catalog = _get_catalog()
    return {"collections": [c.to_dict() for c in catalog.get_children()],
            "links": []}


@app.get("/collections/{collection_id}")
def collection_detail(collection_id: str):
    catalog = _get_catalog()
    for c in catalog.get_children():
        if c.id == collection_id:
            return c.to_dict()
    raise HTTPException(404, f"未找到集合: {collection_id}")


@app.get("/collections/{collection_id}/items")
def collection_items(collection_id: str,
                     datetime: str | None = None,
                     bbox: str | None = Query(None, description="west,south,east,north"),
                     limit: int = Query(100, ge=1, le=1000)):
    return _search(datetime, _parse_bbox(bbox), limit, collection_id)


@app.get("/collections/{collection_id}/items/{item_id}")
def collection_item(collection_id: str, item_id: str):
    catalog = _get_catalog()
    for it in catalog.get_all_items():
        if it.id == item_id and it.collection_id == collection_id:
            return it.to_dict()
    raise HTTPException(404, f"未找到影像: {item_id}")


class SearchBody(BaseModel):
    collections: list[str] | None = None
    datetime: str | None = None
    bbox: list[float] | None = None
    limit: int = 100


@app.post("/search")
def search_post(body: SearchBody):
    return _search(body.datetime, body.bbox, body.limit,
                   body.collections[0] if body.collections else None)


@app.get("/search")
def search_get(datetime: str | None = None,
               bbox: str | None = Query(None, description="west,south,east,north"),
               limit: int = Query(100, ge=1, le=1000)):
    return _search(datetime, _parse_bbox(bbox), limit)


def _parse_bbox(bbox_str: str | None) -> list[float] | None:
    if not bbox_str:
        return None
    parts = [float(x) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise HTTPException(422, "bbox 需为 west,south,east,north 四个数字")
    return parts


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
