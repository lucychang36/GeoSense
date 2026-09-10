"""阶段2 第4月 W2 交付物：tile_server.py —— COG 在线瓦片服务（FastAPI）

核心概念：XYZ 瓦片（XYZ Tiles）
- 地图被切成 256x256 的小图，按三级坐标组织：z=缩放级别，x/y=行列号，
  路径约定 /tiles/{z}/{x}/{y}.png —— WebGIS 前端（MapboxGL/Leaflet）的标准格式。
- Web 墨卡托（EPSG:3857）下，全球范围在级别 z 被分成 2^z × 2^z 个瓦片。

核心概念：CORS（跨域资源共享）—— 瓦片服务必须开
- 前端（:8000）通过 fetch 加载瓦片（Mapbox GL 需要解码/读取像素，
  不是简单的 <img> 标签），浏览器会做同源检查：不同端口的请求
  必须得到 `Access-Control-Allow-Origin` 头，否则报 CORS 错误。
- 瓦片是只读公共资源，这里放开所有来源（allow_origins=["*"]）；
  生产环境可收紧到具体前端域名，TiTiler 等生产瓦片服务同样配置。

运行：
  .venv/bin/uvicorn scripts.tile_server:app --host 127.0.0.1 --port 8001
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.data.cog_reader import cog_info, read_partial_png
from scripts.tile_cache import read_tile_png_cached
from rio_tiler.errors import TileOutsideBounds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COG = os.getenv("COG_PATH", str(PROJECT_ROOT / "data" / "cogs" / "demo_cog.tif"))

app = FastAPI(title="GeoSense COG Tile Server", version="0.1.0")

# 瓦片服务只提供只读栅格，允许所有来源跨域访问（含 OPTIONS 预检）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


def _resolve_cog(path: str | None) -> str:
    """确定要服务的 COG 路径：默认用 data/cogs/demo_cog.tif，也可按请求指定。"""
    cog = path or DEFAULT_COG
    if not Path(cog).exists():
        raise HTTPException(404, f"COG 不存在：{cog}（先运行 make_demo_raster.py + cog_generator.py）")
    return cog


@app.get("/")
def index() -> dict:
    """服务说明（浏览器直接访问时给人看的）。"""
    return {
        "服务": "GeoSense COG 瓦片服务",
        "瓦片": "GET /tiles/{z}/{x}/{y}.png?path=<cog>",
        "元信息": "GET /info?path=<cog>",
        "局部预览": "GET /preview?west=..&south=..&east=..&north=..",
        "默认COG": DEFAULT_COG,
    }


@app.get("/info")
def info(path: str | None = None, request: Request = None) -> Response:
    """返回 COG 元信息（bounds/center/minzoom/maxzoom/波段），前端据此配置图层。

    第12月W1 协商缓存教学：Cache-Control max-age 是「别再问」，ETag/304 是
    「问了但没变就别传」——元信息体积小但请求频繁，304 省的是重复传输。
    ETag = md5(mtime_ns + size)：源文件任何变化都会换指纹，语义与瓦片缓存键一致。
    """
    cog = _resolve_cog(path)
    p = Path(cog)
    etag = 'W/"%s"' % hashlib.md5(f"{p.stat().st_mtime_ns}-{p.stat().st_size}".encode()).hexdigest()[:16]
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    try:
        data = cog_info(cog)
    except Exception as exc:  # noqa: BLE001 —— 接口层兜底
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return JSONResponse(data, headers={"ETag": etag})


@app.get("/tiles/{z}/{x}/{y}.png")
def tile(z: int, x: int, y: int, path: str | None = None,
         bands: str | None = Query(default=None, description="波段组合，如 '1,2,3'")) -> Response:
    """输出一个 XYZ 瓦片 PNG。

    - bands 缺省时按影像波段数自动选（>=3 取 RGB）；
    - 瓦片不可变 → 加 Cache-Control 让浏览器/前端缓存。
    """
    cog = _resolve_cog(path)
    band_tuple = tuple(int(b) for b in bands.split(",")) if bands else None
    try:
        # 第12月W1：瓦片走磁盘缓存（键含源文件 mtime_ns，COG 覆盖自动失效；
        # 缓存层异常 fail-open 降级直读，见 scripts/tile_cache.py）
        png = read_tile_png_cached(cog, z, x, y, bands=band_tuple)
    except TileOutsideBounds:
        raise HTTPException(404, "瓦片超出影像范围")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/preview")
def preview(west: float, south: float, east: float, north: float,
            width: int = 512, height: int = 512, path: str | None = None) -> Response:
    """按 bbox 读取局部并输出 PNG（小范围预览 / 后续栅格分析）。"""
    cog = _resolve_cog(path)
    try:
        png = read_partial_png(cog, (west, south, east, north), width, height)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return Response(content=png, media_type="image/png")
