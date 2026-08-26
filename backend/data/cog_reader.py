"""阶段2 第4月 W2 交付物：cog_reader.py —— 在线 COG 读取封装

核心概念：HTTP Range / vsicurl —— COG 的"按需字节读取"是怎么发生的
- GDAL 打开远程文件（http(s)://...）走的是 vsicurl 驱动：
  它不是下载整个文件，而是按需发 HTTP Range 请求（`Range: bytes=a-b`），
  只取需要的字节区间。
- 普通 GeoTIFF 按行存储 → 读任意小窗口也要拉整行（甚至整个文件）；
  COG 按 256x256 块存储 + 有金字塔 → 读小窗口只拉对应块、缩小时只拉 overview。
  W1 转出的 COG 在这里真正发挥价值。

核心概念：overview 自动选择（rio-tiler 帮你做的事）
- 给定 XYZ 瓦片坐标 (z, x, y) → 转成 Web 墨卡托(3857) bbox → 重投影到影像 CRS
  → 按缩放级别自动选合适的 overview（低 zoom 用金字塔，省流量）→ 读取并重采样。
- 本模块把这一串逻辑封装成三个函数，瓦片服务（tile_server.py）直接调用。
"""
from __future__ import annotations

from typing import Sequence

from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import COGReader

DEFAULT_TILESIZE = 256


def cog_info(path: str) -> dict:
    """返回 COG 元信息：范围、中心、minzoom/maxzoom、波段数等（前端图层配置用）。"""
    with COGReader(path) as cog:
        info = cog.info()
    # rio-tiler 9.x 的 info() 返回 pydantic 模型 → 转成 dict 便于序列化/使用
    return info.model_dump() if hasattr(info, "model_dump") else dict(info)


def _pick_bands(cog: COGReader, bands: Sequence[int] | None) -> tuple[int, ...]:
    """默认波段：>=3 波段取前 3 个做 RGB 渲染，否则取第 1 个。"""
    if bands:
        return tuple(int(b) for b in bands)
    return (1, 2, 3) if cog.dataset.count >= 3 else (1,)


def _to_uint8(img) -> ImageData:
    """把（可能是 16 位的）波段数据线性拉伸成 uint8。

    为什么必须转 8 位？Mapbox GL / 浏览器栅格源只认 8 位 RGBA 的 PNG。

    为什么用百分位拉伸（2%~98%）而不是 min/max？
    - 真实影像里云/高亮目标的 DN 极高（如 16000），若按 min/max 拉伸，
      其余像素会被压到接近全黑；2%~98% 百分位让中间 96% 的像素占满显示范围，
      这是遥感影像显示的标准做法。
    """
    import numpy as np
    from rio_tiler.models import ImageData

    out = np.empty_like(img.data, dtype="uint8")
    for i, band in enumerate(img.data):            # img.data 形状 (bands, h, w)
        valid = band[band != 0] if (band == 0).any() else band   # 排除 NoData 参与统计
        if valid.size == 0:
            valid = band
        lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
        if hi - lo < 1:                            # 极端均匀时退回 min/max
            lo, hi = float(band.min()), float(band.max())
            hi = max(hi, lo + 1)
        out[i] = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
    # rio-tiler 9.x 的 ImageData：掩膜参数叫 alpha_mask（旧版是 mask），且要求 uint8
    return ImageData(out, alpha_mask=img.mask.astype("uint8"),
                     crs=img.crs, bounds=img.bounds)


def read_tile_png(path: str, z: int, x: int, y: int,
                  tilesize: int = DEFAULT_TILESIZE,
                  bands: Sequence[int] | None = None) -> bytes:
    """读取一个 XYZ 瓦片并渲染成 PNG 字节。

    可能抛 TileOutsideBounds（瓦片在影像范围外，服务层转成 404）。
    """
    with COGReader(path) as cog:
        img = cog.tile(x, y, z, tilesize=tilesize, indexes=_pick_bands(cog, bands))
        return _to_uint8(img).render(img_format="PNG")


def read_partial_png(path: str, bbox: tuple[float, float, float, float],
                     width: int = 512, height: int = 512,
                     bands: Sequence[int] | None = None) -> bytes:
    """按地理范围 bbox (west, south, east, north) 读取一块并渲染 PNG。

    用于小范围预览与后续（W3/W4）栅格分析：只读需要的窗口，不读全图。
    """
    with COGReader(path) as cog:
        # rio-tiler 9.x：按 bbox 读局部用 feature()，把 bbox 包装成一个矩形 GeoJSON
        # （旧版叫 partial，9.x 已移除）
        w, s, e, n = bbox
        shape = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
            },
        }
        img = cog.feature(shape, width=width, height=height, indexes=_pick_bands(cog, bands))
        return _to_uint8(img).render(img_format="PNG")
