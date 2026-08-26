"""阶段2 第4月 W1 交付物：make_demo_raster.py —— 合成演示栅格

为什么要"合成"数据？
- 学习栅格/COG 原理不需要真实卫星影像：像素是我们自己生成的，
  每个像素长什么样、放在哪都完全清楚 —— 验证容易、可复现。
- 真实 Sentinel-2 数据受网络限制（本机访问 Copernicus 不稳），
  等 W3 连通性探测后再补真实数据，原理先行、不被网络阻塞。

本脚本生成一张"普通 GeoTIFF"：不切片、无金字塔 —— 恰恰是 COG 的反面教材。
下一节 cog_generator.py 会把它转成 COG，并对比两者元数据差异。

核心概念：栅格数据模型
- 波段（band）：二维数值数组（宽x高）；多光谱影像有多个波段（如 R/G/B/NIR）
- 仿射变换（transform）：像素坐标 → 地理坐标 的映射（左上角 + 像元尺寸）
- CRS：坐标系（这里用 EPSG:4326，方便与现有 POI/行政边界数据对齐）
- NoData：像素的"空值"约定（这里 0 = 无效，如影像边缘的无数据区）

用法：.venv/bin/python scripts/make_demo_raster.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WIDTH, HEIGHT = 1024, 1024          # 像素尺寸
BANDS = 4                           # 模拟多光谱：B1~B4
NODATA = 0                          # 0 = 空值
MINX, MINY = 113.7, 22.4            # 覆盖范围（对齐深圳 bbox，纯演示用）
MAXX, MAXY = 114.7, 22.9


def make_demo_arrays() -> list[np.ndarray]:
    """生成 4 个波段的合成像素，每波段一种模式，肉眼可区分。"""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    cx, cy = WIDTH * 0.55, HEIGHT * 0.45          # 圆形目标中心
    r = 150                                       # 目标半径（模拟"目标地物"）

    b1 = (x / WIDTH * 200).astype("uint16")                       # 水平渐变
    b2 = (y / HEIGHT * 200).astype("uint16")                      # 垂直渐变
    b3 = np.where((x - cx) ** 2 + (y - cy) ** 2 < r**2,
                  250, 60).astype("uint16")                       # 圆形高亮目标
    b4 = ((255 - x / WIDTH * 200) % 256).astype("uint16")         # 反向渐变

    # 两个对角区域设为 NoData（模拟影像边缘的无数据区）
    for arr in (b1, b2, b3, b4):
        arr[0:120, 0:120] = NODATA
        arr[HEIGHT - 120:HEIGHT, WIDTH - 120:WIDTH] = NODATA
    return [b1, b2, b3, b4]


def main() -> int:
    out_dir = PROJECT_ROOT / "data" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo.tif"

    # 左上角 (MINX, MAXY)，每个像元对应 (经度跨度/W, 纬度跨度/H)
    transform = from_origin(MINX, MAXY, (MAXX - MINX) / WIDTH, (MAXY - MINY) / HEIGHT)

    # 关键：故意不写 tiled / overviews —— 生成"普通 GeoTIFF"（逐行存储、无金字塔）
    profile = {
        "driver": "GTiff",
        "width": WIDTH, "height": HEIGHT,
        "count": BANDS,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": NODATA,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, arr in enumerate(make_demo_arrays(), start=1):
            dst.write(arr, i)
        dst.descriptions = [f"B{i}（合成波段 {i}）" for i in range(1, BANDS + 1)]

    with rasterio.open(out_path) as src:
        print(f"✅ 已生成普通 GeoTIFF：{out_path.relative_to(PROJECT_ROOT)}")
        print(f"   尺寸 {src.width}x{src.height}，波段 {src.count}，CRS {src.crs}")
        print(f"   内部切片(tiled)={src.profile.get('tiled')}，"
              f"金字塔(overviews)={src.overviews(1)}")
        print("   —— 注意：tiled=False、无 overviews，这就是 COG 出现之前的老式存储")
    return 0


if __name__ == "__main__":
    sys.exit(main())
