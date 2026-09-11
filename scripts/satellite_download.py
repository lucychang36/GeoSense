"""阶段2 第4月 W3 交付物：satellite_download.py —— 遥感数据仓库构建（合成多时相多光谱）

背景与决策：
- 真实 Sentinel-2 数据来自 earth-search（AWS）STAC 检索 + S3 COG 资产。
  连通性实测：STAC 可达、S3 COG 可下载（本会话已验证 HTTP 206 / GDAL 远程读取）。
- --real 模式：检索 → 用 COG"按需字节读取"远程裁剪 bbox（不下载整景）→ 转本地 COG。
  该能力正是 COG + HTTP Range 的价值：10980x10980 的整景只取需要的窗口。
- 默认模式（合成）：生成与 Sentinel-2 波段结构对齐的模拟影像，保证离线可学、
  可复现；--real 与合成模式产出的仓库格式完全一致。
- 波段约定 v2（2026-09-11 thematic-index-change）：B2,B3,B4,B8,B11,B12 六波段
  ——新增 SWIR（B11/B12，20m 重采样到统一网格）支撑 NDBI 建筑用地指数；
  旧 4 波段 COG 不受影响（SWIR 追加在尾部，前 4 个索引不变）。
- --region：区域预设（REGION_PRESETS），bbox/输出前缀由区域决定，默认深圳湾。

核心概念：Sentinel-2 波段（真彩色 / 假彩色 / 水质）
- 10m 波段：B2(蓝 490nm) B3(绿 560nm) B4(红 665nm) B8(近红外 842nm)
- 真彩色 = R:B4 G:B3 B:B2；假彩色(植被) = R:B8 G:B4 B:B3
- 水质信号：藻类在水体近红外反射率升高 —— 本项目"深圳湾水质"叙事的数据基础

每景内容（合成但"像真的"）：
- 陆地/水体分布（深圳湾形状）+ 城市/植被纹理 + 逐时相变化的"水质"参数
  + 随机云块（云量写入元数据，演示"云量筛选"）+ 噪声 + NoData 边缘

用法：
  .venv/bin/python scripts/satellite_download.py                # 合成数据 → COG 仓库
  .venv/bin/python scripts/satellite_download.py --real         # 真实 Sentinel-2 裁剪入库
  .venv/bin/python scripts/satellite_download.py --probe-real   # 真实数据源连通性探测
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data.cog_reader import read_partial_png
from scripts.cog_generator import convert  # 复用 W1：普通 GeoTIFF → COG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COGS_DIR = PROJECT_ROOT / "data" / "cogs"
TMP_DIR = PROJECT_ROOT / "data" / "tmp"

# 深圳湾覆盖范围（合成场景的地理范围，与项目"深圳湾"叙事一致）
BBOX = (113.88, 22.46, 114.10, 22.60)          # (west, south, east, north)
WIDTH, HEIGHT = 512, 512
NODATA = 0
SEED = 42                                        # 固定随机种子 → 可复现

# 多时相：模拟"过去 5 年"的水质变化观测
DATES = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06"]

# 波段清单（顺序即文件内波段序号）—— v2 六波段：B11/B12 为 SWIR（20m 原生，
# 下载时统一重采样到网格），支撑 NDBI（建筑用地）/NDMI 等指数
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

# 区域预设：bbox + 输出文件前缀。加新区域 = 加一个条目（开放-封闭，同 THEMES 思路）
REGION_PRESETS: dict[str, dict] = {
    "szbay": {
        "bbox": (113.88, 22.46, 114.10, 22.60),
        "prefix": "szbay",
        "label": "深圳湾",
    },
    "zhengzhou_hightech": {
        "bbox": (113.50, 34.73, 113.70, 34.88),   # 郑州高新区 bbox 近似（行政边界裁剪留后续变更）
        "prefix": "zhengzhou",
        "label": "郑州高新区",
    },
}


# ---------------------------------------------------------------------------
# 合成场景生成
# ---------------------------------------------------------------------------
def _land_water_field(width: int, height: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """生成 [陆地分数] 与 [城市/植被纹理] 两个底图场。

    深圳湾在西南、陆地（东北）向海湾延伸；城市与植被用噪声纹理区分。
    """
    y, x = np.mgrid[0:height, 0:width]
    nx, ny = x / width, y / height
    # 海湾椭圆：落在椭圆内视为水体
    bay = ((nx - 0.34) / 0.34) ** 2 + ((ny - 0.74) / 0.26) ** 2 < 1
    # 海岸带平滑过渡（水体与陆地的混合带）
    dist = ((nx - 0.34) / 0.34) ** 2 + ((ny - 0.74) / 0.26) ** 2
    coast = np.clip((dist - 0.8) / 0.4, 0, 1)          # 0=水 1=陆 的连续值
    land = np.where(bay, coast * 0.4, coast)           # 海湾内也留一点浅滩

    # 城市/植被纹理（低频噪声，>0.5 视为植被）
    tex = rng.random((height, width))
    return land, tex


def _band_values(land: np.ndarray, tex: np.ndarray, turbidity: float,
                 rng) -> tuple[list[np.ndarray], float]:
    """按 4 个波段的典型反射率（Sentinel-2 风格 DN）合成像素。

    返回 (波段数组列表, 云量百分比)。
    """
    h, w = land.shape
    # 陆地基值：植被 / 城市按纹理混合（v2 六波段，含 SWIR：城市 SWIR 高、植被 SWIR 低）
    veg = np.stack([0.05, 0.08, 0.06, 0.25, 0.18, 0.09])   # B2,B3,B4,B8,B11,B12 植被
    urb = np.stack([0.12, 0.14, 0.16, 0.14, 0.24, 0.22])   # 城市（SWIR 高 → NDBI 高）
    land_r = veg[None, None, :] * tex[..., None] + urb[None, None, :] * (1 - tex[..., None])
    # 水体基值：清洁水（B8 很低）；浊度升高 → B4 与 B8 上升（藻/泥沙）
    water_r = np.stack([0.07, 0.05, 0.03, 0.02, 0.01, 0.005])
    algae = np.stack([0.06, 0.06, 0.07, 0.16, 0.03, 0.02])  # 藻类光谱（近红外高）
    water_r = water_r + turbidity * algae

    band_float = land[..., None] * land_r + (1 - land[..., None]) * water_r
    # 云块：0~2 个椭圆亮块，覆盖面积计入云量（真实场景来自影像质量标记）
    cloud = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = rng.integers(20, 70), rng.integers(15, 50)
        yy, xx = np.mgrid[0:h, 0:w]
        cloud |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
    cloud_cover = float(cloud.mean() * 100)
    # 云在所有波段都很亮（反射率 ~0.4）
    band_float = np.where(cloud[..., None], 0.4, band_float)

    # 噪声 + 转 uint16（反射率 × 10000，Sentinel-2 风格 DN）
    bands = []
    for b in range(6):
        arr = (band_float[..., b] * 10000 + rng.normal(0, 60, (h, w)))
        arr = np.clip(arr, 1, 10000).astype("uint16")
        arr[0:40, 0:40] = NODATA                      # 边缘 NoData（模拟无效区域）
        bands.append(arr)
    return bands, cloud_cover


def build_scene(date_idx: int, n_dates: int, rng: np.random.Generator):
    """生成一景：返回 (波段数组列表, 浊度, 云量百分比)。"""
    land, tex = _land_water_field(WIDTH, HEIGHT, rng)
    # 水质随时间变化：浊度 = 0.25 + 0.5*|sin| → 模拟"污染-恢复"周期
    turbidity = 0.25 + 0.5 * abs(np.sin(date_idx / n_dates * np.pi))
    bands, cloud_cover = _band_values(land, tex, turbidity, rng)
    return bands, turbidity, cloud_cover


# ---------------------------------------------------------------------------
# 仓库构建
# ---------------------------------------------------------------------------
def _write_and_convert(bands: list[np.ndarray], plain_path: Path, cog_path: Path,
                       bbox: tuple | None = None) -> None:
    """写普通 GeoTIFF → 复用 W1 的 convert 转 COG。

    注意：profile 的尺寸必须用数组实际形状（h, w），而不是模块常量——
    合成场景与真实裁剪场景的尺寸不同（512 vs 1100x700）。
    """
    h, w = bands[0].shape
    west, south, east, north = bbox if bbox else BBOX
    transform = from_origin(west, north, (east - west) / w, (north - south) / h)
    profile = {
        "driver": "GTiff",
        "width": w, "height": h,
        "count": len(bands), "dtype": "uint16",
        "crs": "EPSG:4326", "transform": transform, "nodata": NODATA,
    }
    with rasterio.open(plain_path, "w", **profile) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
        dst.descriptions = list(BANDS)
    convert(plain_path, cog_path)
    plain_path.unlink()          # 清理中间普通 GeoTIFF


# ---------------------------------------------------------------------------
# 真实数据下载（--real）：earth-search STAC 检索 + 远程 COG 裁剪
# ---------------------------------------------------------------------------
# earth-search element84 L2A 资产名 → 波段名；swir16/swir22 为 20m 原生，
# COGReader.feature(width=W,height=H) 统一重采样到目标网格（与 10m 波段同形状）
REAL_BANDS = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8",
              "swir16": "B11", "swir22": "B12"}


def download_real_scene(dt_start: str, dt_end: str, cloud_max: float = 20,
                        tile: str | None = None, region: str = "szbay") -> dict | None:
    """检索一景真实 Sentinel-2（region 预设 bbox、低云量），远程裁剪为本地 COG。

    tile：MGRS tile 过滤（如 "49QGE"）或 "auto"。变化检测需要同 tile 双时相配对——
    不同 tile 轨道视角/网格不同，直接配对会产生大量伪变化。"auto" = 先不带 tile
    检索、取覆盖度最高景的 tile 并固定（第二次调用同一 auto 会收敛到同 tile）。

    返回 manifest 条目；无可匹配影像时返回 None。
    """
    preset = REGION_PRESETS[region]
    bbox = preset["bbox"]
    prefix = preset["prefix"]
    import json as _json
    import urllib.parse as _up
    import urllib.request as _ur
    from rio_tiler.io import COGReader

    # 1) STAC 检索：limit 视场景调整——tile 过滤时需要更大的候选池
    params = _up.urlencode({
        "collections": "sentinel-2-l2a",
        "bbox": ",".join(map(str, bbox)),
        "datetime": f"{dt_start}T00:00:00Z/{dt_end}T23:59:59Z",
        "query": _json.dumps({"eo:cloud_cover": {"lt": cloud_max}}),
        "limit": 30 if tile else 5,
    })
    search_url = f"https://earth-search.aws.element84.com/v1/search?{params}"
    print(f"🔎 STAC 检索（region={region}，bbox={bbox}，云量<{cloud_max}%）…")
    with _ur.urlopen(search_url, timeout=30) as resp:
        features = _json.load(resp).get("features", [])

    def overlap(feat) -> float:
        """景 bbox 与目标 BBOX 的交集面积（覆盖度评分）。"""
        fb = feat["bbox"]
        ox = max(0.0, min(fb[2], bbox[2]) - max(fb[0], bbox[0]))
        oy = max(0.0, min(fb[3], bbox[3]) - max(fb[1], bbox[1]))
        return ox * oy

    if tile == "auto" and features:
        auto_tile = max(features, key=overlap)["id"].split("_")[1]
        print(f"   tile=auto → 取覆盖度最高景的 MGRS tile：{auto_tile}")
        features = [f for f in features if f["id"].split("_")[1] == auto_tile]
    elif tile:
        features = [f for f in features if f["id"].split("_")[1] == tile]
        print(f"   tile 过滤 {tile}：剩余 {len(features)} 景")
    if not features:
        print("   ⚠ 无匹配影像（试试调整时间范围/放宽云量）")
        return None

    item = max(features, key=overlap)
    cover = overlap(item) / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    date = item["properties"]["datetime"][:10]
    cloud = float(item["properties"]["eo:cloud_cover"])
    print(f"   命中 {item['id']}（{date}，云量 {cloud:.1f}%，bbox 覆盖 {cover:.0%}）")
    if cover < 0.5:
        print("   ⚠ 该景对目标区域的覆盖不足 50%，结果可能包含大片 NoData")

    # 2) 远程裁剪：bbox 包成矩形 GeoJSON，用 COG 按需读取（只下需要的字节块）
    w, s, e, n = bbox
    shape = {"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}
    W, H = 1100, 700          # ≈20m 分辨率（bbox 0.22°x0.14°）

    bands: list[np.ndarray] = []
    for key, band_name in REAL_BANDS.items():
        href = item["assets"][key]["href"]
        print(f"   📥 远程读取 {band_name}（{key}）…")
        with COGReader(href) as cog:
            img = cog.feature(shape, width=W, height=H, indexes=[1])
        arr = img.data[0]
        # 掩膜（无效像素置 0）→ 统一 uint16（兼容 NoData=0 的仓库约定）
        arr = np.where(img.mask.astype(bool), arr, 0)
        arr = np.clip(np.nan_to_num(arr, nan=0, posinf=0, neginf=0), 0, 65535)
        bands.append(arr.astype("uint16"))

    # 3) 写本地 COG + 真彩色预览（与合成场景同一流水线、同一波段顺序）
    stem = f"{prefix}_real_{date.replace('-', '')}"
    cog_path = COGS_DIR / f"{stem}.tif"
    _write_and_convert(bands, TMP_DIR / f"{stem}.tif", cog_path, bbox=bbox)
    preview = read_partial_png(str(cog_path), bbox, width=W, height=H, bands=(3, 2, 1))
    (COGS_DIR / f"{stem}.preview.png").write_bytes(preview)

    return {
        "id": item["id"],
        "date": date,
        "region": region,
        "source": "sentinel-2（真实）",
        "bands": list(REAL_BANDS.values()),
        "cloud": round(cloud, 1),
        "bbox": list(bbox),
        "path": str(cog_path.relative_to(PROJECT_ROOT)),
        "preview": str((COGS_DIR / f"{stem}.preview.png").relative_to(PROJECT_ROOT)),
    }


def download_real_mosaic(dt_start: str, dt_end: str, cloud_max: float = 30,
                         width: int = 2200, height: int = 1400,
                         region: str = "szbay") -> dict | None:
    """跨 UTM 分带拼接：下载多景真实 Sentinel-2，合并成覆盖 bbox 的完整 COG。

    背景：深圳湾横跨 UTM 分带（49QGE 覆盖西半、50QKK 覆盖东半），单景必然
    覆盖不全（之前实测 49% NoData）。本函数检索多景，按覆盖度优先合并：
    每个波段在统一网格（同一 bbox / 同一分辨率）上逐像素"先有效者胜"。

    返回 manifest 条目；无可匹配影像时返回 None。
    """
    import json as _json
    import urllib.parse as _up
    import urllib.request as _ur
    from rio_tiler.io import COGReader

    preset = REGION_PRESETS[region]
    bbox = preset["bbox"]
    prefix = preset["prefix"]
    params = _up.urlencode({
        "collections": "sentinel-2-l2a",
        "bbox": ",".join(map(str, bbox)),
        "datetime": f"{dt_start}T00:00:00Z/{dt_end}T23:59:59Z",
        "query": _json.dumps({"eo:cloud_cover": {"lt": cloud_max}}),
        "limit": 10,
    })
    search_url = f"https://earth-search.aws.element84.com/v1/search?{params}"
    print(f"🔎 STAC 检索（region={region}，bbox={bbox}，云量<{cloud_max}%，limit=10）…")
    with _ur.urlopen(search_url, timeout=30) as resp:
        features = _json.load(resp).get("features", [])

    def overlap(feat) -> float:
        fb = feat["bbox"]
        ox = max(0.0, min(fb[2], bbox[2]) - max(fb[0], bbox[0]))
        oy = max(0.0, min(fb[3], bbox[3]) - max(fb[1], bbox[1]))
        return ox * oy

    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    min_cover = 0.1 * bbox_area                       # 至少覆盖 bbox 面积的 10%
    items = [f for f in features if overlap(f) > min_cover]
    items.sort(key=overlap, reverse=True)             # 覆盖优先
    if not items:
        print("   ⚠ 无匹配影像（试试调整时间范围/放宽云量）")
        return None
    print(f"   参与拼接 {len(items)} 景："
          + ", ".join(f"{i['properties']['datetime'][:10]}({overlap(i) / bbox_area:.0%})" for i in items))

    w, s, e, n = bbox
    shape = {"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}
    n_bands = len(REAL_BANDS)
    result = np.zeros((n_bands, height, width), dtype="uint16")
    valid = np.zeros((n_bands, height, width), dtype=bool)

    for item in items:
        print(f"   📥 {item['id'][:40]}…")
        for bi, key in enumerate(REAL_BANDS):
            href = item["assets"][key]["href"]
            with COGReader(href) as cog:
                img = cog.feature(shape, width=width, height=height, indexes=[1])
            arr = img.data[0]
            m = img.mask.astype(bool)
            arr = np.where(m, np.clip(np.nan_to_num(arr, nan=0, posinf=0, neginf=0),
                                      0, 65535), 0).astype("uint16")
            take = m & ~valid[bi]                 # 已有有效像素的保持（先覆盖者胜）
            result[bi][take] = arr[take]
            valid[bi] |= m
        print(f"     累计有效覆盖：{100 * valid.mean():.0f}%")

    cog_path = COGS_DIR / "szbay_real_mosaic.tif"
    _write_and_convert(list(result), TMP_DIR / "szbay_real_mosaic.tif", cog_path)
    preview = read_partial_png(str(cog_path), BBOX, width=width, height=height, bands=(3, 2, 1))
    (COGS_DIR / "szbay_real_mosaic.preview.png").write_bytes(preview)

    return {
        "id": "S2_MOSAIC_SZBAY",
        "date": f"{dt_start}~{dt_end}",
        "source": "sentinel-2（真实拼接）",
        "bands": list(REAL_BANDS.values()),
        "cloud": round(float(min(i["properties"]["eo:cloud_cover"] for i in items)), 1),
        "bbox": list(BBOX),
        "path": str(cog_path.relative_to(PROJECT_ROOT)),
        "preview": str((COGS_DIR / "szbay_real_mosaic.preview.png").relative_to(PROJECT_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="遥感数据 → COG 仓库（合成 / 真实）")
    parser.add_argument("--real", action="store_true",
                        help="下载真实 Sentinel-2（earth-search）裁剪入库")
    parser.add_argument("--mosaic", action="store_true",
                        help="跨 UTM 分带多景拼接（与 --real 同用，覆盖更完整）")
    parser.add_argument("--dt-start", default="2025-07-01", help="真实检索起始日期")
    parser.add_argument("--dt-end", default="2025-08-19", help="真实检索结束日期")
    parser.add_argument("--tile", default=None,
                        help="MGRS tile 过滤（如 49QGE；变化检测时相配对需同 tile；auto=自动取覆盖最高景的 tile）")
    parser.add_argument("--region", default="szbay", choices=sorted(REGION_PRESETS),
                        help="区域预设（bbox + 输出前缀）；默认 szbay 向后兼容")
    parser.add_argument("--cloud", type=float, default=20.0, help="最大云量（%）")
    parser.add_argument("--probe-real", action="store_true",
                        help="只探测真实数据源连通性（Copernicus/AWS STAC）")
    parser.add_argument("--scenes", type=int, default=len(DATES), help="合成模式景数")
    args = parser.parse_args()

    COGS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if args.probe_real:
        probe_real()
        return 0

    # 加载已有 manifest（真实/合成多次运行会追加，不互相覆盖）
    manifest_path = COGS_DIR / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.real:
        entry = (download_real_mosaic(args.dt_start, args.dt_end, region=args.region) if args.mosaic
                 else download_real_scene(args.dt_start, args.dt_end, cloud_max=args.cloud,
                                          tile=args.tile, region=args.region))
        if entry:
            # 同 id 的旧条目先移除（支持重跑更新数据）
            manifest = [m for m in manifest if m["id"] != entry["id"]]
            manifest.append(entry)
            print(f"   ✅ 真实影像已入库：{entry['path']}（云量 {entry['cloud']}%）")
    else:
        rng = np.random.default_rng(SEED)
        preset = REGION_PRESETS[args.region]
        bbox = preset["bbox"]
        prefix = preset["prefix"]
        for i, date in enumerate(DATES[: args.scenes]):
            print(f"🛰 生成第 {i + 1}/{args.scenes} 景：{date}")
            bands, turbidity, cloud_cover = build_scene(i, len(DATES[: args.scenes]), rng)
            cog_path = COGS_DIR / f"{prefix}_{date}.tif"
            _write_and_convert(bands, TMP_DIR / f"{prefix}_{date}.tif", cog_path, bbox=bbox)

            # 真彩色预览（B4,B3,B2），复用 W2 的局部读取（dogfooding）
            west, south, east, north = bbox
            preview = read_partial_png(str(cog_path), (west, south, east, north),
                                       width=WIDTH, height=HEIGHT, bands=(3, 2, 1))
            (COGS_DIR / f"{prefix}_{date}.preview.png").write_bytes(preview)

            manifest.append({
                "id": f"S2_{date.replace('-', '')}" if args.region == "szbay" else f"S2_{prefix}_{date.replace('-', '')}",
                "date": date,
                "region": args.region,
                "source": "sentinel-2（合成）",
                "bands": BANDS,
                "turbidity": round(float(turbidity), 3),     # 合成"水质"参数
                "cloud": round(cloud_cover, 1),              # 云量 %（真实场景来自质量标记）
                "bbox": list(bbox),
                "path": str(cog_path.relative_to(PROJECT_ROOT)),
                "preview": str((COGS_DIR / f"{prefix}_{date}.preview.png").relative_to(PROJECT_ROOT)),
            })
            print(f"   ✅ COG + 真彩色预览已写入：{cog_path.name}（云量 {cloud_cover:.1f}%）")

    # 按 id 去重（幂等：重复运行不产生重复条目）
    seen: set[str] = set()
    manifest = [m for m in manifest if not (m["id"] in seen or seen.add(m["id"]))]

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\n📦 仓库清单：{manifest_path.relative_to(PROJECT_ROOT)}（{len(manifest)} 景）")
    print("\n=== 场景一览 ===")
    for m in manifest:
        extra = f" 浊度={m['turbidity']:.2f}" if "turbidity" in m else ""
        print(f"  {m['date']} [{m.get('source', '?')}]{extra} 云量={m['cloud']:.1f}%")

    # 云量筛选演示（真实数据检索的第一步：只取云量低的景）
    usable = [m for m in manifest if m["cloud"] < 20]
    print(f"\n=== 云量筛选（< 20% 才可用于分析）：{len(usable)}/{len(manifest)} 景 ===")
    print("  " + ", ".join(m["date"] for m in usable))

    print("\n🎉 W3 完成：data/cogs/ 已是可查询的 COG 仓库（第5月将升级为 STAC 目录）")
    return 0


# ---------------------------------------------------------------------------
# 真实数据源连通性探测（预留接口）
# ---------------------------------------------------------------------------
def probe_real() -> None:
    print("\n=== 真实数据源连通性探测 ===")
    endpoints = {
        "Copernicus Data Space STAC": "https://stac.dataspace.copernicus.eu/v1/",
        "AWS Earth Search (Sentinel-2 COG)": "https://earth-search.aws.element84.com/v1",
    }
    for name, url in endpoints.items():
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GeoSense/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"  ✅ {name}: 可达（HTTP {resp.status}，{time.time() - t0:.1f}s）")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {name}: 不可达（{type(exc).__name__}）—— 合成数据继续保进度")


if __name__ == "__main__":
    sys.exit(main())
