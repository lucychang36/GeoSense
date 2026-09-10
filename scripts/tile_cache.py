"""第12月 W1：tile_cache.py —— 瓦片磁盘缓存（design D3）。

核心语义：**不可变内容寻址**。
- 键 = md5(绝对路径)[:12] + 文件 mtime_ns + z/x/y + 波段组合
  → COG 被覆盖（mtime 变）时键自动变，旧缓存永远不可能冒充新数据；
  → md5 进键同时杜绝 `../` 别名与同名不同文件碰撞（safe_cog_path 同款边界思维）。
- 值 = rio-tiler 生成的 PNG 字节。命中直接回文件字节，未命中生成后落盘。
- LRU 上限 MAX_FILES，按 atime 淘汰最旧（os.utime 每次命中刷新）。
  教学点：mtime 键只是最简失效解——「同 mtime 覆盖」理论上可漏网；
  彻底方案是内容 hash，但代价是每次请求都要读源文件，得不偿失，文档化取舍。

缓存自身的任何 IO 异常都不允许打断瓦片服务：降级为直接生成（fail-open）。
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tiles"
MAX_FILES = 512


def cache_key(cog: str, z: int, x: int, y: int, bands: tuple[int, ...] | None) -> str:
    """瓦片缓存键：路径指纹 + 源文件版本（mtime_ns）+ 瓦片坐标 + 波段。"""
    p = Path(cog).resolve()
    path_fp = hashlib.md5(str(p).encode()).hexdigest()[:12]
    version = p.stat().st_mtime_ns          # 源文件被覆盖 → 键变 → 自动失效
    band_s = "-".join(map(str, bands)) if bands else "auto"
    return f"{path_fp}_{version}_{z}_{x}_{y}_{band_s}.png"


def _cache_get(key: str) -> bytes | None:
    f = CACHE_DIR / key
    if not f.is_file():
        return None
    try:
        data = f.read_bytes()
        os.utime(f, None)                    # 刷新 atime → LRU 语义
        return data
    except OSError:
        return None                          # 缓存坏文件 → 当 miss


def _cache_put(key: str, data: bytes) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_DIR / f".tmp_{os.getpid()}_{time.time_ns()}"
        tmp.write_bytes(data)                # 先写临时文件再原子改名，防半写
        tmp.replace(CACHE_DIR / key)
        _evict_if_needed()
    except OSError:
        pass                                 # 磁盘满/只读等 → 缓存静默降级


def _evict_if_needed() -> None:
    files = [f for f in CACHE_DIR.glob("*.png")]
    if len(files) <= MAX_FILES:
        return
    by_atime = sorted(files, key=lambda f: f.stat().st_atime_ns)
    for f in by_atime[: len(files) - MAX_FILES]:
        try:
            f.unlink()
        except OSError:
            pass


def read_tile_png_cached(cog: str, z: int, x: int, y: int,
                         bands: tuple[int, ...] | None = None) -> bytes:
    """tile_server 的瓦片获取入口：命中回缓存，未命中生成并落盘。

    缓存层任何异常都降级为直接生成（fail-open）——缓存是加速器，不是正确性依赖。
    """
    try:
        key = cache_key(cog, z, x, y, bands)
        hit = _cache_get(key)
        if hit is not None:
            return hit
    except OSError:                          # stat 失败等 → 键生成失败也照常生成
        key = None

    from backend.data.cog_reader import read_tile_png
    png = read_tile_png(cog, z, x, y, bands=bands)

    if key:
        _cache_put(key, png)
    return png
