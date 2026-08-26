"""阶段2 第4月 W1 交付物：cog_generator.py —— 普通 GeoTIFF → COG

核心概念：COG（Cloud Optimized GeoTIFF）为什么适合"在线读取"
1. 内部切片（tiling）：普通 GeoTIFF 按行存储，读局部也要扫整行；
   COG 按 256x256 小块存储，读哪个窗口只取对应块 —— 配合 HTTP Range
   请求，服务器"按需返回字节"。
2. overview 金字塔：预生成逐级缩小的影像（2x/4x/8x…），
   需要小尺寸视图时只读 overview，而不是把全分辨率读出来再缩小。
3. 这两点让"只下载需要的字节"成为可能 —— 在线瓦片服务（W2）与
   真实影像访问（W3）的地基。

rio-cogeo 是 rasterio 团队维护的官方转换库，一条 cog_translate 搞定。

用法：
  .venv/bin/python scripts/cog_generator.py data/demo/demo.tif
  .venv/bin/python scripts/cog_generator.py --dir data/demo --out-dir data/cogs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def convert(src: Path, dst: Path, blocksize: int = 256, overview_level: int = 5) -> None:
    """把普通 GeoTIFF 转成 COG。"""
    with rasterio.open(src) as src_ds:
        nodata = src_ds.nodata
    # rio-cogeo 7.x：块大小通过 profile 的 blockxsize/blockysize 控制
    profile = cog_profiles.get("deflate")      # 压缩方案：DEFLATE（无损、通用）
    profile.update(blockxsize=blocksize, blockysize=blocksize)
    cog_translate(
        str(src), str(dst), profile,
        overview_level=overview_level,
        overview_resampling="average",         # 金字塔用平均采样（遥感惯例）
        nodata=nodata,
    )


def report(src: Path, dst: Path) -> None:
    """打印 普通 GeoTIFF vs COG 的元数据对比（本交付物的"讲解"部分）。"""

    def info(path: Path) -> dict:
        with rasterio.open(path) as ds:
            return {
                "tiled": ds.profile.get("tiled"),
                "blocksize": (ds.profile.get("blockysize"), ds.profile.get("blockxsize")),
                "overviews": ds.overviews(1),
                "compression": ds.profile.get("compress"),
                "size_kb": path.stat().st_size / 1024,
            }

    a, b = info(src), info(dst)
    print(f"\n{'项目':<10}{'普通 GeoTIFF':<30}{'COG':<30}")
    for key, label in [("tiled", "内部切片"), ("blocksize", "块大小"),
                       ("overviews", "金字塔"), ("compression", "压缩"),
                       ("size_kb", "文件大小(KB)")]:
        print(f"{label:<10}{str(a[key]):<30}{str(b[key]):<30}")
    print("\n解读：")
    print("- 内部切片(Tiled=True) + 金字塔(overviews) 让 COG 支持按需字节读取；")
    print("- 未转 COG 前这些能力都不存在 —— 下一步 W2 就用 HTTP Range 在线读它。")


def main() -> int:
    parser = argparse.ArgumentParser(description="普通 GeoTIFF → COG")
    parser.add_argument("input", nargs="?", help="单个输入文件")
    parser.add_argument("output", nargs="?", help="单个输出文件（默认 <输入名>_cog.tif）")
    parser.add_argument("--dir", help="批量：输入目录")
    parser.add_argument("--out-dir", help="批量：输出目录（默认 data/cogs）")
    args = parser.parse_args()

    jobs: list[tuple[Path, Path]] = []
    if args.dir:
        in_dir = Path(args.dir)
        out_dir = (Path(args.out_dir) if args.out_dir
                   else PROJECT_ROOT / "data" / "cogs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(p.resolve(), out_dir / f"{p.stem}_cog.tif")
                for p in sorted(in_dir.glob("*.tif"))]
    elif args.input:
        src = Path(args.input).resolve()
        dst = (Path(args.output) if args.output
               else src.with_name(f"{src.stem}_cog.tif")).resolve()
        jobs = [(src, dst)]
    else:
        parser.print_help()
        return 1

    for src, dst in jobs:
        print(f"🔄 {src.relative_to(PROJECT_ROOT)} → {dst.relative_to(PROJECT_ROOT)}")
        convert(src, dst)
        report(src, dst)
    print("\n🎉 全部转换完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
