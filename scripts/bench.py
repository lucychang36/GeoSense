"""第12月 W1：bench.py —— 性能基准工具（基准先行，先测量后优化）。

4 项基准（design D1）：
  B1 瓦片生成冷/热 ×2 轮   —— rio-tiler 开窗延迟（GDAL 块缓存效应）+ T3 磁盘缓存热延迟
  B2 推理延迟 fp32 vs fp16 —— UNet 前向（MPS），含首次调用（kernel 编译）vs 稳态
  B3 模型冷启动            —— 进程内首次 get_unet()（bench 每次是新进程，天然冷启动）
  B4 载荷体积 gzip 前后    —— 报告 md/html + 最大 POI geojson

用法：
  .venv/bin/python scripts/bench.py --save baseline   # 优化前基线 → data/output/bench/baseline.json
  .venv/bin/python scripts/bench.py --save after      # 优化后 → after.json（脚本自动与 baseline 对比）

注意（design D6）：沙箱环境计时受 sitecustomize 注入干扰，本脚本输出是
「同环境相对对比」用的；权威绝对数字请在干净终端跑。
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BENCH_DIR = PROJECT_ROOT / "data" / "output" / "bench"
COGS_DIR = PROJECT_ROOT / "data" / "cogs"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "output" / "reports"
OSM_DIR = PROJECT_ROOT / "data" / "osm"

DEFAULT_COG = "szbay_real_20230708.tif"
N_TILES = 8          # B1 每轮瓦片数
N_FORWARD = 6        # B2 每配置前向次数


def _fmt_ms(ms: float) -> str:
    return f"{ms:8.1f}ms" if ms < 10_000 else f"{ms / 1000:8.2f}s"


def _tile_coords(info: dict, n: int) -> list[tuple[int, int, int]]:
    """从 COG bounds 取 z12 下均匀分布的 n 个瓦片坐标（都落在影像内）。"""
    west, south, east, north = info["bounds"]
    z = 12
    nxy = 2 ** z
    lng = [west + (east - west) * (i + 0.5) / n for i in range(n)]
    lat = [south + (north - south) * (i + 0.5) / n for i in range(n)]

    def _xy(la: float, lo: float) -> tuple[int, int]:
        """经纬度 → z 级 XYZ 瓦片行列号（Web 墨卡托标准公式）。"""
        x = int((lo + 180.0) / 360.0 * nxy)
        lat_r = math.radians(la)
        y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * nxy)
        return x, min(max(y, 0), nxy - 1)

    return [(z, *_xy(la, lo)) for la, lo in zip(lat, lng)]


# ---------------------------------------------------------------- B1 瓦片
def bench_tiles(results: dict) -> None:
    from backend.data.cog_reader import cog_info, read_tile_png

    cog = COGS_DIR / DEFAULT_COG
    info = cog_info(str(cog))
    coords = _tile_coords(info, N_TILES)

    rounds = {}
    for rnd in (1, 2):
        ts = []
        for z, x, y in coords:
            t0 = time.perf_counter()
            read_tile_png(str(cog), z, x, y)
            ts.append((time.perf_counter() - t0) * 1000)
        rounds[rnd] = {"median_ms": round(statistics.median(ts), 1), "total_ms": round(sum(ts), 1)}
    results["B1_tiles_raw"] = {"rounds": rounds,
                               "note": "round1 冷（GDAL 块缓存未热）round2 热；T3 后另有 cached 热延迟"}

    # T3 之后：磁盘缓存热路径（模块存在才测；基线阶段此节缺失是预期）
    try:
        from scripts.tile_cache import read_tile_png_cached
        # 先各请求一次确保已入缓存
        for z, x, y in coords:
            read_tile_png_cached(str(cog), z, x, y)
        ts = []
        for z, x, y in coords:
            t0 = time.perf_counter()
            read_tile_png_cached(str(cog), z, x, y)
            ts.append((time.perf_counter() - t0) * 1000)
        results["B1_tiles_cached_hot"] = {"median_ms": round(statistics.median(ts), 1),
                                          "total_ms": round(sum(ts), 1)}
    except ImportError:
        results["B1_tiles_cached_hot"] = None


# ---------------------------------------------------------------- B2/B3 推理
def bench_inference(results: dict) -> None:
    import torch

    from backend.model_service.loader import get_device, get_unet

    t0 = time.perf_counter()
    model = get_unet()
    results["B3_model_cold_start_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    device = get_device()
    x32 = torch.rand(1, 4, 256, 256, device=device)

    def _time_forward(m, x, n: int) -> dict:
        first = None
        ts = []
        with torch.inference_mode():
            for i in range(n):
                t = time.perf_counter()
                m(x)
                if device == "mps":
                    torch.mps.synchronize()
                dt = (time.perf_counter() - t) * 1000
                if i == 0:
                    first = round(dt, 1)          # 首次含 kernel 编译
                elif i >= 2:                       # 跳过前 2 次热身
                    ts.append(dt)
        return {"first_ms": first, "steady_ms": round(statistics.median(ts), 1)}

    results["B2_unet_fp32_256"] = _time_forward(model, x32, N_FORWARD)

    # fp16 实验（design D2：bench 裁决，负收益是合法结论）
    try:
        import copy
        model16 = copy.deepcopy(model).half()
        x16 = x32.half()
        r = _time_forward(model16, x16, N_FORWARD)
        # 数值 sanity：fp16 输出与 fp32 输出的 argmax 应基本一致
        with torch.inference_mode():
            a = model(x32).argmax(1)
            b = model16(x16).argmax(1)
        agree = (a == b).float().mean().item()
        results["B2_unet_fp16_256"] = {**r, "argmax_agreement": round(agree, 4)}
    except Exception as exc:  # noqa: BLE001 —— fp16 不可用也是结论
        results["B2_unet_fp16_256"] = {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- B4 载荷
def bench_payload(results: dict) -> None:
    items: list[tuple[str, bytes]] = []

    mds = sorted(REPORTS_DIR.glob("report_*.md"))
    if mds:
        biggest = max(mds, key=lambda p: p.stat().st_size)
        items.append((f"report_md({biggest.name[:24]})", biggest.read_bytes()))
    big_osm = max(OSM_DIR.glob("*.json"), key=lambda p: p.stat().st_size, default=None)
    if big_osm:
        items.append((f"poi_geojson({big_osm.name})", big_osm.read_bytes()))

    out = {}
    for name, raw in items:
        gz = gzip.compress(raw)
        out[name] = {"raw_kb": round(len(raw) / 1024, 1),
                     "gzip_kb": round(len(gz) / 1024, 1),
                     "ratio": round(len(gz) / len(raw), 3)}
    results["B4_payload_gzip"] = out


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", choices=["baseline", "after"], required=True,
                    help="结果落盘名（baseline=优化前基线，after=优化后）")
    args = ap.parse_args()

    results: dict = {"meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "cog": DEFAULT_COG}}
    print("== B3/B2 推理（冷启动 → fp32 → fp16）==")
    bench_inference(results)
    print(f"  B3 模型冷启动: {_fmt_ms(results['B3_model_cold_start_ms'])}")
    for k in ("B2_unet_fp32_256", "B2_unet_fp16_256"):
        print(f"  {k}: {results.get(k)}")

    print("== B1 瓦片生成（raw ×2 轮 + cached 热）==")
    bench_tiles(results)
    print(f"  raw: {results['B1_tiles_raw']['rounds']}")
    print(f"  cached_hot: {results.get('B1_tiles_cached_hot')}")

    print("== B4 载荷 gzip 前后 ==")
    bench_payload(results)
    for name, v in results["B4_payload_gzip"].items():
        print(f"  {name}: {v['raw_kb']}KB → gzip {v['gzip_kb']}KB（×{v['ratio']}）")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / f"{args.save}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已落盘 {out}")

    # 与基线对比
    base = BENCH_DIR / "baseline.json"
    if args.save == "after" and base.is_file():
        b = json.loads(base.read_text(encoding="utf-8"))
        print("\n== 对比（after vs baseline，同环境相对值）==")
        rows = [
            ("B1 瓦片 raw 热(r2 median)",
             b.get("B1_tiles_raw", {}).get("rounds", {}).get("2", {}).get("median_ms"),
             results["B1_tiles_raw"]["rounds"].get("2", {}).get("median_ms")),
            ("B1 瓦片 cached 热",
             (b.get("B1_tiles_cached_hot") or {}).get("median_ms"),
             (results.get("B1_tiles_cached_hot") or {}).get("median_ms")),
            ("B2 fp32 稳态", b.get("B2_unet_fp32_256", {}).get("steady_ms"),
             results["B2_unet_fp32_256"]["steady_ms"]),
            ("B3 冷启动", b.get("B3_model_cold_start_ms"), results["B3_model_cold_start_ms"]),
        ]
        for label, old, new in rows:
            if old and new:
                print(f"  {label}: {old} → {new}（×{new / old:.2f}）")
            else:
                print(f"  {label}: baseline={old} after={new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
