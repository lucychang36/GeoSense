"""第12月 W2 交付物：thematic_change.py —— 光谱指数专题框架（OpenSpec 2026-09-10-thematic-index-change）

设计（方案 B+，用户确认）：
- THEMES 专题注册表（纯数据，零逻辑）：每专题 = 指数公式 + 判定阈值 + 联合/排除
  条件 + 语义色。新增专题 = 加一个 dict 条目，零新代码（开放-封闭，同 W2 SEMANTIC
  语义表 / 第9月 task_executors 注入点的项目先例）。
- index_change 泛化引擎（唯一实现）：指数计算 → 联合判定 → 两期差分 → km² 面积。
  建筑用地 / 绿化用地都只是它的配置实例。

防幻觉边界（同 W4 意图分层 / 第11月数字闸门）：
- 专题判定是确定性光谱计算——LLM 只选 theme 枚举，不碰阈值与计算
- 输出指标全部由本模块直接产出，报告模板原样注入（数字不经语言模型转写）

指数公式（归一化到 [-1,1]，除零保护 eps=1e-6）：
- NDBI = (SWIR-NIR)/(SWIR+NIR)          建筑用地：城市 SWIR 反射高
- NDVI = (NIR-Red)/(NIR+Red)            植被活性
- NDWI = (Green-NIR)/(Green+NIR)        水体

诚实边界（design D8）：指数命中 ≠ 土地利用真值——裸土 NDBI 同样偏高，
builtup 用 NDVI<NDBI 联合判定缓解但不根除；报告措辞用"疑似"。

用法：
  .venv/bin/python scripts/thematic_change.py --selftest            # 零网络自测
  .venv/bin/python scripts/thematic_change.py --probe COG.tif       # 指数直方图/分位数（阈值标定）
  .venv/bin/python scripts/thematic_change.py A.tif B.tif builtup   # 两期专题差分 demo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EPS = 1e-6

# 波段约定 v2（与 satellite_download.BANDS 一致；旧 4 波段 COG 走 descriptions 缺省回退）
V2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

# 指数公式表：名称 → (分子正项波段, 负项波段)。所有归一化差值指数同构。
FORMULAS: dict[str, tuple[str, str]] = {
    "ndbi": ("B11", "B8"),
    "ndvi": ("B8", "B4"),
    "ndwi": ("B3", "B8"),
}

# ===========================================================================
# D1：THEMES 专题注册表（纯数据 —— 本文件唯一的"知识"所在处）
# 阈值标注：文献先验 + 真实影像分位数标定（--probe），标定数字回填在注释里
# ===========================================================================
THEMES: dict[str, dict] = {
    "builtup": {
        "label": "建筑用地",
        "main": "ndbi",
        "rule": {"op": "gt", "threshold": 0.0},        # NDBI > 0（文献先验；标定见 T3 回填）
        "assist": {"index": "ndvi", "op": "lt_main"},  # 且 NDVI < NDBI：排农田/裸土（红旗①的工程缓解）
        "exclude": [],                                  # 可扩展：["water"] 待 water 注册实测后启用
        "color": "#5F5E5A",                             # 城市灰（对齐 W2 SEMANTIC）
    },
    "green": {
        "label": "绿化用地",
        "main": "ndvi",
        "rule": {"op": "gt", "threshold": 0.35},       # NDVI > 0.35（文献先验，仅注册未实测）
        "assist": None,
        "exclude": ["builtup", "water"],               # 城市树冠 NDVI 也高，先扣建筑/水体
        "color": "#3B6D11",
    },
    "water": {
        "label": "水域",
        "main": "ndwi",
        "rule": {"op": "gt", "threshold": 0.1},        # NDWI > 0.1（McFeeters 经典值，仅注册未实测）
        "assist": None,
        "exclude": [],
        "color": "#185FA5",
    },
}

AREA_BANDS: list[tuple[float, str]] = [(0.5, "轻微"), (2.0, "中等"), (float("inf"), "显著")]


# ===========================================================================
# D2：泛化引擎
# ===========================================================================
def compute_index(bands: np.ndarray, names: list[str], formula: str) -> np.ndarray:
    """(C,H,W) 反射率数组 + 波段名 → 归一化差值指数（[-1,1]，除零保护）。"""
    pos_name, neg_name = FORMULAS[formula]
    pos = bands[names.index(pos_name)]
    neg = bands[names.index(neg_name)]
    return (pos - neg) / (pos + neg + EPS)


def classify(bands: np.ndarray, names: list[str], theme_key: str) -> np.ndarray:
    """按 THEMES[theme] 规则判定专题掩膜（bool）。

    规则组合 = 主指数阈值 + assist 联合条件（如 builtup 的 NDVI<NDBI）
             + exclude 排除其它专题（递归判定，专题间无环——selftest 断言）。
    """
    th = THEMES[theme_key]
    main = compute_index(bands, names, th["main"])
    if th["rule"]["op"] == "gt":
        mask = main > th["rule"]["threshold"]
    else:
        mask = main < th["rule"]["threshold"]
    if th.get("assist"):
        asst = compute_index(bands, names, th["assist"]["index"])
        if th["assist"]["op"] == "lt_main":
            mask &= asst < main
        elif th["assist"]["op"] == "gt":
            mask &= asst > th["assist"]["threshold"]
    for other in th.get("exclude", []):
        mask &= ~classify(bands, names, other)
    return mask


def _valid_mask(bands: np.ndarray) -> np.ndarray:
    """有效像元：仓库约定 NoData=0，任一波段为 0 即无效（同 real_change_detection 语义）。"""
    return (bands > 0).all(axis=0)


def _cloud_mask(bands: np.ndarray, brightness: float = 0.30) -> np.ndarray:
    """简化云检测：全波段均值亮度超阈（同仓库云"全波段都亮"的合成/真实特性）。

    与 real_change_detection.cloud_mask 语义对齐但自包含（避免本模块拖 torch 依赖）。
    """
    return bands.mean(axis=0) > brightness


def _pixel_area_km2(bounds, height: int, width: int) -> float:
    """经纬度 bbox → 每像元面积 km²（中纬度余弦缩放，教学精度足够）。"""
    w, s, e, n = bounds.left, bounds.bottom, bounds.right, bounds.top
    import math
    mid_lat = math.radians((s + n) / 2)
    px_w_m = (e - w) * 111320.0 * math.cos(mid_lat) / width
    px_h_m = (n - s) * 110540.0 / height
    return px_w_m * px_h_m / 1e6


def _band_names(ds) -> list[str]:
    """优先用 COG 内嵌波段描述；缺失/不完整时回退 v2 固定顺序（旧 4 波段取前 4 个）。"""
    descs = [d for d in (ds.descriptions or ()) if d]
    if len(descs) == ds.count:
        return list(descs)
    return V2_BANDS[: ds.count]


def index_change(cog_a: str | Path, cog_b: str | Path, theme_key: str) -> dict:
    """两期 COG 专题差分：新增/消失/净变化（像元数 + km²）。

    输出字段与 spectral_diff_change 同构（change_ratio/valid_pct），
    cartography / report 零适配成本。
    """
    if theme_key not in THEMES:
        raise ValueError(f"未知专题 {theme_key!r}，可选：{sorted(THEMES)}")
    pa, pb = Path(cog_a), Path(cog_b)
    for p in (pa, pb):
        if not p.is_file():
            raise FileNotFoundError(f"COG 不存在：{p}")
    with rasterio.open(pa) as da:
        bands_a = da.read().astype(np.float32) / 10000.0
        names_a = _band_names(da)
        bounds, shape = da.bounds, (da.height, da.width)
    with rasterio.open(pb) as db_:
        bands_b = db_.read().astype(np.float32) / 10000.0
        names_b = _band_names(db_)
    if bands_a.shape != bands_b.shape:
        raise ValueError(f"两景尺寸不一致：{bands_a.shape} vs {bands_b.shape}（需同 tile 同窗口）")
    if names_a != names_b:
        raise ValueError(f"两景波段表不一致：{names_a} vs {names_b}")

    valid = _valid_mask(bands_a) & _valid_mask(bands_b)
    cloud = _cloud_mask(bands_a) | _cloud_mask(bands_b)
    usable = valid & ~cloud

    mask_a = classify(bands_a, names_a, theme_key) & usable
    mask_b = classify(bands_b, names_b, theme_key) & usable
    gain = mask_b & ~mask_a          # 新增：期 B 命中、期 A 未命中
    loss = mask_a & ~mask_b          # 消失

    n_valid = int(usable.sum())
    px_km2 = _pixel_area_km2(bounds, shape[0], shape[1])
    gain_px, loss_px = int(gain.sum()), int(loss.sum())
    theme = THEMES[theme_key]
    return {
        "method": "index_change",
        "theme": theme_key,
        "bbox": [round(bounds.left, 4), round(bounds.bottom, 4),
                 round(bounds.right, 4), round(bounds.top, 4)],
        "theme_label": theme["label"],
        "threshold": theme["rule"]["threshold"],   # 报告模板兼容字段（主指数阈值）
        "shape": list(shape),
        "valid_px": n_valid,
        "change_px": gain_px + loss_px,
        "change_ratio": round((gain_px + loss_px) / max(n_valid, 1), 4),
        "valid_pct": round(n_valid / (shape[0] * shape[1]), 4),
        "gain_px": gain_px,
        "loss_px": loss_px,
        "gain_km2": round(gain_px * px_km2, 4),
        "loss_km2": round(loss_px * px_km2, 4),
        "net_km2": round((gain_px - loss_px) * px_km2, 4),
        "pixel_area_km2": round(px_km2, 6),
    }


# ===========================================================================
# D4：阈值标定探查（先探查后定值，不拍脑袋）
# ===========================================================================
def probe(cog_path: str | Path) -> dict:
    """打印三指数分位数表 → 人工/脚本据此定阈值并回填 THEMES 注释。"""
    with rasterio.open(cog_path) as ds:
        bands = ds.read().astype(np.float32) / 10000.0
        names = _band_names(ds)
    usable = _valid_mask(bands) & ~_cloud_mask(bands)
    out: dict[str, dict] = {}
    print(f"=== 指数分位数探查：{Path(cog_path).name}（有效像元 {int(usable.sum()):,}）===")
    for f in ("ndbi", "ndvi", "ndwi"):
        idx = compute_index(bands, names, f)[usable]
        qs = {q: round(float(np.percentile(idx, q)), 3) for q in (1, 5, 25, 50, 75, 95, 99)}
        out[f] = qs
        print(f"  {f.upper():5s} " + "  ".join(f"p{q}={v:+.3f}" for q, v in qs.items()))
    return out


# ===========================================================================
# selftest（零网络零数据依赖：合成 GT 驱动）
# ===========================================================================
def _synthetic_scene() -> tuple[np.ndarray, list[str], np.ndarray]:
    """6 波段合成场景（512 真值可精确断言）：

    期 A 全域植被；期 B 中部 40x60 矩形变成城市（NDBI 由负转正、NDVI 转负）
    → 该矩形必须恰好被判为"新增建筑用地"，其余像元无变化。
    """
    h, w = 120, 160
    names = V2_BANDS
    veg = np.array([0.05, 0.08, 0.06, 0.25, 0.18, 0.09])
    urb = np.array([0.12, 0.14, 0.16, 0.14, 0.24, 0.22])
    water = np.array([0.07, 0.05, 0.03, 0.02, 0.01, 0.005])

    def scene(base: np.ndarray, block: np.ndarray | None) -> np.ndarray:
        bands = np.tile(base[None, None, :], (h, w, 1)).astype(np.float32)
        if block is not None:
            bands[30:70, 50:110] = block            # 40x60 = 2400 px 的"新增建成区"
        # 左下角一片水体（water 专题 GT）
        bands[h - 20:, :30] = water
        return np.moveaxis(bands, -1, 0)            # (6,H,W)

    return scene(veg, urb), names, scene(veg, None)


def selftest() -> int:
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"（{detail}）" if detail else ""))
        ok = ok and bool(passed)

    print("thematic_change selftest —— 零网络零数据依赖（合成 GT）")

    # 1. THEMES 注册表结构合法性
    for key, th in THEMES.items():
        check(f"[{key}] main 指数在 FORMULAS", th["main"] in FORMULAS)
        need = {n for f in ([th["main"], th["assist"]["index"]] if th.get("assist") else [th["main"]])
                for n in FORMULAS[f]}
        check(f"[{key}] 所需波段在 v2 表", need <= set(V2_BANDS), str(sorted(need)))
        check(f"[{key}] 阈值在 [-1,1]", -1.0 <= th["rule"]["threshold"] <= 1.0)
        check(f"[{key}] exclude 引用存在", all(e in THEMES for e in th["exclude"]))
    check("exclude 无环（专题图是 DAG）", _no_cycle())

    # 2. index_change 合成 GT：新增建筑恰好 2400 px
    bands_b, names, bands_a = _synthetic_scene()
    # 写成临时 COG 走真实文件路径（覆盖 descriptions 缺省回退逻辑）
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fa, fb = Path(td) / "a.tif", Path(td) / "b.tif"
        _write_tmp_cog(fa, bands_a, names)
        _write_tmp_cog(fb, bands_b, names)
        res = index_change(fa, fb, "builtup")
    check("GT 新增像元 = 2400", res["gain_px"] == 2400, f"gain={res['gain_px']}")
    check("GT 消失像元 = 0", res["loss_px"] == 0, f"loss={res['loss_px']}")
    check("change_ratio 同构在 (0,1)", 0 < res["change_ratio"] < 1, str(res["change_ratio"]))
    check("net_km2 > 0（净新增）", res["net_km2"] > 0, f"{res['net_km2']} km²")
    check("theme_label 注入（报告用）", res["theme_label"] == "建筑用地")
    check("面积公式自洽（gain_px × 像元面积）",
          abs(res["gain_px"] * res["pixel_area_km2"] - res["gain_km2"]) < 5e-4)

    # 3. water 专题：左下角水体 20x30=600 px 命中
    res_w = None
    with tempfile.TemporaryDirectory() as td:
        fa, fb = Path(td) / "a.tif", Path(td) / "b.tif"
        _write_tmp_cog(fa, bands_a, names)
        _write_tmp_cog(fb, bands_b, names)
        res_w = index_change(fa, fb, "water")
    check("water 专题两期稳定（gain=loss=0）", res_w["gain_px"] == 0 and res_w["loss_px"] == 0,
          f"gain={res_w['gain_px']} loss={res_w['loss_px']}")

    # 4. 非法专题名 → 显式报错（不静默回退）
    try:
        index_change("nonexist_a.tif", "nonexist_b.tif", "farmland")
        check("未知专题显式报错", False)
    except ValueError as e:
        check("未知专题显式报错", "farmland" in str(e))

    # 5. 面积档位（报告闸门用）
    def _area_label(km2: float) -> str:
        return next(label for hi, label in AREA_BANDS if abs(km2) < hi)
    check("面积档位代码化", _area_label(0.3) == "轻微" and _area_label(1.0) == "中等"
          and _area_label(3.0) == "显著")

    print(f"\n=== 汇总: {'全部通过' if ok else '存在 FAIL'} ===")
    return 0 if ok else 1


def _no_cycle(seen: set[str] | None = None, key: str | None = None) -> bool:
    """exclude 引用图无环（当前注册表是简单引用，直接检查引用闭包不含自身）。"""
    for k, th in THEMES.items():
        stack, visited = list(th["exclude"]), set()
        while stack:
            cur = stack.pop()
            if cur == k:
                return False
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(THEMES[cur]["exclude"])
    return True


def _write_tmp_cog(path: Path, bands: np.ndarray, names: list[str]) -> None:
    """合成数组 → 最小 GeoTIFF（transform 单位度，无需真 COG——rasterio 直读）。"""
    from rasterio.transform import from_origin
    c, h, w = bands.shape
    profile = {"driver": "GTiff", "width": w, "height": h, "count": c,
               "dtype": "uint16", "crs": "EPSG:4326",
               "transform": from_origin(113.50, 34.88, 0.001, 0.001), "nodata": 0}
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(c):
            dst.write(np.clip(bands[i] * 10000, 1, 65535).astype("uint16"), i + 1)
        dst.descriptions = names


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="光谱指数专题框架（THEMES + index_change）")
    ap.add_argument("--selftest", action="store_true", help="自测（零网络零数据）")
    ap.add_argument("--probe", metavar="COG", help="指数分位数探查（阈值标定）")
    ap.add_argument("cog_a", nargs="?", help="期 A COG 路径")
    ap.add_argument("cog_b", nargs="?", help="期 B COG 路径")
    ap.add_argument("theme", nargs="?", default="builtup", help=f"专题（{sorted(THEMES)}）")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.probe:
        probe(args.probe)
        return 0
    if not (args.cog_a and args.cog_b):
        ap.error("需要 cog_a cog_b（或 --selftest / --probe）")
    res = index_change(args.cog_a, args.cog_b, args.theme)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n{res['theme_label']}：新增 {res['gain_km2']} km² / 消失 {res['loss_km2']} km²"
          f" / 净变化 {res['net_km2']:+} km²（变化占比 {res['change_ratio']:.2%}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
