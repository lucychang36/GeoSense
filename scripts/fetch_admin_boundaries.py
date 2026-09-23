"""region-data-inventory 交付物：fetch_admin_boundaries.py —— 全国行政区边界下载与缓存。

数据源：阿里云 DataV GeoAtlas（areas_v3，公开接口）
  https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json
  100000_full（省级）→ 各省 {adcode}_full（地市级）→ 各市 {adcode}_full（区县级）

产出（design D6）：
  data/admin/china_provinces.geojson   34 省级
  data/admin/china_cities.geojson      ~370 地市级
  data/admin/china_districts.geojson   ~2800 区县级
每个 feature 带 bbox 属性（data_inventory.load_admin_features 直接消费，免几何解析），
几何经 shapely simplify(0.005°≈500m) 降体积（教学精度；bbox 解析不受影响）。

用法：
  .venv/bin/python scripts/fetch_admin_boundaries.py               # 全量下载（~400 请求，2-5 分钟）
  .venv/bin/python scripts/fetch_admin_boundaries.py --from-cache  # 离线重放（只从缓存重建输出，红旗 R3 降级）
  .venv/bin/python scripts/fetch_admin_boundaries.py --no-districts# 只到省市两级（快速模式）

世界级（提案 non-goal）：level 枚举预留 country；导入函数对任意 GeoJSON 通用，
需要时加 fetch 源即可——机制开放，数据按需引入。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "admin"
BASE = "https://geo.datav.aliyun.com/areas_v3/bound/{code}_full.json"
SIMPLIFY_TOL = 0.005  # 度 ≈ 500m（教学精度，bbox 解析不受影响）
RETRIES = 3
TIMEOUT = 30


def fetch(code: str) -> dict:
    """拉一个 _full.json，带重试（红旗 R3：DataV 偶发限流）。"""
    url = BASE.format(code=code)
    last_exc: Exception | None = None
    for i in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 —— 网络类错误统一重试
            last_exc = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"下载失败 {url}: {last_exc}")


def slim_feature(feat: dict) -> dict | None:
    """瘦身：只留 name/adcode/level/bbox/center + 简化几何。"""
    props = feat.get("properties") or {}
    name, adcode, level = props.get("name"), props.get("adcode"), props.get("level")
    if not name or not adcode:
        return None
    geom = feat.get("geometry")
    if not geom:
        return None
    from shapely.geometry import shape, mapping
    g = shape(geom)
    if g.is_empty:
        return None
    g2 = g.simplify(SIMPLIFY_TOL, preserve_topology=True)
    w, s, e, n = g2.bounds
    return {
        "type": "Feature",
        "properties": {"name": name, "adcode": adcode, "level": level,
                       "bbox": [round(w, 5), round(s, 5), round(e, 5), round(n, 5)],
                       "center": props.get("center")},
        "geometry": mapping(g2),
    }


def collect_children(parents: list[dict], child_level: str,
                     side_catch: dict[str, list[dict]] | None = None) -> list[dict]:
    """逐 parent 拉 _full 收集其下级 feature（parent.properties.adcode → {code}_full）。

    side_catch：直辖市陷阱（2026-09-23 实录）——北京/天津/上海/重庆的 110000_full
    里没有 'city' 级子项，区县直接以 'district' 级出现。把 {'district': [...]} 传入
    即可顺路收进区县列表，避免直辖市辖区整层丢失（北京朝阳区缺失的实证）。"""
    out: list[dict] = []
    for i, p in enumerate(parents, 1):
        code = str(p["properties"]["adcode"])
        try:
            data = fetch(code)
        except RuntimeError as exc:
            print(f"  [WARN] 跳过 {p['properties']['name']}({code})：{exc}")
            continue
        for k in (slim_feature(f) for f in data.get("features", [])):
            if not k:
                continue
            lvl = k["properties"]["level"]
            if lvl == child_level:
                out.append(k)
            elif side_catch is not None and lvl in side_catch:
                side_catch[lvl].append(k)
        if i % 10 == 0 or i == len(parents):
            extra = f"（顺路收 {sum(len(v) for v in (side_catch or {}).values())} 直辖市区县）" \
                if side_catch else ""
            print(f"  {i}/{len(parents)} 已拉取，累计 {child_level} {len(out)} 条{extra}")
    return out


def write_geojson(path: Path, feats: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                               ensure_ascii=False), encoding="utf-8")
    print(f"✅ {path.relative_to(ROOT)}：{len(feats)} 条，{path.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true",
                    help="离线重放：跳过下载，直接从缓存文件重建输出（降级路径）")
    ap.add_argument("--no-districts", action="store_true", help="只到省市两级（快速模式）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provinces_f = OUT_DIR / "china_provinces.geojson"
    cities_f = OUT_DIR / "china_cities.geojson"
    districts_f = OUT_DIR / "china_districts.geojson"

    if args.from_cache:
        # 离线重放：缓存即真相（三文件应已存在）
        for f in (provinces_f, cities_f):
            if not f.is_file():
                print(f"❌ 缓存缺失 {f}，无法离线重放")
                return 1
        n_prov = len(json.loads(provinces_f.read_text(encoding="utf-8"))["features"])
        n_city = len(json.loads(cities_f.read_text(encoding="utf-8"))["features"])
        print(f"✅ 离线重放完成（省 {n_prov} / 市 {n_city}）")
        return 0

    t0 = time.time()
    # 1. 省级
    data = fetch("100000")
    provinces = [p for p in (slim_feature(f) for f in data.get("features", []))
                 if p and p["properties"]["level"] == "province"]
    write_geojson(provinces_f, provinces)

    # 2. 地市级（逐省）；直辖市辖区（district 级）顺路收进区县
    print("拉取地市级…")
    districts_direct: list[dict] = []
    cities = collect_children(provinces, "city", side_catch={"district": districts_direct})
    write_geojson(cities_f, cities)

    # 3. 区县级（逐市，~370 请求）+ 直辖市辖区合并
    if not args.no_districts:
        print("拉取区县级（请求最多，预计 2-5 分钟）…")
        districts = collect_children(cities, "district") + districts_direct
        write_geojson(districts_f, districts)

    print(f"🎉 完成，用时 {time.time() - t0:.0f}s。下一步："
          f".venv/bin/python scripts/seed_postgis.py --admin-geojson")
    return 0


if __name__ == "__main__":
    sys.exit(main())
