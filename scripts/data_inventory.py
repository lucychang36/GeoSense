"""region-data-inventory 交付物：data_inventory.py —— 区域数据清单与空间求交解析。

解决什么问题（2026-09-23 金水区案例）：
- 影像元数据活在 manifest.json，agent 的工具层够不着 → 用户问"郑州金水区"时，
  agent 只能说"可用影像仅为深圳湾"，漏掉郑州高新区两期（自己上周刚跑过）。
- 本模块提供唯一真相查询口：manifest 为唯一真相（D2），任意地名/任意 bbox
  通过**空间求交**回答"有没有数据、哪几期、推荐用什么"——region 开放语义（D1），
  全国乃至全球零注册，加数据即生效。

三层结构（design D1/D3/D4）：
  resolve_region    地名 → bbox（admin_boundary 缓存精确/模糊+消歧 → 内置词表回退）
  inventory_query   bbox × manifest 影像 bbox 空间求交 → coverage 分档 + 最近邻推荐
  REGION_META       region → 中文显示文案（封闭最小化；未注册自动生成，开放-封闭）
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "cogs" / "manifest.json"
ADMIN_DIR = ROOT / "data" / "admin"

# ---------------------------------------------------------------------------
# REGION_META：region → 显示文案注册表（开放-封闭：判定靠求交零注册，只有文案登记）
# ---------------------------------------------------------------------------
REGION_META = {
    "szbay": {"label": "深圳湾", "desc": "深圳湾河口-深圳河流域 Sentinel-2 场景库（2019-2025）"},
    "zhengzhou_hightech": {"label": "郑州高新区", "desc": "郑州高新技术产业开发区（bbox 近似，49SGU）"},
}

# 内置地名词表回退（admin_boundary 没有的自定义区域；与下载侧 REGION_PRESETS 分离，
# 本表是"查询侧"的兜底——设计 D3 第三级）
GEO_FALLBACK = {
    "深圳湾": [113.80, 22.40, 114.00, 22.65],
}

# 词表别名 → 数据 region 身份（region-known 判定用）：
# 查询命中数据自属区域时，bbox 近似是已知红线（部分覆盖不否定可用性）；
# 查询非数据自属区域（如 admin 库里的金水区）时，部分覆盖不足以支撑全区口径
ALIAS_REGION = {
    "深圳湾": "szbay",
}

# coverage 面积比分档阈值（D4）：查询 bbox 被影像 bbox 覆盖比例。
# 注意：part 与 none 的分界不再用面积阈值——实测深圳湾自定义框（0.336）与金水区
# 相交（0.337）几乎相同，纯阈值无法区分；改由 region_known 身份判定兜底（见下）。
COVERAGE_FULL = 0.95

# 缓存文件名（fetch_admin_boundaries.py 产出）
ADMIN_FILES = {
    "district": "china_districts.geojson",
    "city": "china_cities.geojson",
    "province": "china_provinces.geojson",
}
# 地名解析优先级：区县 > 市 > 省（更具体者优先，D6）
_LEVEL_PRIORITY = ["district", "city", "province"]

_admin_cache: list[dict] | None = None


def legacy_region(scene: dict) -> str:
    """region 归一：老条目无 region 字段 → 回落 szbay（提案红旗 R2）。"""
    return scene.get("region") or "szbay"


def load_manifest_scenes() -> list[dict]:
    """读 manifest 并归一为求交所需的轻量结构（唯一真相，D2）。"""
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scenes = []
    for e in entries:
        w, s, x, n = e["bbox"]
        scenes.append({
            "id": e["id"],
            "region": legacy_region(e),
            "date": e["date"],
            "cloud": e.get("cloud", 0.0),
            "bbox": [float(w), float(s), float(x), float(n)],
            "bands": e.get("bands", []),
            "path": e["path"],
        })
    return scenes


def load_admin_features(refresh: bool = False) -> list[dict]:
    """读 data/admin/ 缓存的全国边界 → [{name, level, bbox}]（R4：单次加载进程内缓存）。
    缓存缺失时返回 []（解析优雅降级到词表/未解析），不抛异常。"""
    global _admin_cache
    if _admin_cache is not None and not refresh:
        return _admin_cache
    feats: list[dict] = []
    for level in _LEVEL_PRIORITY:
        f = ADMIN_DIR / ADMIN_FILES[level]
        if not f.is_file():
            continue
        gj = json.loads(f.read_text(encoding="utf-8"))
        for feat in gj.get("features", []):
            props = feat.get("properties") or {}
            name = props.get("name")
            bbox = props.get("bbox")  # slim_feature 把 bbox 写进 properties（非 feature 顶层）
            if name and bbox:
                feats.append({"name": name, "level": level, "bbox": [float(v) for v in bbox]})
    _admin_cache = feats
    return feats


def resolve_region(text: str, admin_features: list[dict] | None = None) -> dict:
    """地名 → bbox，三级回退（D3）：
    1. admin 缓存子串匹配（query 含边界名，最长名=最具体优先；同名同级多命中→候选消歧）
    2. 内置词表 GEO_FALLBACK
    3. 未解析 → resolved=False + 诚实指引（全球范围走 bbox 直给）

    返回：{resolved, name, level, bbox, candidates, source}
    """
    result = {"resolved": False, "name": "", "level": "", "bbox": None,
              "candidates": [], "source": ""}
    text = (text or "").strip()
    if not text:
        return result

    feats = load_admin_features() if admin_features is None else admin_features

    # ---- 1. admin 子串匹配（query 文本里出现的边界名）----
    if feats:
        matches = [f for f in feats if f["name"] and f["name"] in text]
        if matches:
            best_len = max(len(m["name"]) for m in matches)
            top = [m for m in matches if len(m["name"]) == best_len]
            if len(top) > 1:
                # 红旗 R1：同名同级歧义（如北京/长春的朝阳区）→ 返回候选，禁止静默取第一个
                result["candidates"] = [
                    {"name": m["name"], "level": m["level"], "bbox": m["bbox"]} for m in top
                ]
                return result
            m = top[0]
            result.update(resolved=True, name=m["name"], level=m["level"],
                          bbox=m["bbox"], source="admin_boundary")
            return result

    # ---- 2. 内置词表 ----
    for alias, bbox in GEO_FALLBACK.items():
        if alias in text:
            result.update(resolved=True, name=alias, level="custom",
                          bbox=list(bbox), source="geo_fallback")
            return result

    # ---- 3. 未解析 ----
    return result


def _box_area(b: list[float]) -> float:
    return max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)


def _inter_ratio(query: list[float], scene: list[float]) -> float:
    """影像 bbox 对查询 bbox 的覆盖面积比（度²面积比，比例量纲无关）。"""
    from shapely.geometry import box
    inter = box(*query).intersection(box(*scene))
    return inter.area / _box_area(query)


def _centroid(b: list[float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _nearest_region_text(query_bbox: list[float], scenes: list[dict]) -> str:
    """最近邻可用区域描述（D4）：按 region 质心欧氏距离取最近，列可用日期。"""
    qc = _centroid(query_bbox)
    by_region: dict[str, list[dict]] = {}
    for s in scenes:
        by_region.setdefault(s["region"], []).append(s)
    best = None
    for region, rs in by_region.items():
        # 质心：region 各景 bbox 中心均值
        ws = sum((r["bbox"][0] + r["bbox"][2]) / 2 for r in rs) / len(rs)
        ys = sum((r["bbox"][1] + r["bbox"][3]) / 2 for r in rs) / len(rs)
        d = ((ws - qc[0]) ** 2 + (ys - qc[1]) ** 2) ** 0.5
        label = REGION_META.get(region, {}).get("label", region)
        dates = sorted({r["date"] for r in rs})
        cand = (d, region, label, dates, rs)
        if best is None or d < best[0]:
            best = cand
    if best is None:
        return "系统中暂无任何影像数据。"
    _, region, label, dates, rs = best
    bb = rs[0]["bbox"]
    return (f"最接近的可用区域：{region}"
            f"（{label}，{len(rs)} 期：{'、'.join(dates)}，bbox {bb[0]}-{bb[2]}E {bb[1]}-{bb[3]}N）")


def _recommendation(query_bbox: list[float], scenes: list[dict]) -> str:
    """无覆盖时的推荐（D4）：事实 + 最近邻替代。"""
    return f"查询区域无影像覆盖。{_nearest_region_text(query_bbox, scenes)}"


def inventory_query(region_text: str = "", bbox=None, scenes: list[dict] | None = None) -> dict:
    """区域数据清单查询（D4 输出协议）。bbox 优先于 region_text。

    返回 JSON 可序列化 dict：
      resolved / query / query_bbox / scenes[{id,region,date,cloud,bbox,intersection,bands}]
      coverage: full | partial | none    recommendation: str    candidates: []
    """
    if scenes is None:
        scenes = load_manifest_scenes()

    # ---- 查询 bbox：显式 bbox 优先（全球范围的支持方式，D3）----
    if bbox is not None:
        if isinstance(bbox, str):
            bbox = [float(v) for v in bbox.split(",")]
        qb = [float(v) for v in bbox]
        qname, qlevel, source, candidates = "自定义bbox", "bbox", "explicit", []
        resolved = True
        alias = None
    else:
        r = resolve_region(region_text)
        if not r["resolved"]:
            out = {"resolved": False, "query": region_text, "query_bbox": None,
                   "scenes": [], "coverage": "none", "candidates": r["candidates"],
                   "recommendation": (
                       "无法解析该地名（未命中行政区边界缓存与内置词表）。"
                       "可改用 bbox 直给：data_inventory_tool(bbox=\"west,south,east,north\")，"
                       "系统支持全国乃至全球任意范围。")}
            return out
        qb = r["bbox"]
        qname, qlevel, source, candidates = r["name"], r["level"], r["source"], r["candidates"]
        alias = qname if r["source"] == "geo_fallback" else None

    # region_known：查询区域是否为数据自属区域（D4 修订——身份判定而非面积阈值）
    label_to_region = {m.get("label"): reg for reg, m in REGION_META.items()}
    known_region = ALIAS_REGION.get(alias or "") or label_to_region.get(qname)

    # ---- 空间求交 ----
    hits = []
    for s in scenes:
        ratio = _inter_ratio(qb, s["bbox"])
        if ratio <= 0:
            continue
        hits.append((s, ratio))
    hits.sort(key=lambda t: (-t[1], t[0]["date"]))

    if not hits:
        cover = "none"
    elif max(r for _, r in hits) >= COVERAGE_FULL:
        cover = "full"
    else:
        cover = "partial"

    # ---- recommendation：分身份给出可用性判断（诚实披露覆盖事实）----
    if not hits:
        rec = _recommendation(qb, scenes)
    elif known_region:
        rec = (f"查询区域「{qname}」为系统数据自属区域（{known_region}，bbox 近似为已知红线）："
               + "；".join(f"{s['id']}（{s['date']}，云 {s['cloud']}%）" for s, _ in hits))
    elif cover == "full":
        rec = (f"查询区域「{qname}」被影像完整覆盖："
               + "；".join(f"{s['id']}（{s['date']}，云 {s['cloud']}%）" for s, _ in hits))
    else:
        max_ratio = max(r for _, r in hits)
        rec = (f"查询区域「{qname}」仅有 {max_ratio:.0%} 被现有影像覆盖（区域主体在数据外），"
               f"不足以支撑全区口径分析。{_nearest_region_text(qb, scenes)}")

    out = {
        "resolved": True,
        "query": region_text or "bbox",
        "region_name": qname,
        "region_level": qlevel,
        "region_source": source,
        "region_known": known_region or "",
        "query_bbox": qb,
        "scenes": [{
            "id": s["id"], "region": s["region"], "date": s["date"],
            "cloud": s["cloud"], "bbox": s["bbox"], "bands": s["bands"],
            "path": s["path"],
            "intersection": ("full" if r >= COVERAGE_FULL else "partial"),
            "coverage_ratio": round(r, 4),
        } for s, r in hits],
        "coverage": cover,
        "candidates": candidates,
        "recommendation": rec,
    }
    return out


# ---------------------------------------------------------------------------
# selftest（设计 D8）：合成场景五分支 + 金水区真实查询，断言一律显式 bool()（项目铁律）
# ---------------------------------------------------------------------------
def selftest() -> int:
    ok = True

    # 合成两区域场景（不依赖磁盘 manifest）
    synth = [
        {"id": "ZZ_2023", "region": "zhengzhou_hightech", "date": "2023-06-26",
         "cloud": 5.2, "bbox": [113.5, 34.73, 113.7, 34.88], "bands": ["B2", "B8"], "path": "x.tif"},
        {"id": "ZZ_2025", "region": "zhengzhou_hightech", "date": "2025-06-27",
         "cloud": 4.4, "bbox": [113.5, 34.73, 113.7, 34.88], "bands": ["B2", "B8"], "path": "y.tif"},
        {"id": "SZ_2023", "region": "szbay", "date": "2023-07-08",
         "cloud": 1.0, "bbox": [113.88, 22.46, 114.10, 22.60], "bands": ["B2", "B8"], "path": "z.tif"},
    ]

    # 1. full：查询 bbox == 影像 bbox
    r = inventory_query(bbox=[113.5, 34.73, 113.7, 34.88], scenes=synth)
    ok = bool(ok and r["resolved"] and r["coverage"] == "full" and len(r["scenes"]) == 2)
    print(f"[{'PASS' if ok else 'FAIL'}] full 覆盖分支（coverage={r['coverage']}, {len(r['scenes'])} 景）")

    # 2. partial：查询区与影像部分相交（比例 ≥ MIN_USABLE_COVERAGE）
    r2 = inventory_query(bbox=[113.6, 34.73, 113.9, 34.93], scenes=synth)
    ok2 = bool(r2["coverage"] == "partial" and r2["scenes"][0]["intersection"] == "partial")
    print(f"[{'PASS' if ok2 else 'FAIL'}] partial 覆盖分支（ratio={r2['scenes'][0]['coverage_ratio'] if r2['scenes'] else 0}）")
    ok = bool(ok and ok2)

    # 2b. 窄条相交 + 非数据自属区域：金水区式 0.18 相交 → partial 事实 + 推荐替代
    #（region_known 身份判定：partial 不否定数据自属区域，但非自属区域要警告不足）
    r2b = inventory_query(bbox=[113.66, 34.76, 113.85, 34.90], scenes=synth)
    ok2b = bool(r2b["coverage"] == "partial" and r2b["scenes"]
                and not r2b["region_known"]
                and "zhengzhou_hightech" in r2b["recommendation"])
    print(f"[{'PASS' if ok2b else 'FAIL'}] 窄条相交非自属区域分支（ratio={r2b['scenes'][0]['coverage_ratio'] if r2b['scenes'] else 0} → 警告+推荐）")
    ok = bool(ok and ok2b)

    # 2c. 数据自属区域（词表别名命中 szbay）：partial 不否定可用性（bbox 近似为已知红线）
    r2c = inventory_query("对比深圳湾的变化", scenes=synth)
    ok2c = bool(r2c["resolved"] and r2c["region_known"] == "szbay"
                and r2c["coverage"] == "partial"
                and "数据自属区域" in r2c["recommendation"])
    print(f"[{'PASS' if ok2c else 'FAIL'}] 自属区域分支（region_known={r2c['region_known']}，partial 不拒绝）")
    ok = bool(ok and ok2c)

    # 3. none + 最近邻推荐：远处查询框 → none，推荐里出现 zhengzhou_hightech（郑州质心比深圳湾近）
    r3 = inventory_query(bbox=[110.0, 30.0, 111.0, 31.0], scenes=synth)
    ok3 = bool(r3["coverage"] == "none" and not r3["scenes"]
               and "zhengzhou_hightech" in r3["recommendation"])
    print(f"[{'PASS' if ok3 else 'FAIL'}] none 分支 + 最近邻推荐（{r3['recommendation'][:40]}…）")
    ok = bool(ok and ok3)

    # 4. 地名歧义消歧（红旗 R1）：两个"朝阳区" → candidates 返回 2 条，不静默取第一个
    fake_admin = [
        {"name": "朝阳区", "level": "district", "bbox": [116.3, 39.8, 116.7, 40.1]},
        {"name": "朝阳区", "level": "district", "bbox": [125.2, 43.7, 125.5, 44.0]},
        {"name": "金水区", "level": "district", "bbox": [113.66, 34.76, 113.85, 34.90]},
    ]
    r4 = resolve_region("对比朝阳区 2015 和 2025 的变化", admin_features=fake_admin)
    ok4 = bool(not r4["resolved"] and len(r4["candidates"]) == 2)
    print(f"[{'PASS' if ok4 else 'FAIL'}] 歧义消歧分支（candidates={len(r4['candidates'])}）")
    ok = bool(ok and ok4)

    # 5. 未解析诚实降级：空 admin + 词表外地名 → resolved=False + bbox 指引
    r5 = resolve_region("杭州西湖区", admin_features=[])
    ok5 = bool(not r5["resolved"] and not r5["candidates"])
    r5b = inventory_query("杭州西湖区", scenes=synth)
    ok5 = bool(ok5 and not r5b["resolved"] and "bbox" in r5b["recommendation"])
    print(f"[{'PASS' if ok5 else 'FAIL'}] 未解析降级分支（指引含 bbox 直给）")
    ok = bool(ok and ok5)

    # 6. 词表回退：深圳湾（admin 缓存没有的自定义区域）
    r6 = resolve_region("对比深圳湾的变化", admin_features=[])
    ok6 = bool(r6["resolved"] and r6["source"] == "geo_fallback")
    print(f"[{'PASS' if ok6 else 'FAIL'}] 词表回退分支（source={r6['source']}）")
    ok = bool(ok and ok6)

    # 7. 真实数据端到端：金水区（admin 缓存）→ partial 事实 + 警告不足 + 推荐 zhengzhou_hightech
    feats = load_admin_features()
    if feats:
        r7 = inventory_query("郑州金水区")
        ok7 = bool(r7["resolved"] and r7["coverage"] == "partial"
                   and not r7["region_known"]
                   and "zhengzhou_hightech" in r7["recommendation"]
                   and "不足以支撑" in r7["recommendation"])
        print(f"[{'PASS' if ok7 else 'FAIL'}] 金水区真实查询（coverage={r7['coverage']}，含推荐）")
        ok = bool(ok and ok7)
    else:
        print("[SKIP] 金水区真实查询（data/admin/ 缓存未生成，跑 fetch_admin_boundaries.py 后重试）")

    print(f"\n=== data_inventory selftest: {'ALL PASS' if ok else 'FAILED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
