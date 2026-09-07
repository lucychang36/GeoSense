#!/usr/bin/env python3
"""text_to_map.py — 第10月 W4：Text-to-Map（自然语言 → Mapbox 样式 JSON）。

学习计划（第10月 AI Cartography 收官）：自然语言→Mapbox样式JSON。

核心思想（explore 方案 B，用户确认）：意图-渲染分层
    LLM 只负责「语义 → 枚举」——把一句话压成受限枚举的制图意图（CartographyIR），
    每个字段都在白名单里（theme 3 选 1、类别 3 选 N、语义词必须命中 W2 SEMANTIC 表）；
    确定性代码负责「枚举 → 语法」——assemble_style() 把 IR 组装成合法的
    Mapbox Style Spec v8 JSON，复用 W2 的 match/interpolate 表达式与 W3 的
    symbol 标注骨架（前两周代码注释里"W4 直接组合"的伏笔在此回收）。

为什么不让 LLM 直出完整 Style JSON？Style Spec 是强 schema 语言，
自由生成必然幻觉非法字段/表达式；分层后每个字段可枚举校验，幻觉面积最小。
这正是业界 Text-to-X 的"约束生成"标准模式：LLM 的输出空间是有限集合，不是字符串。

三主题 × 伏笔回收（design D5）：
    poi  → W2 match 表达式（fill+circle 复用同一表达式对象：表达式与图层类型解耦）
           + W3 place_labels/to_mapbox_labels（标注避让 symbol 层）
    ndvi → W2 interpolate 表达式（raster_to_grid 网格多边形化补连续数据载体）
    cog  → raster-opacity（卫星影像叠加，透明度由一句话控制）

SSE 接入（backend）：text_to_map() 顶层函数返回 {summary, fallback, map_style}，
main.py 的 _stream 检测到 map_style 时发新事件类型 style（全量 JSON，
不经 result 事件的 [:200] 摘要路径）——前端 addSource/addLayer 注入渲染。

用法（项目 .venv，WorkBuddy 会话内加 env -u PYTHONPATH）：
  .venv/bin/python scripts/text_to_map.py --selftest   # 组装器自测（秒级，LLM 不在场）
  .venv/bin/python scripts/text_to_map.py              # 3 条真实 NL → 3 份 Style JSON（走 DeepSeek）
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.auto_symbology import (  # noqa: E402  （W2 引擎，验收⑤零改动）
    SEMANTIC,
    choose_symbology,
    profile_data,
    to_mapbox_style,
)
from scripts.auto_cartography import (  # noqa: E402  （W3 引擎，验收⑤零改动）
    Feature,
    merc_to_lonlat,
    place_labels,
    to_mapbox_labels,
)

# ---------------------------------------------------------------- 常量（白名单即契约）
THEMES = ("poi", "ndvi", "cog")
POI_CLASSES = ("subway", "park", "school")
POI_FILES = {"subway": "poi_subway.json", "park": "poi_park.json", "school": "poi_school.json"}
OSM_DIR = PROJECT_ROOT / "data" / "osm"
COGS_DIR = PROJECT_ROOT / "data" / "cogs"
OUT_DIR = PROJECT_ROOT / "data" / "output" / "text_to_map"

# ndvi color_intent 白名单 = W2 sequential 分支的触发词（choose_symbology 直接消费）
NDVI_INTENTS = ("植被", "风险", "密度", "热度", "NDVI", "")
# poi 语义覆盖的值域 = W2 SEMANTIC 表键（水/植被/城市/新增/消退/持续…）

# region 枚举 → (fit_bbox, 标注放置的 zoom, lat_ref)
REGIONS: dict[str, dict] = {
    "深圳湾": {"bbox": [113.88, 22.46, 114.10, 22.60], "zoom": 12, "lat_ref": 22.53},
    "深圳全市": {"bbox": [113.75, 22.42, 114.65, 22.85], "zoom": 10, "lat_ref": 22.65},
}
COG_TILE_URL = "http://127.0.0.1:8001/tiles/{{z}}/{{x}}/{{y}}.png?path=data/cogs/{cog}"
GRID_N = 40                      # ndvi 网格边长（40×40，教学取舍：中心点采样近似，红旗④）
T2M_PREFIX = "t2m-"              # 前端注入图层/源的 id 前缀（防与既有图层冲突）


def _cog_names() -> list[str]:
    """data/cogs/manifest.json → 白名单文件名列表。"""
    mf = COGS_DIR / "manifest.json"
    if not mf.exists():
        return []
    return [Path(item["path"]).name for item in json.loads(mf.read_text(encoding="utf-8"))
            if "path" in item]


COG_WHITELIST = _cog_names()


# ---------------------------------------------------------------- IR 数据模型
@dataclass
class CartographyIR:
    """制图意图（LLM 的全部输出空间）：每个字段都是受限枚举，不是自由字符串。"""
    theme: str                                   # "poi" | "ndvi" | "cog"
    region: str                                  # REGIONS 键
    rationale: str = ""                          # LLM 一句话决策理由（可审计）
    visible_classes: list[str] = field(default_factory=list)   # ⊆ POI_CLASSES
    label_classes: list[str] = field(default_factory=list)     # ⊆ visible_classes
    color_semantics: dict[str, str] = field(default_factory=dict)  # {class: 语义词∈SEMANTIC}
    color_intent: str = ""                       # ndvi：∈ NDVI_INTENTS
    cog: str = ""                                # cog：∈ COG_WHITELIST
    raster_opacity: float = 0.75                 # cog：[0,1]


DEFAULT_IR = CartographyIR(
    theme="poi", region="深圳全市",
    visible_classes=["subway", "park", "school"], label_classes=["subway"],
    rationale="默认方案：兴趣点分类图 + 地铁标注（LLM 解析失败时回退）",
)


# ---------------------------------------------------------------- 校验三道闸
def validate_ir(ir: CartographyIR) -> list[str]:
    """枚举闸 + 引用闸（parse 闸在 extract_intent 的 JSON 解析处）。返回错误列表，空 = 通过。"""
    errs: list[str] = []
    if ir.theme not in THEMES:
        errs.append(f"theme 非法：{ir.theme!r}（可选 {'/'.join(THEMES)}）")
    if ir.region not in REGIONS:
        errs.append(f"region 非法：{ir.region!r}（可选 {'/'.join(REGIONS)}）")
    if not set(ir.visible_classes) <= set(POI_CLASSES):
        errs.append(f"visible_classes 超集：{ir.visible_classes}（可选 {POI_CLASSES}）")
    if not set(ir.label_classes) <= set(ir.visible_classes):
        errs.append(f"label_classes ⊄ visible_classes：{ir.label_classes}")
    for cls, word in ir.color_semantics.items():
        if cls not in POI_CLASSES:
            errs.append(f"color_semantics 类别非法：{cls!r}")
        if word not in SEMANTIC:
            errs.append(f"color_semantics 语义词未注册：{word!r}（可选 {sorted(SEMANTIC)}）")
    if ir.theme == "ndvi" and ir.color_intent not in NDVI_INTENTS:
        errs.append(f"color_intent 未注册：{ir.color_intent!r}（可选 {list(NDVI_INTENTS)}）")
    if ir.theme == "cog":
        if ir.cog not in COG_WHITELIST:
            errs.append(f"cog 文件名不在白名单：{ir.cog!r}（可选 {COG_WHITELIST}）")
        if not (0.0 <= float(ir.raster_opacity) <= 1.0):
            errs.append(f"raster_opacity 越界：{ir.raster_opacity}（应在 [0,1]）")
    if not ir.rationale.strip():
        errs.append("rationale 为空（决策理由必填——不可审计的样式不生成）")
    return errs


# ---------------------------------------------------------------- LLM 意图抽取
_EXTRACT_PROMPT = """你是制图意图解析器。把用户的自然语言转成制图意图 JSON，只输出 JSON（不要多余文字）。
与制图/地图样式/上图无关的请求 → 只输出 {"error": "not_a_map_request"}。

JSON schema（字段按主题取舍，其余省略）：
{
  "theme": "poi" | "ndvi" | "cog",
  "region": "深圳湾" | "深圳全市",
  "rationale": "一句话决策理由",
  "visible_classes": ["subway","park","school"],      // 仅 poi：想显示哪些兴趣点类别
  "label_classes": ["subway"],                        // 仅 poi：想标注名称的类别（⊆ visible_classes）
  "color_semantics": {"park": "植被"},                // 仅 poi：类别→语义词；语义词只能取：
                                                      //   水/水体/植被/绿植/城市/建成区/新增/变化/消退/持续
  "color_intent": "植被",                             // 仅 ndvi：只能取 植被/风险/密度/热度/NDVI/空串
  "cog": "szbay_real_mosaic.tif",                     // 仅 cog：文件名必须来自下方白名单
  "raster_opacity": 0.75                              // 仅 cog：0~1（"半透明"≈0.5，"微透明"≈0.3）
}

可用 COG 白名单：{cogs}

用户输入：{query}"""


def _llm_json(query: str, feedback: str = "") -> dict:
    """DeepSeek temp=0 + json_object：LLM 只在白名单里选择，不生成任何样式语法。"""
    import os
    from openai import OpenAI

    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from backend.core.config import llm_config

    client = OpenAI(base_url=llm_config.base_url, api_key=os.getenv("DEEPSEEK_API_KEY", ""))
    # 不用 .format()：模板里的 JSON schema 示例含字面花括号，format 会把 {"error": ...}
    # 当占位符解析抛 KeyError '"error"' —— replace 拼接绕开（W4 专属踩坑）。
    prompt = (_EXTRACT_PROMPT
              .replace("{cogs}", "、".join(COG_WHITELIST) or "（无）")
              .replace("{query}", query))
    if feedback:
        prompt += f"\n\n上一次输出未通过校验，错误：{feedback}（请修正后重新输出 JSON）"
    resp = client.chat.completions.create(
        model=llm_config.model, temperature=0,              # 结构化任务 temp=0（项目约定）
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content)


def extract_intent(query: str) -> tuple[CartographyIR | None, str, bool]:
    """NL → CartographyIR。返回 (ir | None, 错误信息, fallback 标记)。

    失败路径（design D3）：单次重试（校验错误回填 prompt）→ 仍失败 → 回退默认 IR
    并标记 fallback=True（诚实暴露降级，不静默）。非制图请求 → (None, not_map, False)。
    """
    feedback = ""
    for attempt in (1, 2):
        try:
            raw = _llm_json(query, feedback)
        except Exception as e:                               # noqa: BLE001 —— 网络/解析错误进重试
            feedback = f"{type(e).__name__}: {e}"
            continue
        if raw.get("error") == "not_a_map_request":
            return None, "not_map", False
        ir = CartographyIR(
            theme=str(raw.get("theme", "")), region=str(raw.get("region", "")),
            rationale=str(raw.get("rationale", "")),
            visible_classes=[str(c) for c in raw.get("visible_classes", [])],
            label_classes=[str(c) for c in raw.get("label_classes", [])],
            color_semantics={str(k): str(v) for k, v in raw.get("color_semantics", {}).items()},
            color_intent=str(raw.get("color_intent", "")),
            cog=str(raw.get("cog", "")),
            raster_opacity=float(raw.get("raster_opacity", 0.75)),
        )
        errs = validate_ir(ir)
        if not errs:
            return ir, "", False
        feedback = "；".join(errs)
    return CartographyIR(**DEFAULT_IR.__dict__), f"解析失败已回退默认方案（{feedback}）", True


# ---------------------------------------------------------------- 数据构建（IO 层）
def _feats_to_point_fc(feats: list[Feature]) -> dict:
    """Feature 列表 → 点 GeoJSON（Polygon/Line 取代表点；circle 层数据）。"""
    out = []
    for f in feats:
        g = f.geom
        x, y = (g.x, g.y) if g.geom_type == "Point" else (g.representative_point().x,
                                                          g.representative_point().y)
        lon, lat = merc_to_lonlat(x, y)
        out.append({"type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                    "properties": {"name": f.name, "class": f.fclass, "priority": f.priority}})
    return {"type": "FeatureCollection", "features": out}


def _merc_to_lonlat_arr(x, y):
    """shapely.ops.transform 用的向量化墨卡托反变换。"""
    lon = np.degrees(np.asarray(x) / 6378137.0)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / 6378137.0)) - np.pi / 2)
    return lon, lat


def _feats_to_poly_fc(feats: list[Feature]) -> dict:
    """Polygon 类 Feature → 面 GeoJSON（fill 层数据）。"""
    from shapely.ops import transform as shp_transform

    out = []
    for f in feats:
        if f.geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        g = shp_transform(_merc_to_lonlat_arr, f.geom)
        out.append({"type": "Feature",
                    "geometry": json.loads(json.dumps(g.__geo_interface__)),
                    "properties": {"name": f.name, "class": f.fclass}})
    return {"type": "FeatureCollection", "features": out}


def raster_to_grid(cog_name: str, n: int = GRID_N) -> tuple[dict, np.ndarray, list[float]]:
    """COG → n×n 网格多边形 + ndvi 值 + fit bbox（design D4/D5）。

    rasterio read(out_shape) 一步 average 重采样到 n×n（比逐点采样快且稳）；
    格多边形在影像 bounds 上均匀切，投影 CRS 时四角 pyproj 转回 WGS84。
    有效掩膜沿用 W2 约定（波段和 > 0.02）；无效格跳过（NaN 不进表达式）。
    """
    import rasterio
    from rasterio.enums import Resampling

    path = COGS_DIR / cog_name
    with rasterio.open(path) as ds:
        b = ds.read(out_shape=(4, n, n), resampling=Resampling.average).astype(np.float32) / 10000.0
        bounds = ds.bounds                                   # (left, bottom, right, top)
        crs = ds.crs
    if crs is not None and not crs.is_geographic:
        from pyproj import Transformer
        tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        corners = [tf.transform(x, y) for x, y in
                   [(bounds.left, bounds.bottom), (bounds.right, bounds.top),
                    (bounds.left, bounds.top), (bounds.right, bounds.bottom)]]
        lngs = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        bbox = [min(lngs), min(lats), max(lngs), max(lats)]
    else:
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]

    ndvi = (b[3] - b[2]) / (b[3] + b[2] + 1e-6)
    valid = b.sum(0) > 0.02
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) / n, (y1 - y0) / n
    feats = []
    for j in range(n):                                       # j: 南→北（ndvi 行 0 = 影像北）
        for i in range(n):
            v = float(ndvi[n - 1 - j, i])
            if not np.isfinite(v) or not valid[n - 1 - j, i]:
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[
                    [x0 + i * dx, y0 + j * dy], [x0 + (i + 1) * dx, y0 + j * dy],
                    [x0 + (i + 1) * dx, y0 + (j + 1) * dy], [x0 + i * dx, y0 + (j + 1) * dy],
                    [x0 + i * dx, y0 + j * dy]]]},
                "properties": {"ndvi": round(v, 4)},
            })
    return {"type": "FeatureCollection", "features": feats}, ndvi[valid], bbox


def _load_feats(cls: str) -> list[Feature]:
    """data/osm/poi_<cls>.json → Feature 列表（W3 load_osm_layer 复用）。"""
    from scripts.auto_cartography import load_osm_layer
    return load_osm_layer(OSM_DIR / POI_FILES[cls], cls)


def build_data_poi(ir: CartographyIR) -> dict:
    feats = [f for cls in ir.visible_classes for f in _load_feats(cls)]
    lat_ref = REGIONS[ir.region]["lat_ref"]
    zoom = REGIONS[ir.region]["zoom"]
    label_feats = [f for f in feats if f.fclass in ir.label_classes]
    placed, dropped = place_labels(label_feats, zoom, lat_ref) if label_feats else ([], [])
    labels_layer = to_mapbox_labels(placed) if placed else None
    pts = [f for f in feats if f.geom.geom_type == "Point"]
    polys = [f for f in feats if f.geom.geom_type in ("Polygon", "MultiPolygon")]
    bbox = REGIONS[ir.region]["bbox"]
    return {
        "theme": "poi", "region": ir.region, "fit_bounds": bbox,
        "points_fc": _feats_to_point_fc(pts if pts else feats),
        "poly_fc": _feats_to_poly_fc(polys) if polys else None,
        "labels_layer": labels_layer,
        "n_placed": len(placed), "n_dropped": len(dropped),
    }


def build_data_ndvi(ir: CartographyIR) -> dict:
    fc, values, bbox = raster_to_grid("szbay_real_mosaic.tif")
    return {"theme": "ndvi", "region": ir.region, "fit_bounds": bbox,
            "ndvi_fc": fc, "ndvi_values": values}


def build_data_cog(ir: CartographyIR) -> dict:
    bbox = next((item["bbox"] for item in json.loads((COGS_DIR / "manifest.json").read_text())
                 if Path(item["path"]).name == ir.cog), REGIONS["深圳湾"]["bbox"])
    return {"theme": "cog", "region": ir.region, "fit_bounds": bbox}


_BUILDERS = {"poi": build_data_poi, "ndvi": build_data_ndvi, "cog": build_data_cog}


# ---------------------------------------------------------------- 组装（纯函数）
def _poi_color_expr(ir: CartographyIR, labels_zh: list[str]) -> tuple[list, dict]:
    """poi 主题的 match 表达式（W2 引擎消费，零改动）。

    W2 profile_data 只认 int 类别值 → values 用 0/1/2、决策后把 class_values
    替换为英文 class 名（消费侧适配，W2 引擎零改动）。返回 (表达式, 语义审计)。
    """
    values = list(range(len(ir.visible_classes)))
    plan = choose_symbology(profile_data(np.array([values]), labels=labels_zh))
    hits: dict[str, str] = {}
    for idx, cls in enumerate(ir.visible_classes):
        word = ir.color_semantics.get(cls)
        if word:
            plan.colors[idx] = SEMANTIC[word]                # W4 语义覆盖层（显式可审计）
            hits[cls] = f"{word}→{SEMANTIC[word]}"
    plan.class_values = list(ir.visible_classes)             # int → 英文（match 键）
    expr = to_mapbox_style(plan, "class")["fill-color"]
    return expr, hits


def assemble_style(ir: CartographyIR, data: dict) -> dict:
    """IR + data → map_style（纯函数：sources/layers/layer_ids/fit_bounds）。

    全部 id 带 t2m- 前缀（前端直接 add/remove，零加工）；layer.source 引用
    sources 里的 id（Mapbox Style Spec v8 要求，inline source 不合法）。
    """
    reg = REGIONS[ir.region]
    style: dict = {"theme": ir.theme, "rationale": ir.rationale,
                   "fit_bounds": data["fit_bounds"], "sources": {}, "layers": [], "layer_ids": []}

    def _reg(obj_id: str) -> str:
        style["layer_ids"].append(obj_id)
        return obj_id

    if ir.theme == "poi":
        labels_zh = {"subway": "地铁", "park": "公园", "school": "学校"}
        expr, hits = _poi_color_expr(ir, [labels_zh[c] for c in ir.visible_classes])
        if data.get("poly_fc") and data["poly_fc"]["features"]:
            sid = _reg(f"{T2M_PREFIX}poi-poly-src")
            style["sources"][sid] = {"type": "geojson", "data": data["poly_fc"]}
            style["layers"].append({
                "id": _reg(f"{T2M_PREFIX}poi-poly-fill"), "type": "fill", "source": sid,
                "paint": {"fill-color": expr, "fill-opacity": 0.35}})
        sid = _reg(f"{T2M_PREFIX}poi-pts-src")
        style["sources"][sid] = {"type": "geojson", "data": data["points_fc"]}
        style["layers"].append({
            "id": _reg(f"{T2M_PREFIX}poi-points"), "type": "circle", "source": sid,
            "paint": {"circle-color": expr,                   # ← 与 fill 复用同一表达式对象
                      "circle-radius": 5,
                      "circle-stroke-color": "#ffffff", "circle-stroke-width": 1}})
        if data.get("labels_layer"):
            lab = data["labels_layer"]
            sid = _reg(f"{T2M_PREFIX}poi-labels-src")
            style["sources"][sid] = lab["source"]
            style["layers"].append({
                "id": _reg(f"{T2M_PREFIX}poi-labels"), "type": "symbol", "source": sid,
                "layout": lab["layout"],
                "paint": {"text-color": "#222222", "text-halo-color": "#ffffff",
                          "text-halo-width": 1.2}})
        style["semantic_hits"] = hits
        style["label_stats"] = {"placed": data.get("n_placed", 0), "dropped": data.get("n_dropped", 0)}

    elif ir.theme == "ndvi":
        # 制图语义 = 单极植被活性 → force sequential（W2 实现裁决的延伸，水面负值是物理性质）
        plan = choose_symbology(profile_data(np.asarray(data["ndvi_values"], dtype=float),
                                             labels=["NDVI", ir.color_intent] if ir.color_intent
                                             else ["NDVI"], force_kind="sequential"))
        fill_expr = to_mapbox_style(plan, "ndvi")["fill-color"]
        sid = _reg(f"{T2M_PREFIX}ndvi-src")
        style["sources"][sid] = {"type": "geojson", "data": data["ndvi_fc"]}
        style["layers"].append({
            "id": _reg(f"{T2M_PREFIX}ndvi-fill"), "type": "fill", "source": sid,
            "paint": {"fill-color": fill_expr, "fill-opacity": 0.8}})
        style["palette"] = {"name": plan.palette_name, "reason": plan.reason}

    elif ir.theme == "cog":
        sid = _reg(f"{T2M_PREFIX}cog-src")
        style["sources"][sid] = {"type": "raster",
                                 "tiles": [COG_TILE_URL.format(cog=ir.cog)],
                                 "tileSize": 256, "minzoom": 0, "maxzoom": 16}
        style["layers"].append({
            "id": _reg(f"{T2M_PREFIX}cog-layer"), "type": "raster", "source": sid,
            "paint": {"raster-opacity": float(ir.raster_opacity)}})

    return style


# ---------------------------------------------------------------- 顶层入口（SSE 工具消费）
def text_to_map(query: str) -> dict:
    """自然语言 → {summary, fallback, rationale, map_style}。

    summary 字段前置（W2 教训：result 事件 [:200] 摘要只看到 JSON 头部）；
    map_style 尾置全量，main.py 检测到后发 SSE style 事件。
    """
    ir, err, fallback = extract_intent(query)
    if ir is None:
        return {"summary": f"[工具错误] 该请求不是制图需求（{err}）。可支持的制图主题："
                           f"poi（兴趣点分类图）/ ndvi（植被指数网格图）/ cog（卫星影像叠加）",
                "fallback": False, "map_style": None}
    try:
        data = _BUILDERS[ir.theme](ir)
        style = assemble_style(ir, data)
    except Exception as e:                                   # noqa: BLE001 —— 数据层错误转文本不炸 ReAct
        return {"summary": f"[工具错误] {type(e).__name__}: {e}（theme={ir.theme}）",
                "fallback": fallback, "map_style": None}
    tag = "（解析失败，已回退默认方案）" if fallback else ""
    return {"summary": f"已生成 {ir.theme} 专题样式：{ir.rationale}{tag}",
            "fallback": fallback, "rationale": ir.rationale, "map_style": style}


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    """组装器全测（LLM 不在场）：校验闸拒绝 + 三主题骨架 + 表达式复用。零数据依赖。"""
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # --- 校验闸（design D3）---
    e1 = validate_ir(CartographyIR(theme="heatmap", region="深圳湾", rationale="x"))
    check("拒非法 theme", any("theme" in e for e in e1), str(e1[:1]))
    e2 = validate_ir(CartographyIR(theme="poi", region="深圳湾", rationale="x",
                                   visible_classes=["subway"], label_classes=["park"]))
    check("拒 label ⊄ visible", any("label_classes" in e for e in e2), str(e2[:1]))
    e3 = validate_ir(CartographyIR(theme="cog", region="深圳湾", rationale="x",
                                   cog="hacker.tif", raster_opacity=1.5))
    check("拒越界 opacity + 白名单外 cog", len(e3) >= 2, str(e3))
    e4 = validate_ir(CartographyIR(theme="poi", region="深圳湾", rationale="x",
                                   visible_classes=["park"], color_semantics={"park": "粉色"}))
    check("拒未注册语义词", any("语义词" in e for e in e4), str(e4[:1]))
    e5 = validate_ir(CartographyIR(theme="ndvi", region="深圳湾", rationale="x", color_intent="粉色"))
    check("拒未注册 color_intent", any("color_intent" in e for e in e5), str(e5[:1]))

    # --- poi assemble（合成数据注入）---
    ir_p = CartographyIR(theme="poi", region="深圳全市",
                         visible_classes=["subway", "park", "school"],
                         label_classes=["subway"], color_semantics={"park": "植被"},
                         rationale="测试")
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [114.0, 22.5]},
         "properties": {"name": "测试站", "class": "subway", "priority": 0}}]}
    data_p = {"fit_bounds": REGIONS["深圳全市"]["bbox"], "points_fc": fc,
              "poly_fc": None, "labels_layer": None, "n_placed": 0, "n_dropped": 0}
    st_p = assemble_style(ir_p, data_p)
    ids = st_p["layer_ids"]
    check("poi 全部 id 带 t2m- 前缀", all(i.startswith(T2M_PREFIX) for i in ids), str(ids))
    check("layer.source 是 id 引用而非 inline", all(isinstance(l["source"], str)
                                                   and l["source"] in st_p["sources"]
                                                   for l in st_p["layers"]), "spec v8 合法")
    pt_layer = next(l for l in st_p["layers"] if l["type"] == "circle")
    check("circle-color 是 match 表达式（W2 复用）",
          pt_layer["paint"]["circle-color"][0] == "match"
          and pt_layer["paint"]["circle-color"][2] == "subway"
          and str(pt_layer["paint"]["circle-color"][3]).startswith(("#", "rgba")), "表达式键=class 名")
    park_idx = ir_p.visible_classes.index("park") + 3
    check("语义覆盖：park → 植被绿（G 通道最大）",
          int(st_p["semantic_hits"]["park"].split("→")[1][3:5], 16)
          > int(st_p["semantic_hits"]["park"].split("→")[1][1:3], 16), str(st_p["semantic_hits"]))

    # --- ndvi assemble（合成网格值）---
    ir_n = CartographyIR(theme="ndvi", region="深圳湾", color_intent="植被", rationale="测试")
    vals = np.linspace(0.05, 0.65, 50)
    grid_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[113.9, 22.5],
            [113.91, 22.5], [113.91, 22.51], [113.9, 22.51], [113.9, 22.5]]]},
         "properties": {"ndvi": 0.3}}]}
    st_n = assemble_style(ir_n, {"fit_bounds": REGIONS["深圳湾"]["bbox"],
                                 "ndvi_fc": grid_fc, "ndvi_values": vals})
    fill_layer = st_n["layers"][0]
    check("ndvi → interpolate 表达式（W2 复用）",
          fill_layer["paint"]["fill-color"][0] == "interpolate", st_n["palette"]["name"])
    breaks = fill_layer["paint"]["fill-color"][3::2]
    check("interpolate 断点升序", breaks == sorted(breaks), f"breaks={[round(b,3) for b in breaks]}")
    check("color_intent=植被 → Greens palette", st_n["palette"]["name"] == "Greens",
          st_n["palette"]["name"])

    # --- cog assemble ---
    ir_c = CartographyIR(theme="cog", region="深圳湾", cog="szbay_real_mosaic.tif",
                         raster_opacity=0.5, rationale="测试")
    st_c = assemble_style(ir_c, {"fit_bounds": REGIONS["深圳湾"]["bbox"]})
    rlayer = st_c["layers"][0]
    check("cog → raster-opacity 生效 + tiles URL 含 path=",
          rlayer["paint"]["raster-opacity"] == 0.5
          and "path=data/cogs/szbay_real_mosaic.tif" in st_c["sources"][rlayer["source"]]["tiles"][0],
          rlayer["id"])

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        con_print = f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
        print(con_print)
    print(f"=== 汇总: {n_pass}/{len(results)} PASS ===")
    return 0 if n_pass == len(results) else 1


# ---------------------------------------------------------------- CLI demo（真实 DeepSeek）
DEMO_QUERIES = [
    ("poi", "给深圳全市做一张兴趣点图，公园按植被绿色显示，并标注地铁站"),
    ("ndvi", "深圳湾的 NDVI 植被指数专题图，植被茂密的地方用绿色"),
    ("cog", "把深圳湾的卫星影像 szbay_real_mosaic 叠加到地图上，半透明"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plans = []
    for expect, query in DEMO_QUERIES:
        print(f"\n[问] {query}")
        r = text_to_map(query)
        print(f"[IR] fallback={r['fallback']}  {r['summary']}")
        if r["map_style"] is None:
            print("[跳过] 无 map_style")
            continue
        st = r["map_style"]
        ok_theme = st["theme"] == expect
        print(f"[样式] theme={st['theme']}（预期 {expect}{' ✓' if ok_theme else ' ✗'}）  "
              f"sources={len(st['sources'])}  layers={[l['id'] for l in st['layers']]}")
        out = OUT_DIR / f"demo_{st['theme']}.json"
        out.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[落盘] {out}（{out.stat().st_size:,} B）")
        plans.append(st)

    # 断言三份主题互异（验收②）
    themes = [s["theme"] for s in plans]
    assert len(set(themes)) == 3 and set(themes) == set(THEMES), f"demo 主题不互异：{themes}"
    print(f"\n三份 Style JSON 主题互异 ✓（{themes}）——NL 变 → 样式变，Text-to-Map 生效")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
