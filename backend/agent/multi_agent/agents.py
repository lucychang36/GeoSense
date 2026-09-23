"""第10月 W1：4 个 Agent + Supervisor 节点实现。

W1 范围：4 Agent 各司其职 + 共享 State。架构层面是"分层协作"，路由用线性顺序边
（planner→data→analysis→cartography→supervisor）。路由决策的"集中性"通过
supervisor 节点在最后汇总体现 —— W2 起会引入条件边（dynamic routing）。

W1 简化（明确标注，不算 hack）：
- Planner 用 LLM 拆解（结构化 JSON）；其它 3 个 agent 是 deterministic 纯函数
- Analysis Agent 只跑 spectral_diff_change（最轻量），不调 U-Net（避免 W1 演示过重）
- Cartography Agent 出 3 面板 matplotlib PNG
- Supervisor 节点是"集中汇报"，不做复杂路由

为什么这样切分：演示"多 Agent 协作"的工程价值（分工、状态共享、串行编排），
不陷入每个 worker 内部的复杂度。第11月再升级到更智能的 routing。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # backend/agent/multi_agent → 上 3 级
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# Planner：LLM 拆解
# ===========================================================================
PLANNER_SYSTEM = """你是 GeoSense 多 Agent 系统的 Planner。把用户问题拆解为结构化 JSON 计划。

字段：
- task_type: "temporal_change" | "index_change" | "segment" | "qa"
  - temporal_change: 时相对比/光谱变化检测（泛化"变化"）
  - index_change: 土地覆盖专题变化（建筑用地/绿化用地/水域等），必须给 theme
  - segment: 单景分割
  - qa: 通用问答（不调工具直接答）
- theme: "builtup"（建筑用地）| "green"（绿化用地）| "water"（水域）——仅 index_change 需要
- cog_a, cog_b: 较早/较晚 COG 文件名（时相对比必填；不确定可留空，Data Agent 兜底）
- reason: 一句话判断理由

只输出 JSON，不要任何其他文字、解释、markdown 包裹。
"""


def _make_llm():
    """复用第3月 graph.py 的 LLM 工厂（DeepSeek，结构化场景 temperature=0）。"""
    from langchain_openai import ChatOpenAI
    from backend.core.config import llm_config
    return ChatOpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model=llm_config.model,
        temperature=0,
    )


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出里抠 JSON —— 处理 ```json ... ``` 包裹 / 多余文字等。"""
    # 优先匹配 ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # 退化：找第一个 { 到最后一个 }
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s:e + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def planner_node(state: dict) -> dict:
    """调 LLM 拆解用户问题为结构化计划。失败时回退到规则匹配（保证 demo 可跑）。"""
    query = state.get("user_query", "")
    raw = ""
    plan: dict = {}
    try:
        llm = _make_llm()
        from langchain_core.messages import SystemMessage, HumanMessage
        resp = llm.invoke([SystemMessage(content=PLANNER_SYSTEM),
                           HumanMessage(content=query)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        plan = _extract_json(raw) or {}
    except Exception as exc:  # noqa: BLE001 —— LLM 失败要回退，不能让图卡住
        raw = f"[planner LLM 失败: {type(exc).__name__}: {exc}]"
        plan = {}

    # 规则回退：问题含两个年份 → temporal_change；含专题关键词 → index_change
    if not plan:
        years = re.findall(r"(?:20)?\d{2}", query)
        years = [("20" + y if len(y) == 2 else y) for y in years if 18 <= (int(y) if len(y) == 2 else int(y) % 100) <= 35]
        if len(set(years)) >= 2:
            theme_kw = {"建筑": "builtup", "绿化": "green", "绿地": "green",
                        "植被": "green", "水域": "water", "水体": "water"}
            theme = next((v for k, v in theme_kw.items() if k in query), None)
            if theme:
                plan = {"task_type": "index_change", "theme": theme, "cog_a": "", "cog_b": "",
                        "reason": f"规则回退：年份 {years} + 专题关键词 → {theme}"}
            else:
                plan = {"task_type": "temporal_change", "cog_a": "", "cog_b": "",
                        "reason": f"规则回退：问题含年份 {years}"}

    log = list(state.get("step_log", []))
    log.append(f"planner → task_type={plan.get('task_type', '?')}, reason={plan.get('reason', '?')[:60]}")
    return {"plan": plan, "planner_raw": raw[:300],
            "current_step": "data", "step_log": log}


# ===========================================================================
# Data Agent：匹配 COG
# ===========================================================================
# 区域关键词 → 文件前缀（thematic-index-change：多区域共存时按用户问题选对数据源）
REGION_KEYWORDS = {"郑州": "zhengzhou", "高新区": "zhengzhou", "深圳湾": "szbay", "深圳": "szbay"}
REGION_LABELS = {"zhengzhou": "郑州高新区", "szbay": "深圳湾"}


def data_node(state: dict) -> dict:
    """从 data/cogs/ 选 cog_a / cog_b。优先用 planner 给的文件名；缺则按区域前缀的
    真实日期 COG 最早/最晚兜底（要求文件名含 8 位连续数字，剔除合成图如 *_mosaic.tif）。
    区域由用户问题关键词或 planner 给的文件名推断；无关键词时回落深圳湾（历史行为）。"""
    cogs_dir = PROJECT_ROOT / "data" / "cogs"
    available = sorted([p.name for p in cogs_dir.glob("*.tif")]) if cogs_dir.exists() else []
    plan = state.get("plan", {})
    query = state.get("user_query", "")
    cog_a, cog_b = (plan.get("cog_a") or ""), (plan.get("cog_b") or "")

    def _date8(name: str) -> int | None:
        m = re.search(r"(\d{8})", name)
        return int(m.group(1)) if m else None

    # 区域推断：问题关键词优先，其次 plan 里 COG 文件名前缀（枚举优先于自由字符串）
    prefix = next((p for kw, p in REGION_KEYWORDS.items() if kw in query), None)
    if prefix is None and cog_a:
        prefix = next((p for p in REGION_LABELS if cog_a.startswith(p + "_")), None)
    region_label = REGION_LABELS.get(prefix or "", "")

    # 兜底：找含 8 位日期的 {prefix}_real COG（剔除 mosaic/合成图）；无前缀回落 szbay
    scan = f"{prefix}_real" if prefix else "szbay_real"
    real_dated = [(n, _date8(n)) for n in available if scan in n]
    real_dated = [(n, d) for n, d in real_dated if d is not None]
    if ((not cog_a or cog_a not in available) or (not cog_b or cog_b not in available)) and len(real_dated) >= 2:
        real_dated.sort(key=lambda x: x[1])
        cog_a, cog_b = real_dated[0][0], real_dated[-1][0]

    log = list(state.get("step_log", []))
    log.append(f"data → cog_a={cog_a}, cog_b={cog_b}（候选 {len(available)} 个，dated {len(real_dated)}，区域={region_label or '默认深圳湾'}）")
    out = {"cog_a": cog_a, "cog_b": cog_b, "cogs_listed": available,
           "current_step": "analysis", "step_log": log}
    if region_label:
        out["region_label"] = region_label
    return out


# ===========================================================================
# Analysis Agent：跑变化检测（复用 W8）
# ===========================================================================
def analysis_node(state: dict) -> dict:
    """调 real_change_detection 的 spectral_diff_change 算变化占比。
    W1 范围克制：只跑最轻量方法（光谐差分），不调 U-Net（避免 W1 demo 几十秒）。
    """
    cog_a = state.get("cog_a", "")
    cog_b = state.get("cog_b", "")
    if not cog_a or not cog_b:
        return {"analysis_error": "缺少 COG 文件名", "current_step": "cartography"}
    if cog_a == cog_b:
        return {"analysis_error": f"cog_a == cog_b ({cog_a})，时相对比需要两景不同", "current_step": "cartography"}

    # 专题分支（thematic-index-change）：index_change → THEMES 光谱指数判定
    plan = state.get("plan", {})
    if plan.get("task_type") == "index_change":
        theme = plan.get("theme") or "builtup"
        try:
            from scripts.thematic_change import THEMES as _THEMES
            from scripts.thematic_change import index_change as _index_change
            if theme not in _THEMES:
                return {"analysis_error": f"未知专题 {theme!r}（可选 {sorted(_THEMES)}）",
                        "current_step": "cartography"}
            cogs_dir = PROJECT_ROOT / "data" / "cogs"
            res = _index_change(cogs_dir / cog_a, cogs_dir / cog_b, theme)
            if state.get("region_label"):
                res["region_label"] = state["region_label"]
            log = list(state.get("step_log", []))
            log.append(f"analysis → index_change[{theme}] net={res['net_km2']:+} km²"
                       f"（gain {res['gain_px']:,} px / loss {res['loss_px']:,} px，valid {res['valid_px']:,} px）")
            return {"analysis_result": res, "current_step": "cartography", "step_log": log}
        except Exception as exc:  # noqa: BLE001
            return {"analysis_error": f"{type(exc).__name__}: {exc}", "current_step": "cartography"}

    try:
        from scripts.real_change_detection import (
            load_pair, valid_mask, cloud_mask, spectral_diff_change,
        )
        from pathlib import Path
        # 临时替换 PAIR 以匹配 data_node 选定的两景
        import scripts.real_change_detection as rcd
        rcd.PAIR = [(cog_a, rcd.COGS_DIR / cog_a),
                    (cog_b, rcd.COGS_DIR / cog_b)]
        bands_a, bands_b = load_pair()
        valid = valid_mask(bands_a) & valid_mask(bands_b)
        cloud = cloud_mask(bands_a) | cloud_mask(bands_b)
        mask_valid = valid & ~cloud
        change = spectral_diff_change(bands_a, bands_b, threshold=0.08)
        change_px = int(change[mask_valid].sum())
        valid_px = int(mask_valid.sum())
        result = {
            "method": "spectral_diff",
            "threshold": 0.08,
            "change_px": change_px,
            "valid_px": valid_px,
            "change_ratio": round(change_px / max(valid_px, 1), 4),
            "valid_pct": round(valid_px / change.size, 4),
        }
        log = list(state.get("step_log", []))
        log.append(f"analysis → spectral_diff change_ratio={result['change_ratio']:.2%}（valid {valid_px:,} px）")
        return {"analysis_result": result, "current_step": "cartography", "step_log": log}
    except Exception as exc:  # noqa: BLE001
        return {"analysis_error": f"{type(exc).__name__}: {exc}", "current_step": "cartography"}


# ===========================================================================
# Cartography Agent：matplotlib 出图
# ===========================================================================
def cartography_node(state: dict) -> dict:
    """3 面板 PNG：cog_a 真彩 | cog_b 真彩 | 变化 mask 叠加。"""
    cog_a = state.get("cog_a", "")
    cog_b = state.get("cog_b", "")
    res = state.get("analysis_result") or {}
    if not cog_a or not cog_b or "change_ratio" not in res:
        return {"current_step": "supervisor"}

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from scripts.unet_segmentation import rgb_preview
    # W2：变化区颜色由符号化引擎按数据性质推导（bool mask → binary → 语义"变化"=红），
    # 替换 W1 硬编码 [0.95, 0.15, 0.15]；reason 进 step_log 可审计。
    from scripts.auto_symbology import choose_symbology, profile_data

    try:
        import rasterio
        with rasterio.open(PROJECT_ROOT / "data" / "cogs" / cog_a) as ds:
            bands_a = ds.read().astype(np.float32) / 10000.0
        with rasterio.open(PROJECT_ROOT / "data" / "cogs" / cog_b) as ds:
            bands_b = ds.read().astype(np.float32) / 10000.0
        # 复用 analysis 算的 change mask（重新跑一次轻量，保证本节点自包含）
        overlay_out = None
        if res.get("method") == "index_change":
            # 专题分支（map-result-linkage D5）：与 analysis 同一 _diff_masks 通路
            # （min_patch_px 过滤后 mask）→ 统计图/报告主数字/地图叠加三者同源
            from scripts.thematic_change import _diff_masks, build_overlay
            theme = res.get("theme", "builtup")
            d = _diff_masks(PROJECT_ROOT / "data" / "cogs" / cog_a,
                            PROJECT_ROOT / "data" / "cogs" / cog_b, theme)
            change = (d["gain_f"] | d["loss_f"]).astype(np.uint8)
            # overlay GeoJSON 落盘（~1MB 不走 SSE/state，D4）；失败不阻塞主流程
            try:
                overlays_dir = PROJECT_ROOT / "data" / "output" / "overlays"
                overlays_dir.mkdir(parents=True, exist_ok=True)
                ov_path = overlays_dir / f"overlay_{time.strftime('%Y%m%d_%H%M%S')}_{theme}.geojson"
                fc = build_overlay(d["gain_f"], d["loss_f"], d["transform"],
                                   d["px_km2"] * 1e6, theme, d["period"])
                ov_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
                tl = res.get("theme_label", theme)
                overlay_out = {
                    "path": str(ov_path),
                    "count": len(fc["features"]),
                    "legend": [
                        {"key": "gain", "color": "#E6323C", "label": f"新增{tl}（疑似）"},
                        {"key": "loss", "color": "#3C78EB", "label": f"消失{tl}（疑似）"},
                    ],
                    "bbox": res.get("bbox"),
                    # 底图联动：直接用被分析的期 B 影像（就是该区域，无需反查 preset）
                    "basemap": {"tile_path": f"data/cogs/{cog_b}", "bounds": res.get("bbox")},
                }
            except Exception as ov_exc:  # noqa: BLE001
                overlay_out = {"error": f"{type(ov_exc).__name__}: {ov_exc}"}
        else:
            from scripts.real_change_detection import spectral_diff_change
            change = spectral_diff_change(bands_a, bands_b, threshold=res.get("threshold", 0.08))
        sym = choose_symbology(profile_data(change.astype(np.uint8), labels=["不变", "变化"]))

        out_dir = PROJECT_ROOT / "data" / "output" / "multi_agent_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_png = out_dir / f"temporal_change__{cog_a.replace('.tif', '')}__{cog_b.replace('.tif', '')}__{ts}.png"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_preview(bands_a)); axes[0].set_title(f"{cog_a} 真彩"); axes[0].axis("off")
        axes[1].imshow(rgb_preview(bands_b)); axes[1].set_title(f"{cog_b} 真彩"); axes[1].axis("off")
        # 变化叠加：底=cog_b 真彩，变化色由符号化引擎推导（binary → 语义"变化"=红）
        bg = rgb_preview(bands_b) * 0.6
        bg[change.astype(bool)] = np.array(matplotlib.colors.to_rgb(sym.colors[-1]))
        axes[2].imshow(bg)
        axes[2].set_title(f"变化 mask（红=差异 {res['change_ratio']:.2%}，配色: {sym.palette_name}）")
        axes[2].axis("off")
        theme_extra = f"｜专题：{res['theme_label']}" if res.get("theme_label") else ""
        fig.suptitle(f"GeoSense Multi-Agent 时相对比{theme_extra}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_png, dpi=100)
        plt.close(fig)

        log = list(state.get("step_log", []))
        log.append(f"cartography → map saved: {out_png.name}（{out_png.stat().st_size:,} B）")
        log.append(f"cartography → symbology: {sym.palette_name}（{sym.reason}）")
        if overlay_out and overlay_out.get("count") is not None:
            log.append(f"cartography → overlay: {overlay_out['count']} 图斑 → "
                       f"{Path(overlay_out['path']).name}（与主数字同源）")
        out = {"map_path": str(out_png), "map_bytes": out_png.stat().st_size,
               "current_step": "supervisor", "step_log": log}
        if overlay_out is not None:
            out["overlay_meta"] = overlay_out
        return out
    except Exception as exc:  # noqa: BLE001
        return {"analysis_error": f"cartography 失败: {type(exc).__name__}: {exc}",
                "current_step": "supervisor"}


# ===========================================================================
# Report Agent（第11月 W1：管道第 5 个 worker）
# ===========================================================================
def report_worker(state: dict) -> dict:
    """消费 analysis_result + map_path 生成分析报告。

    边界（design D4）：analysis_error 存在时短路——不生成空报告
    （回收第9月 safe_cog_path「缺字段不得溜进队列」的边界教训）。
    引擎内部失败也不中断管道：记 step_log，supervisor 照常汇总。
    """
    log = list(state.get("step_log", []))
    if state.get("analysis_error"):
        log.append("report → 短路（analysis_error 存在，不生成空报告）")
        return {"current_step": "supervisor", "step_log": log}
    try:
        from scripts.report_engine import generate_report
        out = generate_report(state)
        log.append(f"report → {Path(out['md_path']).name}（narrative_skipped={out['narrative_skipped']}）")
        return {"report_path": out["md_path"], "report_title": out["title"],
                "narrative_skipped": out["narrative_skipped"],
                "current_step": "supervisor", "step_log": log}
    except Exception as exc:  # noqa: BLE001 —— 报告失败不炸管道
        log.append(f"report → 生成失败: {type(exc).__name__}: {exc}")
        return {"current_step": "supervisor", "step_log": log}


# ===========================================================================
# Supervisor：集中汇总
# ===========================================================================
def supervisor_node(state: dict) -> dict:
    """把 4 个 agent 的产出汇成对用户的最终回复。W1 是简单模板，W2 起可让 LLM 润色。"""
    plan = state.get("plan", {})
    res = state.get("analysis_result") or {}
    map_path = state.get("map_path", "")
    err = state.get("analysis_error")
    log = list(state.get("step_log", []))

    if err:
        answer = f"❌ 分析失败：{err}\n\n执行步骤：\n" + "\n".join(f"  • {x}" for x in log)
    elif res.get("method") == "index_change" and res.get("net_km2") is not None:
        report_line = ""
        if state.get("report_path"):
            report_line = (f"\n• 分析报告：**{state.get('report_title', '已生成')}**\n"
                           f"  - Markdown：{state['report_path']}"
                           f"{'（⚠ LLM 叙事降级，仅数据部分）' if state.get('narrative_skipped') else ''}\n")
        answer = (
            f"✅ 专题变化分析完成：{res.get('theme_label', '专题')}（{state.get('cog_a', '?')} vs {state.get('cog_b', '?')}）\n"
            f"• 方法：光谱指数专题判定（主指数阈值 {res.get('threshold')}，边界：结果为“疑似{res.get('theme_label', '')}”）\n"
            f"• 新增 {res.get('gain_km2')} km²（{res.get('gain_px', 0):,} px）｜消失 {res.get('loss_km2')} km²（{res.get('loss_px', 0):,} px）\n"
            f"• 净变化：**{res.get('net_km2'):+} km²**（变化占比 {res.get('change_ratio'):.2%}）\n"
            f"• 专题图：{map_path or '未生成'}\n"
            + (f"• 地图叠加：{state['overlay_meta']['count']} 个图斑已上图"
               f"（红=新增 / 蓝=消失，点击可查面积）\n" if (state.get("overlay_meta") or {}).get("count") else "")
            + f"{report_line}"
            f"\n执行链路：\n" + "\n".join(f"  • {x}" for x in log)
        )
    elif res.get("change_ratio") is not None:
        report_line = ""
        if state.get("report_path"):
            report_line = (f"\n• 分析报告：**{state.get('report_title', '已生成')}**\n"
                           f"  - Markdown：{state['report_path']}"
                           f"{'（⚠ LLM 叙事降级，仅数据部分）' if state.get('narrative_skipped') else ''}\n")
        answer = (
            f"✅ 时相对比完成（{state.get('cog_a', '?')} vs {state.get('cog_b', '?')}）\n"
            f"• 方法：{res.get('method', '?')}（阈值 {res.get('threshold', '?')}）\n"
            f"• 变化占比：**{res['change_ratio']:.2%}**（{res.get('change_px', 0):,} / {res.get('valid_px', 0):,} 像素）\n"
            f"• 专题图：{map_path or '未生成'}\n"
            f"{report_line}"
            f"\n执行链路：\n" + "\n".join(f"  • {x}" for x in log)
        )
    else:
        answer = f"任务已完成（task_type={plan.get('task_type', '?')}）。\n\n步骤：\n" + "\n".join(f"  • {x}" for x in log)

    return {"final_answer": answer, "current_step": "done"}
