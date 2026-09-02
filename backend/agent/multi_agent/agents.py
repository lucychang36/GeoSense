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
- task_type: "temporal_change" | "segment" | "qa"
  - temporal_change: 时相对比/变化检测
  - segment: 单景分割
  - qa: 通用问答（不调工具直接答）
- cog_a, cog_b: 较早/较晚 COG 文件名（时相对比必填，单景只填 cog_a）
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

    # 规则回退：问题含两个年份 → temporal_change
    if not plan:
        years = re.findall(r"(?:20)?\d{2}", query)
        years = [("20" + y if len(y) == 2 else y) for y in years if 18 <= (int(y) if len(y) == 2 else int(y) % 100) <= 35]
        if len(set(years)) >= 2:
            plan = {"task_type": "temporal_change", "cog_a": "", "cog_b": "",
                    "reason": f"规则回退：问题含年份 {years}"}

    log = list(state.get("step_log", []))
    log.append(f"planner → task_type={plan.get('task_type', '?')}, reason={plan.get('reason', '?')[:60]}")
    return {"plan": plan, "planner_raw": raw[:300],
            "current_step": "data", "step_log": log}


# ===========================================================================
# Data Agent：匹配 COG
# ===========================================================================
def data_node(state: dict) -> dict:
    """从 data/cogs/ 选 cog_a / cog_b。优先用 planner 给的文件名；缺则按 49QGE 日期 COG
    的最早/最晚兜底（要求文件名含 8 位连续数字，剔除合成图如 *_mosaic.tif）。"""
    cogs_dir = PROJECT_ROOT / "data" / "cogs"
    available = sorted([p.name for p in cogs_dir.glob("*.tif")]) if cogs_dir.exists() else []
    plan = state.get("plan", {})
    cog_a, cog_b = (plan.get("cog_a") or ""), (plan.get("cog_b") or "")

    def _date8(name: str) -> int | None:
        m = re.search(r"(\d{8})", name)
        return int(m.group(1)) if m else None

    # 兜底：找含 8 位日期的 szbay_real COG（剔除 mosaic/合成图）
    real_dated = [(n, _date8(n)) for n in available if "szbay_real" in n]
    real_dated = [(n, d) for n, d in real_dated if d is not None]
    if ((not cog_a or cog_a not in available) or (not cog_b or cog_b not in available)) and len(real_dated) >= 2:
        real_dated.sort(key=lambda x: x[1])
        cog_a, cog_b = real_dated[0][0], real_dated[-1][0]

    log = list(state.get("step_log", []))
    log.append(f"data → cog_a={cog_a}, cog_b={cog_b}（候选 {len(available)} 个，dated {len(real_dated)}）")
    return {"cog_a": cog_a, "cog_b": cog_b, "cogs_listed": available,
            "current_step": "analysis", "step_log": log}


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
    import matplotlib.pyplot as plt
    from scripts.unet_segmentation import rgb_preview

    try:
        import rasterio
        with rasterio.open(PROJECT_ROOT / "data" / "cogs" / cog_a) as ds:
            bands_a = ds.read().astype(np.float32) / 10000.0
        with rasterio.open(PROJECT_ROOT / "data" / "cogs" / cog_b) as ds:
            bands_b = ds.read().astype(np.float32) / 10000.0
        # 复用 analysis 算的 change mask（重新跑一次轻量，保证本节点自包含）
        from scripts.real_change_detection import spectral_diff_change
        change = spectral_diff_change(bands_a, bands_b, threshold=res.get("threshold", 0.08))

        out_dir = PROJECT_ROOT / "data" / "output" / "multi_agent_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_png = out_dir / f"temporal_change__{cog_a.replace('.tif', '')}__{cog_b.replace('.tif', '')}__{ts}.png"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_preview(bands_a)); axes[0].set_title(f"{cog_a} 真彩"); axes[0].axis("off")
        axes[1].imshow(rgb_preview(bands_b)); axes[1].set_title(f"{cog_b} 真彩"); axes[1].axis("off")
        # 变化叠加：底=cog_b 真彩，红=变化
        bg = rgb_preview(bands_b) * 0.6
        bg[change.astype(bool)] = np.array([0.95, 0.15, 0.15])
        axes[2].imshow(bg)
        axes[2].set_title(f"变化 mask（红=差异 {res['change_ratio']:.2%}，方法: {res['method']}）")
        axes[2].axis("off")
        fig.suptitle("GeoSense Multi-Agent 时相对比", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_png, dpi=100)
        plt.close(fig)

        log = list(state.get("step_log", []))
        log.append(f"cartography → map saved: {out_png.name}（{out_png.stat().st_size:,} B）")
        return {"map_path": str(out_png), "map_bytes": out_png.stat().st_size,
                "current_step": "supervisor", "step_log": log}
    except Exception as exc:  # noqa: BLE001
        return {"analysis_error": f"cartography 失败: {type(exc).__name__}: {exc}",
                "current_step": "supervisor"}


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
    elif res.get("change_ratio") is not None:
        answer = (
            f"✅ 时相对比完成（{state.get('cog_a', '?')} vs {state.get('cog_b', '?')}）\n"
            f"• 方法：{res.get('method', '?')}（阈值 {res.get('threshold', '?')}）\n"
            f"• 变化占比：**{res['change_ratio']:.2%}**（{res.get('change_px', 0):,} / {res.get('valid_px', 0):,} 像素）\n"
            f"• 专题图：{map_path or '未生成'}\n"
            f"\n执行链路：\n" + "\n".join(f"  • {x}" for x in log)
        )
    else:
        answer = f"任务已完成（task_type={plan.get('task_type', '?')}）。\n\n步骤：\n" + "\n".join(f"  • {x}" for x in log)

    return {"final_answer": answer, "current_step": "done"}
