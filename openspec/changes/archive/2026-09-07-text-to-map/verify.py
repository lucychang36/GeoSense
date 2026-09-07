"""text_to_map 验证脚本（OpenSpec verify 阶段，可重复执行）。

对应变更：openspec/changes/2026-09-07-text-to-map/
验收来源：proposal.md「验收标准」5 条（详见 tasks.md 证据区）。

用法（在项目根目录，用 .venv）：
    .venv/bin/python openspec/changes/2026-09-07-text-to-map/verify.py
        → 直调 5 条（验收 1/2/4/5 + 引擎自检复跑），无需起服务、不调 LLM，~5s
    .venv/bin/python openspec/changes/2026-09-07-text-to-map/verify.py --integration
        → 追加验收 3：要求 8000 端口服务已在跑（uvicorn backend.api.main:app），
          通过 /api/chat SSE 让 LLM 自主触发 text_to_map_tool，断言 style 事件（~30-60s）

退出码：0 = 全过；1 = 有 FAIL；2 = --integration 但服务不在跑。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
# 向上找含 backend/ 的祖先目录作为项目根（归档到 changes/archive/ 后层级变化也不受影响）
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "backend").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

API = "http://127.0.0.1:8000"
QUERY = "给深圳全市做一张兴趣点图，公园按植被绿色显示，并标注地铁站"
DEMO_DIR = PROJECT_ROOT / "data/output/text_to_map"


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


# ---------------------------------------------------------------- 验收 1：引擎
def verify_engine() -> list[bool]:
    """selftest 复跑 + validate_ir 三道闸（LLM 不在场）。"""
    results = []
    from scripts.text_to_map import CartographyIR, COG_WHITELIST, THEMES, validate_ir

    print("\n== 验收 1a：引擎 selftest 复跑（13 断言，LLM 不在场）==")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/text_to_map.py"), "--selftest"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120)
    last = [ln for ln in r.stdout.strip().splitlines() if ln][-1:] or ["(无输出)"]
    results.append(check("scripts/text_to_map.py --selftest 退出码 0", r.returncode == 0,
                         f"rc={r.returncode}，末行：{last[0][:80]}"))

    print("\n== 验收 1b：validate_ir 三道闸拒绝非法 IR ==")
    base = CartographyIR(theme="poi", region="深圳全市", rationale="verify 基线",
                         visible_classes=["park", "subway"], label_classes=["subway"],
                         color_semantics={}, color_intent=None, cog=None, raster_opacity=1.0)
    ok = len(validate_ir(base)) == 0
    results.append(check("基线 IR 合法（0 条错误）", ok, f"errors={validate_ir(base)}"))

    bad_theme = CartographyIR(**{**base.__dict__, "theme": "ui_map"})
    errs = validate_ir(bad_theme)
    ok = bool(errs) and any("theme" in e for e in errs)
    results.append(check("非法 theme 被拒（不在 THEMES 枚举）", ok, f"errors={errs}"))

    bad_cog = CartographyIR(**{**base.__dict__, "theme": "cog", "cog": "foo.tif"})
    errs = validate_ir(bad_cog)
    ok = bool(errs) and any("foo.tif" in e and str(COG_WHITELIST) or "白名单" in e or "cog" in e for e in errs)
    results.append(check("白名单外 cog 被拒", ok, f"errors={errs}"))

    bad_sem = CartographyIR(**{**base.__dict__, "color_semantics": {"park": "未注册语义"}})
    errs = validate_ir(bad_sem)
    ok = bool(errs)
    results.append(check("未注册语义词被拒", ok, f"errors={errs}"))

    bad_label = CartographyIR(**{**base.__dict__, "label_classes": ["school"]})
    errs = validate_ir(bad_label)
    ok = bool(errs)
    results.append(check("label_classes ⊄ visible_classes 被拒", ok, f"errors={errs}"))
    return results


# ---------------------------------------------------------------- 验收 2：工具接入
def verify_tool_wiring() -> list[bool]:
    """注册表 9→10 + graph 系统提示制图规则（静态代码检查，不调 LLM）。"""
    results = []
    print("\n== 验收 2：Agent 工具接入 ==")
    from backend.agent.langchain_tools import SPATIAL_TOOLS, text_to_map_tool

    ok = text_to_map_tool in SPATIAL_TOOLS
    results.append(check("text_to_map_tool 已注册进 SPATIAL_TOOLS", ok,
                         f"共 {len(SPATIAL_TOOLS)} 个工具（W3 前 9 → 现 {len(SPATIAL_TOOLS)}）"))
    results.append(check("工具总数 9→10", len(SPATIAL_TOOLS) == 10,
                         f"len(SPATIAL_TOOLS)={len(SPATIAL_TOOLS)}"))

    import backend.agent.graph as g
    prompt = g.SPATIAL_SYSTEM_PROMPT
    ok = "text_to_map_tool" in prompt and ("制图" in prompt or "专题图" in prompt or "地图" in prompt)
    results.append(check("SPATIAL_SYSTEM_PROMPT 含制图路由规则", ok,
                         f"提示长度 {len(prompt)} 字符，规则数≈{prompt.count('。')} 句"))

    from backend.api import main as api_main
    src = Path(api_main.__file__).read_text(encoding="utf-8")
    ok = '_sse("style"' in src and "map_style" in src
    results.append(check("main.py 有 style SSE 发射分支（全量、绕开 [:200] 摘要）", ok,
                         f"含 style 发射={'_sse' in src and 'style' in src}，含 map_style={'map_style' in src}"))

    fe = (PROJECT_ROOT / "frontend/index.html").read_text(encoding="utf-8")
    ok = ("event === 'style'" in fe or "ev.event === 'style'" in fe) and "applyStyle" in fe
    results.append(check("前端 handleEvent 有 style 分支 + applyStyle 方法", ok,
                         f"style 分支={'style' in fe and 'applyStyle' in fe}，"
                         f"清旧={'removeLayer' in fe or 't2m-' in fe}"))
    return results


# ---------------------------------------------------------------- 验收 4：落盘产物
def verify_artifacts() -> list[bool]:
    """demo 三份 Style JSON：主题互异 + spec v8 结构一致性。"""
    results = []
    print("\n== 验收 4：demo 落盘产物（data/output/text_to_map/）==")
    if not DEMO_DIR.is_dir():
        results.append(check("demo 目录存在", False, f"{DEMO_DIR} 不存在；先跑 scripts/text_to_map.py"))
        return results

    themes, names = {}, ["demo_poi.json", "demo_ndvi.json", "demo_cog.json"]
    for name in names:
        p = DEMO_DIR / name
        if not p.is_file():
            results.append(check(f"{name} 存在", False, "缺文件"))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        themes[d["theme"]] = d

        # spec v8 结构：layers[].source 必须是 sources 里的 id（字符串引用）
        src_keys = set(d.get("sources", {}))
        ok = all(isinstance(l.get("source"), str) and l["source"] in src_keys
                 for l in d.get("layers", [])) and bool(d.get("layers"))
        results.append(check(f"{name} layers[].source 为 id 引用且在 sources 中", ok,
                             f"theme={d['theme']}，layers={[(l['id'], l['type']) for l in d['layers']]}"))

        # layer_ids 按设计包含 sources+layers 两类 id（前端清旧层两者都删），须为 layers/sources id 的超集且带 t2m- 前缀
        ids = [l["id"] for l in d["layers"]]
        all_ids = set(ids) | set(d.get("sources", {}))
        li = set(d.get("layer_ids", []))
        ok = all_ids <= li and all(i.startswith("t2m-") for i in li)
        results.append(check(f"{name} layer_ids 覆盖 sources+layers 且带 t2m- 前缀", ok,
                             f"layer_ids={d.get('layer_ids')}"))

        # fit_bounds 是 4 元数组
        fb = d.get("fit_bounds")
        ok = isinstance(fb, list) and len(fb) == 4 and all(isinstance(v, (int, float)) for v in fb)
        results.append(check(f"{name} fit_bounds 合法", ok, f"fit_bounds={fb}"))

    ok = len(themes) == 3
    results.append(check("三条 NL → 三主题互异", ok, f"themes={sorted(themes)}"))

    if "poi" in themes:
        d = themes["poi"]
        ok = bool(d.get("semantic_hits")) and d.get("label_stats", {}).get("placed", 0) > 0
        results.append(check("poi 含语义覆盖 + 标注放置统计", ok,
                             f"semantic_hits={d.get('semantic_hits')}，label_stats={d.get('label_stats')}"))
    if "ndvi" in themes:
        d = themes["ndvi"]
        n_feat = len(d["sources"].get("t2m-ndvi-src", {}).get("data", {}).get("features", []))
        ok = n_feat > 0 and bool(d.get("palette", {}).get("name"))
        results.append(check("ndvi 网格要素 >0 且 palette 已定", ok,
                             f"features={n_feat}，palette={d.get('palette', {}).get('name')}"))
    return results


# ---------------------------------------------------------------- 验收 3：SSE 集成
def _sse_request(query: str) -> str:
    req = urllib.request.Request(
        f"{API}/api/chat",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read().decode("utf-8")


def verify_integration() -> list[bool]:
    """验收 3：起服务后走 /api/chat SSE，断言 style 事件全量下发。"""
    try:
        urllib.request.urlopen(f"{API}/api/config", timeout=5)
    except Exception:
        print("\n[集成前置] 8000 端口无服务。请先另开终端：\n"
              "  cd <项目根> && env -u PYTHONPATH .venv/bin/python -m uvicorn backend.api.main:app --port 8000\n"
              "然后重跑本脚本 --integration。")
        sys.exit(2)

    results = []
    print(f"\n== 验收 3：SSE 集成（/api/chat 问「{QUERY}」）==")
    print("  等待 LLM 决策 + POI 加载 + 标注避让…（通常 30-60s）")
    t0 = time.time()
    sse = _sse_request(QUERY)

    events: dict[str, list[str]] = {}
    for block in sse.strip().split("\n\n"):
        ev = data = None
        for ln in block.split("\n"):
            if ln.startswith("event: "):
                ev = ln[7:]
            elif ln.startswith("data: "):
                data = ln[6:]
        if ev and data is not None:
            events.setdefault(ev, []).append(data)

    tool_evs = [d for d in events.get("tool", []) if "text_to_map_tool" in d]
    ok = bool(tool_evs)
    results.append(check("出现 tool 事件 name=text_to_map_tool（LLM 自主触发）", ok,
                         (f"events={sorted(events)}" if not ok else f"（{time.time()-t0:.1f}s）")))

    styles = events.get("style", [])
    ok = bool(styles)
    detail = f"style 事件数={len(styles)}"
    st = None
    if ok:
        try:
            st = json.loads(styles[0])
            detail = f"theme={st.get('theme')}，layer_ids={st.get('layer_ids')}"
        except json.JSONDecodeError as e:
            ok, detail = False, f"style data JSON 解析失败: {e}"
    results.append(check("style 事件可达且 JSON 可解析", ok, detail))

    if st:
        src_keys = set(st.get("sources", {}))
        ok = (st.get("theme") == "poi"
              and all(isinstance(l.get("source"), str) and l["source"] in src_keys
                      for l in st.get("layers", []))
              and isinstance(st.get("fit_bounds"), list) and len(st["fit_bounds"]) == 4)
        results.append(check("style 载荷 spec v8 合法（source=id 引用 + fit_bounds）", ok,
                             f"sources={sorted(src_keys)}，"
                             f"layers={[(l['id'], l['type']) for l in st.get('layers', [])]}"))

    ok = bool(events.get("answer")) or bool(events.get("done"))
    results.append(check("流正常收尾（answer/done）", ok,
                         f"事件统计: {dict((k, len(v)) for k, v in events.items())}"))
    return results


def main() -> int:
    print("text_to_map verify —— 验收 1/2/4 直调（--integration 追加验收 3）")
    res = verify_engine()
    res += verify_tool_wiring()
    res += verify_artifacts()

    if "--integration" in sys.argv:
        res += verify_integration()

    n_pass = sum(res)
    print(f"\n=== 汇总: {n_pass}/{len(res)} PASS ===")
    if all(res):
        print("✓ 全部验收通过")
        return 0
    print("✗ 存在 FAIL，见上方明细")
    return 1


if __name__ == "__main__":
    sys.exit(main())
