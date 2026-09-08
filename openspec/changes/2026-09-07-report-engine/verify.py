"""report_engine 验证脚本（OpenSpec verify 阶段，可重复执行）。

对应变更：openspec/changes/2026-09-07-report-engine/
验收来源：proposal.md「验收标准」（详见 tasks.md 证据区）。

用法（在项目根目录，用 .venv）：
    .venv/bin/python openspec/changes/2026-09-07-report-engine/verify.py
        → 直调验收（引擎数字闸门 / 工具接入 / 下载路由 / 落盘产物），
          零网络零真实 LLM（stub 驱动），~10s
    .venv/bin/python openspec/changes/2026-09-07-report-engine/verify.py --integration
        → 追加端到端：运行 scripts/e2e_test.py（自起 8010 服务，
          NL → planner 路由 → report 事件 → 下载 → 数字断言，1-3 min，依赖 DeepSeek）

退出码：0 = 全过；1 = 有 FAIL。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
# 向上找含 backend/ 的祖先目录作为项目根（归档到 changes/archive/ 后层级变化也不受影响）
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "backend").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "data" / "output" / "reports"


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


class _StubLLM:
    """假 LLM client：返回固定叙事；记录收到的 prompt 供数字闸门断言。"""

    model = "stub"

    def __init__(self):
        self.prompts: list[str] = []

    @property
    def chat(self):
        outer = self

        class _Chat:
            completions = outer

        return _Chat

    def create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])

        class _Msg:
            content = "监测期内该区域呈现显著变化趋势，可能与潮位和季节因素有关。建议结合多期影像复核。"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _BoomLLM:
    """永远抛异常的 client：模拟 LLM 宕机。"""

    model = "boom"

    @property
    def chat(self):
        class _Chat:
            completions = self

        return _Chat

    def create(self, **kw):
        raise RuntimeError("LLM down (verify 模拟)")


def _fake_state() -> dict:
    return {
        "user_query": "对比深圳湾 2023 和 2025 的水域变化",
        "cog_a": "szbay_real_20230708.tif",
        "cog_b": "szbay_real_20250727.tif",
        "analysis_result": {
            "change_ratio": 0.1182,
            "change_px": 226325,
            "valid_px": 334583,
            "valid_pct": 100.0,
            "method": "spectral_diff",
            "threshold": 0.08,
        },
        "map_path": "",
    }


# ---------------------------------------------------------------- 验收 1：引擎 + 数字闸门
def verify_engine() -> list[bool]:
    results = []
    from scripts.report_engine import (
        QUAL_BANDS,
        build_narrative_prompt,
        collect,
        qualitative_projection,
    )

    print("\n== 验收 1a：引擎 selftest 复跑（10 断言，LLM 不在场）==")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/report_engine.py"), "--selftest"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=180)
    last = [ln for ln in r.stdout.strip().splitlines() if ln][-1:] or ["(无输出)"]
    results.append(check("scripts/report_engine.py --selftest 退出码 0", r.returncode == 0,
                         f"rc={r.returncode}，末行：{last[0][:80]}"))

    print("\n== 验收 1b：定性投影档位（QUAL_BANDS 边界）==")
    cases = {"0.01 → 轻微": ("轻微", 0.01), "0.05 → 中等": ("中等", 0.05), "0.50 → 显著": ("显著", 0.50)}
    for label, (want, v) in cases.items():
        got = qualitative_projection({"change_ratio": v}).get("magnitude")
        results.append(check(f"change_ratio={v} 投影为「{want}」", got == want,
                             f"got={got}，bands={QUAL_BANDS}"))

    print("\n== 验收 1c：数字闸门（LLM prompt 不含原始数字）==")
    state = _fake_state()
    qual = collect(state)["qualitative"]
    prompt = build_narrative_prompt(qual, "2023-07-08 → 2025-07-27", "深圳湾")
    leaks = [s for s in ("11.82", "0.1182", "226325", "334583") if s in prompt]
    results.append(check("叙事 prompt 无 state 原始数字泄漏", not leaks,
                         f"泄漏={leaks or '无'}（时间/地名为设计允许）"))
    ok = qual.get("magnitude") in {"轻微", "中等", "显著"} and bool(qual.get("coverage"))
    results.append(check("qualitative 只含档位词与覆盖描述", ok, f"qual={qual}"))

    print("\n== 验收 1d：generate_report 全流程（stub LLM，不出图）==")
    from scripts.report_engine import generate_report
    stub = _StubLLM()
    out = generate_report(state, llm_client=stub, make_charts=False)
    md = Path(out["md_path"]).read_text(encoding="utf-8")
    ok = "11.82%" in md and "226325" in md
    results.append(check("md 含 state 真实数字（模板注入）", ok, f"md={out['md_path']}"))
    ok = (not out["narrative_skipped"]) and "潮位" in md
    results.append(check("stub 叙事进入 md 且 narrative_skipped=False", ok,
                         f"narrative_skipped={out['narrative_skipped']}"))
    ok = bool(stub.prompts) and not leaks_check(stub.prompts)
    results.append(check("stub 收到的 prompt 同样无数字泄漏", ok,
                         f"prompts={len(stub.prompts)} 条，泄漏={leaks_check(stub.prompts) or '无'}"))

    print("\n== 验收 1e：LLM 宕机降级（可见，不静默）==")
    out2 = generate_report(state, llm_client=_BoomLLM(), make_charts=False)
    md2 = Path(out2["md_path"]).read_text(encoding="utf-8")
    ok = out2["narrative_skipped"] is True
    results.append(check("narrative_skipped=True（显式标记）", ok,
                         f"narrative_skipped={out2['narrative_skipped']}"))
    ok = ("降级" in md2 or "⚠" in md2) and "11.82%" in md2
    results.append(check("降级报告仍含核心数字与 ⚠ 提示", ok,
                         f"⚠={'⚠' in md2}，数字={'11.82%' in md2}"))

    # 清理 verify 自身产物（不加重 selftest 污染问题）
    for o in (out, out2):
        for k in ("md_path", "html_path"):
            p = o.get(k)
            if p and Path(p).is_file():
                Path(p).unlink()
    return results


def leaks_check(prompts: list[str]) -> list[str]:
    return [s for p in prompts for s in ("11.82", "0.1182", "226325", "334583") if s in p]


# ---------------------------------------------------------------- 验收 2：双链路接入
def verify_wiring() -> list[bool]:
    results = []
    print("\n== 验收 2：工具/图/SSE/前端接线（静态检查，不调 LLM）==")

    from backend.agent.langchain_tools import SPATIAL_TOOLS, report_tool
    results.append(check("report_tool 已注册进 SPATIAL_TOOLS", report_tool in SPATIAL_TOOLS,
                         f"共 {len(SPATIAL_TOOLS)} 个工具"))
    results.append(check("工具总数 10→11", len(SPATIAL_TOOLS) == 11,
                         f"len(SPATIAL_TOOLS)={len(SPATIAL_TOOLS)}"))

    import backend.agent.graph as g
    ok = "report_tool" in g.SPATIAL_SYSTEM_PROMPT and "报告" in g.SPATIAL_SYSTEM_PROMPT
    results.append(check("SPATIAL_SYSTEM_PROMPT 含报告路由规则", ok,
                         f"规则句数≈{g.SPATIAL_SYSTEM_PROMPT.count('。')}"))

    ag = (PROJECT_ROOT / "backend/agent/multi_agent/agents.py").read_text(encoding="utf-8")
    ok = "def report_worker" in ag and "analysis_error" in ag.split("def report_worker")[1][:800]
    results.append(check("report_worker 存在且对 analysis_error 短路", ok,
                         f"短路={'analysis_error' in ag.split('def report_worker')[1][:800]}"))

    gr = (PROJECT_ROOT / "backend/agent/multi_agent/graph.py").read_text(encoding="utf-8")
    ok = ('g.add_node("report"' in gr and 'g.add_edge("cartography", "report")' in gr
          and 'g.add_edge("report", "supervisor")' in gr)
    results.append(check("multi_agent 图线性插入 cartography→report→supervisor", ok,
                         f"add_node={'report' in gr}，edges={gr.count('report')} 处引用"))

    st = (PROJECT_ROOT / "backend/agent/multi_agent/state.py").read_text(encoding="utf-8")
    ok = all(f in st for f in ("report_path", "report_title", "narrative_skipped"))
    results.append(check("state 含 report_path/report_title/narrative_skipped", ok,
                         f"字段齐={ok}"))

    mn = (PROJECT_ROOT / "backend/api/main.py").read_text(encoding="utf-8")
    ok = '_sse("report"' in mn and '"/api/reports/{name}"' in mn
    results.append(check("main.py 有 report SSE 发射 + /api/reports/{name} 路由", ok,
                         f"sse={'_sse(\"report\"' in mn}，route={'/api/reports/{name}' in mn}"))

    fe = (PROJECT_ROOT / "frontend/index.html").read_text(encoding="utf-8")
    ok = "ev.event === 'report'" in fe and "reportOf" in fe and "下载 Markdown" in fe
    results.append(check("前端 handleEvent report 分支 + 交付物卡片", ok,
                         f"reportOf={'reportOf' in fe}，下载链接={'下载 Markdown' in fe}"))
    return results


# ---------------------------------------------------------------- 验收 3：下载路由（直调，无需起服）
def verify_route() -> list[bool]:
    results = []
    print("\n== 验收 3：/api/reports/{name} 路由净化（直调函数）==")
    from backend.api.main import download_report

    r = download_report("../../etc/passwd")
    results.append(check("../ 穿越被拒 400", getattr(r, "status_code", None) == 400,
                         f"status={getattr(r, 'status_code', None)}"))

    r = download_report("report.md.txt")
    results.append(check("白名单外后缀被拒 400", getattr(r, "status_code", None) == 400,
                         f"status={getattr(r, 'status_code', None)}"))

    r = download_report("no_such_report.md")
    results.append(check("不存在的文件 404", getattr(r, "status_code", None) == 404,
                         f"status={getattr(r, 'status_code', None)}"))

    real = sorted(REPORTS_DIR.glob("*.md"))
    if real:
        r = download_report(real[-1].name)
        ok = getattr(r, "status_code", None) == 200 and real[-1].name in str(getattr(r, "path", ""))
        results.append(check("真实报告文件 200 可下载", ok,
                             f"status={getattr(r, 'status_code', None)}，file={real[-1].name}"))
    else:
        results.append(check("reports 目录存在可下载样例", False, "目录为空，先跑 CLI demo"))
    return results


# ---------------------------------------------------------------- 验收 4：落盘产物
def verify_artifacts() -> list[bool]:
    results = []
    print("\n== 验收 4：真实报告产物（data/output/reports/）==")
    mds = sorted(REPORTS_DIR.glob("report_*.md"))
    results.append(check("存在报告产物", bool(mds), f"{len(mds)} 份 md"))
    if not mds:
        return results

    # 取含真实叙事的那份（体积最大 = 非 selftest 降级产物）
    md_path = max(mds, key=lambda p: p.stat().st_size)
    md = md_path.read_text(encoding="utf-8")
    ok = "67.64%" in md
    results.append(check("报告含 baseline 真实数字 67.64%", ok, f"file={md_path.name}"))
    ok = "/Users/" not in md
    results.append(check("图表链接全为相对路径（无绝对路径泄漏）", ok,
                         f"绝对路径={'/Users/' in md}"))
    ok = "数字来源" in md or "未经任何语言模型转写" in md
    results.append(check("方法说明含数字来源声明（可追溯）", ok,
                         f"含声明={ok}"))
    return results


# ---------------------------------------------------------------- 验收 5：端到端（--integration）
def verify_integration() -> list[bool]:
    results = []
    print("\n== 验收 5：端到端（复用 scripts/e2e_test.py，自起 8010 服务）==")
    print("  NL → planner 路由 → report 事件 → 下载 → 数字断言…（通常 1-3 min，依赖 DeepSeek）")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/e2e_test.py")],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=420)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln][-2:] or ["(无输出)"]
    results.append(check("e2e_test.py 退出码 0", r.returncode == 0,
                         f"rc={r.returncode}（{time.time()-t0:.0f}s），末行：{tail[-1][:90]}"))
    n_line = next((ln for ln in r.stdout.splitlines() if "e2e 汇总" in ln), "")
    ok = "7/7" in n_line
    results.append(check("e2e 7/7 全过", ok, n_line.strip()[:90]))
    return results


def main() -> int:
    print("report_engine verify —— 验收 1-4 直调（--integration 追加端到端）")
    res = verify_engine()
    res += verify_wiring()
    res += verify_route()
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
