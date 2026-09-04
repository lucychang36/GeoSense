"""temporal_change_tool 验证脚本（OpenSpec verify 阶段，可重复执行）。

对应变更：openspec/changes/2026-09-04-agent-temporal-change-tool/
验收来源：proposal.md「验收标准」4 条。

用法（在项目根目录，用 .venv）：
    .venv/bin/python openspec/changes/2026-09-04-agent-temporal-change-tool/verify.py
        → 直调 3 条（验收 1/2/4），无需起服务，~6s
    .venv/bin/python openspec/changes/2026-09-04-agent-temporal-change-tool/verify.py --integration
        → 追加验收 3：要求 8000 端口服务已在跑（uvicorn backend.api.main:app），
          通过 /api/chat SSE 让 LLM 自主触发工具，断言 tool 事件 + result 摘要（~10-40s）

退出码：0 = 全过；1 = 有 FAIL；2 = --integration 但服务不在跑。
"""
from __future__ import annotations

import json
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
QUERY = "对比 2023 和 2025 深圳湾水域变化"

COG_A, COG_B = "szbay_real_20230708.tif", "szbay_real_20250727.tif"
POSTCLASS_EXPECTED = 0.119   # 与第8月 W2 线下 11.9% / 服务端 11.87% 一致（±0.005）
SPECTRAL_RANGE = (0.60, 0.75)  # 光谱差分 baseline ~67% 量级


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def verify_direct() -> list[bool]:
    """验收 1/2/4：不经服务，函数级直调。"""
    from backend.agent.langchain_tools import SPATIAL_TOOLS, temporal_change_tool

    results = []
    print("\n== 验收 1：postclass 直调（U-Net 分类后比较）==")
    t0 = time.time()
    r1 = temporal_change_tool.invoke({"cog_a": COG_A, "cog_b": COG_B})
    d1 = json.loads(r1)
    ok = abs(d1["change_ratio"] - POSTCLASS_EXPECTED) < 0.005
    results.append(check(
        "change_ratio≈0.119", ok,
        f"method={d1['method']} change_ratio={d1['change_ratio']} "
        f"n_change={d1['n_change']} n_valid={d1['n_valid']}（{time.time()-t0:.1f}s）"))
    # transition 物理合理性：水→水 应为主对角最大项（海岸线稳定先验）
    tr = d1["transition"]
    ok_t = tr[0][0] == max(tr[0])
    results.append(check(
        "transition 水行主对角最大（海岸稳定先验）", ok_t,
        f"transition 水行={tr[0]}"))

    print("\n== 验收 4：spectral 直调（光谱差分 baseline）==")
    t0 = time.time()
    r4 = temporal_change_tool.invoke({"cog_a": COG_A, "cog_b": COG_B, "method": "spectral"})
    d4 = json.loads(r4)
    ok = SPECTRAL_RANGE[0] < d4["change_ratio"] < SPECTRAL_RANGE[1]
    results.append(check(
        "change_ratio∈(0.60,0.75)", ok,
        f"method={d4['method']} change_ratio={d4['change_ratio']}（{time.time()-t0:.1f}s）"))

    print("\n== 验收 2：失败路径（不存在的 COG）==")
    r2 = temporal_change_tool.invoke({"cog_a": "szbay_2099_nonexist.tif", "cog_b": COG_B})
    ok = r2.startswith("[工具错误]") and COG_A in r2
    results.append(check(
        "返回 [工具错误] 且带可用 COG 候选、不抛异常", ok,
        f"{r2[:90]}…"))
    return results


def _sse_request() -> str:
    """POST /api/chat，返回原始 SSE 文本。"""
    req = urllib.request.Request(
        f"{API}/api/chat",
        data=json.dumps({"query": QUERY}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read().decode("utf-8")


def verify_integration() -> list[bool]:
    """验收 3：起服务后走 /api/chat SSE，让 LLM 自主触发工具。"""
    # 服务探活
    try:
        urllib.request.urlopen(f"{API}/api/model/health", timeout=5)
    except Exception:
        print("\n[集成前置] 8000 端口无服务。请先另开终端：\n"
              "  cd <项目根> && env -u PYTHONPATH .venv/bin/uvicorn backend.api.main:app --port 8000\n"
              "然后重跑本脚本 --integration。")
        sys.exit(2)

    results = []
    print(f"\n== 验收 3：SSE 集成（/api/chat 问「{QUERY}」）==")
    print("  等待 LLM 决策 + U-Net 推理…（通常 10-40s）")
    t0 = time.time()
    sse = _sse_request()

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

    tool_evs = [d for d in events.get("tool", []) if "temporal_change_tool" in d]
    ok1 = bool(tool_evs) and '"cog_a": "szbay_real_20230708.tif"' in (tool_evs[0] if tool_evs else "")
    results.append(check(
        "出现 tool 事件 name=temporal_change_tool（LLM 自主触发）", ok1,
        (f"args={tool_evs[0]}" if tool_evs else "未触发；events=" + str(sorted(events))[:200])
        + f"（{time.time()-t0:.1f}s）"))

    result_evs = events.get("result", [])
    ok2 = False
    detail2 = "无 result 事件"
    for d in result_evs:
        try:
            payload = json.loads(d)          # 外层 data 是完整合法 JSON
        except json.JSONDecodeError:
            continue
        summ = payload.get("summary", "")    # summary 可能被 main.py 截断（[:200]），故只做子串检查
        if "change_ratio" in summ and "0.1187" in summ:
            ok2 = True
            detail2 = summ[:150]
            break
        detail2 = summ[:150]
    results.append(check(
        "result 摘要含 change_ratio≈0.118", ok2, detail2))

    ok3 = bool(events.get("done")) or bool(events.get("answer"))
    results.append(check("流正常收尾（answer/done）", ok3,
                         f"事件统计: {dict((k, len(v)) for k, v in events.items())}"))
    return results


def main() -> int:
    print("temporal_change_tool verify —— 验收 1/2/4 直调（--integration 追加验收 3）")
    res = verify_direct()

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
