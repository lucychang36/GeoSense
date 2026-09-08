"""第11月 W4：端到端测试 —— 自然语言 → 多工具 Agent → 报告交付（OpenSpec 2026-09-07-report-engine）。

覆盖学习计划「终极示例交互」的最小闭环：
    一句自然语言 → planner 路由 report_tool → multi_agent 管道（分析→制图→报告）
    → SSE report 事件 → /api/reports/{name} 下载 → md 含真实数字

用法（项目根目录）：
    .venv/bin/python scripts/e2e_test.py          # 自起服务（127.0.0.1:8010）+ 全链路断言
退出码：0 = 全过；1 = 有 FAIL。

注意：属集成测试——依赖 DEEPSEEK_API_KEY、本地 COG、网络（LLM/底图），运行 1-3 分钟；
不进 selftest（design D6）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PORT = 8010
API = f"http://127.0.0.1:{PORT}"
QUERY = "对比深圳湾 2023 和 2025 的水域变化，并生成分析报告"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"（{detail}）" if detail else ""))
    return ok


def wait_server(timeout: float = 60) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"{API}/api/config", timeout=3)
            return True
        except Exception:
            time.sleep(1.5)
    return False


def main() -> int:
    results: list[bool] = []

    # ---- 起服务（子进程；WORKBUDDY 沙箱内须 env -u PYTHONPATH，脚本外运行无此问题）----
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print(f"[e2e] 起服务 uvicorn :{PORT} …")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.main:app", "--port", str(PORT)],
        cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not check("服务就绪（/api/config 200）", wait_server()):
            return 1

        # ---- SSE 全链路 ----
        print(f"[e2e] POST /api/chat：「{QUERY}」（含完整分析管道，预计 1-3 分钟）…")
        req = urllib.request.Request(
            f"{API}/api/chat",
            data=json.dumps({"query": QUERY}).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")

        events: dict[str, list[dict]] = {}
        for block in raw.strip().split("\n\n"):
            ev = data = None
            for ln in block.split("\n"):
                if ln.startswith("event: "):
                    ev = ln[7:]
                elif ln.startswith("data: "):
                    try:
                        data = json.loads(ln[6:])
                    except json.JSONDecodeError:
                        data = None
            if ev and data is not None:
                events.setdefault(ev, []).append(data)

        tool_names = [d.get("name") for d in events.get("tool", [])]
        results.append(check("tool 事件含 report_tool（LLM 自主路由）",
                             "report_tool" in tool_names,
                             f"tools={tool_names}（{time.time()-t0:.0f}s）"))

        reports = events.get("report", [])
        results.append(check("report 事件到达", bool(reports),
                             f"事件统计: {dict((k, len(v)) for k, v in events.items())}"))
        if not reports:
            return 1 if not all(results) else 0
        rep = reports[0]

        results.append(check("report 载荷合法（title + md_url + change_ratio）",
                             bool(rep.get("title")) and bool(rep.get("md_url"))
                             and rep.get("change_ratio") is not None,
                             f"title={rep.get('title')}，change_ratio={rep.get('change_ratio')}"))

        # ---- 下载路由（含净化负例）----
        md_url = rep.get("md_url", "")
        results.append(check("GET md_url 200", bool(md_url) and
                             urllib.request.urlopen(f"{API}{md_url}", timeout=10).status == 200,
                             md_url))
        md_text = urllib.request.urlopen(f"{API}{md_url}", timeout=10).read().decode("utf-8")
        ratio_pct = f"{rep['change_ratio'] * 100:.2f}%"
        results.append(check(f"报告内容含真实数字 {ratio_pct}（数字闸门端到端）",
                             ratio_pct in md_text))

        # 净化负例：越界路径不得 200
        try:
            urllib.request.urlopen(f"{API}/api/reports/..%2F..%2FREADME.md", timeout=10)
            evil_ok = False
        except urllib.error.HTTPError as e:
            evil_ok = e.code in (400, 404)
        except Exception:
            evil_ok = False
        results.append(check("路径净化负例（../ 穿越被拒）", evil_ok))

        results.append(check("流正常收尾（done）", bool(events.get("done"))))
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    n_pass = sum(results)
    print(f"\n=== e2e 汇总: {n_pass}/{len(results)} PASS ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
