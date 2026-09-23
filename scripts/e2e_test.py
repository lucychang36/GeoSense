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
# 专题场景（thematic-index-change）：--zhengzhou 切换；断言专题标题/面积行/口径修正
QUERY_ZZ = "对比郑州高新区 2023 和 2025 的建筑用地变化，并生成分析报告"


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zhengzhou", action="store_true",
                    help="跑郑州高新区建筑用地专题场景（默认深圳湾回归场景）")
    args = ap.parse_args()
    query = QUERY_ZZ if args.zhengzhou else QUERY
    results: list[bool] = []

    # ---- 起服务（子进程；WORKBUDDY 沙箱内须 env -u PYTHONPATH，脚本外运行无此问题）----
    # 端口占用预检（2026-09-23 实录）：残留旧服务占 PORT 会让 Popen 的 uvicorn 静默
    # bind 失败（stderr=DEVNULL），wait_server 探到旧服务 → 断言全打到旧代码（假绿/假红）
    import socket
    _probe = socket.socket()
    try:
        _probe.bind(("127.0.0.1", PORT))
    except OSError:
        print(f"[e2e] ❌ 端口 {PORT} 已被占用——先停掉残留服务（lsof -i :{PORT}）再跑，"
              f"否则测试会打到旧进程而非本次起的服务")
        return 1
    finally:
        _probe.close()
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
        print(f"[e2e] POST /api/chat：「{query}」（含完整分析管道，预计 1-3 分钟）…")
        req = urllib.request.Request(
            f"{API}/api/chat",
            data=json.dumps({"query": query}).encode(),
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
        ratio_pct = f"{round(rep['change_ratio'] * 100, 2)}%"   # 与 metrics 同口径（round 非 .2f）
        results.append(check(f"报告内容含真实数字 {ratio_pct}（数字闸门端到端）",
                             ratio_pct in md_text))
        if args.zhengzhou:
            results.append(check("专题标题（郑州高新区建筑用地变化分析报告）",
                                 "郑州高新区建筑用地变化分析报告" in rep.get("title", "")
                                 and "郑州高新区建筑用地变化分析报告" in md_text,
                                 rep.get("title", "")))
            results.append(check("专题面积行（新增/消失/净变化 km²）",
                                 "新增面积" in md_text and "净变化面积" in md_text))
            results.append(check("专题边界声明（疑似 建筑用地）",
                                 "疑似 建筑用地" in md_text))

            # ---- map-result-linkage：overlay 事件 + 面积一致性 + basemap 联动 ----
            overlays = events.get("overlay", [])
            results.append(check("overlay 事件到达（通用协议 url/fit_bounds/legend/render_hint）",
                                 bool(overlays) and all(k in overlays[0] for k in
                                     ("url", "fit_bounds", "legend", "render_hint")),
                                 json.dumps(overlays[0], ensure_ascii=False)[:140] if overlays else "无事件"))
            if overlays:
                ov = overlays[0]
                n_feat = 0
                try:
                    fc = json.loads(urllib.request.urlopen(f"{API}{ov['url']}", timeout=15).read().decode())
                    n_feat = len(fc["features"])
                    g = sum(f["properties"]["area_m2"] for f in fc["features"]
                            if f["properties"]["kind"] == "gain") / 1e6
                    l = sum(f["properties"]["area_m2"] for f in fc["features"]
                            if f["properties"]["kind"] == "loss") / 1e6
                    import re as _re
                    mg = _re.search(r"\| 新增面积 \| ([\d.]+) km²", md_text)
                    ml = _re.search(r"\| 消失面积 \| ([\d.]+) km²", md_text)
                    ok_area = bool(mg and ml) and abs(g - float(mg.group(1))) < 0.01 \
                        and abs(l - float(ml.group(1))) < 0.01
                    results.append(check(f"overlay 面积总和 == 报告主数字"
                                         f"（新增 {g:.2f} / 消失 {l:.2f} km²，{n_feat} 图斑）", ok_area))
                except Exception as exc:  # noqa: BLE001
                    results.append(check("overlay 下载与面积对账", False, str(exc)))
                results.append(check("overlay 图斑规模合理", 0 < n_feat < 20000, f"{n_feat}"))
                results.append(check("basemap 联动（期 B 郑州影像）",
                                     "zhengzhou" in (ov.get("basemap") or {}).get("tile_path", "")))
                results.append(check("图例含疑似限定词",
                                     all("疑似" in lg.get("label", "") for lg in ov.get("legend", []))))
                results.append(check("md 图斑过滤声明在场", "图斑过滤" in md_text))
                # 净化负例：overlays 路由同样拒穿越
                try:
                    urllib.request.urlopen(f"{API}/api/overlays/..%2F..%2FREADME.md", timeout=10)
                    evil2 = False
                except urllib.error.HTTPError as e:
                    evil2 = e.code in (400, 404)
                except Exception:
                    evil2 = False
                results.append(check("overlays 路径净化负例（../ 穿越被拒）", evil2))

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
