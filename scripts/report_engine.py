"""第11月 W1：自动报告生成引擎 —— 数字闸门 + 双模板 + 图表（OpenSpec 变更 2026-09-07-report-engine）。

设计核心（design D1，W4-D1「语义-语法分层」同款思想）
--------------------------------------------------------
**LLM 在报告链路里拿不到任何原始数字。**

    analysis_result（含 change_ratio 等数值）
        │  collect()：代码做定性投影（轻微/中等/显著 档位映射）
        ▼
    llm_narrative(qualitative)：DeepSeek 只收定性输入，写解读叙事（无数值可编）
        │
        ▼  Jinja2 双模板（md + html）从 context 直接注入数字
    最终报告 = 代码注入的数字 + LLM 的定性解读

失败路径（design D1）：叙事 LLM 失败 → 返回 None，报告头部标记降级（不静默）；
analysis_error 存在时上游 worker 短路（见 multi_agent/agents.py report_worker）。

用法（项目根目录，.venv）：
    .venv/bin/python scripts/report_engine.py --selftest   # 零网络零 LLM（stub 驱动）
    .venv/bin/python scripts/report_engine.py "对比深圳湾 2023 和 2025 的水域变化"
                                                        # 真实管道 → data/output/reports/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "data" / "output" / "reports"
TEMPLATES_DIR = Path(__file__).resolve().parent / "report_templates"

# ===========================================================================
# 中文字体（复用 unet_segmentation.py:255-266 已趟坑的模式：macOS PingFang → Linux Noto）
# ===========================================================================


def setup_cjk_font() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    for fp in ("/System/Library/Fonts/PingFang.ttc",
               "/System/Library/Fonts/STHeiti Light.ttc",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


# ===========================================================================
# D1：定性投影（数字闸门第一道闸）
# ===========================================================================

# change_ratio 档位：代码化的映射规则，LLM 输入只有档位词（design D1）
QUAL_BANDS: list[tuple[float, str]] = [(0.02, "轻微"), (0.08, "中等"), (float("inf"), "显著")]

# 专题净变化面积档位（km²，index_change 方法专用；同 D1 代码化思想）
AREA_BANDS: list[tuple[float, str]] = [(0.5, "轻微"), (2.0, "中等"), (float("inf"), "显著")]


def qualitative_projection(analysis_result: dict) -> dict:
    """analysis_result → 定性档位。返回值不含任何原始数值，作为 LLM 叙事的唯一输入。

    index_change 结果（含 net_km2）额外给出方向与面积档位；旧 spectral_diff
    state 无 net_km2 → 输出与历史版本逐字节一致（向后兼容，selftest 断言）。
    """
    ratio = float(analysis_result.get("change_ratio", 0.0))
    magnitude = next(label for hi, label in QUAL_BANDS if ratio < hi)
    qual = {
        "direction": "增加",          # spectral_diff 差分为正 → 变化像元增多（语义：扰动增强）
        "magnitude": magnitude,
        "coverage": "较高" if float(analysis_result.get("valid_pct", 0)) > 0.5 else "有限",
    }
    if "net_km2" in analysis_result:
        net = float(analysis_result["net_km2"])
        qual["direction"] = "增加" if net >= 0 else "减少"
        qual["area"] = next(label for hi, label in AREA_BANDS if abs(net) < hi)
    return qual


def build_narrative_prompt(qual: dict, period: str, region: str) -> str:
    """叙事 prompt —— 只允许定性词与时间/地名，禁止出现任何数值。

    selftest 会断言本函数输出不含 analysis_result 的原始数字（数字闸门）。
    """
    area_line = f"净变化面积：{qual['area']}\n" if qual.get("area") else ""
    return (
        "你是遥感分析报告的撰写者。请根据以下定性结论写一段 2-3 句的中文解读，"
        "面向非遥感专业的读者，说明变化可能的原因类别（如潮位/季节/人类活动），"
        "并给出一句话的后续建议。\n"
        f"区域：{region}\n监测期：{period}\n"
        f"变化幅度：{qual['magnitude']}（{qual['direction']}趋势）\n"
        f"{area_line}"
        f"有效观测覆盖：{qual['coverage']}\n"
        "注意：不要编造任何具体数值、面积或百分比；不要使用 markdown。"
    )


def llm_narrative(qual: dict, period: str, region: str, client=None) -> str | None:
    """调 DeepSeek 写解读段落。失败返回 None（降级可见，不静默）。

    client 参数供 selftest 注入 stub；None 时用项目 llm_config 现连。
    """
    try:
        if client is None:
            from openai import OpenAI
            from backend.core.config import llm_config
            client = OpenAI(base_url=llm_config.base_url,
                            api_key=llm_config.api_key,
                            timeout=60)
            model = llm_config.model
        else:
            model = getattr(client, "model", "stub")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_narrative_prompt(qual, period, region)}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:                                        # noqa: BLE001 —— 降级可见
        return None


# ===========================================================================
# D3：图表（Matplotlib 统计图 + Mapbox Static 底图）
# ===========================================================================


def render_charts(context: dict, out_dir: Path, mapbox_token: str | None = None) -> dict:
    """统计图 PNG（必出）+ 专题底图（可选，失败跳过）。返回相对 context 的路径字典。"""
    import matplotlib.pyplot as plt
    setup_cjk_font()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = context["report_id"]
    m = context["metrics"]

    # 统计图：变化 vs 未变像元（面积量纲统一为像元数，避免依赖分辨率元数据）
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(["变化像元", "有效像元"], [m["change_px"], m["valid_px"]],
                  color=["#C4432B", "#7A8894"], width=0.5)
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("像元数")
    ax.set_title(f"变化检测统计（{m['method']}，阈值 {m['threshold']}）")
    fig.tight_layout()
    stats_png = out_dir / f"{stem}_stats.png"
    fig.savefig(stats_png, dpi=110)
    plt.close(fig)

    charts = {"stats": stats_png.name}

    # Mapbox Static 底图（design D3：可选增强，失败跳过不致命）
    if mapbox_token and context.get("bbox"):
        try:
            import urllib.request
            bbox = context["bbox"]                      # [w, s, e, n]
            url = (f"https://api.mapbox.com/styles/v1/mapbox/dark-v11/static/"
                   f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]/640x360@2x"
                   f"?access_token={mapbox_token}&attribution=false&logo=false")
            req = urllib.request.Request(url, headers={"User-Agent": "GeoSense/1.0"})
            basemap_png = out_dir / f"{stem}_basemap.png"
            basemap_png.write_bytes(urllib.request.urlopen(req, timeout=20).read())
            charts["basemap"] = basemap_png.name
        except Exception:                                # noqa: BLE001 —— 可选增强
            pass
    return charts


# ===========================================================================
# D2：Jinja2 双模板同 context 渲染
# ===========================================================================

SECTIONS = ["概述", "解读", "核心指标", "图表", "方法说明", "附录"]


def render_report(context: dict) -> tuple[Path, Path]:
    """同一 context 渲染 md + html 双模板（design D2）。返回两个文件路径。"""
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)),
                      autoescape=False, keep_trailing_newline=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = context["report_id"]
    md_path = REPORTS_DIR / f"{stem}.md"
    html_path = REPORTS_DIR / f"{stem}.html"
    md_path.write_text(env.get_template("report.md.j2").render(**context), encoding="utf-8")
    html_path.write_text(env.get_template("report.html.j2").render(**context), encoding="utf-8")
    return md_path, html_path


# ===========================================================================
# 全流程入口
# ===========================================================================


def collect(state: dict) -> dict:
    """multi_agent state → 报告 context（定性投影 + 数字上下文）。"""
    res = dict(state.get("analysis_result") or {})
    if not res:
        raise ValueError("state 缺少 analysis_result，无法生成报告")
    cog_a, cog_b = state.get("cog_a", "?"), state.get("cog_b", "?")

    def _pretty(name: str) -> str:
        m = re.search(r"(\d{8})", name)
        return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else name

    # 专题参数化（2026-09-11 thematic-index-change）：region/theme 从 analysis_result
    # 读取；旧 state 无这些字段 → 回落深圳湾/水域，标题与历史版本逐字节一致
    region = res.get("region_label") or "深圳湾"
    theme_label = res.get("theme_label") or "水域"
    period = f"{_pretty(cog_a)} → {_pretty(cog_b)}"
    qual = qualitative_projection(res)
    metrics = {
        "change_ratio_pct": round(res.get("change_ratio", 0.0) * 100, 2),
        "change_px": res.get("change_px", 0),
        "valid_px": res.get("valid_px", 0),
        "valid_pct": round(res.get("valid_pct", 0.0) * 100, 1),
        "method": res.get("method", "?"),
        "threshold": res.get("threshold", "?"),
    }
    if res.get("method") == "index_change":
        metrics.update({
            "theme_label": theme_label,
            "gain_km2": res.get("gain_km2", 0.0),
            "loss_km2": res.get("loss_km2", 0.0),
            "net_km2": res.get("net_km2", 0.0),
        })
    return {
        "report_id": f"report_{time.strftime('%Y%m%d_%H%M%S')}",
        "title": f"{region}{theme_label}变化分析报告（{_pretty(cog_a)} → {_pretty(cog_b)}）",
        "region": region,
        "theme_label": theme_label,
        "period": period,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "metrics": metrics,
        "qualitative": qual,
        "narrative": None,                # generate_report 里填充
        "narrative_skipped": False,
        "map_path": state.get("map_path", ""),
        # bbox 优先取 analysis_result（真实下载时随影像入库）；缺省深圳湾（历史行为）
        "bbox": res.get("bbox") or [113.88, 22.46, 114.1, 22.6],
        "deliverables": [],
    }


def generate_report(state: dict, llm_client=None, mapbox_token: str | None = None,
                    make_charts: bool = True) -> dict:
    """全流程：collect → charts → narrative → render。返回落盘信息 dict。"""
    context = collect(state)
    if make_charts:
        context["charts"] = render_charts(context, REPORTS_DIR, mapbox_token)
    else:
        context["charts"] = {}
    narrative = llm_narrative(context["qualitative"], context["period"],
                              context["region"], client=llm_client)
    if narrative:
        context["narrative"] = narrative
    else:
        context["narrative_skipped"] = True

    # 交付物清单（模板附录）：图表 + 时相图（与报告同目录，相对链接）
    # 时相图产在 multi_agent_maps/，绝对路径会让 html 相对链接断掉 → 拷进报告目录
    deliverables = [v for v in context.get("charts", {}).values()]
    mp = context.get("map_path")
    if mp and Path(mp).is_file():
        local = REPORTS_DIR / Path(mp).name
        if local.resolve() != Path(mp).resolve():
            local.write_bytes(Path(mp).read_bytes())
        context["map_path"] = local.name
        deliverables.append(local.name)
    context["deliverables"] = deliverables

    md_path, html_path = render_report(context)
    return {
        "title": context["title"],
        "change_ratio": (state.get("analysis_result") or {}).get("change_ratio"),
        "md_path": str(md_path),
        "html_path": str(html_path),
        "narrative_skipped": context["narrative_skipped"],
        "charts": context.get("charts", {}),
    }


# ===========================================================================
# selftest（零网络零 LLM：stub 驱动全流程）
# ===========================================================================


class _StubLLM:
    """假 LLM client：返回固定叙事；记录收到的 prompt 供数字闸门断言。"""

    model = "stub"

    def __init__(self):
        self.prompts: list[str] = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])

        class _Msg:
            content = "监测期内该区域呈现显著变化趋势，可能与潮位和季节因素有关。建议结合多期影像复核。"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _synthetic_state() -> dict:
    return {
        "cog_a": "szbay_real_20230708.tif",
        "cog_b": "szbay_real_20250727.tif",
        "analysis_result": {"method": "spectral_diff", "threshold": 0.08,
                            "change_px": 8421, "valid_px": 71234,
                            "change_ratio": 0.1182, "valid_pct": 0.6234},
        "map_path": "",
    }


def _headings(text: str, is_html: bool) -> set:
    if is_html:
        return set(re.findall(r"<h2>(.*?)</h2>", text))
    return set(re.findall(r"^## (.+)$", text, re.M))


def selftest() -> int:
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"（{detail}）" if detail else ""))
        ok = ok and passed

    print("report_engine selftest —— 零网络零 LLM（stub 驱动）")

    # 1. 数字闸门：叙事 prompt 不含原始数值
    state = _synthetic_state()
    qual = qualitative_projection(state["analysis_result"])
    prompt = build_narrative_prompt(qual, "2023-07-08 → 2025-07-27", "深圳湾")
    leaked = [tok for tok in ("0.1182", "8421", "71234", "11.82", "0.6234") if tok in prompt]
    check("数字闸门：narrative prompt 不含原始数值", not leaked, f"泄漏={leaked}")
    check("定性投影产出档位词", qual["magnitude"] == "显著", str(qual))

    # 2. 全流程（stub LLM）：md 含真实数字 + 叙事在场
    out = generate_report(state, llm_client=_StubLLM(), make_charts=False)
    md = Path(out["md_path"]).read_text(encoding="utf-8")
    check("md 含 state 真实数字（change_ratio 11.82%）", "11.82%" in md)
    check("md 含变化像元数 8,421 或 8421", ("8,421" in md) or ("8421" in md))
    check("叙事在场（stub 返回）", not out["narrative_skipped"] and "潮位" in md)
    html = Path(out["html_path"]).read_text(encoding="utf-8")
    check("html 同样含真实数字", "11.82%" in html)

    # 3. 双模板章节标题集合一致（design 红旗②）
    hs_md, hs_html = _headings(md, False), _headings(html, True)
    check("双模板章节标题集合一致", hs_md == hs_html, f"md={sorted(hs_md)} html={sorted(hs_html)}")

    # 4. 叙事缺席降级：报告仍完整 + 标记
    class _NoneLLM(_StubLLM):
        def create(self, **kw):
            self.prompts.append(kw["messages"][0]["content"])
            raise RuntimeError("stub 模拟 LLM 宕机")

    out2 = generate_report(state, llm_client=_NoneLLM(), make_charts=False)
    md2 = Path(out2["md_path"]).read_text(encoding="utf-8")
    check("LLM 宕机 → narrative_skipped 标记", out2["narrative_skipped"])
    check("降级报告仍含核心数字与章节", "11.82%" in md2 and "## 核心指标" in md2
          and "自动叙事生成失败" in md2)
    hs2 = _headings(md2, False)
    check("降级报告章节完整（缺解读）", "解读" not in hs2 and {"概述", "核心指标", "图表", "方法说明", "附录"} <= hs2,
          str(sorted(hs2)))


    # 5. 专题参数化（thematic-index-change）：index_change state → region/theme 注入 + 面积闸门
    themed_state = {
        "cog_a": "zhengzhou_real_20230615.tif",
        "cog_b": "zhengzhou_real_20250618.tif",
        "analysis_result": {
            "method": "index_change", "theme": "builtup", "theme_label": "建筑用地",
            "region_label": "郑州高新区", "threshold": 0.0,
            "change_px": 15200, "valid_px": 90200, "change_ratio": 0.1685,
            "valid_pct": 0.8111, "gain_px": 15200, "loss_px": 0,
            "gain_km2": 2.41, "loss_km2": 0.0, "net_km2": 2.41,
            "pixel_area_km2": 1.586e-4,
        },
        "map_path": "",
    }
    qual3 = qualitative_projection(themed_state["analysis_result"])
    check("专题档位：direction=增加 + area=显著",
          qual3["direction"] == "增加" and qual3.get("area") == "显著", str(qual3))
    prompt3 = build_narrative_prompt(qual3, "2023-06-15 → 2025-06-18", "郑州高新区")
    leaked3 = [tok for tok in ("2.41", "0.1685", "15200", "90200") if tok in prompt3]
    check("专题闸门：面积只有档位词，原始数字不进 prompt", not leaked3, f"泄漏={leaked3}")
    out3 = generate_report(themed_state, llm_client=_StubLLM(), make_charts=False)
    md3 = Path(out3["md_path"]).read_text(encoding="utf-8")
    check("专题标题注入（郑州高新区建筑用地）",
          out3["title"].startswith("郑州高新区建筑用地变化分析报告"), out3["title"])
    check("专题面积行注入（新增 2.41 km²）", "新增面积" in md3 and "2.41" in md3)
    check("专题边界声明在场（疑似建筑用地）", "疑似 建筑用地" in md3)
    html3 = Path(out3["html_path"]).read_text(encoding="utf-8")
    check("html 同步专题面积行", "净变化面积" in html3)

    # 6. 向后兼容：旧 spectral_diff state 的标题与历史版本一致（回落深圳湾/水域）
    out4_title = collect(_synthetic_state())["title"]
    check("旧 state 回落深圳湾水域（逐字节兼容）",
          out4_title == "深圳湾水域变化分析报告（2023-07-08 → 2025-07-27）", out4_title)

    print(f"\n=== 汇总: {'全部通过' if ok else '存在 FAIL'} ===")
    return 0 if ok else 1


# ===========================================================================
# CLI demo：跑真实 multi_agent 管道 → 报告
# ===========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description="第11月 W1 报告生成引擎")
    ap.add_argument("query", nargs="?",
                    default="对比深圳湾 2023 和 2025 的水域变化",
                    help="用户问题（驱动 multi_agent 管道）")
    ap.add_argument("--selftest", action="store_true", help="自测（零网络零 LLM）")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    print(f"[1/2] 运行 multi_agent 管道：{args.query}")
    from backend.agent.multi_agent import get_multi_agent
    final = get_multi_agent().invoke({"user_query": args.query, "step_log": []})
    if final.get("analysis_error"):
        print(f"✗ 管道失败：{final['analysis_error']}")
        return 1
    print(f"      change_ratio={(final.get('analysis_result') or {}).get('change_ratio')}")

    print("[2/2] 生成报告（DeepSeek 叙事 + Mapbox 底图）")
    token = ""
    try:
        from dotenv import dotenv_values
        token = (dotenv_values(PROJECT_ROOT / ".env") or {}).get("MAPBOX_TOKEN", "")
    except Exception:                                    # noqa: BLE001
        pass
    out = generate_report(final, mapbox_token=token or None)

    print(f"\n✓ 报告已生成（narrative_skipped={out['narrative_skipped']}）")
    for k, v in (("md", out["md_path"]), ("html", out["html_path"])):
        p = Path(v)
        print(f"  [{k}] {p}（{p.stat().st_size:,} B）")
    print(f"  [charts] {out['charts']}")

    # 交付自检：md 数字与 state 一致
    md = Path(out["md_path"]).read_text(encoding="utf-8")
    # 与 collect 的 metrics 同口径（round 而非 .2f——0.074 → "7.4%"，格式必须逐字一致）
    ratio_pct = f"{round(final['analysis_result']['change_ratio'] * 100, 2)}%"
    assert ratio_pct in md, f"数字闸门自检失败：md 缺 {ratio_pct}"
    print(f"  [自检] md 含真实数字 {ratio_pct} ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
