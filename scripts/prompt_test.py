"""W2 交付物验证：prompt_test.py —— 逐个实测 5 种 Prompt 模板

运行方式：
  .venv/bin/python scripts/prompt_test.py
（未配置 Key 时进入 DRY-RUN：展示每个模板组装后的 messages 结构）

核心概念：结构化任务的 temperature 约定
  - P2/P3/P5 输出 JSON 给程序解析 → temperature=0（要确定性，不要创意）
  - P1 知识问答 → temperature=0.3（略灵活）
  - P4 规划 → temperature=0.2（推理任务需要一点探索空间）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from backend.agent.prompts import (
    GIS_QA_SYSTEM,
    INTENT_PARSE_SYSTEM,
    NL2SQL_SYSTEM,
    PLANNING_SYSTEM,
    TEXT_TO_STYLE_SYSTEM,
)
from backend.core.config import llm_config
from backend.core.llm import LLMClient

console = Console()

# 注入给 NL2SQL 的数据库 Schema（对应阶段1数据准备里的两张表）
DEMO_SCHEMA = """
CREATE TABLE schools (id SERIAL PRIMARY KEY, name VARCHAR(200), type VARCHAR(50), geom GEOMETRY(Point, 4326));
CREATE TABLE admin_boundary (id SERIAL PRIMARY KEY, name VARCHAR(100), level VARCHAR(20), geom GEOMETRY(Polygon, 4326));
""".strip()

# (编号, 技术, system_prompt, 测试问题, temperature, 输出是否应为JSON)
CASES = [
    ("P1 知识问答", "Zero-shot", GIS_QA_SYSTEM,
     "PostGIS 里 ST_DWithin 和 ST_Distance 有什么区别？性能上呢？", 0.3, False),
    ("P2 NL→SQL", "Few-shot + JSON", NL2SQL_SYSTEM.replace("{schema}", DEMO_SCHEMA),
     "查询南山区3公里内的所有中学", 0.0, True),
    ("P3 意图解析", "Few-shot + JSON", INTENT_PARSE_SYSTEM,
     "帮我找福田区附近2公里的医院", 0.0, True),
    ("P4 任务规划", "Chain-of-Thought", PLANNING_SYSTEM,
     "分析深圳湾过去3年的水质变化", 0.2, False),
    ("P5 Text→样式", "结构化输出", TEXT_TO_STYLE_SYSTEM,
     "用蓝色渐变显示人口密度，密度越高颜色越深", 0.0, True),
]


def main() -> None:
    if not llm_config.available:
        console.print(Panel(
            "DRY-RUN 模式：展示每个模板实际发送的 messages（配置 DEEPSEEK_API_KEY 后实测）",
            border_style="yellow",
        ))
        for name, tech, system, question, temp, _ in CASES:
            console.rule(f"[bold cyan]{name} · {tech} · temperature={temp}")
            console.print(Panel(system, title="system", border_style="dim", height=10))
            console.print(Panel(question, title="user", border_style="blue"))
        return

    client = LLMClient()
    passed = 0
    for name, tech, system, question, temp, expect_json in CASES:
        console.rule(f"[bold cyan]{name} · {tech} · temperature={temp}")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        answer, usage = client.chat(messages, temperature=temp)
        console.print(Panel(answer, title=f"GeoSense（{usage.get('total_tokens', '?')} tokens）",
                            border_style="green"))
        # 结构化模板的质量门禁：输出必须能被 json.loads 解析
        if expect_json:
            try:
                json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
                console.print("[green]✓ JSON 格式校验通过")
                passed += 1
            except json.JSONDecodeError:
                console.print("[red]✗ JSON 解析失败 —— 需要加固 Prompt（格式校验未过）")
        else:
            passed += 1
    console.rule(f"[bold green]{passed}/{len(CASES)} 个模板验证通过")


if __name__ == "__main__":
    main()
