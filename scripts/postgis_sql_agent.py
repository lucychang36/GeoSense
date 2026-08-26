"""阶段1 补课交付物：postgis_sql_agent.py —— 自然语言 → PostGIS SQL 端到端演示

两种运行方式：
  1. 检查模式（不需要 API Key / 不需要 LLM）：
       .venv/bin/python scripts/postgis_sql_agent.py --check
     验证四件事：数据库连通、安全查询执行、注入攻击全部被拦截、合法 SQL 自动补 LIMIT。
  2. 完整模式（需要 .env 配置 DEEPSEEK_API_KEY）：
       .venv/bin/python scripts/postgis_sql_agent.py
     Agent 走完整流水线：自然语言 → spatial_sql 生成 SQL → 校验 → 执行 → GeoJSON。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from backend.db.connection import query
from backend.db.sql_validator import SQLValidationError, validate

console = Console()

# ---- 注入攻击测试用例：每一条都必须被 sql_validator 拦截 ----
INJECTION_ATTEMPTS = [
    "SELECT * FROM schools; DROP TABLE schools;--",           # 多语句注入
    "SELECT * FROM schools WHERE name='x'; DELETE FROM poi;",  # 字符串后的注入
    "DROP TABLE schools",                                     # 直接 DDL
    "SELECT * FROM pg_catalog.pg_user",                       # 读系统表
    "SELECT pg_read_file('/etc/passwd')",                     # 危险函数
    "UPDATE schools SET name='hack' WHERE id=1",              # 写操作
    "SELECT * FROM schools LIMIT 999999",                     # 超限 LIMIT
    "SELECT * FROM secret_table",                             # 白名单外表
]


def run_check() -> int:
    """检查模式：连通性 + 安全查询 + 注入拦截。返回 0=全部通过。"""
    ok = True

    # 1. 连通性 + 数据就绪
    try:
        rows = query("SELECT count(*) AS n FROM poi")
        console.print(f"[green]✅ 数据库连通，poi 表 {rows[0]['n']} 条[/]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]❌ 数据库连接失败：{exc}[/]")
        console.print("[yellow]  请先：docker compose up -d && .venv/bin/python scripts/seed_postgis.py[/]")
        return 1

    # 2. 一条安全的空间查询能正常执行（市民中心周边 3km 学校）
    rows = query(
        """
        SELECT name, ST_AsGeoJSON(geom) AS geom
        FROM poi
        WHERE ST_DWithin(geom::geography,
                         ST_SetSRID(ST_MakePoint(114.0579, 22.5431), 4326)::geography, 3000)
        ORDER BY name
        LIMIT 20
        """
    )
    names = "、".join(r["name"] for r in rows)
    console.print(f"[green]✅ 安全查询执行成功，命中 {len(rows)} 条：{names}[/]")

    # 3. 注入测试：8 条攻击必须全部被拦截
    for payload in INJECTION_ATTEMPTS:
        try:
            validate(payload)
            console.print(f"[red]❌ 未拦截（危险！）：{payload}[/]")
            ok = False
        except SQLValidationError as exc:
            console.print(f"[green]✅ 已拦截[/] {payload[:46]}… → {exc}")

    # 4. 合法 SQL 通过校验（缺 LIMIT 自动补）
    safe = validate("SELECT name, ST_AsGeoJSON(geom) AS geom FROM schools WHERE type = '学校'")
    console.print(f"[green]✅ 合法 SQL 通过并自动补 LIMIT：[/]{safe}")

    console.rule("[bold green]检查完成")
    return 0 if ok else 1


DEMO_QUERIES = [
    "查询深圳市民中心 (114.0579, 22.5431) 周边 3 公里内的所有学校，按距离排序列出",
    "南山区内有多少所学校？列出学校名称",
    "统计 poi 表中各类型 POI 的数量",
]


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def run_agent() -> None:
    """完整模式：LangGraph Agent 流式演示。"""
    from backend.agent.graph import get_spatial_agent
    from backend.core.config import llm_config

    if not llm_config.available:
        console.print("[yellow]请先在 .env 配置 DEEPSEEK_API_KEY[/]")
        return
    agent = get_spatial_agent()
    for q in DEMO_QUERIES:
        console.rule(f"[bold cyan]{q[:24]}…")
        console.print(Panel(q, title="用户提问", border_style="blue"))
        prev = 0
        for step in agent.stream({"messages": [("user", q)]}, stream_mode="values"):
            msgs = step["messages"]
            for m in msgs[prev:]:
                _print_message(m)
            prev = len(msgs)
    console.rule("[bold green]演示完成：自然语言 → PostGIS SQL → 结果")


def _print_message(m) -> None:
    from langchain_core.messages import AIMessage, ToolMessage

    if isinstance(m, AIMessage):
        content = text_of(m.content).strip()
        if m.tool_calls:
            for call in m.tool_calls:
                console.print(f"  [yellow]🤔 调用[/] {call['name']}({str(call['args'])[:80]})")
            if content:
                console.print(f"  [dim]💭 {content[:70]}[/]")
        elif content:
            console.print(Panel(content, title="最终回答", border_style="green"))
    elif isinstance(m, ToolMessage):
        console.print(f"  [cyan]👁 结果：{m.content[:120]}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="PostGIS SQL Agent 演示")
    parser.add_argument("--check", action="store_true", help="检查模式（不调 LLM）")
    args = parser.parse_args()
    sys.exit(run_check() if args.check else run_agent())


if __name__ == "__main__":
    main()
