"""W1 交付物：llm_basics.py —— 跑通 LLM API 调用（DeepSeek）

四个实验，对应本周要理解的四个核心概念：
  实验1  单轮对话       → messages 结构（system/user/assistant）
  实验2  多轮对话       → 无状态 API + 历史消息回传机制
  实验3  流式输出       → SSE 逐 token 推送（打字机效果）
  实验4  Token 用量     → 计费单位与上下文窗口

运行方式：
  1. cp .env.example .env 并填入 DEEPSEEK_API_KEY
  2. python scripts/llm_basics.py
（未配置 Key 时进入 DRY-RUN 模式：打印将要发送的请求结构，不真正调用 API）
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本可以直接运行：把项目根目录加入 import 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from backend.core.config import llm_config
from backend.core.llm import LLMClient

console = Console()

# 系统提示词：给 GeoSense 定义"人格"。这就是最简单的 Prompt Engineering。
SYSTEM_PROMPT = (
    "你是 GeoSense，一名空间智能分析助手。"
    "你精通 GIS、遥感、PostGIS 空间分析，回答要准确、简洁、有工程视角。"
)


def demo_1_single_turn(client: LLMClient) -> None:
    """实验1：单轮对话。"""
    console.rule("[bold cyan]实验1：单轮对话（messages 结构）")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "用一句话解释什么是空间索引，并举一个 PostGIS 中的例子。"},
    ]
    console.print(Panel(str(messages), title="发送给 API 的 messages", border_style="dim"))
    answer, usage = client.chat(messages)
    console.print(Panel(answer, title="GeoSense 回答", border_style="green"))
    console.print(f"[yellow]Token 用量：{usage}")


def demo_2_multi_turn(client: LLMClient) -> None:
    """实验2：多轮对话 —— 关键：模型无记忆，历史由我们维护。"""
    console.rule("[bold cyan]实验2：多轮对话（无状态 API 的记忆机制）")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 第1轮
    q1 = "深圳南山区的经纬度大概是多少？"
    a1, _ = client.chat(messages + [{"role": "user", "content": q1}])
    console.print(f"[bold]你：[/]{q1}\n[green]GeoSense：[/]{a1}\n")

    # 第2轮：把第1轮的问答都放进历史，模型才知道"那里"指哪
    messages += [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": "那里适合用什么投影坐标系做面积计算？为什么？"},
    ]
    a2, usage = client.chat(messages)
    console.print(f"[bold]你：[/]那里适合用什么投影坐标系做面积计算？为什么？\n"
                  f"[green]GeoSense：[/]{a2}")
    console.print(f"\n[yellow]Token 用量：{usage} —— 注意 prompt_tokens 比第1轮大，"
                  f"因为历史消息也占 Token、也计费！")


def demo_3_stream(client: LLMClient) -> None:
    """实验3：流式输出，逐块接收。"""
    console.rule("[bold cyan]实验3：流式输出（SSE）")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "列举 3 个空间分析中常用的距离度量方式。"},
    ]
    console.print("[green]GeoSense（流式）：[/]", end="")
    for chunk in client.chat_stream(messages):
        print(chunk, end="", flush=True)  # 边收边打印 = 打字机效果
    print()


def demo_4_token_budget(client: LLMClient) -> None:
    """实验4：观察 temperature 与 max_tokens 的影响。"""
    console.rule("[bold cyan]实验4：生成参数（temperature / max_tokens）")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "详细说明 GeoJSON 与 Shapefile 的区别。"},
    ]
    _, usage_full = client.chat(messages)
    _, usage_limited = client.chat(messages, max_tokens=50)
    console.print(f"[yellow]完整回答：{usage_full}")
    console.print(f"[yellow]max_tokens=50：{usage_limited} —— completion_tokens 被硬截断")


def main() -> None:
    if not llm_config.available:
        # DRY-RUN：没有 Key 也能看清"将发送什么"，便于先理解结构
        console.print(Panel(
            "未检测到有效 DEEPSEEK_API_KEY，进入 DRY-RUN 模式。\n\n"
            "配置步骤：\n"
            "1. 到 https://platform.deepseek.com/ 创建 API Key\n"
            "2. cp .env.example .env，把 Key 填进去\n"
            "3. 重新运行本脚本\n\n"
            "真正调用时，每个实验发送的请求体核心结构如下：",
            title="DRY-RUN 模式", border_style="yellow",
        ))
        console.print({
            "model": llm_config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<你的问题>"},
            ],
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
        })
        return

    client = LLMClient()
    console.print(f"[dim]已连接 {llm_config.base_url}，模型 {llm_config.model}[/]")
    demo_1_single_turn(client)
    demo_2_multi_turn(client)
    demo_3_stream(client)
    demo_4_token_budget(client)
    console.rule("[bold green]W1 完成：LLM API 调用全部跑通")


if __name__ == "__main__":
    main()
