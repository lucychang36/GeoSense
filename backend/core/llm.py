"""LLM 客户端封装 —— GeoSense 所有大模型调用的唯一入口。

核心概念：为什么要封装一层？
- 业务代码（Agent/RAG）只面对 chat() / chat_stream() 两个方法，
  不关心底层是 DeepSeek 还是 GPT-4o。将来换模型 = 改配置，不改业务代码。
- 这一层是后续加"重试、限流、成本统计、LangSmith 追踪"的统一挂点。

核心概念：messages 消息列表
- LLM API 是无状态的：它不"记得"你上次说过什么。
- 多轮对话的本质 = 每次请求都把完整历史 messages 重新发一遍。
- 三种角色：system（系统设定，定义人格与规则）、user（用户输入）、
  assistant（模型之前的回复，由我们负责追加进历史）。
"""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from .config import LLMConfig, llm_config


class LLMClient:
    """OpenAI 兼容协议的 LLM 客户端。"""

    def __init__(self, config: LLMConfig = llm_config):
        self.config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def chat(self, messages: list[dict], **kwargs) -> tuple[str, dict]:
        """一次性对话：等模型生成完，返回 (完整文本, token用量)。

        返回值中的 usage 包含 prompt_tokens / completion_tokens / total_tokens，
        这是理解"Token 计费"的第一手数据。
        """
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        content = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage else {}
        return content, usage

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """流式对话：模型每生成一小段就立刻吐出来（SSE 流）。

        核心概念：流式输出（Streaming / SSE）
        - LLM 是逐 token 生成的。非流式 = 等全部生成完一次性返回，用户干等。
        - 流式 = 服务器用 Server-Sent Events 逐块推送，前端边收边渲染，
          这就是 ChatGPT 那种"打字机效果"，首 token 延迟（TTFT）大幅降低。
        """
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def chat_with_tools(self, messages: list[dict], tools: list[dict], **kwargs):
        """带工具表的对话：返回完整的 Choice 对象（调用方检查 tool_calls）。

        核心概念：Function Calling 的返回约定
        - finish_reason == "tool_calls" → 模型想调工具：
          message.tool_calls 里有 [{id, function.name, function.arguments(JSON字符串)}]
          模型可能一次要求调多个工具（parallel tool calls）。
        - finish_reason == "stop" → 模型直接给出了最终文本回答。
        工具调用场景一律 temperature=0：参数必须精确，不需要创意。
        """
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return resp.choices[0]
