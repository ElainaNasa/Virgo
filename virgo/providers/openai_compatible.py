"""OpenAI 兼容的本地模型 HTTP 适配器，仅使用标准库。"""

from __future__ import annotations

import json
import threading
from collections import deque
from contextlib import contextmanager
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any

from virgo.providers.protocol import is_dsml_tool_protocol_residue
from virgo.types import ModelResponse, Usage


# 组织并发额度为 1：进程内所有适配器实例共用队列，覆盖各拓扑及不同密钥。
_KIMI_CONDITION = threading.Condition()
_KIMI_QUEUE = deque()


@contextmanager
def _kimi_request_slot():
    ticket = object()
    with _KIMI_CONDITION:
        _KIMI_QUEUE.append(ticket)
        try:
            _KIMI_CONDITION.wait_for(lambda: _KIMI_QUEUE[0] is ticket)
        except BaseException:
            _KIMI_QUEUE.remove(ticket)
            _KIMI_CONDITION.notify_all()
            raise
    try:
        yield
    finally:
        with _KIMI_CONDITION:
            _KIMI_QUEUE.remove(ticket)
            _KIMI_CONDITION.notify_all()


class OpenAICompatibleProvider:
    work_throttle_at_transport = True
    def __init__(self, base_url: str, api_key: str | None, timeout: float = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, *, system: str, messages: list[dict[str, Any]], model: str, temperature: float = 0.2, response_format: str | None = "json_object", tools: list[dict[str, Any]] | None = None) -> ModelResponse:
        kwargs = dict(system=system, messages=messages, model=model, temperature=temperature,
                      response_format=response_format, tools=tools)
        if urlparse(self.base_url).hostname in {"api.moonshot.cn", "api.moonshot.ai"}:
            with _kimi_request_slot():
                return self._complete(**kwargs)
        return self._complete(**kwargs)

    @staticmethod
    def _read_stream(response):
        """聚合 SSE 增量，保留工具调用和用量；中断流不得作为完整回复交付。"""
        content, reasoning = [], []
        tools = {}
        usage = {}
        finish_reason = None
        done = False
        event_lines = []

        def consume(data):
            nonlocal usage, finish_reason, done
            if data == "[DONE]":
                done = True
                return
            chunk = json.loads(data)
            if not isinstance(chunk, dict) or chunk.get("error"):
                raise RuntimeError("Kimi 流式接口返回错误对象")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for part in delta.get("tool_calls", []):
                    index = part["index"]
                    tool = tools.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if part.get("id"):
                        tool["id"] = part["id"]
                    if part.get("type"):
                        tool["type"] = part["type"]
                    function = part.get("function") or {}
                    if function.get("name"):
                        tool["function"]["name"] = function["name"]
                    tool["function"]["arguments"] += function.get("arguments") or ""

        for line in response:
            line = line.decode("utf-8").rstrip("\r\n")
            if not line:
                if event_lines:
                    consume("\n".join(event_lines))
                    event_lines.clear()
                if done:
                    break
            elif line.startswith("data:"):
                event_lines.append(line[5:].lstrip(" "))
        if event_lines and not done:
            consume("\n".join(event_lines))
        if not done or finish_reason is None:
            raise ConnectionError("Kimi 流式连接提前结束，未收到完整结束标记")
        message = {"content": "".join(content), "tool_calls": [tools[i] for i in sorted(tools)]}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        return {"choices": [{"message": message, "finish_reason": finish_reason}], "usage": usage}

    def _complete(self, *, system, messages, model, temperature, response_format, tools):
        # 已进入Kimi队列槽位后再限速，避免排队出队时请求挤在一起。
        from virgo.providers.work_request_control import throttle_work_transport
        throttle_work_transport()
        # K3.1 Pro 服务端固定只接受 temperature=1。
        if model == "kimi-k3.1-pro":
            temperature = 1
        # 模型池保留版本标识；仅在 DeepSeek 官方接口使用官方模型名称。
        if urlparse(self.base_url).hostname == "api.deepseek.com" and model == "deepseek-v4.1-flash":
            model = "deepseek-flash"
        payload: dict[str, Any] = {"model": model, "messages": [{"role": "system", "content": system}, *messages], "temperature": temperature}
        # Kimi 工具阶段使用原生函数协议；仅在无工具交付阶段约束 JSON 正文。
        if response_format and not (model.startswith("kimi-") and tools):
            payload["response_format"] = {"type": response_format}
        if tools:
            payload["tools"] = tools
            if model.startswith("kimi-"):
                payload["tool_choice"] = "auto"
        streaming = model.startswith("kimi-")
        if streaming:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        headers = {"Content-Type": "application/json"}
        if streaming:
            headers["Accept"] = "text/event-stream"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(), headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = self._read_stream(response) if streaming else json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"OpenAI 兼容服务返回 HTTP {exc.code}：{body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenAI 兼容模型服务调用失败：{exc}") from exc
        if not isinstance(raw, dict) or raw.get("error"):
            raise RuntimeError("Chat Completions 返回错误对象，未产生有效回答")
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("Chat Completions 响应缺少有效 choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Chat Completions 响应缺少 message")
        usage_raw = raw.get("usage", {})
        content = message.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict) and part.get("type") in {"text", "output_text"})
        if not isinstance(content, str):
            raise RuntimeError("Chat Completions content 类型不受支持")
        if message.get("refusal") and not content:
            raise RuntimeError("Chat Completions 模型拒绝了当前请求")
        if is_dsml_tool_protocol_residue(content):
            # 该内容是模型服务泄漏的工具协议，不能作为 Agent 最终交付物。
            raw = {**raw, "_virgo_protocol_residue": "dsml_tool_call"}
            content = ""
        return ModelResponse(
            content=content,
            tool_calls=message.get("tool_calls", []) or [],
            raw=raw,
            usage=Usage(
                input_tokens=int(usage_raw.get("prompt_tokens", 0)),
                output_tokens=int(usage_raw.get("completion_tokens", 0)),
                duration_ms=int((time.perf_counter() - started) * 1000),
            ),
        )
