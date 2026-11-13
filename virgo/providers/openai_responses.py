"""OpenAI Responses API 适配器，用于 InferAI 等 Responses 兼容服务。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from virgo.providers.protocol import is_dsml_tool_protocol_residue
from virgo.types import ModelResponse, Usage
from virgo.providers.openai_compatible import OpenAICompatibleProvider


class _ResponsesBadGateway(RuntimeError):
    """Responses 返回502，立即交给兼容协议降级。"""



class OpenAIResponsesProvider:
    def __init__(self, base_url: str, api_key: str, timeout: float = 60, max_retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def complete(self, *, system: str, messages: list[dict[str, Any]], model: str, temperature: float = 0.2, response_format: str | None = "json_object", tools: list[dict[str, Any]] | None = None) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "input": self._input_items(system, messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {}),
                }
                for tool in tools
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
            ]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        request = urllib.request.Request(f"{self.base_url}/responses", data=json.dumps(payload).encode(), headers=headers, method="POST")
        started = time.perf_counter()
        try:
            raw = self._request_with_retry(request)
        except _ResponsesBadGateway:
            # 原始消息（含工具结果）仍采用 Chat 格式，直接复用而非重建对话。
            fallback = OpenAICompatibleProvider(self.base_url, self.api_key, self.timeout)
            response = fallback.complete(system=system, messages=messages, model=model,
                                         temperature=temperature, response_format=response_format, tools=tools)
            response.raw = {**response.raw, "_virgo_protocol_fallback": {
                "from": "responses", "to": "chat_completions", "http_status": 502,
            }}
            response.usage.duration_ms = int((time.perf_counter() - started) * 1000)
            return response
        content, tool_calls = self._parse_output(raw)
        if is_dsml_tool_protocol_residue(content):
            raw = {**raw, "_virgo_protocol_residue": "dsml_tool_call"}
            content = ""
        usage_raw = raw.get("usage", {})
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=raw,
            usage=Usage(
                input_tokens=int(usage_raw.get("input_tokens", usage_raw.get("prompt_tokens", 0))),
                output_tokens=int(usage_raw.get("output_tokens", usage_raw.get("completion_tokens", 0))),
                duration_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    def _request_with_retry(self, request: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:1000]
                if exc.code == 502:
                    raise _ResponsesBadGateway("Responses 服务返回 HTTP 502") from exc
                last_error = RuntimeError(f"Responses 服务返回 HTTP {exc.code}：{body}")
                retryable = exc.code == 429 or 500 <= exc.code < 600
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = RuntimeError(f"Responses 模型服务调用失败：{exc}")
                retryable = True
            if not retryable or attempt >= self.max_retries:
                # 已在传输层完成重试，避免执行器再次放大请求次数。
                last_error.retries_exhausted = self.max_retries > 0 and retryable
                raise last_error
            time.sleep(2**attempt)
        raise last_error or RuntimeError("Responses 模型服务调用失败")

    @staticmethod
    def _input_items(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            if message.get("role") == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": message.get("content", "")})
                continue
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message["tool_calls"]:
                    function = call.get("function", call)
                    items.append({"type": "function_call", "call_id": call.get("id"), "name": function.get("name"), "arguments": function.get("arguments", "{}")})
                continue
            items.append({"role": message.get("role", "user"), "content": message.get("content", "")})
        return items

    @staticmethod
    def _parse_output(raw: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        text: list[str] = []
        calls: list[dict[str, Any]] = []
        for item in raw.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                calls.append({"id": item.get("call_id") or item.get("id"), "type": "function", "function": {"name": item.get("name"), "arguments": item.get("arguments", "{}")}})
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                        text.append(str(part.get("text", "")))
        return "\n".join(text) or str(raw.get("output_text", "")), calls
