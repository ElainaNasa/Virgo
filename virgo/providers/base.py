"""模型服务适配器接口。"""

from __future__ import annotations

from typing import Any, Protocol

from virgo.types import ModelResponse


class ModelProvider(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.2,
        response_format: str | None = "json_object",
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse: ...
