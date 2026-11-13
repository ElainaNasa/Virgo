"""工具注册表、权限和统一返回结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    executor: Callable[[dict[str, Any], str], dict[str, Any]]
    read_only: bool = True
    max_output_chars: int = 12000
    timeout_seconds: float = 30.0

    def as_model_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.tool_id in self._tools:
            raise ValueError(f"工具重复注册：{spec.tool_id}")
        self._tools[spec.tool_id] = spec

    def get(self, tool_id: str) -> ToolSpec:
        if tool_id not in self._tools:
            raise KeyError(f"未注册工具：{tool_id}")
        return self._tools[tool_id]

    def list(self, allowed: list[str] | None = None) -> list[ToolSpec]:
        return [tool for tool_id, tool in self._tools.items() if allowed is None or tool_id in allowed]

    def schemas(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        return [tool.as_model_schema() for tool in self.list(allowed)]
