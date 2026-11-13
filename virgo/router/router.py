"""三级决策路由器。"""

from __future__ import annotations

import json
import re
from typing import Any

from virgo.models import ModelPool
from virgo.providers.base import ModelProvider
from virgo.router.prompts import INITIAL_CONTEXT_PROMPT, MODE_ROUTER_PROMPT, MODEL_ROUTER_PROMPT, POST_MODE_PROMPT, ROLE_ROUTER_PROMPT
from virgo.types import InitialContext, Usage


class DecisionRouter:
    def __init__(self, provider: ModelProvider, model_pool: ModelPool, model_id: str = "deepseek-v4.1-flash", initial_context_provider: ModelProvider | None = None, initial_context_model: str | None = None, fallback_provider: ModelProvider | None = None, fallback_model: str | None = None) -> None:
        self.provider = provider
        self.model_pool = model_pool
        self.model_id = model_id
        self.initial_context_provider = initial_context_provider or provider
        self.initial_context_model = initial_context_model or model_id
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model

    def initial_context(self, task: str, constraints: list[str]) -> tuple[InitialContext, dict[str, Any]]:
        messages = [{"role": "user", "content": f"原始任务：{task}\n约束：{constraints}"}]
        response = self.initial_context_provider.complete(system=INITIAL_CONTEXT_PROMPT, messages=messages, model=self.initial_context_model)
        try:
            data = self._json(response.content)
            context = InitialContext.from_dict(data)
        except ValueError as exc:
            # JSON 不合法或字段校验失败均重试一次；上下文构建任务不能因缺字段
            # 直接终止整个 Agent 任务。
            messages.append({"role": "user", "content": f"上一条 InitialContext 响应无效：{exc}。不要调用工具、不要输出 tool_call、function 或 DSML 标记。只返回一个合法 JSON 对象，且必须包含 task_description、tasktype、domains、difficulty（1-100 整数）、estimated_duration_minutes、recommended_budget、decompose、constraints。"})
            repaired = self.initial_context_provider.complete(system=INITIAL_CONTEXT_PROMPT, messages=messages, model=self.initial_context_model, tools=None)
            response.usage = Usage(
                input_tokens=response.usage.input_tokens + repaired.usage.input_tokens,
                output_tokens=response.usage.output_tokens + repaired.usage.output_tokens,
                cost=response.usage.cost + repaired.usage.cost,
                calls=response.usage.calls + repaired.usage.calls,
                duration_ms=response.usage.duration_ms + repaired.usage.duration_ms,
            )
            data = self._json(repaired.content)
            context = InitialContext.from_dict(data)
        return context, {"input": task, "output": data, "usage": response.usage.as_dict(), "initial_context_model": self.initial_context_model}

    def select_mode(self, payload: dict[str, Any], candidates: list[str]) -> tuple[str, dict[str, Any]]:
        result = self._call(MODE_ROUTER_PROMPT, {**payload, "candidate_modes": list(candidates), "_routing_target": "mode"})
        mode = result.get("mode", "Single")
        return (mode if mode in candidates else candidates[0]), result

    def select_roles(self, payload: dict[str, Any], candidates: list[str], count: int = 1) -> tuple[list[str], dict[str, Any]]:
        result = self._call(ROLE_ROUTER_PROMPT, {**payload, "candidate_roles": list(candidates), "count": count, "_routing_target": "roles"})
        roles = [role for role in result.get("roles", []) if role in candidates]
        return (roles[:count] or [candidates[0]], result)

    def select_model(self, payload: dict[str, Any], candidates: list[str]) -> tuple[str, dict[str, Any]]:
        task_id = payload.get('task_id') or (payload.get('task') or {}).get('task_id')
        blocked = self.model_pool.channel_failures(task_id)
        candidates = [m for m in candidates if m == 'Stop' or self.model_pool.channel_available(task_id, m)]
        if not candidates:
            raise RuntimeError('本任务所有候选模型渠道已失败，停止重复调用')
        payload = {**payload, 'failed_model_channels': blocked,
                   'excluded_models': self.model_pool.excluded_models(task_id)}
        result = self._call(MODEL_ROUTER_PROMPT, {**payload, "candidate_models": list(candidates), "model_tables": self.model_pool.as_tables(), "_routing_target": "model_id"})
        proposed_model = result.get("model_id", candidates[0])
        model_id = proposed_model if proposed_model in candidates else candidates[0]
        result["proposed_model_id"] = proposed_model
        result["model_id"] = model_id
        return model_id, result

    def select_next_mode(self, payload: dict[str, Any], candidates: list[str]) -> tuple[str, dict[str, Any]]:
        result = self._call(POST_MODE_PROMPT, {**payload, "candidate_modes": list(candidates), "_routing_target": "next_mode"})
        mode = result.get("next_mode", "Stop")
        return (mode if mode in candidates or mode == "Stop" else "Stop"), result

    def _call(self, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
        active_model = self.model_id
        fallback: dict[str, str] | None = None
        try:
            response = self.provider.complete(system=prompt, messages=messages, model=active_model)
        except Exception as exc:
            if not self.fallback_provider or not self.fallback_model:
                raise
            fallback = {"from_model": self.model_id, "to_model": self.fallback_model, "error_type": type(exc).__name__, "error": str(exc)}
            active_model = self.fallback_model
            response = self.fallback_provider.complete(system=prompt, messages=messages, model=active_model)
        result = self._json(response.content)
        result["_usage"] = response.usage.as_dict()
        result["_router_model"] = active_model
        result["_router_input"] = payload
        if fallback:
            result["_router_fallback"] = fallback
        return result

    @staticmethod
    def _json(value: str) -> dict[str, Any]:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", value, flags=re.S)
            if match:
                return json.loads(match.group(0))
        raise ValueError(f"路由器未返回合法 JSON：{value[:300]}")
