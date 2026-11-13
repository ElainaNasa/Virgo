"""Jev 结构化决策模型适配器。

Jev 不使用 Chat Completions，而是接收 ``state`` 和 keyed ``questions``，返回
typed answers。这里把现有模式、角色、模型路由请求转换为 choice 问题，再恢复
为 DecisionRouter 现有的 JSON 结构。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from virgo.router.mode_definitions import mode_choice_criteria
from virgo.types import ModelResponse, Usage, is_long_duration_tier


MEDIUM_BUDGET_COST = 6.0

DECISION_RECOMMENDATIONS = {
    "chainstop_select_recommendations": "对于链式协作（Mode Chain）中终止角色Chainstop的选择，前序Work Agent给出的判断具有一定的参照价值，具体是否停止还是要看Work Agent的工作结果与题目要求的相符程度，绝对不能因为Reviewer类似的角色给出所谓OK的判断就过早终止。",
    "model_decision_recommendation": "请注意在一次任务决策中模型选择的多样性，注意到你在大部分任务的模型决策中多次的、重复的选择了deepseek-v4.1-flash模型，这不符合我们系统构建的初衷。存在相似价格、能力的别家模型，存在更便宜但是适合本次任务的模型、存在符合预算限制效果更好的模型。",
    "mode_decision_recommendation": "允许在合适的条件下做出“探索性”的模式决策以增加模式的多样性。",
}


def append_decision_recommendations(state):
    """在 Jev 决策上下文中追加提示，保留原有字符串或列表，不重复追加。"""
    recommendations = state.setdefault("recommendations", {})
    if not isinstance(recommendations, dict):
        # 保留非标准的原推荐内容，再建立结构化推荐项。
        recommendations = {"previous_recommendations": recommendations}
        state["recommendations"] = recommendations
    for key, text in DECISION_RECOMMENDATIONS.items():
        # 兼容已有上下文将该项直接放在顶层的情况，原位置也保留新旧内容。
        containers = [recommendations]
        if key in state:
            containers.append(state)
        for container in containers:
            values = container[key] if isinstance(container.get(key), list) else [container[key]] if key in container else []
            values = list(values)
            if text not in values:
                values.append(text)
            container[key] = values



class JevProvider:
    def __init__(self, base_url: str, api_key: str, model: str = "jev-latest", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, *, system: str, messages: list[dict[str, str]], model: str, temperature: float = 0.2, response_format: str | None = "json_object", tools: list[dict[str, Any]] | None = None) -> ModelResponse:
        payload = self._request_payload(system, messages)
        started = time.perf_counter()
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Jev API 返回 HTTP {exc.code}：{body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Jev API 调用失败：{exc}") from exc
        answers = raw.get("answers", {})
        result = self._answers_to_router_json(payload["questions"], answers)
        usage_raw = raw.get("usage", {})
        return ModelResponse(
            content=json.dumps(result, ensure_ascii=False),
            raw=raw,
            usage=Usage(
                input_tokens=int(usage_raw.get("input_tokens", usage_raw.get("prompt_tokens", 0))),
                output_tokens=int(usage_raw.get("output_tokens", usage_raw.get("completion_tokens", 0))),
                cost=float(usage_raw.get("cost", 0.0) or 0.0),
                duration_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    def list_models(self) -> list[dict[str, Any]]:
        """只读验证当前密钥可访问的 Jev 模型，不记录或输出密钥。"""
        models_url = self.base_url.rsplit("/v1/", 1)[0] + "/v1/models" if "/v1/" in self.base_url else self.base_url.rstrip("/") + "/models"
        request = urllib.request.Request(models_url, headers={"Authorization": f"Bearer {self.api_key}"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Jev 模型查询返回 HTTP {exc.code}：{body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Jev 模型查询失败：{exc}") from exc
        models = raw.get("models", [])
        if not isinstance(models, list):
            raise RuntimeError("Jev 模型查询响应缺少 models 数组")
        return [item for item in models if isinstance(item, dict)]

    def _request_payload(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        raw_content = messages[-1].get("content", "") if messages else "{}"
        try:
            state = json.loads(raw_content)
        except json.JSONDecodeError:
            state = {"text": raw_content}
        append_decision_recommendations(state)
        target = state.get("_routing_target")
        candidate_field = {"mode": "candidate_modes", "next_mode": "candidate_modes", "roles": "candidate_roles", "model_id": "candidate_models"}.get(target)
        if not candidate_field:
            raise ValueError("Jev 路由请求缺少 candidate_modes、candidate_roles 或 candidate_models")
        candidates = state.get(candidate_field) or []
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Jev 路由请求缺少有效的 {candidate_field}")
        question_key = "role" if target == "roles" else target
        count = max(1, int(state.get("count", 1)))
        questions: dict[str, Any] = {}
        if question_key == "role" and count > 1:
            selected = set(state.get("selected_roles", []))
            for index in range(count):
                available = [str(item) for item in candidates if str(item) not in selected]
                questions[f"role_{index + 1}"] = self._choice_question(f"选择第 {index + 1} 个角色。{self._policy(state)}", available)
                if available:
                    selected.add(available[0])
        else:
            choices = [str(item) for item in candidates]
            criteria = mode_choice_criteria(choices) if question_key in {"mode", "next_mode"} else None
            questions[question_key] = self._choice_question(self._instruction(question_key, state), choices, criteria)
        return {"model": self.model, "state": json.dumps({"system": system, "routing_input": state}, ensure_ascii=False, default=str), "questions": questions}

    @staticmethod
    def _choice_question(instructions: str, candidates: list[str], criteria: dict[str, str] | None = None) -> dict[str, Any]:
        return {"type": "choice", "instructions": instructions, "criteria": criteria or {candidate: candidate for candidate in candidates}}

    @staticmethod
    def _policy(state: dict[str, Any]) -> str:
        policy = (
            "提示：为防止单次费用过高，每个工作 Agent 的工具调用总次数最多为 50 次。"
            "难度字段使用 1-100 整数：1-20 为简单直接任务，21-40 为常规少步骤任务，"
            "41-60 为多步骤或需工具验证任务，61-80 为复杂专业或多模块任务，81-100 为高不确定性研究或复杂系统任务。"
            "对可能长程或高难度的任务，若预算充足，应优先考虑多智能体协作拓扑。"
            "每次测试的总开销应优先落在推荐预算的 1/3 至 5/6 区间；"
            "若前序失败且实际开销明显低于该预期、剩余预算充足，应偏向选择能以更高费用提升结果质量的选项。"
        )
        initial_context = state.get("task", {}).get("initial_context", state.get("initial_context", {}))
        try:
            if int(initial_context.get("difficulty", 0)) >= 50:
                policy += "难度可能较高，谨慎选择Single模式，工具调用次数可能不够。"
        except (TypeError, ValueError):
            pass
        if is_long_duration_tier(initial_context.get("estimated_duration_minutes", ">15 & <30 minutes")):
            policy += "预计时长超过 30 分钟，属于长程任务；预算允许时优先考虑协作拓扑，避免选择 Single 模式。"
        budget = state.get("budget", {})
        total_cost = budget.get("total", {}).get("cost") if isinstance(budget, dict) else None
        try:
            high_budget = float(total_cost) >= MEDIUM_BUDGET_COST
        except (TypeError, ValueError):
            high_budget = False
        if high_budget:
            policy += (
                "费用预算达到 medium 或更高档位时，单题给定的尝试成本相当高，应偏向复杂拓扑（DAG、Tree 或 FullMesh）、"
                "更多角色和更强模型，以取得良好的工作效果，不要采用简单拓扑或过早停止。"
                "前序尝试表明，对于复杂问题和长程问题，简单拓扑以及只花费较少费用预算的过早停止效果很差，"
                "大部分情况下无法正确解决问题。"
            )
        return policy

    @staticmethod
    def _instruction(question_key: str, state: dict[str, Any]) -> str:
        policy = JevProvider._policy(state)
        if question_key == "mode":
            return (
                "逐项比对候选项中的协作定义、任务结构、预算和前序模式后选择最合适的模式。候选排列顺序不代表偏好。"
                "Single 模式中的单个智能体节点工具调用次数限制为 30 次；对于长程任务，尽量不要选择 Single 模式。"
                + policy
            )
        if question_key == "next_mode":
            return "评价当前协作结果，逐项比对候选项中的协作定义后选择下一步模式；候选排列顺序不代表偏好。仅在当前阶段已完成且无需继续时选择 Stop。" + policy
        if question_key == "role":
            return "根据当前任务阶段和上下文选择最合适的角色。" + policy
        return "根据角色能力、任务类型、模型能力和预算选择最合适的模型。" + policy

    @staticmethod
    def _answers_to_router_json(questions: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
        if "mode" in questions or "next_mode" in questions or "model_id" in questions:
            key = next(iter(questions))
            answer = answers.get(key, {})
            value = answer.get("choice") if isinstance(answer, dict) else answer
            return {key: value, "reason": "Jev 结构化决策结果。"}
        roles: list[str] = []
        for key in questions:
            answer = answers.get(key, {})
            value = answer.get("choice") if isinstance(answer, dict) else answer
            if value:
                roles.append(str(value))
        return {"roles": roles, "reason": "Jev 结构化决策结果。"}
