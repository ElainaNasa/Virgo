"""拓扑执行器的公共能力。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from virgo.context.system import ContextSystem
from virgo.controller.agent import AgentExecutor
from virgo.router import DecisionRouter
from virgo.types import ModeResult
from virgo.types import Usage
from virgo.roles import expand_domain_specialists


RECOVERY_ROLE_CANDIDATES = [
    "MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst",
    "Researcher", "ToolOperator", "DomainSpecialist", "Reviewer", "ReportWriter",
]
STRUCTURAL_RECOVERY_ROLES = {"SpecialPlanner", "Coordinator", "SubtaskCoordinator", "Synthesizer"}


class TopologyExecutor(ABC):
    mode = ""

    def __init__(self, context: ContextSystem, agents: AgentExecutor, router: DecisionRouter, model_id: str = "deepseek-v4.1-flash") -> None:
        self.context = context
        self.agents = agents
        self.router = router
        self.model_id = model_id
        self._subtask_id = None
        self._subtask_context = None
        self._subtask_constraints = []
        self._initial_result_ids: list[str] = []
        self._initial_node_ids: list[str] = []
        self._max_agents = 10
        self._default_parent_node_id: str | None = None
        self._default_depth: int | None = None

    def configure_subtask(self, subtask_id: str, initial_context: dict[str, Any], constraints: list[str] | None = None, upstream_result_ids: list[str] | None = None, upstream_node_ids: list[str] | None = None, max_agents: int | None = None, parent_node_id: str | None = None, depth: int | None = None) -> None:
        """为嵌套模式绑定独立子任务上下文。"""
        self._subtask_id = subtask_id
        self._subtask_context = initial_context
        self._subtask_constraints = constraints or []
        self._initial_result_ids = upstream_result_ids or []
        self._initial_node_ids = upstream_node_ids or []
        self._default_parent_node_id = parent_node_id
        self._default_depth = depth
        if max_agents is not None:
            self._max_agents = max(1, max_agents)

    @abstractmethod
    def run(self, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult: ...

    def node(self, task_id: str, mode_instance_id: str, role: str, assignment: str, *, parent: str | None = None, depth: int | None = None, model_id: str | None = None, round: int = 0, subtask_id: str | None = None, initial_context: dict[str, Any] | None = None, constraints: list[str] | None = None) -> str:
        selected_model = model_id or self.model_id
        if not self.agents.model_pool.channel_available(task_id, selected_model):
            payload = self.routing_context(task_id, {
                "phase": "blocked_model_reselection", "selected_mode": self.mode,
                "selected_role": role, "blocked_model": selected_model,
                "instruction": "原模型已熔断或因连续无进展被排除，必须选择健康候选模型。",
            })
            candidates = [model.model_id for model in self.agents.model_pool.all()]
            selected_model, decision = self.router.select_model(payload, candidates)
            self.record_router_decision(task_id, "blocked_model_reselection", payload, decision)
        return self.context.add_node(task_id, mode_instance_id=mode_instance_id, role=role, assignment=assignment, parent_node_id=parent if parent is not None else self._default_parent_node_id, depth=depth if depth is not None else self._default_depth, model_id=selected_model, subtask_id=subtask_id if subtask_id is not None else self._subtask_id, initial_context=initial_context if initial_context is not None else self._subtask_context, constraints=constraints if constraints is not None else self._subtask_constraints, round=round)

    def call(self, task_id: str, node_id: str, *, settle_budget: bool = True) -> tuple[str | None, dict[str, Any] | str, str]:
        result_id, output = self.agents.run(task_id, node_id, settle_budget=settle_budget)
        return result_id, output.task_output, output.status

    def route_recovery(self, task_id: str, *, failed_node_id: str, failed_result_id: str | None, failed_role: str, failed_status: str) -> tuple[str, str]:
        """为失败节点选择一次替代角色与模型。"""
        snapshot = self.context.store.get(task_id)
        node = next(item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == failed_node_id)
        assignment = next((item for item in snapshot.get("agent_assignments", []) if item.get("assignment_id") == node.get("active_assignment_id")), {})
        initial = node.get("initial_context") or snapshot.get("initial_context") or {}
        preserve_role = self.mode == "Parallel" or failed_role in STRUCTURAL_RECOVERY_ROLES
        candidates = [failed_role] if preserve_role else ([role for role in RECOVERY_ROLE_CANDIDATES if role != failed_role] or list(RECOVERY_ROLE_CANDIDATES))
        payload = self.routing_context(
            task_id,
            {
                "task_id": task_id,
                "mode": self.mode,
                "phase": "agent_recovery",
                "failed_node_id": failed_node_id,
                "failed_role": failed_role,
                "failed_status": failed_status,
                "failed_model_error": self._recovery_model_error(task_id, failed_node_id, assignment),
                "candidate_roles": candidates,
                "remaining_budget": snapshot["budget"]["remaining"],
            },
            focused_result_ids=[failed_result_id] if failed_result_id else [],
        )
        roles, role_decision = self.router.select_roles(payload, candidates, 1)
        role = roles[0] if roles and roles[0] in candidates else candidates[0]
        role = failed_role if preserve_role else expand_domain_specialists([role], str(initial.get("domains", "")))[0]
        self.record_router_decision(task_id, "agent_recovery_role_decision", payload, role_decision)
        model_candidates = [model.model_id for model in self.agents.model_pool.candidates(initial.get("recommended_budget", "low"), initial.get("tasktype", "Other"))]
        model_payload = {**payload, "selected_role": role, "candidate_models": model_candidates}
        model, model_decision = self.router.select_model(model_payload, model_candidates)
        self.record_router_decision(task_id, "agent_recovery_model_decision", model_payload, model_decision)
        return role, model

    def _recovery_model_error(self, task_id: str, failed_node_id: str, assignment: dict[str, Any]) -> dict[str, Any] | None:
        """将模型接口故障显式交给恢复模型路由，而非依赖无结果节点的隐式上下文。"""
        trace = getattr(self.agents, "trace", None)
        events = trace.read(task_id) if trace and callable(getattr(trace, "read", None)) else []
        for event in reversed(events):
            if event.get("event_type") != "agent_call_failed" or event.get("node_id") != failed_node_id:
                continue
            output = event.get("output") if isinstance(event.get("output"), dict) else {}
            error = (output.get("runtime") or {}).get("error") or {}
            message = str(error.get("message") or "模型调用失败")
            return {
                "failure_kind": "model_api_error",
                "failed_model": assignment.get("llm", {}).get("model_id"),
                "error_type": message.split(":", 1)[0] or "RuntimeError",
                "message": message,
                "instruction": "前序节点在模型接口调用阶段失败。选择恢复模型时应考虑该错误，避免无理由重复使用同一不可用模型。",
            }
        return None

    def record_router_decision(self, task_id: str, event_type: str, context: dict[str, Any], decision: dict[str, Any]) -> None:
        """记录路由决策并结算该次本地路由模型调用。"""
        usage = decision.get("_usage", {})
        if usage:
            value = Usage(input_tokens=int(usage.get("input_tokens", 0)), output_tokens=int(usage.get("output_tokens", 0)), calls=int(usage.get("calls", 1)), duration_ms=int(usage.get("duration_ms", 0)))
            value.cost = self.agents.model_pool.estimate_cost("deepseek-v4.1-flash", value.input_tokens, value.output_tokens)
            try:
                self.agents.ledger.reserve(task_id, 0.0)
                self.agents.ledger.settle(task_id, value, 0.0)
            except Exception:
                pass
        self.agents.trace.record(task_id, event_type, context=decision.get("_router_input") or context, output=decision, budget=self.context.store.get(task_id)["budget"])

    def failed_decision_trace(self, task_id: str) -> list[dict[str, Any]]:
        """向嵌入拓扑的路由阶段提供此前测试失败的简化决策轨迹。"""
        return self.context.failed_decision_trace(task_id)

    def routing_context(self, task_id: str, payload: dict[str, Any], *, focused_result_ids: list[str] | None = None) -> dict[str, Any]:
        """在所有拓扑路由调用中注入统一的完整任务和结果上下文。"""
        context = self.context.routing_context(task_id, focused_result_ids=focused_result_ids)
        payload = dict(payload)
        payload.pop("task_id", None)
        if isinstance(payload.get("mode"), str):
            payload["selected_mode"] = payload.pop("mode")
        context.update(payload)
        return context

    @staticmethod
    def join(outputs: list[str]) -> str:
        return "\n\n".join(output for output in outputs if output)
