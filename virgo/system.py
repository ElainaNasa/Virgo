"""Virgo 实验系统总装配与任务生命周期。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virgo.config import Settings
from virgo.context import ContextStore, ContextSystem
from virgo.controller import BudgetLedger
from virgo.controller.agent import AgentExecutor
from virgo.models import ModelPool
from virgo.roles import expand_domain_specialists
from virgo.observability import ExecutionLogger
from virgo.providers import build_execution_provider, build_initial_context_provider, build_router_provider
from virgo.router import DecisionRouter
from virgo.topology import DAGExecutor, FullMeshExecutor, ParallelExecutor, SingleExecutor, TreeExecutor, ChainExecutor
from virgo.topology.chain import CHAIN_WORKER_ROLE_CANDIDATES, MIN_CHAIN_NODES
from virgo.tools import ToolExecutor, ToolRegistry, register_builtin_tools
from virgo.trace import TraceRecorder
from virgo.types import InitialContext, TaskRequest, Usage


ROLE_CANDIDATES = {
    "Single": ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "DomainSpecialist", "ReportWriter"],
    "Chain": ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "ToolOperator", "DomainSpecialist", "Reviewer", "ReportWriter", "ChainStop"],
    "Parallel": ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "DomainSpecialist", "Reviewer", "Synthesizer"],
    "Tree": ["Coordinator", "SubtaskCoordinator"],
    "DAG": ["SpecialPlanner"],
    "FullMesh": ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "DomainSpecialist", "Reviewer", "Synthesizer", "FullMeshStop"],
}
TOP_LEVEL_MODE_CANDIDATES = ["Chain", "Parallel", "Tree", "DAG", "FullMesh"]
POST_MODE_CANDIDATES = {
    # 兼容历史任务与 ModelEscalation 的内部 Single 执行器；它不是顶层候选。
    "Single": ["Parallel", "FullMesh", "ModelEscalation", "Stop"],
    "Chain": ["Parallel", "FullMesh", "Stop"],
    "Parallel": ["Single", "FullMesh", "Stop"],
    "Tree": ["Single", "Parallel", "FullMesh", "Stop"],
    "DAG": ["Single", "Chain", "FullMesh", "Stop"],
    "FullMesh": ["Single", "Parallel", "Stop"],
}
MAX_TOP_LEVEL_STAGES = 3


@dataclass
class RunResult:
    task_id: str
    status: str
    output: dict[str, Any]
    snapshot: dict[str, Any]


class VirgoSystem:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        self.execution_log = ExecutionLogger(self.settings.log_dir or self.settings.state_dir / "logs", level=self.settings.log_level, console=self.settings.log_console)
        self.execution_log.emit("system_initialized", status="ready", workspace=str(self.settings.workspace), state_dir=str(self.settings.state_dir), offline=self.settings.offline)
        group = self.settings.storage_group
        if group is not None and (not group or not all(char.isalnum() or char in "_-" for char in group)):
            raise ValueError("storage_group 只能包含字母、数字、下划线和连字符")
        state_dir = self.settings.state_dir / group if group else self.settings.state_dir
        trace_dir = self.settings.state_dir / "traces"
        if group:
            trace_dir = trace_dir / group
        run_id = self.settings.storage_run_id
        if run_id is not None:
            if not group or not run_id or not all(char.isalnum() or char in "_-" for char in run_id):
                raise ValueError("storage_run_id 需要 storage_group，且只能包含字母、数字、下划线和连字符")
            state_dir = state_dir / run_id
            trace_dir = trace_dir / run_id
        self.store = ContextStore(state_dir, logger=self.execution_log)
        self.context = ContextSystem(self.store, collaboration_tool_call_limit=self.settings.collaboration_tool_call_limit)
        self.context.workspace = self.settings.workspace
        self.context.workspace_excluded_paths = [self.settings.state_dir]
        if self.settings.log_dir:
            self.context.workspace_excluded_paths.append(self.settings.log_dir)
        self.model_pool = ModelPool(deepseek_v4_flash_pricing=self.settings.deepseek_v4_flash_pricing)
        self.trace = TraceRecorder(trace_dir, logger=self.execution_log)
        self.ledger = BudgetLedger(self.store, logger=self.execution_log)
        self.tools = register_builtin_tools(ToolRegistry())
        self.tool_executor = ToolExecutor(self.tools, str(self.settings.workspace), self.trace, benchmark_task_loader=(lambda task_id: self.store.get(task_id)["task"]["original_task"]) if self.settings.storage_group else None, workspace_resolver=self.context.tool_workspace_for_node)
        self.provider = build_execution_provider(self.settings)
        self.router_provider = build_router_provider(self.settings)
        self.initial_context_provider = build_initial_context_provider(self.settings)
        execution_model = self.settings.execution_model or self.settings.local_model
        router_mode = self.settings.router_provider
        router_model = self.settings.router_model if router_mode in {"cloud", "openai", "openai_compatible", "openrouter", "deepseek", "zhipu", "minimax", "qwen", "mimo", "moonshot"} else (self.settings.jev_model if router_mode == "jev" else (self.settings.local_model if router_mode == "local" or (router_mode == "auto" and not self.settings.offline) else "deepseek-v4.1-flash"))
        router_fallback_provider = self.provider if router_mode == "jev" else None
        router_fallback_model = execution_model if router_mode == "jev" else None
        self.router = DecisionRouter(self.router_provider, self.model_pool, router_model, initial_context_provider=self.initial_context_provider, initial_context_model=self.settings.initial_context_model or execution_model, fallback_provider=router_fallback_provider, fallback_model=router_fallback_model)
        self.execution_log.emit("providers_configured", status="ready", execution_provider=type(self.provider).__name__, execution_model=execution_model, execution_mode=self.settings.execution_provider, initial_context_provider=type(self.initial_context_provider).__name__, initial_context_model=self.settings.initial_context_model or execution_model, initial_context_mode=self.settings.initial_context_provider, router_provider=type(self.router_provider).__name__, router_model=self.router.model_id, router_mode=self.settings.router_provider, router_fallback_model=router_fallback_model)
        self.agents = AgentExecutor(
            self.context,
            self.ledger,
            self.provider,
            self.model_pool,
            self.trace,
            self.tools,
            self.tool_executor,
            execution_model=execution_model,
            max_tool_calls=self.settings.max_tool_calls,
            context_compaction_threshold_tokens=self.settings.context_compaction_threshold_tokens,
        )

    def submit(self, request: TaskRequest) -> RunResult:
        snapshot = self.context.create_task(request)
        task_id = snapshot["task"]["task_id"]
        self.execution_log.emit("task_started", task_id=task_id, status="running", budget_tier=request.budget_tier)
        self.trace.record(task_id, "task_created", context={"task": snapshot["task"], "budget": snapshot["budget"]}, budget=snapshot["budget"])
        if request.initial_context is not None:
            initial = request.initial_context if isinstance(request.initial_context, InitialContext) else InitialContext.from_dict(request.initial_context)
            decision = None
            self.trace.record(
                task_id,
                "initial_context_reused",
                context={"task": request.task, "failed_trace_attempts": len(request.failed_decision_trace)},
                output=initial.as_dict(),
                budget=snapshot["budget"],
            )
            self.execution_log.emit("initial_context_reused", task_id=task_id, status="reused", failed_trace_attempts=len(request.failed_decision_trace))
        else:
            with self.execution_log.stage("initial_context", task_id=task_id):
                initial, decision = self.router.initial_context(request.task, list(request.constraints))
        self.context.set_initial_context(task_id, initial)
        budget_policy = self.context.adopt_recommended_budget_tier(
            task_id,
            request.budget_tier,
            initial.recommended_budget,
            has_explicit_cost_limit=request.budget_cost is not None,
        )
        if budget_policy:
            self.trace.record(task_id, "budget_tier_adopted", context=budget_policy, output={"status": "applied", "reason": "InitialContext 推荐预算低于用户给定档位。"}, budget=budget_policy["budget"])
            self.execution_log.emit("budget_tier_adopted", task_id=task_id, status="applied", **{key: value for key, value in budget_policy.items() if key != "budget"})
        if decision is not None:
            self._charge_router(task_id, decision.get("usage", {}))
            self.trace.record(task_id, "initial_context_decision", context={"task": request.task, "constraints": request.constraints}, output=initial.as_dict(), budget=self.store.get(task_id)["budget"], usage=decision.get("usage"))
        initial_mode_candidates, excluded_initial_modes = self._retry_initial_mode_candidates(request.failed_decision_trace)
        initial_mode_input = self.context.route_context(task_id, "mode", candidate_modes=initial_mode_candidates)
        initial_mode_input.update(
            {
                "retry_initial_mode_exclusions": excluded_initial_modes,
                "retry_initial_mode_rule": "上一失败尝试的初始宏模式不参与本次首次模式选择。",
            }
        )
        with self.execution_log.stage("top_level_mode_routing", task_id=task_id):
            current_mode, mode_decision = self.router.select_mode(initial_mode_input, initial_mode_candidates)
        self.trace.record(task_id, "mode_decision", context=self._router_input(mode_decision, initial_mode_input), output=mode_decision, budget=self.store.get(task_id)["budget"])
        self._charge_router(task_id, mode_decision.get("_usage", {}))
        source_result_ids: list[str] = []
        outputs: list[str] = []
        terminal_status: str | None = None
        last_completed_mode: str | None = None
        last_roles: list[str] = []
        last_selected_model: str | None = None
        for stage in range(MAX_TOP_LEVEL_STAGES):
            is_model_escalation = False
            if stage > 0:
                completed_mode = last_completed_mode or current_mode
                post_input = self.context.routing_context(task_id, focused_result_ids=source_result_ids)
                post_input.update({
                    "current_mode": completed_mode,
                    "previous_output": outputs[-1] if outputs else "",
                    "previous_result_ids": source_result_ids[-1:] if source_result_ids else [],
                    "candidate_modes": POST_MODE_CANDIDATES[completed_mode],
                    "remaining_budget": self.store.get(task_id)["budget"]["remaining"],
                })
                next_mode, mode_decision = self.router.select_next_mode(post_input, POST_MODE_CANDIDATES[completed_mode])
                self.trace.record(task_id, "post_mode_decision", context=self._router_input(mode_decision, post_input), output=mode_decision, budget=self.store.get(task_id)["budget"])
                self._charge_router(task_id, mode_decision.get("_usage", {}))
                if next_mode == "Stop":
                    break
                is_model_escalation = next_mode == "ModelEscalation"
                current_mode = "Single" if is_model_escalation else next_mode
            if is_model_escalation:
                roles = last_roles or ["MathSpecialist"]
                escalation_input = {"task_id": task_id, "source_mode": last_completed_mode, "source_model": last_selected_model, "reused_roles": roles, "remaining_budget": self.store.get(task_id)["budget"]["remaining"]}
                self.trace.record(task_id, "model_escalation_started", context=escalation_input, output={"status": "running"}, budget=self.store.get(task_id)["budget"])
                model_candidates = self.model_pool.escalation_candidates(last_selected_model, initial.tasktype)
            else:
                roles, role_decision = self._select_roles(task_id, current_mode)
                roles = expand_domain_specialists(roles, initial.domains)
                if roles != role_decision.get("roles"):
                    role_decision = {**role_decision, "roles": roles, "domain_instances": [role for role in roles if role.startswith("DomainSpecialist[")]}
                self.trace.record(task_id, "role_decision", context=self._router_input(role_decision, {"mode": current_mode, "candidate_roles": ROLE_CANDIDATES[current_mode]}), output=role_decision, budget=self.store.get(task_id)["budget"])
                self._charge_router(task_id, role_decision.get("_usage", {}))
                model_candidates = [model.model_id for model in self.model_pool.candidates(initial.recommended_budget, initial.tasktype)]
            model_input = self.context.route_context(task_id, "model", mode=current_mode, roles=roles)
            if is_model_escalation:
                model_input.update({"model_escalation": True, "previous_selected_model": last_selected_model})
            if current_mode == "Parallel" and not is_model_escalation:
                # Parallel 在拓扑内部为同一角色路由多个模型，这里跳过冗余的顶层单模型选择。
                selected_model = model_candidates[0]
            else:
                selected_model, model_decision = self.router.select_model(model_input, model_candidates)
                model_event = "model_escalation_model_decision" if is_model_escalation else "model_decision"
                self.trace.record(task_id, model_event, context=self._router_input(model_decision, model_input), output=model_decision, budget=self.store.get(task_id)["budget"])
                self._charge_router(task_id, model_decision.get("_usage", {}))
            mode_id = self.context.create_mode(
                task_id,
                current_mode,
                decision_mode="ModelEscalation" if is_model_escalation else current_mode,
            )
            with self.execution_log.stage("topology_execution", task_id=task_id, mode=current_mode, mode_instance_id=mode_id):
                executor = self._topology(current_mode, selected_model)
                previous = self.context.upstream_results(self.store.get(task_id), source_result_ids)
                executor._initial_result_ids = [item["result_id"] for item in previous]
                executor._initial_node_ids = [item["node_id"] for item in previous]
                result = executor.run(task_id, mode_id, roles)
            self.context.finish_mode(task_id, mode_id, result)
            source_result_ids.extend(result.result_ids)
            outputs.append(result.output)
            last_completed_mode = current_mode
            last_roles = roles
            last_selected_model = selected_model
            if is_model_escalation:
                self.trace.record(task_id, "model_escalation_completed", context={"source_model": model_input.get("previous_selected_model"), "selected_model": selected_model, "mode_instance_id": mode_id}, output={"status": result.status, "result_ids": result.result_ids}, budget=self.store.get(task_id)["budget"])
            if result.status not in {"completed", "partial"}:
                terminal_status = result.status
                break
            if result.status == "partial":
                terminal_status = "partial"
        status = terminal_status or ("completed" if outputs else "blocked")
        output = outputs[-1] if outputs else "任务未产生输出。"
        snapshot = self.context.finish_task(task_id, status, output, source_result_ids)
        output = snapshot["state"]
        self.trace.record(task_id, "task_completed", context={"task_id": task_id}, output={"status": status, "task_output": output, "source_result_ids": source_result_ids}, budget=self.store.get(task_id)["budget"])
        self.execution_log.emit("task_finished", task_id=task_id, status=status, result_count=len(source_result_ids), remaining_budget=self.store.get(task_id)["budget"]["remaining"])
        return RunResult(task_id, status, output, snapshot)

    def _select_roles(self, task_id: str, mode: str) -> tuple[list[str], dict[str, Any]]:
        candidates = ROLE_CANDIDATES[mode]
        if mode == "Parallel":
            candidates = [role for role in candidates if role != "Synthesizer"]
        if mode == "Single":
            count = 1
        elif mode == "Parallel":
            count = 1
        elif mode == "FullMesh":
            count = 1
        elif mode == "Chain":
            payload = self.context.route_context(task_id, "role", mode=mode, roles=CHAIN_WORKER_ROLE_CANDIDATES)
            payload.update({"chain_phase": "initial", "minimum_chain_nodes": MIN_CHAIN_NODES, "chain_stop_allowed": False})
            roles, decision = self.router.select_roles(payload, CHAIN_WORKER_ROLE_CANDIDATES, 1)
            selected = roles[0] if roles and roles[0] in CHAIN_WORKER_ROLE_CANDIDATES else CHAIN_WORKER_ROLE_CANDIDATES[0]
            if roles != [selected]:
                decision = {
                    **decision,
                    "roles": [selected],
                    "constraint_override": {
                        "rule": "minimum_chain_nodes",
                        "minimum": MIN_CHAIN_NODES,
                        "completed_nodes": 0,
                        "rejected_roles": roles,
                    },
                }
            return [selected], decision
        elif mode == "Tree":
            return self.router.select_roles(self.context.route_context(task_id, "role", mode=mode, roles=["Coordinator"]), ["Coordinator"], 1)
        elif mode == "DAG":
            return self.router.select_roles(self.context.route_context(task_id, "role", mode=mode, roles=["SpecialPlanner"]), ["SpecialPlanner"], 1)
        return self.router.select_roles(self.context.route_context(task_id, "role", mode=mode, roles=candidates), candidates, count)

    def _topology(self, mode: str, model_id: str) -> Any:
        classes = {"Single": SingleExecutor, "Chain": ChainExecutor, "Parallel": ParallelExecutor, "Tree": TreeExecutor, "DAG": DAGExecutor, "FullMesh": FullMeshExecutor}
        return classes[mode](self.context, self.agents, self.router, model_id)

    @staticmethod
    def _retry_initial_mode_candidates(failed_trace: tuple[dict[str, Any], ...]) -> tuple[list[str], list[str]]:
        """排除先前失败尝试的首个宏模式，促使重试探索其他拓扑。"""
        excluded: list[str] = []
        for attempt in failed_trace:
            mode_runs = attempt.get("mode_runs")
            if not isinstance(mode_runs, list) or not mode_runs or not isinstance(mode_runs[0], dict):
                continue
            mode = mode_runs[0].get("mode")
            if isinstance(mode, str) and mode in TOP_LEVEL_MODE_CANDIDATES and mode not in excluded:
                excluded.append(mode)
        candidates = [mode for mode in TOP_LEVEL_MODE_CANDIDATES if mode not in excluded]
        # 所有拓扑都曾失败时重新开放完整集合，避免重试失去路由入口。
        return (candidates or list(TOP_LEVEL_MODE_CANDIDATES), excluded)

    @staticmethod
    def _router_input(decision: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        """轨迹记录模型实际收到的路由载荷，而不是调用后的派生状态。"""
        return dict(decision.get("_router_input") or fallback)

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.store.get(task_id)

    def get_trace(self, task_id: str) -> dict[str, Any]:
        return self.trace.summary(task_id)

    def get_runtime_log(self, task_id: str | None = None) -> list[dict[str, Any]]:
        return self.trace.read_runtime_log(task_id)

    def _charge_router(self, task_id: str, usage: dict[str, Any]) -> None:
        if not usage:
            return
        model_id = self.router.model_id if self.router.model_id in {item.model_id for item in self.model_pool.all()} else "deepseek-v4.1-flash"
        model = self.model_pool.get(model_id)
        value = Usage(input_tokens=int(usage.get("input_tokens", 0)), output_tokens=int(usage.get("output_tokens", 0)), calls=int(usage.get("calls", 1)), duration_ms=int(usage.get("duration_ms", 0)))
        reported_cost = float(usage.get("cost", 0.0) or 0.0)
        value.cost = reported_cost or self.model_pool.estimate_cost(model.model_id, value.input_tokens, value.output_tokens)
        if model_id != self.router.model_id:
            self.execution_log.emit("router_cost_fallback", level="WARNING", task_id=task_id, status="estimated_with_fallback", router_model=self.router.model_id, billing_model=model_id)
        try:
            self.ledger.reserve(task_id, 0.0)
            self.ledger.settle(task_id, value, 0.0)
        except Exception:
            # 当前路由调用已经发生，保留已记录的调用事实；后续执行器会阻止新的调用。
            return
