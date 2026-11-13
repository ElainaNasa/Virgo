"""支持嵌入 Single、Chain、Parallel 的 DAG 拓扑。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from virgo.context.factory import default_limits
from virgo.roles import get_role
from virgo.types import DURATION_TIERS, InitialContext, ModeResult, Usage
from .chain import ChainExecutor
from .langgraph_runtime import END, START, GraphState, LangGraphTopologyExecutor, new_graph
from .single import SingleExecutor
from .parallel import ParallelExecutor, PARALLEL_ROLE_CANDIDATES


NESTED_MODES = {"Single", "Chain", "Parallel"}
NESTED_ROLE_CANDIDATES = {
    "Parallel": [role for role in PARALLEL_ROLE_CANDIDATES if role != "Synthesizer"],
    "Single": ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "DomainSpecialist", "ReportWriter"],
    "Chain": ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "ToolOperator", "DomainSpecialist", "Reviewer", "ReportWriter"],
}


class DAGExecutor(LangGraphTopologyExecutor):
    mode = "DAG"

    def run(self, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult:
        graph = new_graph()

        def planner(state: GraphState):
            return self._plan_with_retries(state)

        def execute_plan(state: GraphState):
            plan = state.get("plan", [])
            subtask_results = self._execute_plan(task_id, mode_instance_id, plan)
            result_ids = [result_id for item in subtask_results for result_id in item.get("result_ids", [])]
            return {"result_ids": result_ids, "subtask_results": subtask_results, "outputs": [item.get("output", "") for item in subtask_results], "statuses": [item.get("status", "blocked") for item in subtask_results]}

        def summarize(state: GraphState):
            updates, _, _, status = self.agent_state(
                state,
                "Synthesizer",
                "汇总 DAG 中所有嵌入模式的终端结果，明确报告失败或阻塞的子任务。",
                input_result_ids=state.get("result_ids", []),
                input_node_ids=state.get("node_ids", []),
            )
            updates["final_output"] = updates["outputs"][-1]
            updates["status"] = status
            return updates

        def route_after_planner(state: GraphState):
            return "execute_plan" if state.get("status") in {"running", "completed", "partial"} else END

        graph.add_node("planner", planner)
        graph.add_node("execute_plan", execute_plan)
        graph.add_node("summarize", summarize)
        graph.add_edge(START, "planner")
        graph.add_conditional_edges("planner", route_after_planner, {"execute_plan": "execute_plan", END: END})
        graph.add_edge("execute_plan", "summarize")
        graph.add_edge("summarize", END)
        return self.invoke(self.compile(graph), task_id, mode_instance_id, roles)

    def _plan_with_retries(self, state: GraphState) -> dict[str, Any]:
        """将规划格式错误反馈给模型，最多额外纠正两次。"""
        assignment = "规划 DAG 子任务、依赖关系、嵌入模式、角色、预期输出和验收标准。"
        requirements = (
            "必须返回完整、可解析的 JSON 规划，tasks 必须是非空对象数组；"
            "每个 initial_context 必须是对象，difficulty 必须是 1 到 100 的整数。"
            "estimated_duration_minutes 必须逐字选择以下枚举之一："
            + json.dumps(DURATION_TIERS, ensure_ascii=False)
            + "。禁止翻译 minutes、使用中文单位或自创时间区间。返回前必须逐字段检查，确保严格符合要求。"
        )
        updates: dict[str, Any] = {"result_ids": [], "node_ids": [], "outputs": [], "statuses": []}
        result_id = None
        node = None
        for attempt in range(3):
            current, result_id, node, status = self.agent_state(
                state, "SpecialPlanner", assignment + "\n" + requirements,
                input_result_ids=[result_id] if result_id else None,
                input_node_ids=[node] if node else None,
            )
            for key in updates:
                updates[key].extend(current.get(key, []))
            try:
                plan = self._read_plan(state["task_id"], result_id)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                error = str(exc)
                retry = attempt < 2 and self.agents.ledger.available(state["task_id"])
                self.agents.trace.record(
                    state["task_id"], "dag_plan_validation_failed",
                    context={"planner_result_id": result_id, "attempt": attempt + 1},
                    output={"error": error, "retrying": retry},
                )
                if retry:
                    assignment = "上一份规划未通过格式校验：" + error + "。请修正并重新返回完整规划，不要只解释错误。"
                    continue
                updates.update(status="failed", final_output="DAG 规划格式校验失败：" + error, plan=[])
                return updates
            updates.update(root_node_id=node, root_result_id=result_id, plan=plan,
                           planner_result_id=result_id, status=status)
            return updates
        return updates

    def _read_plan(self, task_id: str, planner_result_id: str | None) -> list[dict[str, Any]]:
        if planner_result_id:
            snapshot = self.context.store.get(task_id)
            result = snapshot.get("agent_outputs", {}).get(planner_result_id, {}).get("output", {})
            raw_plan = (result.get("plan") or result.get("tasks")) if isinstance(result, dict) else None
            if not raw_plan and isinstance(result, dict):
                try:
                    candidate = json.loads(result.get("task_output", ""))
                    raw_plan = candidate.get("tasks") or candidate.get("plan")
                except (json.JSONDecodeError, TypeError):
                    raw_plan = None
            if isinstance(raw_plan, list):
                plan = self._normalize_plan(raw_plan)
                if plan:
                    return plan
        raise ValueError("规划缺少非空 tasks/plan 数组，必须返回完整 JSON 规划")

    @staticmethod
    def _normalize_plan(raw_plan: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_plan, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"tasks[{index - 1}] 必须是对象")
            node_id = str(item.get("node_id") or f"dag:T{index}")
            if node_id in seen:
                continue
            seen.add(node_id)
            context = InitialContext.from_dict(item.get("initial_context", {"task_description": item.get("description", "子任务"), "tasktype": "Other", "domains": "General", "difficulty": 40, "estimated_duration_minutes": ">15 & <30 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"})).as_dict()
            normalized.append({"node_id": node_id, "description": str(item.get("description", context["task_description"])), "dependencies": [str(value) for value in item.get("dependencies", [])], "required_capability": str(item.get("required_capability", "Other")), "initial_context": context, "expected_output": str(item.get("expected_output", "完成子任务并提供结果。")), "acceptance_criteria": [str(value) for value in item.get("acceptance_criteria", ["生成可供上游使用的结果"])], "constraints": [str(value) for value in item.get("constraints", [])]})
        valid_ids = {item["node_id"] for item in normalized}
        for item in normalized:
            item["dependencies"] = [dependency for dependency in item["dependencies"] if dependency in valid_ids and dependency != item["node_id"]]
        return normalized

    def _execute_plan(self, task_id: str, parent_mode_instance_id: str, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending = {item["node_id"]: item for item in plan}
        finished: dict[str, dict[str, Any]] = {}
        all_results: list[dict[str, Any]] = []
        while pending:
            ready = [item for item in pending.values() if all(dependency in finished and finished[dependency]["status"] in {"completed", "partial"} for dependency in item["dependencies"])]
            if not ready:
                all_results.extend({"subtask_id": item["node_id"], "mode": item.get("mode", "not_started"), "status": "blocked", "output": "", "result_ids": [], "runtime_error": {"kind": "dependency", "message": "前置子任务失败或 DAG 依赖无法满足。"}} for item in pending.values())
                break
            with ThreadPoolExecutor(max_workers=min(3, len(ready))) as pool:
                futures = {pool.submit(self._run_nested_subtask, task_id, parent_mode_instance_id, item, [finished[dependency] for dependency in item["dependencies"]]): item for item in ready}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"subtask_id": item["node_id"], "mode": item.get("mode", "not_started"), "status": "failed", "output": "", "result_ids": [], "runtime_error": {"kind": "execution", "message": str(exc)}}
                    finished[item["node_id"]] = result
                    all_results.append(result)
                    pending.pop(item["node_id"], None)
        return all_results

    def _run_nested_subtask(self, task_id: str, parent_mode_instance_id: str, spec: dict[str, Any], upstream: list[dict[str, Any]]) -> dict[str, Any]:
        mode, roles, model_id = self._route_subtask(task_id, spec, upstream)
        child_mode_id = self.context.create_mode(task_id, mode, parent_id=parent_mode_instance_id, limits=default_limits(max_agents=min(5, max(1, len(roles))), allow_nesting=False))
        executor_type = {"Single": SingleExecutor, "Chain": ChainExecutor, "Parallel": ParallelExecutor}[mode]
        executor = executor_type(self.context, self.agents, self.router, model_id)
        snapshot = self.context.store.get(task_id)
        dependencies = self.context.upstream_results(snapshot, [rid for item in upstream for rid in item.get("result_ids", [])])
        executor.configure_subtask(spec["node_id"], spec["initial_context"],
            [*spec.get("constraints", []), *spec.get("acceptance_criteria", []), spec["expected_output"]],
            upstream_result_ids=[item["result_id"] for item in dependencies],
            upstream_node_ids=[item["node_id"] for item in dependencies])
        try:
            result = executor.run(task_id, child_mode_id, roles)
        except Exception as exc:
            result = ModeResult(mode, "failed", "", [], error={"kind": "execution", "message": str(exc)})
        self.context.finish_mode(task_id, child_mode_id, result)
        return {"subtask_id": spec["node_id"], "mode": mode, "roles": roles, "model_id": model_id, "status": result.status, "output": result.output, "result_ids": result.result_ids, "state": result.output, "runtime_error": result.error, "upstream_subtask_ids": [item["subtask_id"] for item in upstream]}

    def _route_subtask(self, task_id: str, spec: dict[str, Any], upstream: list[dict[str, Any]]) -> tuple[str, list[str], str]:
        """嵌入模式的三次路由；Planner 的 required_capability 只作为上下文输入。"""
        focused_result_ids = [
            result_id
            for item in upstream
            for result_id in item.get("result_ids", [])
            if isinstance(result_id, str)
        ]
        route_payload = self.routing_context(
            task_id,
            {"task_id": task_id, "subtask": spec, "upstream_results": self.context.upstream_results(self.context.store.get(task_id), focused_result_ids), "remaining_budget": self.context.store.get(task_id)["budget"]["remaining"]},
            focused_result_ids=focused_result_ids,
        )
        mode, mode_decision = self.router.select_mode(route_payload, sorted(NESTED_MODES))
        self._record_nested_decision(task_id, "nested_mode_decision", route_payload, mode_decision)
        candidates = NESTED_ROLE_CANDIDATES[mode]
        count = 2 if mode == "Chain" else 1
        roles, role_decision = self.router.select_roles({**route_payload, "selected_mode": mode}, candidates, count)
        roles = [role for role in roles if role in candidates and role != "Synthesizer"] or [candidates[0]]
        self._record_nested_decision(task_id, "nested_role_decision", {**route_payload, "selected_mode": mode}, role_decision)
        model_candidates = [model.model_id for model in self.agents.model_pool.candidates(spec["initial_context"].get("recommended_budget", "low"), spec["initial_context"].get("tasktype", "Other"))]
        model_id, model_decision = self.router.select_model({**route_payload, "selected_mode": mode, "selected_roles": roles}, model_candidates)
        self._record_nested_decision(task_id, "nested_model_decision", {**route_payload, "selected_mode": mode, "selected_roles": roles}, model_decision)
        return mode, roles, model_id

    def _record_nested_decision(self, task_id: str, event_type: str, context: dict[str, Any], decision: dict[str, Any]) -> None:
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

    @staticmethod
    def _fallback_plan() -> list[dict[str, Any]]:
        context = {"task_description": "完成 DAG 主任务", "tasktype": "Other", "domains": "General", "difficulty": 40, "estimated_duration_minutes": ">15 & <30 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"}
        return [{"node_id": "dag:T1", "description": "完成主任务", "dependencies": [], "required_capability": "Programming", "initial_context": context, "expected_output": "主任务结果", "acceptance_criteria": ["生成结果"], "constraints": []}, {"node_id": "dag:T2", "description": "审查主任务结果", "dependencies": ["dag:T1"], "required_capability": "Review", "initial_context": context, "expected_output": "审查结果", "acceptance_criteria": ["完成审查"], "constraints": []}]
