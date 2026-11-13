"""Tree 协作拓扑。

层级严格限制为三层：根 Coordinator、SubtaskCoordinator、叶节点执行者。
根协调器决定分支数量；每个分支的执行方式由协作模式路由器决定，允许的
嵌入模式为 Single、Chain、FullMesh，嵌入模式不能继续创建子模式。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from virgo.context.factory import default_limits
from virgo.types import InitialContext, ModeResult
from .chain import ChainExecutor
from .full_mesh import FullMeshExecutor
from .langgraph_runtime import END, START, GraphState, LangGraphTopologyExecutor, new_graph
from .single import SingleExecutor


TREE_EMBEDDED_MODES = {"Single", "Chain", "FullMesh"}
TREE_LEAF_ROLE_CANDIDATES = ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "ToolOperator", "DomainSpecialist", "Reviewer", "Synthesizer", "ReportWriter"]


class TreeExecutor(LangGraphTopologyExecutor):
    mode = "Tree"

    def run(self, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult:
        graph = new_graph()

        def root_plan(state: GraphState):
            root_payload = self.routing_context(
                task_id,
                {"task_id": task_id, "mode": "Tree", "phase": "root_coordinator", "candidate_roles": ["Coordinator"]},
            )
            root_role, role_decision = self.router.select_roles(root_payload, ["Coordinator"], 1)
            self.record_router_decision(task_id, "role_decision", root_payload, role_decision)
            root_model = self._route_model(task_id, "Coordinator", "root_coordinator")
            updates, root_result_id, root_node, status = self.agent_state(state, root_role[0], "将整体任务拆分为最多三个分支，明确每个分支的目标、输入、预期输出和验收标准。", model_id=root_model, depth=1, round=0)
            assignments = self._read_assignments(task_id, root_result_id)
            manager_payload = self.routing_context(
                task_id,
                {"task_id": task_id, "mode": "Tree", "phase": "branch_manager_count", "branch_count": len(assignments), "candidate_roles": ["SubtaskCoordinator"]},
                focused_result_ids=[root_result_id] if root_result_id else [],
            )
            manager_roles, manager_decision = self.router.select_roles(manager_payload, ["SubtaskCoordinator"], len(assignments))
            self.record_router_decision(task_id, "role_decision", manager_payload, {**manager_decision, "selected_count": len(assignments), "roles": manager_roles})
            updates.update({"root_node_id": root_node, "root_result_id": root_result_id, "tree_assignments": assignments, "status": status})
            return updates

        def execute_branches(state: GraphState):
            assignments = state.get("tree_assignments", [])[:3]
            results: list[dict[str, Any]] = []
            pending = {item["subtask_id"]: item for item in assignments}
            finished = {}
            while pending:
                ready = [item for item in pending.values() if all(
                    dep in finished and finished[dep]["status"] in {"completed", "partial"}
                    for dep in item.get("depends_on", []))]
                if not ready:
                    results.extend({"subtask_id": item["subtask_id"], "status": "blocked", "output": "",
                                    "result_ids": [], "node_ids": [], "runtime_error": {"kind": "dependency", "message": "分支依赖缺失、失败或存在环。"}}
                                   for item in pending.values())
                    break
                with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                    futures = {pool.submit(self._execute_branch, state["task_id"], state["mode_instance_id"],
                        state.get("root_node_id"), state.get("root_result_id"), item,
                        [finished[dep] for dep in item.get("depends_on", [])]): item for item in ready}
                    for future in as_completed(futures):
                        item = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {"subtask_id": item["subtask_id"], "status": "failed", "output": "",
                                      "result_ids": [], "node_ids": [], "runtime_error": {"kind": "execution", "message": str(exc)}}
                        results.append(result)
                        finished[item["subtask_id"]] = result
                        pending.pop(item["subtask_id"])
            return {"subtask_results": results, "result_ids": [rid for item in results for rid in item.get("result_ids", [])], "node_ids": [nid for item in results for nid in item.get("node_ids", [])], "outputs": [item.get("output", "") for item in results], "statuses": [item.get("status", "failed") for item in results]}

        def root_summarize(state: GraphState):
            root_model = self._route_model(task_id, "Coordinator", "root_summary")
            input_result_ids = state.get("result_ids", [])
            input_node_ids = state.get("node_ids", [])
            updates, _, _, status = self.agent_state(state, "Coordinator", "验收所有分支结果，整合兼容结论，并明确报告未完成或失败的分支。", input_result_ids=input_result_ids, input_node_ids=input_node_ids, model_id=root_model, depth=1, round=0)
            updates["final_output"] = updates["outputs"][-1]
            branch_statuses = [item.get("status", "failed") for item in state.get("subtask_results", [])]
            if status not in {"completed", "partial"}:
                updates["status"] = status
            elif any(value == "failed" for value in branch_statuses):
                updates["status"] = "partial"
            elif any(value == "blocked" for value in branch_statuses):
                updates["status"] = "partial"
            elif any(value == "partial" for value in branch_statuses):
                updates["status"] = "partial"
            else:
                updates["status"] = "completed"
            return updates

        def route_after_root(state: GraphState):
            return "execute_branches" if state.get("status") in {"running", "completed", "partial"} else END

        graph.add_node("root_plan", root_plan)
        graph.add_node("execute_branches", execute_branches)
        graph.add_node("root_summarize", root_summarize)
        graph.add_edge(START, "root_plan")
        graph.add_conditional_edges("root_plan", route_after_root, {"execute_branches": "execute_branches", END: END})
        graph.add_edge("execute_branches", "root_summarize")
        graph.add_edge("root_summarize", END)
        return self.invoke(self.compile(graph), task_id, mode_instance_id, roles)

    def _read_assignments(self, task_id: str, result_id: str | None) -> list[dict[str, Any]]:
        raw: Any = None
        if result_id:
            snapshot = self.context.store.get(task_id)
            output = snapshot.get("agent_outputs", {}).get(result_id, {}).get("output", {})
            if isinstance(output, dict):
                raw = output.get("assignments")
                if not raw:
                    try:
                        value = json.loads(output.get("task_output", ""))
                        raw = value.get("assignments")
                    except (json.JSONDecodeError, TypeError):
                        raw = None
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw or [], start=1):
            if not isinstance(item, dict):
                continue
            subtask_id = str(item.get("subtask_id") or f"tree:T{index}")
            context = InitialContext.from_dict(item.get("initial_context", {"task_description": item.get("description", "树形分支任务"), "tasktype": "Other", "domains": "General", "difficulty": 40, "estimated_duration_minutes": ">15 & <30 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"})).as_dict()
            normalized.append({"subtask_id": subtask_id, "description": str(item.get("description", context["task_description"])), "initial_context": context, "acceptance_criteria": [str(value) for value in item.get("acceptance_criteria", ["完成分支任务并通过验收"])], "depends_on": [str(value) for value in item.get("depends_on", [])], "constraints": [str(value) for value in item.get("constraints", [])]})
            if len(normalized) == 3:
                break
        return normalized or self._fallback_assignments()

    @staticmethod
    def _fallback_assignments() -> list[dict[str, Any]]:
        base = {"task_description": "完成一个树形分支任务", "tasktype": "Other", "domains": "General", "difficulty": 40, "estimated_duration_minutes": ">15 & <30 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"}
        return [{"subtask_id": "tree:T1", "description": "完成主要任务", "initial_context": base, "acceptance_criteria": ["生成主要结果"], "depends_on": [], "constraints": []}, {"subtask_id": "tree:T2", "description": "审查主要任务和边界条件", "initial_context": {**base, "task_description": "审查主要任务和边界条件", "tasktype": "Writing,Programming"}, "acceptance_criteria": ["给出审查结论"], "depends_on": [], "constraints": []}]

    def _execute_branch(self, task_id: str, tree_mode_instance_id: str, root_node_id: str | None, root_result_id: str | None, spec: dict[str, Any], upstream: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        branch_id = spec["subtask_id"]
        manager_model = self._route_model(task_id, "SubtaskCoordinator", "branch_manager", spec)
        manager_node = self.node(task_id, tree_mode_instance_id, role="SubtaskCoordinator", assignment="管理本分支，判断直接完成或请求受限的叶节点执行。", parent=root_node_id, depth=2, model_id=manager_model, subtask_id=branch_id, initial_context=spec["initial_context"], constraints=[*spec.get("constraints", []), *spec.get("acceptance_criteria", [])], round=0)
        dependency_ids = [rid for item in upstream or [] for rid in item.get("result_ids", [])]
        inherited_ids = [*([root_result_id] if root_result_id else []), *dependency_ids]
        if inherited_ids:
            dependencies = self.context.upstream_results(self.context.store.get(task_id), inherited_ids)
            self.context.set_node_inputs(task_id, manager_node, [item["result_id"] for item in dependencies])
            for item in dependencies:
                if item["node_id"] != root_node_id:
                    self.context.connect(task_id, item["node_id"], manager_node, "dependency")
            if root_node_id:
                self.context.connect(task_id, root_node_id, manager_node, "delegation")
        manager_result_id, manager_output = self.agents.run(task_id, manager_node)
        request = manager_output.extra.get("execution_request") if manager_output.extra else None
        if not isinstance(request, dict) or request.get("request_type") != "route_subtask":
            return {"subtask_id": branch_id, "status": manager_output.status, "output": manager_output.task_output, "result_ids": [manager_result_id] if manager_result_id else [], "node_ids": [manager_node], "mode": "direct", "runtime_error": manager_output.error}

        mode, leaf_roles, leaf_model = self._route_nested_execution(task_id, spec, request)
        child_mode_id = self.context.create_mode(task_id, mode, parent_id=tree_mode_instance_id, limits=default_limits(max_agents=3, allow_nesting=False))
        executor_type = {"Single": SingleExecutor, "Chain": ChainExecutor, "FullMesh": FullMeshExecutor}[mode]
        executor = executor_type(self.context, self.agents, self.router, leaf_model)
        executor.configure_subtask(branch_id, spec["initial_context"], spec.get("constraints", []), upstream_result_ids=[manager_result_id] if manager_result_id else [], upstream_node_ids=[manager_node], max_agents=3, parent_node_id=manager_node, depth=3)
        try:
            nested_result = executor.run(task_id, child_mode_id, leaf_roles)
        except Exception as exc:
            nested_result = ModeResult(mode, "failed", "", [], error={"kind": "execution", "message": str(exc)})
        self.context.finish_mode(task_id, child_mode_id, nested_result)
        if nested_result.result_ids:
            self.context.set_node_inputs(task_id, manager_node, nested_result.result_ids)
        if not self.agents.model_pool.channel_available(task_id, manager_model):
            manager_node = self.node(task_id, tree_mode_instance_id, role="SubtaskCoordinator",
                                     assignment="验收本分支执行结果并报告结论。", parent=root_node_id,
                                     depth=2, model_id=manager_model, subtask_id=branch_id,
                                     initial_context=spec["initial_context"])
            self.context.set_node_inputs(task_id, manager_node, nested_result.result_ids)
        validation_result_id, validation_output = self.agents.run(task_id, manager_node)
        final_result_id = validation_result_id or (nested_result.result_ids[-1] if nested_result.result_ids else manager_result_id)
        branch_status = nested_result.status if nested_result.status in {"failed", "blocked"} else (validation_output.status if validation_result_id else nested_result.status)
        return {"subtask_id": branch_id, "status": branch_status, "output": validation_output.task_output if validation_result_id else nested_result.output, "result_ids": [rid for rid in [manager_result_id, *nested_result.result_ids, validation_result_id] if rid], "node_ids": [manager_node], "mode": mode, "roles": leaf_roles, "model_id": leaf_model, "final_result_id": final_result_id, "runtime_error": nested_result.error or (validation_output.error if validation_result_id else None)}

    def _route_nested_execution(self, task_id: str, spec: dict[str, Any], request: dict[str, Any]) -> tuple[str, list[str], str]:
        allowed_modes = [mode for mode in request.get("limits", {}).get("allowed_modes", TREE_EMBEDDED_MODES) if mode in TREE_EMBEDDED_MODES] or sorted(TREE_EMBEDDED_MODES)
        payload = self.routing_context(
            task_id,
            {"task_id": task_id, "mode": "Tree", "phase": "branch_execution", "subtask": spec, "execution_request": request, "candidate_modes": allowed_modes, "remaining_budget": self.context.store.get(task_id)["budget"]["remaining"]},
        )
        mode, mode_decision = self.router.select_mode(payload, allowed_modes)
        self.record_router_decision(task_id, "mode_decision", payload, mode_decision)
        count = 1 if mode == "Single" else (1 if mode == "Chain" else 1)
        roles, role_decision = self.router.select_roles({**payload, "selected_mode": mode, "candidate_roles": TREE_LEAF_ROLE_CANDIDATES, "count": count}, TREE_LEAF_ROLE_CANDIDATES, count)
        roles = [role for role in roles if role in TREE_LEAF_ROLE_CANDIDATES] or ["Programmer"]
        self.record_router_decision(task_id, "role_decision", {**payload, "selected_mode": mode, "candidate_roles": TREE_LEAF_ROLE_CANDIDATES, "count": count}, role_decision)
        model_id = self._route_model(task_id, roles[0], "branch_execution", spec, mode=mode, roles=roles)
        return mode, roles, model_id

    def _route_model(self, task_id: str, role: str, phase: str, spec: dict[str, Any] | None = None, *, mode: str = "Tree", roles: list[str] | None = None) -> str:
        snapshot = self.context.store.get(task_id)
        context = (spec or {}).get("initial_context") or snapshot.get("initial_context") or {}
        candidates = [model.model_id for model in self.agents.model_pool.candidates(context.get("recommended_budget", "low"), context.get("tasktype", "Other"))]
        payload = self.routing_context(
            task_id,
            {"task_id": task_id, "mode": mode, "phase": phase, "role": role, "selected_roles": roles or [role], "candidate_models": candidates, "remaining_budget": snapshot["budget"]["remaining"]},
        )
        model_id, decision = self.router.select_model(payload, candidates)
        self.record_router_decision(task_id, "model_decision", payload, decision)
        return model_id
