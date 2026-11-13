"""同角色、多模型独立求解并进行候选比较的并行拓扑。"""

from __future__ import annotations

from typing import Any

from virgo.types import ModeResult
from virgo.roles import parse_role_instance
from virgo.context.workspaces import ParallelWorkspaces
from .langgraph_runtime import END, START, LangGraphTopologyExecutor, new_graph


PARALLEL_ROLE_CANDIDATES = ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "DomainSpecialist", "Reviewer", "Synthesizer"]


class ParallelExecutor(LangGraphTopologyExecutor):
    mode = "Parallel"

    def run(self, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult:
        graph = new_graph()
        role = next((
            role for role in roles
            if parse_role_instance(role)[0] in PARALLEL_ROLE_CANDIDATES
            and parse_role_instance(role)[0] != "Synthesizer"
        ), "MathSpecialist")
        model_pool_candidates = [
            model.model_id
            for model in self.agents.model_pool.all()
        ]
        if len(model_pool_candidates) < 2:
            raise ValueError("Parallel 需要至少两个不同的模型")

        branch_models: list[str] = []
        max_models = min(4, len(model_pool_candidates))
        for index in range(max_models):
            available_models = [model for model in model_pool_candidates if model not in branch_models]
            candidates = list(available_models)
            if len(branch_models) >= 2 and len(branch_models) < max_models:
                candidates.append("Stop")
            payload = self.routing_context(
                task_id,
                {
                    "task_id": task_id,
                    "mode": "Parallel",
                    "phase": "parallel_model_selection",
                    "selected_role": role,
                    "candidate_models": candidates,
                    "selected_models": list(branch_models),
                    "selection_index": index + 1,
                    "minimum_models": 2,
                    "maximum_models": max_models,
                    "instruction": "为同一角色的同一个完整任务选择不同模型，使各模型独立求解。至少选择两个，最多选择四个；已选择至少两个后可选择 Stop。",
                    "remaining_budget": self.context.store.get(task_id)["budget"]["remaining"],
                },
                focused_result_ids=self._initial_result_ids,
            )
            model_id, decision = self.router.select_model(payload, candidates)
            self.record_router_decision(task_id, "parallel_model_decision", payload, decision)
            if model_id == "Stop":
                break
            if model_id not in branch_models:
                branch_models.append(model_id)

        if len(branch_models) < 2:
            raise ValueError(f"Parallel 模型路由只选出 {len(branch_models)} 个不同模型，至少需要 2 个")
        branch_roles = [role] * len(branch_models)

        source = getattr(self.context, "workspace", ".")
        if self._default_parent_node_id:
            source = self.context.workspace_for_node(task_id, self._default_parent_node_id)
        workspaces = ParallelWorkspaces(self.context, task_id, mode_instance_id, source, len(branch_models))
        for index, model_id in enumerate(branch_models):
            name = f"branch_{index}"

            def branch(state, model_id=model_id, candidate=index):
                assignment = (
                    "独立解决完整任务并生成候选结果。与其他并行节点使用相同角色和任务，"
                    "不得将任务拆成不同职责，不等待或照搬其他候选；给出结论、依据、必要验证和限制。"
                )
                return self.agent_state(state, role, assignment, model_id=model_id, settle_budget=False, workspace_binding=workspaces.binding(candidate))[0]

            graph.add_node(name, branch)
            graph.add_edge(START, name)
            graph.add_edge(name, "synthesize")

        def synthesize(state):
            task_id_value = state["task_id"]
            self.agents.settle_result_ids(task_id_value, state.get("result_ids", []))
            self.agents.trace.record(task_id_value, "parallel_phase_complete", context={"phase": "parallel", "role": role, "models": branch_models, "result_ids": state.get("result_ids", [])}, output={"status": "completed", "result_ids": state.get("result_ids", [])}, budget=self.context.store.get(task_id_value)["budget"])
            summary_payload = self.routing_context(
                task_id_value,
                {"task_id": task_id_value, "mode": "Parallel", "phase": "summary", "parallel_result_ids": state.get("result_ids", []), "candidate_roles": PARALLEL_ROLE_CANDIDATES, "remaining_budget": self.context.store.get(task_id_value)["budget"]["remaining"]},
                focused_result_ids=state.get("result_ids", []),
            )
            summary_roles, role_decision = self.router.select_roles(summary_payload, PARALLEL_ROLE_CANDIDATES, 1)
            self.record_router_decision(task_id_value, "role_decision", summary_payload, role_decision)
            summary_role = summary_roles[0]
            model_candidates = [model.model_id for model in self.agents.model_pool.all()]
            summary_model, model_decision = self.router.select_model({**summary_payload, "selected_role": summary_role, "candidate_models": model_candidates}, model_candidates)
            self.record_router_decision(task_id_value, "model_decision", {**summary_payload, "selected_role": summary_role, "candidate_models": model_candidates}, model_decision)
            summary_binding = workspaces.prepare_summary()
            updates, _, _, status = self.agent_state(
                state,
                summary_role,
                "比较同一角色由不同模型独立产生的候选结果，核查分歧、关键依据和验证证据，识别可能的共同错误。不要只按多数投票或拼接答案；形成有依据的最终结果，并明确仍未解决的分歧。",
                input_result_ids=state.get("result_ids", []),
                input_node_ids=state.get("node_ids", []),
                model_id=summary_model,
                workspace_binding=summary_binding,
            )
            if status in {"completed", "partial"}:
                try:
                    publication = workspaces.publish()
                    if updates["result_ids"]:
                        updates["outputs"][-1] = self.context.publish_output_paths(
                            task_id_value, updates["result_ids"][-1], publication["path_map"])
                    self.agents.trace.record(task_id_value, "parallel_workspace_published",
                        context={"mode_instance_id": mode_instance_id}, output=publication)
                except (OSError, RuntimeError) as exc:
                    status = "failed"
                    output_state = updates["outputs"][-1]
                    if isinstance(output_state, dict):
                        output_state["possible_issue"].append(f"汇总产物发布失败：{exc}")
                        output_state["remaining"].append("解决父工作区冲突后重新发布并验证汇总产物。")
                        if updates["result_ids"]:
                            self.context.record_publication_failure(task_id_value, updates["result_ids"][-1], output_state)
                    self.agents.trace.record(task_id_value, "parallel_workspace_publish_failed",
                        context={"mode_instance_id": mode_instance_id}, output={"error": str(exc)})
            updates["final_output"] = updates["outputs"][-1]
            updates["status"] = status
            self.agents.trace.record(task_id_value, "parallel_summary_complete", context={"phase": "summary", "role": summary_role, "model_id": summary_model, "input_result_ids": state.get("result_ids", [])}, output={"status": status, "result_id": updates["result_ids"][0] if updates["result_ids"] else None}, budget=self.context.store.get(task_id_value)["budget"])
            return updates

        graph.add_node("synthesize", synthesize)
        return self.invoke(self.compile(graph), task_id, mode_instance_id, branch_roles)

    def _task_type(self, task_id: str) -> str:
        return str((self.context.store.get(task_id).get("initial_context") or {}).get("tasktype", "Other"))

    def _budget_tier(self, task_id: str) -> str:
        return str((self.context.store.get(task_id).get("initial_context") or {}).get("recommended_budget", "low"))
