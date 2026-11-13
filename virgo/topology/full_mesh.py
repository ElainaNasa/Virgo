"""全连接协作拓扑。

角色路由器逐次增加讨论参与者，直到选出 FullMeshStop 或达到 5 人上限。
讨论阶段采用同步轮次：下一轮的每个 Agent 都只读取上一轮的完整消息快照。
"""

from __future__ import annotations

from virgo.types import ModeResult
from .langgraph_runtime import END, START, LangGraphTopologyExecutor, new_graph


FULL_MESH_ROLE_CANDIDATES = ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "DomainSpecialist", "Reviewer", "Synthesizer", "FullMeshStop"]

FULL_MESH_ROUND_ONE_ASSIGNMENT = """FullMesh 第一轮 assignment：你是 FullMesh 协作拓扑的第一轮参与者。以原始任务为准，结合共同上下文和本节点角色，独立形成有依据的候选结论；具备执行条件时完成必要操作与验证。区分已确认事实、假设、候选方案、已执行操作和验证结论。交付：核心结论及候选方案、关键依据和适用条件、已执行操作及验证结果、未验证假设与风险、下一轮应检查的具体问题。重要观点或问题使用稳定标识（如 claim_1、issue_1），并关联证据。不要读取其他参与者尚未正式交付的内容，不要把计划说成已完成。遵守原始任务、权限、预算和产物路径。"""

FULL_MESH_ROUND_TWO_ASSIGNMENT = """FullMesh 第二轮讨论 assignment：你是 FullMesh 协作拓扑的第二轮参与者。以原始任务为准，交叉审查上一轮正式交付，优先处理影响结论的冲突、证据缺口、接口不一致、可复现异常和未完成关键工作。重要支持、反驳或修正须引用来源节点/结果 ID、观点或问题标识及证据；独立核查关键结论，不把他人声称通过当作自己的验证。对实质分歧说明不同假设或证据，并据检查结果保留、修正或撤回观点；问题须闭环为已修复并验证、证据排除、确认不适用或仍未解决。遵守本节点授权与产物所有权。交付本轮新增事实、操作和验证、当前推荐方案、已关闭问题，以及未解决事项和具体下一步。"""

FULL_MESH_SYNTHESIZER_ASSIGNMENT = """FullMesh Synthesizer 最终汇总 assignment：依据原始任务、全部正式轮次结果和实际证据形成可直接使用的最终交付，不按人数或自称完成状态投票。核对结果是否偏离原始任务、是否存在不同假设或产物版本。区分有证据的共识、未解决冲突、单节点但证据充分的发现、已撤回/证伪/被替代内容及未完成工作。关键结论须关联证据；必要且授权时针对性核查，并只报告实际执行的验证。若修正最终产物，重新验证受影响部分。逐项闭环验收问题；保留歧义、风险和阻塞及具体后续动作。最终输出须包含：面向原始任务的答案/产物、关键证据、实际操作和最终版本验证、冲突取舍理由、未解决问题及下一步。汇总完成不等于整体通过。遵守权限、预算和产物路径。"""


class FullMeshExecutor(LangGraphTopologyExecutor):
    mode = "FullMesh"

    def run(self, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult:
        participant_roles = self._select_participants(task_id, roles)
        if not participant_roles:
            return ModeResult(self.mode, "completed", "FullMeshStop 已终止讨论。", [])
        graph = new_graph()
        round_one_names: list[str] = []
        round_two_names: list[str] = []
        for index, role in enumerate(participant_roles):
            first_name = f"round_1_{index}"
            second_name = f"round_2_{index}"
            round_one_names.append(first_name)
            round_two_names.append(second_name)

            def round_one(state, role=role, participant=index):
                model_id = self._route_model(state["task_id"], role, "discussion_round_1", state.get("result_ids", []))
                return self.agent_state(state, role, FULL_MESH_ROUND_ONE_ASSIGNMENT, model_id=model_id, settle_budget=False, round=1, participant_id=f"participant_{participant}")[0]

            def round_two(state, role=role, participant=index):
                model_id = self._route_model(state["task_id"], role, "discussion_round_2", state.get("result_ids", []))
                return self.agent_state(state, role, FULL_MESH_ROUND_TWO_ASSIGNMENT, input_result_ids=state.get("result_ids", []), input_node_ids=state.get("node_ids", []), model_id=model_id, settle_budget=False, edge_type="communication", round=2, participant_id=f"participant_{participant}")[0]

            graph.add_node(first_name, round_one)
            graph.add_node(second_name, round_two)
            graph.add_edge(START, first_name)

        # 所有第二轮节点都等待第一轮的全部节点完成，形成真正的同步通信轮。
        for second_name in round_two_names:
            for first_name in round_one_names:
                graph.add_edge(first_name, second_name)
        
        def synthesize(state):
            task_id_value = state["task_id"]
            self.agents.settle_result_ids(task_id_value, state.get("result_ids", []))
            self.agents.trace.record(task_id_value, "fullmesh_discussion_complete", context={"phase": "discussion", "roles": participant_roles, "rounds": 2, "result_ids": state.get("result_ids", [])}, output={"status": "completed", "result_ids": state.get("result_ids", [])}, budget=self.context.store.get(task_id_value)["budget"])
            summary_payload = self.routing_context(
                task_id_value,
                {"task_id": task_id_value, "mode": "FullMesh", "phase": "summary", "discussion_result_ids": state.get("result_ids", []), "candidate_roles": ["Synthesizer"], "remaining_budget": self.context.store.get(task_id_value)["budget"]["remaining"]},
                focused_result_ids=state.get("result_ids", []),
            )
            summary_roles, role_decision = self.router.select_roles(summary_payload, ["Synthesizer"], 1)
            self.record_router_decision(task_id_value, "role_decision", summary_payload, role_decision)
            summary_role = summary_roles[0]
            model_candidates = [model.model_id for model in self.agents.model_pool.candidates(self._budget_tier(task_id_value), self._task_type(task_id_value))]
            summary_model, model_decision = self.router.select_model({**summary_payload, "selected_role": summary_role, "candidate_models": model_candidates}, model_candidates)
            self.record_router_decision(task_id_value, "model_decision", {**summary_payload, "selected_role": summary_role, "candidate_models": model_candidates}, model_decision)
            updates, _, _, status = self.agent_state(state, summary_role, FULL_MESH_SYNTHESIZER_ASSIGNMENT, input_result_ids=state.get("result_ids", []), input_node_ids=state.get("node_ids", []), model_id=summary_model)
            updates["final_output"] = updates["outputs"][-1]
            updates["status"] = status
            self.agents.trace.record(task_id_value, "fullmesh_summary_complete", context={"phase": "summary", "role": summary_role, "model_id": summary_model}, output={"status": status, "result_id": updates["result_ids"][0] if updates["result_ids"] else None}, budget=self.context.store.get(task_id_value)["budget"])
            return updates

        graph.add_node("synthesize", synthesize)
        for name in round_two_names:
            graph.add_edge(name, "synthesize")
        return self.invoke(self.compile(graph), task_id, mode_instance_id, participant_roles)

    def _select_participants(self, task_id: str, initial_roles: list[str]) -> list[str]:
        selected = []
        if initial_roles:
            first = initial_roles[0]
            if first in FULL_MESH_ROLE_CANDIDATES and first != "FullMeshStop":
                selected.append(first)
        while len(selected) < min(5, self._max_agents):
            payload = self.routing_context(
                task_id,
                {"task_id": task_id, "mode": "FullMesh", "phase": "participant_selection", "selected_roles": selected, "candidate_roles": FULL_MESH_ROLE_CANDIDATES, "remaining_budget": self.context.store.get(task_id)["budget"]["remaining"]},
            )
            next_roles, decision = self.router.select_roles(payload, FULL_MESH_ROLE_CANDIDATES, 1)
            self.record_router_decision(task_id, "role_decision", payload, decision)
            next_role = next_roles[0] if next_roles else "FullMeshStop"
            if next_role == "FullMeshStop":
                break
            if next_role in selected:
                available = [role for role in FULL_MESH_ROLE_CANDIDATES if role not in selected and role != "FullMeshStop"]
                if not available:
                    break
                next_role = available[0]
            selected.append(next_role)
        return selected

    def _route_model(self, task_id: str, role: str, phase: str, result_ids: list[str]) -> str:
        candidates = [model.model_id for model in self.agents.model_pool.candidates(self._budget_tier(task_id), self._task_type(task_id))]
        payload = self.routing_context(
            task_id,
            {"task_id": task_id, "mode": "FullMesh", "phase": phase, "role": role, "input_result_ids": result_ids, "candidate_models": candidates, "remaining_budget": self.context.store.get(task_id)["budget"]["remaining"]},
            focused_result_ids=result_ids,
        )
        model, decision = self.router.select_model(payload, candidates)
        self.record_router_decision(task_id, "model_decision", payload, decision)
        return model

    def _task_type(self, task_id: str) -> str:
        return str((self.context.store.get(task_id).get("initial_context") or {}).get("tasktype", "Other"))

    def _budget_tier(self, task_id: str) -> str:
        return str((self.context.store.get(task_id).get("initial_context") or {}).get("recommended_budget", "low"))
