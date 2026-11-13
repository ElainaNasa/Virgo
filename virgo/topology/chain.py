"""动态链式协作拓扑。

每次 Agent 完成后，Agent 执行器先结算本次调用；随后本图才调用角色路由器
选择下一个角色，再调用模型路由器选择下一次计价模型。Chain 至少执行两个
工作节点；第二个节点完成后，ChainStop 才会成为合法的停止选择。若某节点
失败或阻塞且预算仍可用，系统强制进行一次恢复路由，避免直接丢弃后续角色。
"""

from __future__ import annotations

from typing import Any

from virgo.types import ModeResult
from virgo.roles import parse_role_instance
from .langgraph_runtime import END, START, LangGraphTopologyExecutor, new_graph


CHAIN_ROLE_CANDIDATES = ["MathSpecialist", "AlgorithmDesigner", "Programmer", "DataAnalyst", "Researcher", "ToolOperator", "DomainSpecialist", "Reviewer", "ReportWriter", "ChainStop"]
CHAIN_WORKER_ROLE_CANDIDATES = [role for role in CHAIN_ROLE_CANDIDATES if role != "ChainStop"]
MIN_CHAIN_NODES = 2


class ChainExecutor(LangGraphTopologyExecutor):
    mode = "Chain"

    def run(self, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult:
        graph = new_graph()
        initial_role = roles[0] if roles and parse_role_instance(roles[0])[0] in CHAIN_WORKER_ROLE_CANDIDATES else "Programmer"

        def execute(state):
            index = state.get("index", 0)
            role = initial_role if index == 0 else state.get("next_role")
            if role == "ChainStop" or not role:
                return {"status": "completed", "next_role": "ChainStop", "final_output": state.get("final_output", "")}
            if index > 0 and state.get("next_model"):
                self.model_id = state["next_model"]
            updates, result_id, node, status = self.agent_state(
                state,
                role,
                "承接当前协作阶段的前序结果，完成一个对整体任务有价值且可核查的处理阶段。",
                input_result_ids=[state["previous_result_id"]] if state.get("previous_result_id") else [],
                input_node_ids=[state["previous_node_id"]] if state.get("previous_node_id") else [],
            )
            updates.update({"index": index + 1, "previous_result_id": result_id, "previous_node_id": node, "final_output": updates["outputs"][-1], "status": status, "next_role": "ChainStop", "next_model": None})
            completed_nodes = index + 1
            minimum_not_met = completed_nodes < MIN_CHAIN_NODES
            recoverable_failure = status in {"failed", "blocked"} and self.agents.ledger.available(task_id)
            if (status in {"completed", "partial"} or minimum_not_met or recoverable_failure) and completed_nodes < max(state.get("max_steps", 10), MIN_CHAIN_NODES):
                next_role, next_model = self._route_next(
                    task_id,
                    role,
                    result_id,
                    completed_nodes,
                    force_worker=recoverable_failure,
                    previous_status=status,
                )
                updates["next_role"] = next_role
                updates["next_model"] = next_model
            return updates

        def route(state):
            if state.get("next_role") in {None, "ChainStop"}:
                return END
            if state.get("index", 0) >= max(state.get("max_steps", 10), MIN_CHAIN_NODES):
                return END
            return "execute"

        graph.add_node("execute", execute)
        graph.add_edge(START, "execute")
        graph.add_conditional_edges("execute", route, {"execute": "execute", END: END})
        return self.invoke(self.compile(graph), task_id, mode_instance_id, roles)

    def _route_next(self, task_id: str, previous_role: str, previous_result_id: str | None, completed_nodes: int, *, force_worker: bool = False, previous_status: str = "completed") -> tuple[str, str | None]:
        stop_allowed = completed_nodes >= MIN_CHAIN_NODES and not force_worker
        candidates = CHAIN_ROLE_CANDIDATES if stop_allowed else CHAIN_WORKER_ROLE_CANDIDATES
        payload = {
            **self.context.chain_route_context(task_id, previous_result_id, previous_role),
            "candidate_roles": candidates,
            "completed_chain_nodes": completed_nodes,
            "minimum_chain_nodes": MIN_CHAIN_NODES,
            "chain_stop_allowed": stop_allowed,
            "chain_recovery": force_worker,
            "previous_node_status": previous_status,
        }
        next_roles, role_decision = self.router.select_roles(payload, candidates, 1)
        next_role = next_roles[0] if next_roles else candidates[0]
        if next_role not in candidates:
            enforced_role = candidates[0]
            role_decision = {
                **role_decision,
                "roles": [enforced_role],
                "constraint_override": {
                    "rule": "minimum_chain_nodes",
                    "minimum": MIN_CHAIN_NODES,
                    "completed_nodes": completed_nodes,
                    "rejected_role": next_role,
                },
            }
            next_role = enforced_role
        self.record_router_decision(task_id, "chain_role_decision", payload, role_decision)
        if next_role == "ChainStop":
            return next_role, None
        snapshot = self.context.store.get(task_id)
        initial = snapshot.get("initial_context") or {}
        model_candidates = [model.model_id for model in self.agents.model_pool.candidates(initial.get("recommended_budget", "low"), initial.get("tasktype", "Other"))]
        model, model_decision = self.router.select_model({**payload, "selected_role": next_role, "candidate_models": model_candidates}, model_candidates)
        self.record_router_decision(task_id, "chain_model_decision", {**payload, "selected_role": next_role, "candidate_models": model_candidates}, model_decision)
        return next_role, model
