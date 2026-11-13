"""LangGraph 拓扑运行时。

LangGraph 负责状态图的节点编排和检查点；任务上下文、预算账本和轨迹仍由
Virgo 的业务模块维护，避免把权威状态复制到图状态中。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from virgo.topology.base import TopologyExecutor
from virgo.types import ModeResult


class GraphState(TypedDict, total=False):
    task_id: str
    mode_instance_id: str
    roles: list[str]
    result_ids: Annotated[list[str], operator.add]
    node_ids: Annotated[list[str], operator.add]
    outputs: Annotated[list[dict[str, Any] | str], operator.add]
    statuses: Annotated[list[str], operator.add]
    index: int
    previous_result_id: str | None
    previous_node_id: str | None
    root_node_id: str | None
    root_result_id: str | None
    planner_result_id: str | None
    plan: list[dict[str, Any]]
    subtask_results: Annotated[list[dict[str, Any]], operator.add]
    tree_assignments: list[dict[str, Any]]
    next_role: str | None
    next_model: str | None
    max_steps: int
    final_output: dict[str, Any] | str
    status: str


class LangGraphTopologyExecutor(TopologyExecutor):
    """所有拓扑共享的 LangGraph 编译和结果收集逻辑。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpointer = MemorySaver()

    def compile(self, builder: StateGraph) -> Any:
        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, graph: Any, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult:
        initial: GraphState = {
            "task_id": task_id,
            "mode_instance_id": mode_instance_id,
            "roles": roles,
            "result_ids": [],
            "node_ids": [],
            "outputs": [],
            "statuses": [],
            "index": 0,
            "previous_result_id": None,
            "previous_node_id": None,
            "root_node_id": None,
            "root_result_id": None,
            "planner_result_id": None,
            "plan": [],
            "subtask_results": [],
            "tree_assignments": [],
            "next_role": None,
            "next_model": None,
            "max_steps": self._max_agents,
            "final_output": "",
            "status": "running",
        }
        state = graph.invoke(initial, config={"configurable": {"thread_id": f"{task_id}:{mode_instance_id}"}})
        statuses = state.get("statuses", [])
        status = state.get("status") or (statuses[-1] if statuses else "completed")
        if status == "running":
            status = "completed"
        return ModeResult(self.mode, status, state.get("final_output") or (state.get("outputs") or [""])[-1], state.get("result_ids", []))

    def agent_state(self, state: GraphState, role: str, assignment: str, *, parent: str | None = None, depth: int | None = None, input_result_ids: list[str] | None = None, input_node_ids: list[str] | None = None, model_id: str | None = None, settle_budget: bool = True, edge_type: str = "dependency", round: int = 0, subtask_id: str | None = None, initial_context: dict[str, Any] | None = None, constraints: list[str] | None = None, participant_id: str | None = None, workspace_binding: dict[str, Any] | None = None) -> tuple[dict[str, Any], str | None, str, str]:
        node = self.node(state["task_id"], state["mode_instance_id"], role, assignment, parent=parent, depth=depth, model_id=model_id, round=round, subtask_id=subtask_id, initial_context=initial_context, constraints=constraints)
        if workspace_binding:
            self.context.bind_workspace(state["task_id"], node, workspace_binding)
        if participant_id is not None:
            self.context.store.update(state["task_id"], lambda snapshot: next(
                item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == node
            ).update(participant_id=participant_id))
        input_result_ids = input_result_ids or self._initial_result_ids
        input_node_ids = input_node_ids or self._initial_node_ids
        if input_result_ids:
            self.context.set_node_inputs(state["task_id"], node, input_result_ids)
            for source_node in input_node_ids:
                self.context.connect(state["task_id"], source_node, node, edge_type)
        result_id, output, status = self.call(state["task_id"], node, settle_budget=settle_budget)
        updates = {"result_ids": [result_id] if result_id else [], "node_ids": [node], "outputs": [output], "statuses": [status]}
        if status not in {"failed", "blocked"} or not self.agents.ledger.available(state["task_id"]):
            return updates, result_id, node, status

        recovery_role, recovery_model = self.route_recovery(
            state["task_id"],
            failed_node_id=node,
            failed_result_id=result_id,
            failed_role=role,
            failed_status=status,
        )
        recovery_evidence = self._recovery_tool_evidence(state["task_id"], node)
        recovery_node = self.node(
            state["task_id"], state["mode_instance_id"], recovery_role,
            f"接手失败节点 {node} 的工作，基于失败证据完成原分配：{assignment}\n\n"
            f"【前序工具证据摘要】\n{recovery_evidence}\n"
            "除非证据不足、工作区已变化或需验证修改效果，不得重复前序已完成的读取；应优先实施、验证或收尾。",
            parent=parent, depth=depth, model_id=recovery_model, round=round,
            subtask_id=subtask_id, initial_context=initial_context, constraints=constraints,
        )
        if workspace_binding:
            self.context.bind_workspace(state["task_id"], recovery_node, workspace_binding)
        if participant_id is not None:
            self.context.store.update(state["task_id"], lambda snapshot: next(
                item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == recovery_node
            ).update(participant_id=participant_id))
        if self.mode == "Parallel":
            self.context.store.update(state["task_id"], lambda snapshot: next(
                item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == recovery_node
            ).update(candidate_id=node))
        recovery_result_ids = list(dict.fromkeys([*input_result_ids, *([result_id] if result_id else [])]))
        self.context.set_node_inputs(state["task_id"], recovery_node, recovery_result_ids)
        for source_node in [*input_node_ids, node]:
            self.context.connect(state["task_id"], source_node, recovery_node, "recovery")
        recovery_result_id, recovery_output, recovery_status = self.call(state["task_id"], recovery_node, settle_budget=settle_budget)
        self.agents.trace.record(
            state["task_id"],
            "agent_recovery_complete",
            context={"failed_node_id": node, "recovery_node_id": recovery_node, "recovery_role": recovery_role, "recovery_model": recovery_model},
            output={"status": recovery_status, "result_id": recovery_result_id},
            budget=self.context.store.get(state["task_id"])["budget"],
            node_id=recovery_node,
        )
        updates = {
            "result_ids": [value for value in (result_id, recovery_result_id) if value],
            "node_ids": [node, recovery_node],
            "outputs": [output, recovery_output],
            "statuses": [status, recovery_status],
        }
        return updates, recovery_result_id, recovery_node, recovery_status

    def _recovery_tool_evidence(self, task_id: str, failed_node_id: str) -> str:
        """从可审计轨迹提取失败节点的最近工具事实，供恢复节点直接承接。"""
        trace = getattr(self.agents, "trace", None)
        events = trace.read(task_id) if trace and callable(getattr(trace, "read", None)) else []
        summaries: list[str] = []
        for event in events:
            if event.get("event_type") != "tool_call":
                continue
            context = event.get("context") if isinstance(event.get("context"), dict) else {}
            if context.get("node_id") != failed_node_id:
                continue
            output = event.get("output") if isinstance(event.get("output"), dict) else {}
            data = output.get("data") if isinstance(output.get("data"), dict) else {}
            tool_id = context.get("tool_id", "unknown")
            arguments = context.get("arguments", {})
            status = output.get("status", "unknown")
            if status == "succeeded":
                details = data.get("content") or data.get("stdout") or data.get("matches") or data.get("changed_paths") or "已返回结果"
            else:
                error = output.get("error") if isinstance(output.get("error"), dict) else {}
                details = error.get("message") or data.get("stderr") or "调用未成功"
            text = str(details).replace("\n", " ")[:500]
            summaries.append(f"- {tool_id}({arguments}) -> {status}: {text}")
        if not summaries:
            return "未找到可复用的工具调用记录；先检查工作区状态后再继续。"
        return "\n".join(summaries[-20:])


def new_graph() -> StateGraph:
    return StateGraph(GraphState)


__all__ = ["END", "START", "GraphState", "LangGraphTopologyExecutor", "new_graph"]
