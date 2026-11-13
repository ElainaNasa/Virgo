"""上下文系统：集中组织路由输入、Agent 输入和执行结果。"""

from __future__ import annotations

import json
import hashlib
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from virgo.context.factory import create_snapshot, default_limits, money_limits, new_id, recommended_tier_is_lower
from virgo.task_state import combine_states, normalize_state
from virgo.state_protocol import StateProtocolError, validate_state, validate_artifacts
from virgo.context.store import ContextStore
from virgo.roles import get_role
from virgo.types import AgentOutput, InitialContext, TaskRequest
from virgo.tools.sandbox import SANDBOX_WORKSPACE


PLANNING_ROLES = {"InitialContextBuilder", "SpecialPlanner", "Coordinator", "SubtaskCoordinator", "Synthesizer"}
COLLABORATION_ACCEPTANCE_CRITERIA = [
    "在已有工作基础上取得有效进展，明确新增的事实、操作、产物或结论；无实质新增的复述不计为进展。",
    "为关键结论和完成声明提供可核查的依据，区分已执行、已验证、待验证和仅提出的方案。",
    "如实报告任务完成程度；未取得进展或执行受阻时，说明具体原因及已尝试的措施，不得将节点结束等同于任务完成。",
    "未完成时，明确剩余事项、必要上下文和下一步可执行动作，使后续节点能够直接承接。",
    "按照约定的结构化格式输出结果。",
]


class ContextSystem:
    def __init__(self, store: ContextStore, collaboration_tool_call_limit: int = 30) -> None:
        self.store = store
        self.workspace_publish_lock = threading.RLock()
        self.collaboration_tool_call_limit = max(1, collaboration_tool_call_limit)

    def create_task(self, request: TaskRequest) -> dict[str, Any]:
        return self.store.create(create_snapshot(request))

    def set_initial_context(self, task_id: str, context: InitialContext | dict[str, Any]) -> dict[str, Any]:
        value = context.as_dict() if isinstance(context, InitialContext) else context
        def mutate(snapshot: dict[str, Any]) -> None:
            snapshot["initial_context"] = value
            snapshot["execution_state"]["status"] = "ready"
        return self.store.update(task_id, mutate)

    def adopt_recommended_budget_tier(self, task_id: str, requested_tier: str, recommended_tier: str, *, has_explicit_cost_limit: bool) -> dict[str, Any] | None:
        """在推荐档位更低时收紧总预算；显式数值上限不受推荐值覆盖。"""
        if has_explicit_cost_limit or not recommended_tier_is_lower(recommended_tier, requested_tier):
            return None
        recommended_cost = money_limits(recommended_tier, None)

        def mutate(snapshot: dict[str, Any]) -> None:
            budget = snapshot["budget"]
            current_total = budget["total"]["cost"]
            if current_total is not None and recommended_cost is not None and recommended_cost >= current_total:
                return
            budget["total"]["cost"] = recommended_cost
            budget["remaining"]["cost"] = recommended_cost

        snapshot = self.store.update(task_id, mutate)
        return {
            "requested_tier": requested_tier,
            "effective_tier": recommended_tier,
            "effective_cost_limit": snapshot["budget"]["total"]["cost"],
            "budget": snapshot["budget"],
        }

    def route_context(self, task_id: str, router_kind: str, *, mode: str | None = None, roles: list[str] | None = None, candidate_modes: list[str] | None = None) -> dict[str, Any]:
        common = self.routing_context(task_id)
        if router_kind == "mode":
            common["candidate_modes"] = list(candidate_modes or ["Single", "Chain", "Parallel", "Tree", "DAG", "FullMesh"])
        elif router_kind == "role":
            common["selected_mode"] = mode
            common["candidate_roles"] = roles or []
            common["previous_roles"] = [item["role"]["name"] for item in common["agent"]]
        elif router_kind == "model":
            common["selected_mode"] = mode
            common["selected_roles"] = roles or []
            common["candidate_models"] = "由模型池能力表和价格表提供"
        return common

    def routing_context(self, task_id: str, *, focused_result_ids: list[str] | None = None) -> dict[str, Any]:
        """构造所有路由器共享的完整、可审计决策上下文。

        路由器能够看到完整任务要求、初始上下文、预算、已执行拓扑、配置、
        全部结果以及本次阶段重点结果；工作 Agent 输入仍由 build_agent_input 单独构造。
        """
        snapshot = self.store.get(task_id)
        initial = snapshot["initial_context"]
        if not initial:
            raise RuntimeError("初始上下文尚未生成")
        active_ids = [item["result_id"] for item in snapshot.get("state_snapshots", {}).values()]
        all_results = self._routing_results(snapshot, active_ids or list(snapshot.get("agent_outputs", {})))
        budget_total = snapshot["budget"]["total"]["cost"]
        difficulty = initial.get("difficulty")
        try:
            high_difficulty = int(difficulty) >= 50
        except (TypeError, ValueError):
            high_difficulty = False
        instances = snapshot["mode_state"]["instances"]
        nodes = snapshot["execution_state"]["nodes"]
        assignment_agents = {item["assignment_id"]: item["agent_id"] for item in snapshot["agent_assignments"]}
        agents = []
        for assignment in snapshot["agent_assignments"]:
            instance = next((item for item in instances if item["mode_instance_id"] == assignment["mode_instance_id"]), {})
            collaborative = instance.get("execution_mode", instance.get("mode")) != "Single" and assignment["role"]["name"] not in PLANNING_ROLES
            agents.append({
                "agent_id": assignment["agent_id"],
                "mode_instance_id": assignment["mode_instance_id"],
                "node_id": assignment["node_id"],
                "role": assignment["role"],
                "llm": assignment["llm"],
                "tool": assignment["tools"],
                "tool_calls_limit": self.collaboration_tool_call_limit if collaborative else getattr(self, "max_tool_calls", 50),
                "context_window": 200_000,
            })
        return {
            "task": {
                "task_id": task_id,
                "original_task": snapshot["task"]["original_task"],
                "initial_context": {key: value for key, value in initial.items() if key != "task_description"},
                **{key: value for key, value in snapshot["task"].items() if key not in {"task_id", "original_task", "constraints", "acceptance_criteria"}},
            },
            "budget": snapshot["budget"],
            "mode": {
                "active_mode_instance_id": snapshot["mode_state"]["active_mode_instance_id"],
                "instances": [{key: value for key, value in item.items() if key != "limits"} for item in instances],
                "nodes": [{**{key: value for key, value in node.items() if key not in {"limits", "permissions", "status", "active_assignment_id"}}, "agent_id": assignment_agents.get(node.get("active_assignment_id"))} for node in nodes],
                "edges": snapshot["execution_state"]["edges"],
            },
            "agent": agents,
            "results": all_results,
            "recommendations": {
                "tool_call_limit": self.collaboration_tool_call_limit,
                "budget_usage_recommendations": (
                    {"minimum": round(budget_total / 3, 8), "maximum": round(budget_total * 5 / 6, 8)}
                    if isinstance(budget_total, (int, float)) else None
                ),
                "progress_recommendations": "根据新增证据、可用产物、验证结果和未解决问题判断进展，不以输出长度、讨论轮数或工具调用数量代替进展。",
                "failure_recommendations": "失败后先区分能力不足、信息不足、工具或环境异常、任务分配不当、预算耗尽及验证不通过，再决定后续动作。失败且花费偏低时考虑提高投入。",
                "high_difficuty_recommendations": "难度可能较高，谨慎选择Single模式，工具调用次数可能不够" if high_difficulty else None,
            },
        }

    @staticmethod
    def _routing_results(snapshot: dict[str, Any], result_ids: list[str]) -> list[dict[str, Any]]:
        assignments = {
            item.get("assignment_id"): item
            for item in snapshot.get("agent_assignments", [])
            if isinstance(item, dict) and item.get("assignment_id")
        }
        records: list[dict[str, Any]] = []
        for result_id in result_ids:
            result = snapshot.get("agent_outputs", {}).get(result_id)
            if not isinstance(result, dict):
                continue
            assignment = assignments.get(result.get("assignment_id"), {})
            records.append(
                {
                    "result_id": result_id,
                    "node_id": result.get("node_id"),
                    "mode_instance_id": result.get("mode_instance_id"),
                    "role": assignment.get("role", {}).get("name"),
                    "model_id": assignment.get("llm", {}).get("model_id"),
                    "attempt": result.get("attempt"),
                    "round": result.get("round"),
                    "state_accepted": result.get("state_accepted", True),
                    "state": normalize_state(result.get("output", {}).get("task_output")),
                    "output": result.get("output", {}),
                    "usage": result.get("usage", {}),
                }
            )
        return records

    def failed_decision_trace(self, task_id: str) -> list[dict[str, Any]]:
        """返回重试任务传入的失败轨迹，供内部路由阶段复用。"""
        value = self.store.get(task_id)["task"].get("failed_decision_trace", [])
        return list(value) if isinstance(value, list) else []

    def chain_route_context(self, task_id: str, previous_result_id: str | None, previous_role: str) -> dict[str, Any]:
        """构造链式下一步路由的完整任务与前序结果上下文。"""
        payload = self.routing_context(task_id, focused_result_ids=[previous_result_id] if previous_result_id else [])
        payload.update(
            {
                "current_mode": "Chain",
                "previous_role": previous_role,
                "previous_result": next((result for result in payload["results"] if result["result_id"] == previous_result_id), None),
                "remaining_budget": payload["budget"]["remaining"],
            }
        )
        return payload

    def create_mode(self, task_id: str, mode: str, parent_id: str | None = None, limits: dict[str, Any] | None = None, *, decision_mode: str | None = None) -> str:
        """登记路由语义模式及其实际复用的执行拓扑。"""
        instance_id = new_id("mode")
        def mutate(snapshot: dict[str, Any]) -> None:
            snapshot["mode_state"]["active_mode_instance_id"] = instance_id
            snapshot["mode_state"]["instances"].append(
                {
                    "mode_instance_id": instance_id,
                    "mode": decision_mode or mode,
                    "execution_mode": mode,
                    "parent_mode_instance_id": parent_id,
                    "status": "running",
                    "limits": limits or default_limits(snapshot["budget"]["remaining"]["cost"]),
                }
            )
        self.store.update(task_id, mutate)
        return instance_id

    def add_node(self, task_id: str, *, mode_instance_id: str, role: str | None = None, assignment: str = "", acceptance: list[str] | None = None, parent_node_id: str | None = None, initial_context: dict[str, Any] | None = None, depth: int | None = None, permissions: dict[str, Any] | None = None, model_id: str = "deepseek-v4.1-flash", subtask_id: str | None = None, constraints: list[str] | None = None, round: int = 0) -> str:
        from virgo.roles import parse_role_instance

        template_role, domain = parse_role_instance(role) if role else (None, None)
        node_id = new_id("node")
        def mutate(snapshot: dict[str, Any]) -> None:
            node = {"node_id": node_id, "subtask_id": subtask_id, "mode_instance_id": mode_instance_id, "parent_node_id": parent_node_id, "depth": depth, "round": round, "status": "pending", "active_assignment_id": None, "initial_context": initial_context, "domain": domain, "constraints": constraints or [], "input_result_ids": [], "output_result_ids": [], "limits": default_limits(snapshot["budget"]["remaining"]["cost"]), "permissions": permissions or {"allowed_modes": [], "allowed_roles": [], "allowed_models": [], "allowed_tools": []}}
            snapshot["execution_state"]["nodes"].append(node)
            if template_role:
                assignment_id = new_id("assignment")
                model = {"provider": "local", "model_id": model_id, "inference_config": {"temperature": 0.2, "max_output_tokens": 4096}}
                effective_acceptance = list(acceptance or [])
                snapshot["agent_assignments"].append({"assignment_id": assignment_id, "mode_instance_id": mode_instance_id, "node_id": node_id, "agent_id": new_id("agent"), "role": {"name": template_role, "display_name": role or template_role, "domain": domain, "config_ref": "builtin-roles@1.0.0"}, "llm": model, "tools": get_role(template_role).tools, "assignment": assignment, "acceptance_criteria": effective_acceptance})
                node["active_assignment_id"] = assignment_id
                node["status"] = "ready"
        self.store.update(task_id, mutate)
        return node_id

    def workspace_for_node(self, task_id: str, node_id: str) -> str:
        snapshot = self.store.get(task_id)
        node = next(item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == node_id)
        return str(Path(node.get("workspace") or getattr(self, "workspace", ".")).resolve())

    def tool_workspace_for_node(self, task_id: str, node_id: str) -> str:
        """Parallel 若漏绑隔离目录则拒绝执行，不能退回共享工作区。"""
        snapshot = self.store.get(task_id)
        node = next(item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == node_id)
        instance = next(item for item in snapshot["mode_state"]["instances"]
                        if item["mode_instance_id"] == node["mode_instance_id"])
        if instance.get("execution_mode", instance["mode"]) == "Parallel" and not node.get("workspace"):
            raise PermissionError("Parallel 节点尚未绑定独立工作区，拒绝使用共享目录。")
        return self.workspace_for_node(task_id, node_id)

    def bind_workspace(self, task_id: str, node_id: str, binding: dict[str, Any]) -> None:
        """在调用模型前绑定节点目录，执行过程中不修改全局工作区。"""
        def mutate(snapshot):
            node = next(item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == node_id)
            node.update(deepcopy(binding))
        self.store.update(task_id, mutate)

    @staticmethod
    def localize_state(state, mappings):
        """将上游文件引用重定位到本节点可见的副本；不改历史记录。"""
        result = deepcopy(state)
        def relocate(value):
            if not isinstance(value, str):
                return value
            for source, target in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
                if value == source or value.startswith(source + "/"):
                    return target + value[len(source):]
            return value
        for artifact in result["artifacts"]:
            artifact["path"] = relocate(artifact["path"])
        for check in result["verification"]:
            check["target"] = relocate(check["target"])
            check["evidence"] = relocate(check["evidence"])
        return result

    def build_agent_input(self, task_id: str, node_id: str) -> dict[str, Any]:
        snapshot = self.store.get(task_id)
        node = next(node for node in snapshot["execution_state"]["nodes"] if node["node_id"] == node_id)
        assignment = next(a for a in snapshot["agent_assignments"] if a["assignment_id"] == node["active_assignment_id"])
        context = node["initial_context"] or snapshot["initial_context"]
        if not context:
            raise RuntimeError("节点缺少可解析的初始上下文")
        upstream = self.upstream_results(snapshot, node["input_result_ids"])
        constraints = [*snapshot["task"]["constraints"], *node["constraints"],
                       *snapshot["task"]["acceptance_criteria"], *assignment["acceptance_criteria"]]
        extracted = context.get("constraints")
        if isinstance(extracted, str):
            constraints.extend(extracted.splitlines())
        elif isinstance(extracted, list):
            constraints.extend(extracted)
        description = snapshot["task"]["original_task"]
        if node.get("initial_context"):
            description += "\n\n【自足子任务】\n" + context["task_description"]
        # 只传任务特有约束；通用执行协议已由系统提示词承载。
        generic = {*COLLABORATION_ACCEPTANCE_CRITERIA, "输出与用户任务目标一致",
                   "记录可复盘的决策轨迹", "生成结构化结果"}
        constraints = list(dict.fromkeys(
            item.strip() for item in constraints if isinstance(item, str)
            and item.strip() and item.strip().lower() not in {"none", "null"}
            and item.strip() not in generic and item.strip() not in description
        ))
        if node.get("participant_id"):
            own = [record["result_id"] for record in upstream
                   if snapshot["agent_outputs"][record["result_id"]].get("snapshot_key") == self._snapshot_key(snapshot, node)]
            if own:
                constraints.append("当前候选的前序快照为 " + ", ".join(own) + "；其他来源仅作同级参考，输出只维护自己的候选状态。")
        return {
            "task": {"description": description, "constraints": list(dict.fromkeys(constraints))},
            "state": self.localize_state(combine_states(upstream), node.get("input_path_map", {})),
            "environment": {
                "workspace": ".",
                "writable_paths": ["."],
                "constraints": [
                    "仅使用本次提供的工具及其授权范围；文件写入不得越出指定工作区。",
                    "文件工具使用相对于本节点工作区的路径，例如 solution.py；产物和验证目标也可登记相对路径。",
                    f"shell_execute/python_execute 在禁网容器中执行，工作区挂载及命令当前目录均为 {SANDBOX_WORKSPACE}，也可通过 $VIRGO_WORKSPACE 引用；宿主机绝对路径不适用于容器。网页工具受独立访问策略约束。",
                    *(["本节点使用独立工作区副本；原始任务中的项目路径以当前 workspace 为准。不得访问父目录或同级候选目录。"]
                      if node.get("workspace_kind") == "parallel_candidate" else []),
                    *([".candidate_inputs 内是候选产物的导出副本；在当前工作区生成最终交付，禁止将候选目录当作最终交付目录。"]
                      if node.get("workspace_kind") == "parallel_summary" else []),
                    *getattr(self, "environment_constraints", []),
                ],
            },
            "limits": {"tools_remaining": self.node_tool_call_limit(task_id, node_id)
                       or getattr(self, "max_tool_calls", 50)},
        }

    def upstream_results(self, snapshot: dict[str, Any], result_ids: list[str]) -> list[dict[str, Any]]:
        """只传各来源最新的状态快照，原始记录持续保留以供审计。"""
        latest = {}
        requested = set()
        queue = list(result_ids)
        visited = set()
        while queue:
            result_id = queue.pop()
            if result_id in visited:
                continue
            visited.add(result_id)
            record = snapshot["agent_outputs"].get(result_id, {})
            if record.get("state_accepted") is False:
                queue.extend(record.get("preserved_result_ids", []))
            else:
                requested.add(result_id)
        superseded = {old for rid in requested for old in snapshot["agent_outputs"].get(rid, {}).get("supersedes", [])}
        requested -= superseded
        for result_id, record in snapshot["agent_outputs"].items():
            if result_id not in requested:
                continue
            if not record:
                continue
            output = record.get("output", {})
            key = record.get("snapshot_key", record["node_id"])
            latest[key] = {"result_id": result_id, "node_id": record["node_id"],
                           "state": validate_state(output.get("task_output"))}
        return list(latest.values())

    @staticmethod
    def _snapshot_key(snapshot, node):
        """链按承接范围替换；候选、分支和同级参与者分别保留。"""
        instance = next(item for item in snapshot["mode_state"]["instances"]
                        if item["mode_instance_id"] == node["mode_instance_id"])
        mode = instance.get("execution_mode", instance["mode"])
        scope = node.get("subtask_id") or node.get("parent_node_id") or "root"
        if mode in {"Chain", "Single"}:
            identity = scope
        elif mode == "Tree":
            identity = scope
        elif mode == "FullMesh":
            identity = node.get("participant_id") or node["node_id"]
        elif mode == "Parallel":
            identity = node.get("candidate_id") or node["node_id"]
        else:
            identity = node["node_id"]
        return f"{node['mode_instance_id']}:{identity}"

    def surviving_artifacts(self, artifacts: list[dict[str, Any]], *, workspace=None) -> list[dict[str, Any]]:
        """保留仍存在的工作区产物，按位置去重并拒绝路径越界。"""
        kept = {}
        workspace = workspace or getattr(self, "workspace", None)
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("location") or artifact.get("path"), str):
                continue
            location = artifact.get("location") or artifact["path"]
            if location.startswith("workspace://") and workspace is not None:
                root = Path(workspace).resolve()
                path = (root / location.removeprefix("workspace://")).resolve()
                if root not in path.parents or not path.is_file():
                    continue
            elif workspace is not None:
                root = Path(workspace).resolve()
                path = (root / location).resolve()
                if root not in path.parents or not path.is_file():
                    continue
            if workspace is not None:
                try:
                    with path.open("rb") as stream:
                        stream.read(1)
                except OSError:
                    continue
            kept[location] = artifact
        return list(kept.values())

    def inherited_artifacts(self, upstream, *, workspace=None, mappings=None):
        """维护有效继承产物；兼容旧引用并归一为五字段状态的产物格式。"""
        workspace = workspace or getattr(self, "workspace", ".")
        inherited = {}
        for record in upstream:
            if record.get("state_accepted") is False:
                continue
            output = record.get("output") or {}
            candidates = list(output.get("pervious_artifacts") or [])
            state = output.get("task_output")
            if isinstance(state, dict):
                candidates.extend(state.get("artifacts") or [])
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                if {"path", "version", "summary"} <= item.keys():
                    artifact = {key: value for key, value in item.items() if key in {"path", "version", "summary", "source_result_id"}}
                    candidate = normalize_state(None)
                    candidate["artifacts"] = [artifact]
                else:
                    candidate = normalize_state(None, artifacts=[item])
                try:
                    candidate = validate_state(candidate)
                    candidate = self.localize_state(candidate, mappings or {})
                    candidate = validate_artifacts(candidate, workspace)
                    for artifact in candidate["artifacts"]:
                        version = artifact["version"]
                        digest = version.removeprefix("sha256:") if isinstance(version, str) else None
                        if digest and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                            if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != digest.lower():
                                continue
                        inherited[artifact["path"]] = artifact
                except (StateProtocolError, OSError):
                    continue
        return list(inherited.values())

    def node_tool_call_limit(self, task_id: str, node_id: str) -> int | None:
        """读取内部执行限制，不向工作 Agent 输入暴露调度元数据。"""
        snapshot = self.store.get(task_id)
        node = next(item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == node_id)
        assignment = next(item for item in snapshot["agent_assignments"] if item["assignment_id"] == node["active_assignment_id"])
        instance = next(item for item in snapshot["mode_state"]["instances"] if item["mode_instance_id"] == node["mode_instance_id"])
        if instance.get("execution_mode", instance["mode"]) != "Single" and assignment["role"]["name"] not in PLANNING_ROLES:
            return self.collaboration_tool_call_limit
        return None

    def record_output(self, task_id: str, node_id: str, output: AgentOutput, usage: dict[str, Any], call_id: str) -> str:
        result_id = new_id("result")
        snapshot = self.store.get(task_id)
        node = next(item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == node_id)
        effective = self.upstream_results(snapshot, node["input_result_ids"])
        upstream = [snapshot["agent_outputs"][item["result_id"]] for item in effective]
        workspace = self.workspace_for_node(task_id, node_id)
        output.extra["pervious_artifacts"] = self.inherited_artifacts(upstream, workspace=workspace,
                                                                    mappings=node.get("input_path_map", {}))
        preserved_ids = list(node["input_result_ids"])
        if not preserved_ids:
            current = snapshot.get("state_snapshots", {}).get(self._snapshot_key(snapshot, node))
            if current:
                preserved_ids = [current["result_id"]]
        previous = self.localize_state(combine_states(self.upstream_results(snapshot, preserved_ids)), node.get("input_path_map", {}))
        accepted = output.extra.get("protocol_validation", {}).get("accepted", True)
        if accepted:
            try:
                state = validate_state(output.task_output)
                state = self.localize_state(state, node.get("input_path_map", {}))
                state = validate_artifacts(state, workspace)
            except StateProtocolError as exc:
                accepted = False
                output.extra["protocol_validation"] = exc.as_dict()
        if not accepted:
            state = deepcopy(previous)
            output.status = "failed"
            output.error = {"kind": "protocol", "message": "状态接收失败", "details": output.extra["protocol_validation"]}
        else:
            output.extra["protocol_validation"] = {"accepted": True, "stage": "artifacts", "errors": [], "answer_correctness_checked": False}
        output.task_output = state
        data = output.as_dict()
        def mutate(snapshot: dict[str, Any]) -> None:
            node = next(node for node in snapshot["execution_state"]["nodes"] if node["node_id"] == node_id)
            assignment = next(a for a in snapshot["agent_assignments"] if a["assignment_id"] == node["active_assignment_id"])
            snapshot["agent_outputs"][result_id] = {"mode_instance_id": node["mode_instance_id"], "node_id": node_id, "agent_id": assignment["agent_id"], "assignment_id": assignment["assignment_id"], "call_id": call_id, "attempt": len(node["output_result_ids"]) + 1, "round": node["round"], "output": data, "usage": usage}
            key = self._snapshot_key(snapshot, node)
            snapshot["agent_outputs"][result_id]["snapshot_key"] = key
            snapshot["agent_outputs"][result_id]["state_accepted"] = accepted
            if not accepted:
                snapshot["agent_outputs"][result_id]["preserved_result_ids"] = preserved_ids
                snapshot["agent_outputs"][result_id]["supersedes"] = []
                node["output_result_ids"].append(result_id)
                node["status"] = "failed"
                return
            active = snapshot.setdefault("state_snapshots", {})
            integrates = (assignment["role"]["name"] in {"Synthesizer", "ReportWriter"}
                          and not node.get("participant_id")) or (
                assignment["role"]["name"] == "Coordinator" and bool(node["input_result_ids"])) or (
                assignment["role"]["name"] == "SubtaskCoordinator" and bool(node["output_result_ids"]))
            supersedes = []
            effective_inputs = [item["result_id"] for item in self.upstream_results(snapshot, node["input_result_ids"])]
            source_ids = list(dict.fromkeys([*node["input_result_ids"], *effective_inputs]))
            for source in source_ids:
                source_record = snapshot["agent_outputs"].get(source, {})
                if integrates or source_record.get("snapshot_key") == key:
                    supersedes.extend([source, *source_record.get("supersedes", [])])
            snapshot["agent_outputs"][result_id]["supersedes"] = list(dict.fromkeys(supersedes))
            if integrates:
                # 汇总取代其实际整合的快照，不删除历史输出。
                for source in source_ids:
                    old_key = snapshot["agent_outputs"].get(source, {}).get("snapshot_key")
                    active.pop(old_key, None)
            active[key] = {
                "result_id": result_id, "node_id": node_id, "state": deepcopy(output.task_output)}
            node["output_result_ids"].append(result_id)
            node["status"] = "completed" if output.status == "completed" else output.status
        self.store.update(task_id, mutate)
        return result_id

    def publish_output_paths(self, task_id, result_id, mappings):
        """发布后更新正式引用；模型原始交付继续保留在调用轨迹中。"""
        def mutate(snapshot):
            record = snapshot["agent_outputs"][result_id]
            output = record["output"]
            output["task_output"] = self.localize_state(output["task_output"], mappings)
            inherited = normalize_state(None)
            inherited["artifacts"] = output.get("pervious_artifacts", [])
            output["pervious_artifacts"] = self.localize_state(inherited, mappings)["artifacts"]
            output["publication_path_map"] = deepcopy(mappings)
            for current in snapshot.get("state_snapshots", {}).values():
                if current["result_id"] == result_id:
                    current["state"] = deepcopy(output["task_output"])
        snapshot = self.store.update(task_id, mutate)
        return snapshot["agent_outputs"][result_id]["output"]["task_output"]

    def record_publication_failure(self, task_id, result_id, state):
        """发布失败必须同步到正式快照，不能仅在图状态中标记失败。"""
        def mutate(snapshot):
            record = snapshot["agent_outputs"][result_id]
            record["output"]["task_output"] = deepcopy(state)
            record["output"]["runtime"] = {"status": "failed", "error": {"kind": "publication", "message": "汇总产物发布失败"}}
            next(node for node in snapshot["execution_state"]["nodes"]
                 if node["node_id"] == record["node_id"])["status"] = "failed"
            for current in snapshot.get("state_snapshots", {}).values():
                if current["result_id"] == result_id:
                    current["state"] = deepcopy(state)
        self.store.update(task_id, mutate)

    def set_node_inputs(self, task_id: str, node_id: str, result_ids: list[str]) -> None:
        def mutate(snapshot: dict[str, Any]) -> None:
            node = next(node for node in snapshot["execution_state"]["nodes"] if node["node_id"] == node_id)
            node["input_result_ids"] = list(dict.fromkeys(result_ids))
        self.store.update(task_id, mutate)

    def finish_task(self, task_id, status, state, source_result_ids):
        """结束任务，只保存最终 state；空产物列表回填有效 pervious_artifacts。"""
        def mutate(snapshot):
            sources = self.upstream_results(snapshot, source_result_ids)
            records = [snapshot["agent_outputs"][item["result_id"]] for item in sources]
            inherited = self.inherited_artifacts(records)
            # 内部停止消息不是状态，更不能被包装成任务答案。
            current = validate_state(state) if isinstance(state, dict) else combine_states(sources)
            if not current["artifacts"]:
                current["artifacts"] = deepcopy(inherited)
            snapshot["state"] = current
            snapshot["pervious_artifacts"] = inherited
            snapshot["execution_state"]["status"] = status
            snapshot["execution_state"]["source_result_ids"] = [item["result_id"] for item in sources]
        return self.store.update(task_id, mutate)

    def connect(self, task_id: str, source: str, target: str, edge_type: str = "dependency") -> None:
        self.store.update(task_id, lambda snapshot: snapshot["execution_state"]["edges"].append({"source_node_id": source, "target_node_id": target, "edge_type": edge_type}))

    def finish_mode(self, task_id: str, mode_instance_id: str, result: Any) -> None:
        def mutate(snapshot: dict[str, Any]) -> None:
            instance = next(i for i in snapshot["mode_state"]["instances"] if i["mode_instance_id"] == mode_instance_id)
            instance["status"] = result.status
            snapshot["mode_state"]["active_mode_instance_id"] = instance["parent_mode_instance_id"]
        self.store.update(task_id, mutate)

