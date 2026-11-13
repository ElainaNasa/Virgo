"""失败决策轨迹的压缩与归档数据构造。"""

from __future__ import annotations

import re
from typing import Any


def actual_cost(snapshot: dict[str, Any]) -> float:
    """计算一次任务尝试已结算的实际费用。"""
    budget = snapshot.get("budget", {})
    total = budget.get("total", {}).get("cost")
    remaining = budget.get("remaining", {}).get("cost")
    if total is None or remaining is None:
        return 0.0
    return round(max(0.0, float(total) - float(remaining)), 8)


def compact_failed_decision_trace(
    snapshot: dict[str, Any],
    *,
    attempt: int,
    failure: dict[str, Any],
) -> dict[str, Any]:
    """从权威快照构造可发送给后续路由器的简化失败轨迹。"""
    assignments = {
        item.get("node_id"): item
        for item in snapshot.get("agent_assignments", [])
        if isinstance(item, dict) and item.get("node_id")
    }
    nodes = [item for item in snapshot.get("execution_state", {}).get("nodes", []) if isinstance(item, dict)]
    nodes_by_mode: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        mode_id = node.get("mode_instance_id")
        if isinstance(mode_id, str):
            nodes_by_mode.setdefault(mode_id, []).append(node)

    mode_instances = [
        item
        for item in snapshot.get("mode_state", {}).get("instances", [])
        if isinstance(item, dict) and isinstance(item.get("mode_instance_id"), str)
    ]
    node_aliases: dict[str, str] = {}
    for node_index, node in enumerate(nodes, start=1):
        if isinstance(node.get("node_id"), str):
            node_aliases[node["node_id"]] = f"N{node_index}"

    mode_runs: list[dict[str, Any]] = []
    for mode_index, instance in enumerate(mode_instances, start=1):
        mode_id = instance["mode_instance_id"]
        simplified_nodes = []
        for node in nodes_by_mode.get(mode_id, []):
            node_id = str(node.get("node_id"))
            assignment = assignments.get(node_id, {})
            simplified_nodes.append(
                {
                    "id": node_aliases.get(node_id, node_id),
                    "role": assignment.get("role", {}).get("name"),
                    "model_id": assignment.get("llm", {}).get("model_id"),
                }
            )
        mode_node_ids = {node.get("node_id") for node in nodes_by_mode.get(mode_id, [])}
        edges = []
        for edge in snapshot.get("execution_state", {}).get("edges", []):
            source = edge.get("source_node_id")
            target = edge.get("target_node_id")
            if source in mode_node_ids and target in mode_node_ids:
                edges.append([node_aliases.get(str(source), str(source)), node_aliases.get(str(target), str(target))])
        mode_run = {
            "id": f"M{mode_index}",
            "mode": instance.get("mode"),
            "nodes": simplified_nodes,
            "edges": edges,
            "exit": "continue" if mode_index < len(mode_instances) else "stop",
        }
        execution_mode = instance.get("execution_mode")
        if isinstance(execution_mode, str) and execution_mode != mode_run["mode"]:
            mode_run["execution_mode"] = execution_mode
        mode_runs.append(mode_run)

    return {
        "attempt": attempt,
        "mode_runs": mode_runs,
        "failure": _failure_fields(failure),
        "cost": actual_cost(snapshot),
    }


def _failure_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(value.get("type", "execution_failed")),
        "summary": str(value.get("summary", "任务执行或验收失败。"))[:1000],
        "evidence": summarize_evidence(str(value.get("evidence", "None"))),
        "location": value.get("location") if isinstance(value.get("location"), str) else None,
    }


def failure_from_assessment(
    *,
    execution_status: str,
    final_state: dict[str, Any] | None,
    compile_check: dict[str, Any],
    official_test: dict[str, Any],
    location: str | None,
) -> dict[str, Any]:
    """将执行、编译或公开测试结果归一为可供下次路由使用的失败说明。"""
    if compile_check.get("status") != "passed":
        if execution_status != "completed":
            issues = (final_state or {}).get("possible_issue", [])
            evidence = "；".join(str(item) for item in issues) or "；".join(str(item) for item in (final_state or {}).get("confirmed", [])) or "任务未产生有效交付。"
            return {
                "type": "execution_partial" if execution_status == "partial" else "execution_failed",
                "summary": f"任务执行状态为 {execution_status}，且没有可测试的有效提交。",
                "evidence": evidence,
                "location": location,
            }
        return {
            "type": "submission_invalid",
            "summary": "未产生可供测试的有效 Python 提交。",
            "evidence": str(compile_check.get("reason", "代码提取或编译失败。")),
            "location": location,
        }
    official_status = official_test.get("status")
    if official_status == "not_run":
        issues = (final_state or {}).get("possible_issue", [])
        evidence = "；".join(str(item) for item in issues) or str(official_test.get("reason", "测试未运行。"))
        return {
            "type": "execution_partial" if execution_status == "partial" else "execution_failed",
            "summary": f"任务执行状态为 {execution_status}，公开测试未运行。",
            "evidence": evidence,
            "location": location,
        }
    if official_status == "timeout":
        return {
            "type": "test_timeout",
            "summary": "公开测试在时间上限内未完成。",
            "evidence": str(official_test.get("evidence") or official_test.get("reason")),
            "location": location,
        }
    return {
        "type": "answer_incorrect",
        "summary": "最终提交未通过公开测试。",
        "evidence": summarize_evidence(str(official_test.get("evidence") or official_test.get("reason") or "公开测试失败。")),
        "location": location,
    }


def summarize_evidence(value: str) -> str:
    """从 unittest 输出提取失败用例、异常或断言，避免向路由器复制完整日志。"""
    normalized = value.strip()
    if len(normalized) <= 1000:
        return normalized or "None"
    signals: list[str] = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:FAIL|ERROR): test_", line) or re.match(r"^test_[^(]+", line):
            signals.append(line)
        elif re.match(r"^(?:AssertionError|[A-Za-z_]*Error|[A-Za-z_]*Exception):", line):
            signals.append(line)
    if signals:
        return "\n".join(dict.fromkeys(signals))[:1000]
    return normalized[-1000:]
