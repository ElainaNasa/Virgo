"""创建符合任务上下文契约的初始状态。"""

from __future__ import annotations

import uuid
from typing import Any

from virgo.types import BudgetLimits, TaskRequest
from virgo.task_state import empty_state


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def money_limits(tier: str, explicit_cost: float | None) -> float | None:
    if explicit_cost is not None:
        return explicit_cost
    return {"xlow": 1.0, "low": 3.0, "medium": 6.0, "high": 12.0, "xhigh": 25.0}.get(tier, 3.0)


BUDGET_TIER_ORDER = {"xlow": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}


def recommended_tier_is_lower(recommended_tier: str, requested_tier: str) -> bool:
    """仅比较已定义预算档位；未知推荐值不改变用户的预算设置。"""
    return (
        recommended_tier in BUDGET_TIER_ORDER
        and requested_tier in BUDGET_TIER_ORDER
        and BUDGET_TIER_ORDER[recommended_tier] < BUDGET_TIER_ORDER[requested_tier]
    )


def create_snapshot(request: TaskRequest) -> dict[str, Any]:
    task_id = new_id("task")
    cost_limit = money_limits(request.budget_tier, request.budget_cost)
    return {
        "schema_version": "1.0.0",
        "revision": 0,
        "registry_snapshot": {"modes": "builtin-modes@1.0.0", "roles": "builtin-roles@1.0.0", "models": "builtin-models@1.0.0", "tools": "builtin-tools@1.0.0"},
        "task": {
            "task_id": task_id,
            "original_task": request.task,
            "constraints": list(request.constraints),
            "acceptance_criteria": list(request.acceptance_criteria),
            "failed_decision_trace": list(request.failed_decision_trace),
        },
        "budget": {
            "currency": "CNY",
            "total": {"cost": cost_limit, "tokens": None, "calls": None, "time_seconds": None},
            "remaining": {"cost": cost_limit, "tokens": None, "calls": None, "time_seconds": None},
            "reserved": {"tokens": 0, "calls": 0, "time_seconds": 0.0},
        },
        "initial_context": None,
        "mode_state": {"active_mode_instance_id": None, "instances": []},
        "agent_assignments": [],
        "execution_state": {"status": "pending", "nodes": [], "edges": [], "source_result_ids": []},
        "agent_outputs": {},
        "state_snapshots": {},
        "state": empty_state(),
        "pervious_artifacts": [],
    }


def default_limits(cost_limit: float | None = None, *, max_agents: int | None = None, max_rounds: int | None = None, allow_nesting: bool = True) -> dict[str, Any]:
    return BudgetLimits(cost_limit=cost_limit, max_agents=max_agents, max_rounds=max_rounds, allow_further_nesting=allow_nesting).as_dict()
