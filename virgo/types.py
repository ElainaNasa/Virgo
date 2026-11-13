"""跨模块共享的数据类型。

这些类型只表达业务事实，不包含调度策略，便于以后替换存储或模型服务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DURATION_TIERS = ("<5 minutes", ">5 & <15 minutes", ">15 & <30 minutes", ">30 & <45 minutes", ">45 & <60 minutes", ">60 minutes")
_DURATION_TIER_ALIASES = {
    "<5minutes": "<5 minutes",
    "<15minutes": ">5 & <15 minutes",
    "<30minutes": ">15 & <30 minutes",
    "<60minutes": ">45 & <60 minutes",
    ">60minutes": ">60 minutes",
    "15-30minutes": ">15 & <30 minutes",
    "30-60minutes": ">45 & <60 minutes",
    "60+minutes": ">60 minutes",
}


def normalize_duration_tier(value: Any) -> str:
    """规范化时长档位；兼容历史快照中的分钟整数。"""
    raw = str(value).strip()
    if raw in DURATION_TIERS:
        return raw
    if raw in _DURATION_TIER_ALIASES:
        return _DURATION_TIER_ALIASES[raw]
    try:
        minutes = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"初始上下文 estimated_duration_minutes 非法：{value!r}") from exc
    if minutes <= 5:
        return "<5 minutes"
    if minutes < 15:
        return ">5 & <15 minutes"
    if minutes < 30:
        return ">15 & <30 minutes"
    if minutes < 45:
        return ">30 & <45 minutes"
    if minutes < 60:
        return ">45 & <60 minutes"
    return ">60 minutes"


def is_long_duration_tier(value: Any) -> bool:
    """超过 30 分钟的预计时长视为长程任务。"""
    return normalize_duration_tier(value) in {">30 & <45 minutes", ">45 & <60 minutes", ">60 minutes"}


@dataclass(frozen=True)
class InitialContext:
    task_description: str
    tasktype: str
    domains: str
    difficulty: int
    recommended_budget: str
    decompose: str
    constraints: str
    estimated_duration_minutes: str = ">15 & <30 minutes"

    def __post_init__(self) -> None:
        raw_difficulty = str(self.difficulty).strip()
        if raw_difficulty.endswith("/100"):
            raw_difficulty = raw_difficulty.removesuffix("/100").strip()
        try:
            score = int(raw_difficulty)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"初始上下文 difficulty 非法：{self.difficulty!r}") from exc
        object.__setattr__(self, "difficulty", max(1, min(100, score)))
        object.__setattr__(self, "estimated_duration_minutes", normalize_duration_tier(self.estimated_duration_minutes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_description": self.task_description,
            "tasktype": self.tasktype,
            "domains": self.domains,
            "difficulty": self.difficulty,
            "recommended_budget": self.recommended_budget,
            "decompose": self.decompose,
            "constraints": self.constraints,
            "estimated_duration_minutes": self.estimated_duration_minutes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InitialContext":
        if "difficulty" not in value:
            raise ValueError("初始上下文缺少 difficulty")
        raw_difficulty = str(value["difficulty"]).strip()
        if raw_difficulty.endswith("/100"):
            raw_difficulty = raw_difficulty.removesuffix("/100").strip()
        try:
            difficulty = int(raw_difficulty)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"初始上下文 difficulty 非法：{value.get('difficulty')!r}") from exc
        return cls(
            task_description=str(value.get("task_description", "")),
            tasktype=str(value.get("tasktype", "Other")),
            domains=str(value.get("domains", "General")),
            difficulty=max(1, min(100, difficulty)),
            recommended_budget=str(value.get("recommended_budget", "low")),
            decompose=str(value.get("decompose", "not recommend")),
            constraints=str(value.get("constraints", "None")),
            estimated_duration_minutes=normalize_duration_tier(value.get("estimated_duration_minutes", ">15 & <30 minutes")),
        )


@dataclass(frozen=True)
class BudgetLimits:
    cost_limit: float | None = None
    token_limit: int | None = None
    call_limit: int | None = None
    time_limit_seconds: float | None = None
    max_agents: int | None = None
    max_rounds: int | None = None
    max_retries: int | None = None
    allow_further_nesting: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "cost_limit": self.cost_limit,
            "currency": "CNY",
            "token_limit": self.token_limit,
            "call_limit": self.call_limit,
            "time_limit_seconds": self.time_limit_seconds,
            "max_agents": self.max_agents,
            "max_rounds": self.max_rounds,
            "max_retries": self.max_retries,
            "allow_further_nesting": self.allow_further_nesting,
        }


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    calls: int = 1
    duration_ms: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.tokens,
            "cost": round(self.cost, 8),
            "calls": self.calls,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ModelResponse:
    content: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """业务状态唯一保存在 task_output；status/error 仅为运行时执行元数据。"""
    task_output: dict[str, Any]
    status: str = "completed"
    error: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from copy import deepcopy
        self.task_output = deepcopy(self.task_output)

    def as_dict(self) -> dict[str, Any]:
        legacy = {"task_phase", "task_description", "status", "CoT_summary", "cot_summary",
                  "verification", "artifacts", "possible_issue", "progress", "remaining_work", "next_action",
                  "runtime", "task_output"}
        return {**{key: value for key, value in self.extra.items() if key not in legacy},
                "task_output": self.task_output,
                "runtime": {"status": self.status, "error": self.error}}


@dataclass
class ModeResult:
    mode: str
    status: str
    output: dict[str, Any] | str
    result_ids: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    next_mode: str | None = None


@dataclass(frozen=True)
class TaskRequest:
    task: str
    budget_tier: str = "low"
    budget_cost: float | None = None
    constraints: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    failed_decision_trace: tuple[dict[str, Any], ...] = ()
    initial_context: InitialContext | dict[str, Any] | None = None

