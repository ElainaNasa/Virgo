"""预算预留与结算。"""

from __future__ import annotations

import threading
from typing import Any

from virgo.context.store import ContextStore
from virgo.observability import ExecutionLogger
from virgo.types import Usage


class BudgetExceeded(RuntimeError):
    """当前预算不允许发起新的模型调用。"""


class BudgetLedger:
    def __init__(self, store: ContextStore, logger: ExecutionLogger | None = None) -> None:
        self.store = store
        self._lock = threading.RLock()
        self.logger = logger

    def reserve(self, task_id: str, estimated_cost: float = 0.0) -> str:
        reservation_id = f"reservation_{id(self)}_{threading.get_ident()}"
        with self._lock:
            def mutate(snapshot: dict[str, Any]) -> None:
                remaining = snapshot["budget"]["remaining"]["cost"]
                if remaining is not None and remaining - estimated_cost < 0:
                    raise BudgetExceeded(f"预算不足：剩余 {remaining:.6f} CNY，预计 {estimated_cost:.6f} CNY")
                snapshot["budget"]["reserved"]["calls"] += 1
            self.store.update(task_id, mutate)
        if self.logger:
            self.logger.emit("budget_reserved", task_id=task_id, status="reserved", reservation_id=reservation_id, estimated_cost=estimated_cost)
        return reservation_id

    def settle(self, task_id: str, usage: Usage, reservation_cost: float = 0.0) -> dict[str, Any]:
        with self._lock:
            def mutate(snapshot: dict[str, Any]) -> None:
                budget = snapshot["budget"]
                budget["reserved"]["calls"] = max(0, budget["reserved"]["calls"] - 1)
                if budget["remaining"]["cost"] is not None:
                    budget["remaining"]["cost"] = round(budget["remaining"]["cost"] - usage.cost, 8)
                if budget["remaining"]["tokens"] is not None:
                    budget["remaining"]["tokens"] -= usage.tokens
                if budget["remaining"]["calls"] is not None:
                    budget["remaining"]["calls"] -= usage.calls
            budget = self.store.update(task_id, mutate)["budget"]
            if self.logger:
                self.logger.emit("budget_settled", task_id=task_id, status="settled", cost=usage.cost, tokens=usage.tokens, calls=usage.calls, remaining=budget["remaining"])
            return budget

    def release(self, task_id: str, reservation_cost: float = 0.0) -> None:
        with self._lock:
            def mutate(snapshot: dict[str, Any]) -> None:
                snapshot["budget"]["reserved"]["calls"] = max(0, snapshot["budget"]["reserved"]["calls"] - 1)
            self.store.update(task_id, mutate)
            if self.logger:
                self.logger.emit("budget_released", task_id=task_id, status="released", reservation_cost=reservation_cost)

    def available(self, task_id: str) -> bool:
        budget = self.store.get(task_id)["budget"]
        remaining = budget["remaining"]["cost"]
        return remaining is None or remaining >= 0
