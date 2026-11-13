"""决策与执行轨迹记录器。"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from virgo.observability import ExecutionLogger


class TraceRecorder:
    def __init__(self, trace_dir: str | Path, logger: ExecutionLogger | None = None) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.logger = logger

    @staticmethod
    def _timestamp() -> str:
        """按本地时区记录年月日时分，月日不补零。"""
        now = datetime.now().astimezone()
        return f"{now.year}.{now.month}.{now.day}.{now.hour:02d}.{now.minute:02d}"

    def record(self, task_id: str, event_type: str, *, context: Any = None, output: Any = None, budget: dict[str, Any] | None = None, **extra: Any) -> None:
        base = {"timestamp": self._timestamp(), "event_type": event_type, "task_id": task_id}
        if event_type.endswith("decision"):
            # JSONL 按文本查看时优先露出路由决策，完整上下文仍紧随其后保留。
            event = {**base, "output": output, "context": context, "budget": budget, **extra}
        else:
            event = {**base, "context": context, "output": output, "budget": budget, **extra}
        path = self.trace_dir / f"{task_id}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if self.logger:
            status = ((output.get("runtime") or {}).get("status") or output.get("status")) if isinstance(output, dict) else None
            context_fields = context if isinstance(context, dict) else {}
            level = "ERROR" if event_type.endswith("failed") else ("WARNING" if event_type.endswith("denied") else "INFO")
            self.logger.emit(
                event_type,
                level=level,
                message=f"执行事件：{event_type}",
                task_id=task_id,
                status=status,
                phase=context_fields.get("phase"),
                mode=context_fields.get("mode") or extra.get("mode"),
                node_id=extra.get("node_id") or context_fields.get("node_id"),
                mode_instance_id=extra.get("mode_instance_id") or context_fields.get("mode_instance_id"),
                **{key: value for key, value in extra.items() if key not in {"mode", "mode_instance_id", "node_id"}},
            )
            if isinstance(output, dict) and output.get("_router_fallback"):
                fallback = output["_router_fallback"]
                self.logger.emit("router_fallback", level="WARNING", message="首选路由模型调用失败，已切换到回退模型", task_id=task_id, status="fallback", from_model=fallback.get("from_model"), to_model=fallback.get("to_model"), error_type=fallback.get("error_type"), error=fallback.get("error"), phase=context_fields.get("phase"), mode=context_fields.get("mode"))

    def read(self, task_id: str) -> list[dict[str, Any]]:
        path = self.trace_dir / f"{task_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summary(self, task_id: str) -> dict[str, Any]:
        events = self.read(task_id)
        return {"task_id": task_id, "events": len(events), "decisions": sum(event["event_type"].endswith("decision") for event in events), "success": any(event["event_type"] == "task_completed" for event in events), "events_data": events}

    def read_runtime_log(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """读取运行日志；指定 task_id 时只返回该任务的日志。"""
        path = self.logger.log_path if self.logger else None
        if path is None or not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if task_id is None or record.get("task_id") == task_id:
                records.append(record)
        return records
