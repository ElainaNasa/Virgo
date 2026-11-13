"""任务状态快照存储。

存储采用 JSON 文件加 revision 乐观锁，便于实验中复盘和替换为数据库。
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from virgo.observability import ExecutionLogger


class ContextStore:
    def __init__(self, state_dir: str | Path, logger: ExecutionLogger | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.logger = logger

    def path_for(self, task_id: str) -> Path:
        return self.state_dir / f"{task_id}.json"

    def create(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            path = self.path_for(snapshot["task"]["task_id"])
            if path.exists():
                raise FileExistsError(f"任务已存在：{snapshot['task']['task_id']}")
            self._write(path, snapshot)
            if self.logger:
                self.logger.emit("state_created", task_id=snapshot["task"]["task_id"], revision=snapshot["revision"], status="created")
            return copy.deepcopy(snapshot)

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            with self.path_for(task_id).open(encoding="utf-8") as file:
                return json.load(file)

    def update(self, task_id: str, mutate: Any, expected_revision: int | None = None) -> dict[str, Any]:
        with self._lock:
            snapshot = self.get(task_id)
            if expected_revision is not None and snapshot["revision"] != expected_revision:
                raise RuntimeError(f"状态 revision 冲突：期望 {expected_revision}，实际 {snapshot['revision']}")
            try:
                mutate(snapshot)
            except Exception as exc:
                if self.logger:
                    self.logger.emit("state_update_failed", level="ERROR", task_id=task_id, status="failed", error_type=type(exc).__name__, error=str(exc), revision=snapshot["revision"])
                raise
            snapshot["revision"] += 1
            self._write(self.path_for(task_id), snapshot)
            if self.logger:
                self.logger.emit("state_committed", task_id=task_id, revision=snapshot["revision"], status="committed")
            return copy.deepcopy(snapshot)

    def list_tasks(self) -> list[str]:
        return sorted(path.stem for path in self.state_dir.glob("*.json"))

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(value, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
