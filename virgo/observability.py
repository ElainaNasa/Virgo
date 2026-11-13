"""运行日志与阶段状态跟踪。

日志用于定位运行问题，实验轨迹仍由 ``TraceRecorder`` 保存完整上下文和模型输出。
这里的日志只写摘要和关联标识，避免把大段任务上下文重复写入运行日志。
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator


class JsonLogFormatter(logging.Formatter):
    """将每条标准日志格式化为一行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", None)
        if payload is None:
            payload = {"message": record.getMessage()}
        return json.dumps(payload, ensure_ascii=False, default=str)


class ExecutionLogger:
    """集中记录代码执行阶段和异常状态。"""

    def __init__(self, log_dir: str | Path, *, level: str = "INFO", console: bool = False) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "virgo-execution.log"
        logger_name = f"virgo.execution.{self.log_dir.resolve()}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False
        self._logger.handlers.clear()
        formatter = JsonLogFormatter()
        file_handler = RotatingFileHandler(self.log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def emit(self, event: str, *, level: str = "INFO", message: str | None = None, **fields: Any) -> None:
        payload = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            "level": level.upper(),
            "message": message or event,
            **fields,
        }
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(numeric_level, payload.get("message", event), extra={"payload": payload})

    @contextmanager
    def stage(self, stage: str, **fields: Any) -> Iterator[str]:
        """记录阶段的开始、成功或失败，并返回本阶段唯一 ID。"""
        stage_id = f"stage_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        self.emit("stage_started", stage=stage, stage_id=stage_id, status="running", **fields)
        try:
            yield stage_id
        except Exception as exc:
            self.emit("stage_failed", level="ERROR", stage=stage, stage_id=stage_id, status="failed", duration_ms=int((time.perf_counter() - started) * 1000), error_type=type(exc).__name__, error=str(exc), **fields)
            raise
        else:
            self.emit("stage_completed", stage=stage, stage_id=stage_id, status="completed", duration_ms=int((time.perf_counter() - started) * 1000), **fields)

    def close(self) -> None:
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
