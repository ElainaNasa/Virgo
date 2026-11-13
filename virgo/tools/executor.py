"""工具权限、参数校验、超时和调用日志。"""

from __future__ import annotations

import json
import time
import uuid
import urllib.error
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from virgo.tools.registry import ToolRegistry
from virgo.tools.patch import PatchError
from virgo.tools.evidence import archive_tool_result, process_evidence


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, workspace: str, recorder: Any | None = None, blocked_tools: set[str] | None = None, benchmark_task_loader: Any | None = None, workspace_resolver: Any | None = None) -> None:
        self.registry = registry
        self.workspace = workspace
        self.workspace_resolver = workspace_resolver
        self.recorder = recorder
        self.benchmark_task_loader = benchmark_task_loader
        self.blocked_tools = frozenset(blocked_tools or ())

    def execute(self, *, task_id: str, node_id: str, agent_id: str, tool_id: str, arguments: dict[str, Any], allowed_tools: list[str]) -> dict[str, Any]:
        call_id = f"tool_call_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        raw_data = None
        workspace = None
        prohibited_lookup = False
        if self.benchmark_task_loader and tool_id in {"web_search", "web_fetch"}:
            from virgo.tools.benchmark_policy import answer_lookup_request
            prohibited_lookup = answer_lookup_request(arguments, self.benchmark_task_loader(task_id))
        if prohibited_lookup or tool_id not in allowed_tools or tool_id in self.blocked_tools:
            result = self._result(call_id, "denied", error={"code": "tool_not_allowed", "message": "禁止搜索题号、原题、题解或答案；允许查询通用知识和技术文档。" if prohibited_lookup else f"工具未授权：{tool_id}", "retryable": False})
            if self.recorder:
                self.recorder.record(task_id, "tool_call_denied", context={"node_id": node_id, "agent_id": agent_id, "tool_id": tool_id, "arguments": arguments}, output=result)
            return result
        try:
            spec = self.registry.get(tool_id)
            self._validate(arguments, spec.parameters)
            workspace = self.workspace_resolver(task_id, node_id) if self.workspace_resolver else self.workspace
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(spec.executor, arguments, workspace)
                data = future.result(timeout=spec.timeout_seconds)
                raw_data = data
            duration_ms = int((time.perf_counter() - started) * 1000)
            if self._execution_failed(data):
                # 子进程能返回结果不代表其执行成功。保留 stdout/stderr 供诊断，
                # 但绝不把非零退出码作为写入或验证已经完成的证据。
                result = self._result(
                    call_id,
                    "failed",
                    data=data,
                    error={
                        "code": "command_failed",
                        "message": f"命令执行失败，退出码为 {data['returncode']}。请检查 stdout 和 stderr；后续步骤可能未执行。",
                        "retryable": False,
                    },
                    duration_ms=duration_ms,
                )
            else:
                text = json.dumps(data, ensure_ascii=False, default=str)
                truncated = len(text) > spec.max_output_chars
                artifacts = self._artifacts(call_id, tool_id, data)
                from pathlib import Path
                if Path(workspace).resolve() != Path(self.workspace).resolve():
                    for artifact in artifacts:
                        artifact["location"] = str(Path(workspace) / artifact["location"].removeprefix("workspace://"))
                result = self._result(
                    call_id,
                    "succeeded",
                    data=text[:spec.max_output_chars] if truncated else data,
                    artifacts=artifacts,
                    truncated=truncated,
                    duration_ms=duration_ms,
                )
        except TimeoutError as exc:
            raw_data = {"returncode": None, "timed_out": True,
                        "stdout": getattr(exc, "stdout", ""), "stderr": getattr(exc, "stderr", "")}
            if tool_id in {"web_search", "web_fetch"}:
                result = self._web_unavailable(call_id, TimeoutError("网页访问超时"), started)
            else:
                result = self._result(call_id, "timeout", error={"code": "timeout", "message": "工具执行超时", "retryable": spec.read_only if 'spec' in locals() else False}, duration_ms=int((time.perf_counter() - started) * 1000))
        except PatchError as exc:
            result = self._result(call_id, "failed", error=exc.details, duration_ms=int((time.perf_counter() - started) * 1000))
        except Exception as exc:
            if tool_id in {"web_search", "web_fetch"} and isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
                result = self._web_unavailable(call_id, exc, started)
            else:
                result = self._result(call_id, "failed", error={"code": "tool_error", "message": str(exc), "retryable": False}, duration_ms=int((time.perf_counter() - started) * 1000))
        evidence = process_evidence(tool_id, arguments, raw_data, result.get("error"))
        if evidence:
            result["decision_evidence"] = evidence
        archive_error = None
        if self.recorder is not None and workspace is not None:
            try:
                result.update(archive_tool_result(workspace, node_id, call_id, tool_id, arguments, result, raw_data))
            except OSError as exc:
                archive_error = f"{type(exc).__name__}: {exc}"
                result["archive_error"] = archive_error
        if self.recorder:
            # 归档失败时在审计轨迹中保留原始数据，不能静默丢失完整结果。
            if archive_error:
                self.recorder.record(task_id, "tool_full_result", context={"node_id": node_id, "tool_call_id": call_id},
                                     output={"data": raw_data, "archive_error": archive_error})
            self.recorder.record(task_id, "tool_call", context={"node_id": node_id, "agent_id": agent_id, "tool_id": tool_id, "arguments": arguments}, output=result)
        return result

    def _web_unavailable(self, call_id: str, exc: Exception, started: float) -> dict[str, Any]:
        """网页访问异常作为非致命反馈交给模型，不推断整项任务失败。"""
        code = getattr(exc, "code", None)
        message = "网页访问未成功；网站可能不支持自动程序访问、暂时不可达或存在网络异常。"
        if code in {401, 403}:
            message = f"网站拒绝了当前请求（HTTP {code}），可能需要授权或不支持自动程序访问。"
        return self._result(
            call_id, "unavailable",
            data={"non_fatal": True, "guidance": "更换允许的资料来源，或基于已有信息继续。不要反复访问同一失败地址；不能仅因网页不可用就将节点判为 blocked/failed，只有必要信息确实缺失且无替代途径时才报告阻塞。"},
            error={"code": "web_access_unavailable", "message": message, "detail": str(exc), "retryable": False},
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def limit_reached(self, *, task_id: str, node_id: str, agent_id: str, tool_id: str | None, maximum: int) -> dict[str, Any]:
        """拒绝超出 Agent 工具调用上限的请求，并保留完整调用日志。"""
        call_id = f"tool_call_{uuid.uuid4().hex[:12]}"
        result = self._result(
            call_id,
            "denied",
            error={
                "code": "tool_call_limit_reached",
                "message": f"当前 Agent 的实际工具调用上限为 {maximum} 次；请基于已返回结果提交最终输出。",
                "retryable": False,
            },
        )
        if self.recorder:
            self.recorder.record(
                task_id,
                "tool_call_denied",
                context={"node_id": node_id, "agent_id": agent_id, "tool_id": tool_id, "arguments": {}, "reason": "tool_call_limit_reached", "maximum": maximum},
                output=result,
            )
        return result

    @staticmethod
    def _validate(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是对象")
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f"缺少工具参数：{missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = set(arguments) - set(properties)
            if unknown:
                raise ValueError(f"未知工具参数：{sorted(unknown)}")
        for name, value in arguments.items():
            if name in properties:
                ToolExecutor._validate_value(name, value, properties[name])

    @staticmethod
    def _validate_value(name: str, value: Any, schema: dict[str, Any]) -> None:
        expected = schema.get("type")
        type_matches = {
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }
        if expected and not type_matches.get(expected, True):
            raise ValueError(f"参数 {name} 必须是 {expected}")
        if isinstance(value, str) and "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ValueError(f"参数 {name} 长度不足")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"参数 {name} 小于最小值")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"参数 {name} 大于最大值")
        if isinstance(value, list):
            if "maxItems" in schema and len(value) > int(schema["maxItems"]):
                raise ValueError(f"参数 {name} 项数超过上限")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    ToolExecutor._validate_value(f"{name}[{index}]", item, item_schema)

    @staticmethod
    def _artifacts(call_id: str, tool_id: str, data: Any) -> list[dict[str, str]]:
        """只登记工具已实际写入的工作区文件。"""
        if tool_id != "file_write" or not isinstance(data, dict) or not isinstance(data.get("path"), str):
            return []
        location = f"workspace://{data['path']}"
        sha256 = data.get("sha256")
        size = data.get("bytes")
        description = f"由 file_write 实际写入的文件；bytes={size}"
        if isinstance(sha256, str):
            description += f"；sha256={sha256}"
        artifact_type = "code" if data["path"].endswith(".py") else "file"
        return [{"id": f"artifact_{call_id}", "type": artifact_type, "location": location, "description": description}]

    @staticmethod
    def _execution_failed(data: Any) -> bool:
        """识别具有进程语义的工具结果，避免混淆请求完成与命令成功。"""
        return isinstance(data, dict) and isinstance(data.get("returncode"), int) and data["returncode"] != 0

    @staticmethod
    def _result(call_id: str, status: str, data: Any = None, error: dict[str, Any] | None = None, artifacts: list[dict[str, str]] | None = None, truncated: bool = False, duration_ms: int = 0) -> dict[str, Any]:
        return {"tool_call_id": call_id, "status": status, "data": data, "error": error, "artifacts": artifacts or [], "truncated": truncated, "usage": {"duration_ms": duration_ms, "cost": 0, "currency": "CNY"}}
