"""统一 Agent 执行器。

路由模型只决定配置，执行模型由 provider 注入；默认 provider 可指向本地模型或离线演示模型。
"""

from __future__ import annotations

from virgo.task_state import is_task_state, normalize_state, empty_state
from virgo.roles.prompt_data import build_role_prompt
from virgo.state_protocol import StateProtocolError, validate_output, validate_state

import json
import hashlib
from pathlib import Path
import re
import time
import uuid
from dataclasses import replace
from typing import Any

from virgo.context.system import ContextSystem
from virgo.controller.budget import BudgetExceeded, BudgetLedger
from virgo.controller.tool_history import NodeToolHistory, validate_tool_pairs, recent_complete_exchanges, HISTORY_MARKER
from virgo.models import ModelPool
from virgo.providers.base import ModelProvider
from virgo.roles import get_role
from virgo.roles.runtime_prompts import programmer_code_benchmark_prompt, programmer_swebench_prompt
from virgo.tools import ToolExecutor, ToolRegistry
from virgo.trace import TraceRecorder
from virgo.types import AgentOutput, ModelResponse, Usage


TOOL_FEEDBACK_TOTAL_CHARS = 48_000
TOOL_FEEDBACK_RESULT_CHARS = 4_000
CONTEXT_COMPACTION_SYSTEM_PROMPT = """你是 Agent 执行上下文压缩器。将给定对话压缩为一份可供同一 Agent 继续执行的准确工作摘要。
必须保留：原始任务和交付物、硬约束、当前角色与分配、已读取的关键事实、源码路径和实际改动、测试命令与结果、产物、工具错误、尚未完成的工作。
不要编造事实，不要调用工具，不要输出 DSML 或工具协议。只输出一个紧凑的 JSON 对象。"""
FINAL_RESPONSE_RECOVERY_PROMPT = """上一条响应未被接受：它是内部 DSML 工具协议残片、空正文，或最终 JSON 缺少有效的 task_output。
已完成的工具结果仍在当前对话中。不要调用任何工具，不要输出 DSML 或其他工具协议标记。
请立即仅返回一个符合当前角色输出契约的最终 JSON 对象。task_output 必须是五字段状态对象，包含 confirmed、possible_issue、artifacts、verification、remaining 五个数组；不要回显输入上下文或上游结果对象。"""
TOOL_BUDGET_EXHAUSTED_PROMPT = """本节点的工具调用预算已经耗尽。此前所有已请求工具都已返回结果。
禁止继续调用工具或输出工具协议。请基于当前对话中的真实工具结果，立即只返回一个符合当前角色输出契约的最终 JSON 对象；明确已完成部分、未完成部分和交给后续协作节点的建议。"""


class AgentExecutor:
    def __init__(self, context: ContextSystem, ledger: BudgetLedger, provider: ModelProvider, model_pool: ModelPool, trace: TraceRecorder, tools: ToolRegistry | None = None, tool_executor: ToolExecutor | None = None, execution_model: str = "deepseek-v4.1-flash", max_tool_calls: int = 50, context_compaction_threshold_tokens: int = 200_000) -> None:
        self.context = context
        self.ledger = ledger
        from virgo.providers.offline import OfflineProvider
        self._work_request_throttle = not isinstance(provider, OfflineProvider)
        self.provider = provider
        self.model_pool = model_pool
        self.trace = trace
        self.tools = tools
        self.tool_executor = tool_executor
        self.execution_model = execution_model
        self.max_tool_calls = max(1, max_tool_calls)
        self.context.max_tool_calls = self.max_tool_calls
        self.context_compaction_threshold_tokens = max(1, context_compaction_threshold_tokens)

    def run(self, task_id: str, node_id: str, *, settle_budget: bool = True) -> tuple[str | None, AgentOutput]:
        payload = self.context.build_agent_input(task_id, node_id)
        snapshot = self.context.store.get(task_id)
        assignment = next(a for a in snapshot["agent_assignments"] if a["assignment_id"] == next(n for n in snapshot["execution_state"]["nodes"] if n["node_id"] == node_id)["active_assignment_id"])
        is_bigcodebench = "BigCodeBench" in payload["task"]["description"]
        is_swebench = "SWE-bench" in payload["task"]["description"]
        is_code_benchmark = is_bigcodebench or is_swebench
        assignment = {
            **assignment,
            "tools": list(assignment["tools"]),
        }
        tool_call_limit = self.context.node_tool_call_limit(task_id, node_id)
        role_name = assignment["role"]["name"]
        billing_model = assignment["llm"]["model_id"]
        self.trace.record(task_id, "agent_call_start", context=payload, budget=snapshot["budget"], node_id=node_id, role=role_name, model=billing_model)
        if getattr(self.tool_executor, "benchmark_task_loader", None):
            payload["environment"]["constraints"].append("允许联网查询通用知识、算法与 API 文档；禁止通过题号、原题文本、题解或答案获取当前题目的解答，禁止访问本地数据集答案。")
        messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
        try:
            self.ledger.reserve(task_id, 0.0)
            collaborative_worker = tool_call_limit is not None
            system_prompt = self._system_prompt(
                role_name,
                build_role_prompt(role_name,
                    next(item for item in snapshot["mode_state"]["instances"] if item["mode_instance_id"] == assignment["mode_instance_id"])["execution_mode"],
                    first_round=next(n for n in snapshot["execution_state"]["nodes"] if n["node_id"] == node_id).get("round", 0) == 1)
                    + ("\n当前专业领域：" + assignment["role"]["domain"] if assignment["role"].get("domain") else ""),
                assignment["tools"],
                is_bigcodebench=is_bigcodebench,
                is_swebench=is_swebench,
                collaborative_worker=collaborative_worker,
            )
            node = next(item for item in snapshot["execution_state"]["nodes"] if item["node_id"] == node_id)
            if node.get("workspace_kind") == "parallel_summary":
                system_prompt += "\n本次调用是 Parallel 的汇总阶段，候选已完成。使用 .candidate_inputs 中的导出副本比较候选；在当前 workspace 正式实现或复制所选择的最终产物并验证。不要继续执行独立候选阶段的职责，不要把导入目录当作正式交付目录。"
            self.trace.record(task_id, "agent_prompt_ready", context={"system_prompt": system_prompt},
                              node_id=node_id, role=role_name, model=billing_model,
                              mode=next(item for item in snapshot["mode_state"]["instances"]
                                        if item["mode_instance_id"] == assignment["mode_instance_id"])["execution_mode"])
            messages, compaction_usage = self._compact_context_if_needed(task_id, node_id, system_prompt, messages, billing_model)
            response = self._complete_model(task_id, node_id, system=system_prompt, messages=messages, model=billing_model, tools=self.tools.schemas(assignment["tools"]) if self.tools else None)
            response.usage = self._sum_usage(compaction_usage, response.usage)
            self._record_protocol_response(task_id, node_id, response, tools_enabled=bool(self.tools))
            response, actual_artifacts = self._complete_tools(task_id, node_id, assignment, system_prompt, messages, response, billing_model, tool_call_limit=tool_call_limit)
            response.usage.cost = self.model_pool.estimate_cost(billing_model, response.usage.input_tokens, response.usage.output_tokens)
            if settle_budget:
                self.ledger.settle(task_id, response.usage, 0.0)
            else:
                # 并行阶段先保存真实用量，阶段汇总节点再一次性结算。
                self.ledger.release(task_id, 0.0)
            output = self._parse_output(response, role_name, payload["task"]["description"])
            if collaborative_worker:
                self._ensure_collaboration_handoff(output)
            if is_code_benchmark and role_name not in {"SpecialPlanner", "Synthesizer", "Reviewer", "Coordinator", "SubtaskCoordinator"}:
                # 只将没有工具操作、产物或验证的阶段性回复视为无进展。
                no_progress = output.status in {"blocked", "failed"} or (
                    not output.task_output["confirmed"] and not output.task_output["artifacts"]
                    and not output.task_output["verification"]
                    and not any(message.get("role") == "tool" for message in messages)
                )
                if self.model_pool.record_progress(task_id, billing_model, no_progress):
                    self.trace.record(task_id, "agent_model_no_progress_blocked",
                                      context={"model": billing_model, "threshold": 2},
                                      output={"status": "excluded_for_task"}, node_id=node_id)
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            result_id = self.context.record_output(task_id, node_id, output, response.usage.as_dict(), call_id)
            self.trace.record(task_id, "agent_call_complete", context=payload, output=output.as_dict(), budget=self.context.store.get(task_id)["budget"], node_id=node_id, result_id=result_id, call_id=call_id)
            return result_id, output
        except BudgetExceeded as exc:
            output = AgentOutput(payload["state"], status="blocked", error={"kind": "budget", "message": str(exc)})
            self.trace.record(task_id, "agent_call_denied", context=payload, output=output.as_dict(), budget=self.context.store.get(task_id)["budget"], node_id=node_id)
            output.task_output = self._preserve_interrupted_state(payload["state"], output)
            return None, output
        except Exception as exc:
            self.ledger.release(task_id, 0.0)
            output = AgentOutput(payload["state"], status="failed", error={"kind": "execution", "message": str(exc)})
            self.trace.record(task_id, "agent_call_failed", context=payload, output=output.as_dict(), budget=self.context.store.get(task_id)["budget"], node_id=node_id)
            output.task_output = self._preserve_interrupted_state(payload["state"], output)
            return None, output

    @staticmethod
    def _preserve_interrupted_state(previous, output):
        from copy import deepcopy
        return deepcopy(previous)

    @staticmethod
    def _available_tools(assigned_tools: list[str] | tuple[str, ...], is_code_benchmark: bool) -> list[str]:
        """保留角色工具，评测答案查询由工具执行层按请求拦截。"""
        return list(assigned_tools)

    def _complete_model(self, task_id: str, node_id: str, **kwargs: Any) -> ModelResponse:
        """瞬时接口异常先重试原请求，耗尽后交给节点恢复。"""
        for attempt in range(3):
            try:
                pool = getattr(self, 'model_pool', None)
                if pool is not None and not pool.channel_available(task_id, kwargs.get('model')):
                    raise RuntimeError('本任务该模型渠道已被禁用，不再发送请求')
                if getattr(self, '_work_request_throttle', False):
                    from virgo.providers.work_request_control import wait_for_work_request, work_request_scope
                    if not getattr(self.provider,'work_throttle_at_transport',False):
                        wait_for_work_request()
                    if pool is not None and not pool.channel_available(task_id, kwargs.get('model')):
                        raise RuntimeError('等待期间该模型渠道已失败，不再发送请求')
                if getattr(self, '_work_request_throttle', False):
                    with work_request_scope(lambda: pool is None or pool.channel_available(task_id, kwargs.get("model"))):
                        return self.provider.complete(**kwargs)
                return self.provider.complete(**kwargs)
            except Exception as exc:
                message = str(exc).lower()
                http = re.search(r"http\s+(\d{3})", message)
                code = int(http.group(1)) if http else None
                timeout = isinstance(exc, TimeoutError) or "timed out" in message or "timeout" in message
                transport_error = isinstance(exc, ConnectionError) or any(term in message for term in (
                    "connection reset", "connection refused", "connection aborted", "remote end closed",
                    "temporary failure", "urlerror", "name resolution", "nodename nor servname",
                    "unexpected_eof_while_reading", "eof occurred in violation of protocol",
                    "remote disconnected", "connection broken",
                ))
                retryable = (code in {408, 429} or (code is not None and 500 <= code < 600)) if code is not None else (
                    timeout or transport_error
                )
                if not retryable or attempt == 2 or getattr(exc, "retries_exhausted", False):
                    if (timeout or code is not None or transport_error) and getattr(self,'model_pool',None) is not None:
                        self.model_pool.block_channel(task_id, kwargs.get('model'), str(exc))
                        if getattr(self,'trace',None):
                            self.trace.record(task_id,'agent_model_channel_blocked',context={'model':kwargs.get('model'),'reason':str(exc)},output={'status':'blocked_for_task'},node_id=node_id)
                    raise
                # 免费额度可能把限流表现为读超时；超时冷却300秒，显式429仍冷却60秒。
                delay = 300 if timeout or code == 408 else 60 if code == 429 else 2 ** attempt
                if getattr(self, "trace", None):
                    self.trace.record(
                        task_id,
                        "agent_model_timeout_retry" if timeout else "agent_model_api_retry",
                        context={"node_id": node_id, "model": kwargs.get("model"), "tools_enabled": bool(kwargs.get("tools")), "retry": attempt + 1, "delay_seconds": delay},
                        output={"status": "retrying_same_request", "error": str(exc)},
                        node_id=node_id,
                    )
                time.sleep(delay)

    def settle_result_ids(self, task_id: str, result_ids: list[str]) -> None:
        """按结果集合一次性结算一个并行阶段的模型用量。"""
        snapshot = self.context.store.get(task_id)
        usages = [snapshot["agent_outputs"][result_id].get("usage", {}) for result_id in result_ids if result_id in snapshot.get("agent_outputs", {})]
        usage = Usage(
            input_tokens=sum(int(item.get("input_tokens", 0)) for item in usages),
            output_tokens=sum(int(item.get("output_tokens", 0)) for item in usages),
            calls=sum(int(item.get("calls", 1)) for item in usages),
            duration_ms=sum(int(item.get("duration_ms", 0)) for item in usages),
            cost=sum(float(item.get("cost", 0.0)) for item in usages),
        )
        if usage.calls:
            self.ledger.reserve(task_id, 0.0)
            self.ledger.settle(task_id, usage, 0.0)
    def _complete_tools(self, task_id: str, node_id: str, assignment: dict[str, Any], system: str, messages: list[dict[str, Any]], response: ModelResponse, model: str | None = None, tool_call_limit: int | None = None) -> tuple[ModelResponse, list[dict[str, Any]]]:
        model = model or self.execution_model
        total_usage = response.usage
        actual_artifacts: list[dict[str, Any]] = []
        max_tool_calls = max(1, tool_call_limit if tool_call_limit is not None else getattr(self, "max_tool_calls", 50))
        executed_tool_calls = 0
        patch_failures = 0
        workspace = None
        context = getattr(self, "context", None)
        if context is not None and callable(getattr(context, "workspace_for_node", None)):
            workspace = context.workspace_for_node(task_id, node_id)
        if not isinstance(workspace, (str, Path)):
            workspace = None
        if workspace is None and isinstance(getattr(context, "workspace", None), (str, Path)):
            workspace = context.workspace
        role_name = assignment.get("role", {}).get("name")
        history = NodeToolHistory(node_id, workspace)
        for _ in range(max_tool_calls):
            if not response.tool_calls or not self.tool_executor:
                if self._needs_final_response_recovery(response, role_name, workspace, check_artifacts=True):
                    response = self._recover_final_response(task_id, node_id, system, messages, response, model,
                                                           role_name=role_name, workspace=workspace)
                    total_usage = self._sum_usage(total_usage, response.usage)
                response.usage = total_usage
                return response, actual_artifacts
            tool_results: list[dict[str, Any]] = []
            for call in response.tool_calls:
                function = call.get("function", call)
                name = function.get("name")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                before = history.before_call(name, arguments)
                if executed_tool_calls >= max_tool_calls:
                    result = self.tool_executor.limit_reached(
                        task_id=task_id,
                        node_id=node_id,
                        agent_id=assignment["agent_id"],
                        tool_id=name,
                        maximum=max_tool_calls,
                    )
                else:
                    result = self.tool_executor.execute(
                        task_id=task_id,
                        node_id=node_id,
                        agent_id=assignment["agent_id"],
                        tool_id=name,
                        arguments=arguments,
                        allowed_tools=assignment["tools"],
                    )
                    executed_tool_calls += 1
                if name == "file_patch":
                    patch_failures = patch_failures + 1 if result.get("status") == "failed" else 0
                    if patch_failures >= 2:
                        result["recovery_guidance"] = (
                            "file_patch 已连续失败至少两次。停止同类盲目重试；先用 file_read/file_read_lines "
                            "检查当前文件，确认前次是否已经修改；依据错误检查 hunk 行数和上下文。"
                            "工具不要求 Git 仓库，无需 git init、暂存或提交。"
                            "仅在掌握完整且未截断的文件内容时考虑 file_write；大文件继续按实际上下文重建最小补丁。"
                        )
                result = self._associate_tool_result(call, result)
                result = history.observe(call, result, before)
                tool_results.append(result)
                actual_artifacts.extend(result.get("artifacts", []))
            self._record_tool_protocol_round(task_id, node_id, response, tool_results, max_tool_calls)
            feedback, _ = self._compact_tool_results(tool_results, 0)
            messages = [*messages, *self.tool_followup_messages(response, feedback)]
            messages = history.compact(messages)
            tools_enabled = bool(self.tools and executed_tool_calls < max_tool_calls)
            if not tools_enabled:
                messages.append({"role": "user", "content": TOOL_BUDGET_EXHAUSTED_PROMPT})
                self._record_tool_budget_exhausted(task_id, node_id, response, tool_results, max_tool_calls)
            messages, compaction_usage = self._compact_context_if_needed(task_id, node_id, system, messages, model)
            response = self._complete_model(
                task_id, node_id,
                system=system,
                messages=messages,
                model=model,
                tools=self.tools.schemas(assignment["tools"]) if tools_enabled else None,
            )
            total_usage = self._sum_usage(total_usage, self._sum_usage(compaction_usage, response.usage))
            self._record_protocol_response(task_id, node_id, response, tools_enabled=tools_enabled)
        # 达到工具上限后，以一次无工具调用要求模型输出最终结构化结果。
        if response.tool_calls:
            # 对未处理的工具调用补回限额结果，再禁用工具要求最终文本。
            tool_results = []
            for call in response.tool_calls:
                function = call.get("function", call)
                tool_results.append(
                    self._associate_tool_result(call, self.tool_executor.limit_reached(
                        task_id=task_id,
                        node_id=node_id,
                        agent_id=assignment["agent_id"],
                        tool_id=function.get("name"),
                        maximum=max_tool_calls,
                    ))
                )
            feedback, _ = self._compact_tool_results(tool_results, 0)
            messages = [*messages, *self.tool_followup_messages(response, feedback)]
            messages = history.compact(messages)
            messages.append({"role": "user", "content": TOOL_BUDGET_EXHAUSTED_PROMPT})
            self._record_tool_protocol_round(task_id, node_id, response, tool_results, max_tool_calls)
            self._record_tool_budget_exhausted(task_id, node_id, response, tool_results, max_tool_calls)
            messages, compaction_usage = self._compact_context_if_needed(task_id, node_id, system, messages, model)
            response = self._complete_model(task_id, node_id, system=system, messages=messages, model=model, tools=None)
            total_usage = self._sum_usage(total_usage, self._sum_usage(compaction_usage, response.usage))
            self._record_protocol_response(task_id, node_id, response, tools_enabled=False)
        if self._needs_final_response_recovery(response, role_name, workspace, check_artifacts=True):
            response = self._recover_final_response(task_id, node_id, system, messages, response, model,
                                                   role_name=role_name, workspace=workspace)
            total_usage = self._sum_usage(total_usage, response.usage)
        response.usage = total_usage
        return response, actual_artifacts

    def _record_protocol_response(self, task_id: str, node_id: str, response: ModelResponse, *, tools_enabled: bool) -> None:
        if not getattr(self, "trace", None):
            return
        self.trace.record(
            task_id,
            "agent_model_response",
            context={"node_id": node_id, "tools_enabled": tools_enabled, "tool_choice": "auto" if tools_enabled else "none"},
            output=self._protocol_metadata(response),
            node_id=node_id,
        )

    def _record_tool_protocol_round(self, task_id: str, node_id: str, response: ModelResponse, tool_results: list[dict[str, Any]], maximum: int) -> None:
        if not getattr(self, "trace", None):
            return
        call_ids = [str(call.get("id")) for call in response.tool_calls if call.get("id")]
        returned_request_ids = [str(result.get("model_tool_call_id")) for result in tool_results if result.get("model_tool_call_id")]
        execution_ids = [str(result.get("tool_call_id")) for result in tool_results if result.get("tool_call_id")]
        associations = [
            {"model_tool_call_id": result.get("model_tool_call_id"), "execution_tool_call_id": result.get("tool_call_id")}
            for result in tool_results
        ]
        self.trace.record(
            task_id,
            "agent_tool_protocol_round",
            context={"node_id": node_id, "requested_tool_call_ids": call_ids, "returned_model_tool_call_ids": returned_request_ids, "execution_tool_call_ids": execution_ids, "associations": associations, "all_tool_results_matched": sorted(call_ids) == sorted(returned_request_ids), "tool_limit": maximum},
            output=self._protocol_metadata(response),
            node_id=node_id,
        )

    @staticmethod
    def _associate_tool_result(call: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """将内部执行记录显式关联回模型请求 ID，二者不要求字面相等。"""
        request_id = call.get("id")
        if not request_id:
            raise ValueError("模型工具调用缺少 id，无法关联执行结果")
        return {**result, "model_tool_call_id": str(request_id)}

    def _record_tool_budget_exhausted(self, task_id: str, node_id: str, response: ModelResponse, tool_results: list[dict[str, Any]], maximum: int) -> None:
        if not getattr(self, "trace", None):
            return
        self.trace.record(
            task_id,
            "agent_tool_budget_exhausted",
            context={"node_id": node_id, "tool_limit": maximum, "requested_tool_calls": len(response.tool_calls), "returned_tool_results": len(tool_results), "finalization_tools_enabled": False, "finalization_tool_choice": "none"},
            output={"status": "finalization_requested", **self._protocol_metadata(response)},
            node_id=node_id,
        )

    @staticmethod
    def _protocol_metadata(response: ModelResponse) -> dict[str, Any]:
        raw = response.raw if isinstance(response.raw, dict) else {}
        choice = raw.get("choices", [{}])[0] if isinstance(raw.get("choices"), list) and raw.get("choices") else {}
        return {
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "content_chars": len(response.content),
            "content_preview": response.content[:200],
            "tool_calls": [{"id": call.get("id"), "name": call.get("function", call).get("name")} for call in response.tool_calls],
            "responses_status": raw.get("status"),
            "responses_output_types": [item.get("type") for item in raw.get("output", []) if isinstance(item, dict)],
            "protocol_residue": raw.get("_virgo_protocol_residue"),
        }

    @staticmethod
    def _validate_final_response(response, role_name=None, workspace=None, *, check_artifacts=False):
        if response.tool_calls or response.raw.get("_virgo_protocol_residue"):
            raise StateProtocolError("json", ["最终交付仍包含工具调用或工具协议残片。"])
        return validate_output(AgentExecutor._unwrap_json_fence(response.content), role=role_name,
                               workspace=workspace, check_artifacts=check_artifacts)

    @staticmethod
    def _needs_final_response_recovery(response, role_name=None, workspace=None, *, check_artifacts=False):
        try:
            AgentExecutor._validate_final_response(response, role_name, workspace, check_artifacts=check_artifacts)
            return False
        except StateProtocolError:
            return True

    def _recover_final_response(self, task_id, node_id, system, messages, failed_response, model=None,
                                *, role_name=None, workspace=None):
        """格式或引用不合格时最多重试一次；错误不得进入有效状态。"""
        try:
            self._validate_final_response(failed_response, role_name, workspace, check_artifacts=True)
            return failed_response
        except StateProtocolError as exc:
            error = exc
        if getattr(self, "trace", None):
            self.trace.record(task_id, "agent_output_rejected", context={"node_id": node_id, "attempt": 0},
                output={"protocol_validation": error.as_dict(), "raw_response": failed_response.content}, node_id=node_id)
        messages = list(messages)
        if isinstance(failed_response.content, str) and failed_response.content.strip() and not failed_response.tool_calls and not failed_response.raw.get("_virgo_protocol_residue"):
            messages.append({"role": "assistant", "content": failed_response.content})
        feedback = error.as_dict()
        feedback["instruction"] = "已有 state 保持不变。请重新返回完整协议对象，不要把错误说明当作新的任务状态。"
        if role_name == "SpecialPlanner":
            feedback["instruction"] = "已有状态保持不变，请重新返回角色要求的完整 DAG 派发 JSON。"
        messages.append({"role": "user", "content": FINAL_RESPONSE_RECOVERY_PROMPT + "\n" + json.dumps(feedback, ensure_ascii=False)})
        if getattr(self, "trace", None):
            self.trace.record(task_id, "agent_protocol_recovery", context={"node_id": node_id, "reason": error.stage,
                              "errors": error.errors}, output={"status": "retrying_final_response"}, node_id=node_id)
        messages, compaction_usage = self._compact_context_if_needed(task_id, node_id, system, messages, model or self.execution_model)
        response = self._complete_model(task_id, node_id, system=system, messages=messages, model=model or self.execution_model, tools=None)
        response.usage = self._sum_usage(compaction_usage, response.usage)
        try:
            self._validate_final_response(response, role_name, workspace, check_artifacts=True)
        except StateProtocolError as exc:
            response.raw = {**response.raw, "_virgo_invalid_final_output": True,
                            "_virgo_protocol_validation": {**exc.as_dict(), "recovery_attempts": 1}}
            if getattr(self, "trace", None):
                self.trace.record(task_id, "agent_output_rejected", context={"node_id": node_id, "attempt": 1},
                    output={"protocol_validation": exc.as_dict(), "raw_response": response.content}, node_id=node_id)
        return response

    def _compact_context_if_needed(self, task_id: str, node_id: str, system: str, messages: list[dict[str, Any]], model: str | None = None) -> tuple[list[dict[str, Any]], Usage]:
        """上下文达到阈值时用一次无工具调用生成可审计摘要。"""
        validate_tool_pairs(messages)
        estimated_tokens = self._estimate_context_tokens(system, messages)
        threshold = max(1, getattr(self, "context_compaction_threshold_tokens", 200_000))
        if estimated_tokens < threshold:
            return messages, Usage(calls=0)
        compaction_usage = Usage(calls=0)
        try:
            response = self._complete_model(
                task_id, node_id,
                system=CONTEXT_COMPACTION_SYSTEM_PROMPT,
                messages=messages,
                model=model or self.execution_model,
                tools=None,
            )
            compaction_usage = response.usage
            summary = response.content.strip()
        except Exception as exc:
            response = None
            summary = ""
            compaction_error = f"{type(exc).__name__}: {exc}"
        else:
            compaction_error = None
        if not summary or (response is not None and response.raw.get("_virgo_protocol_residue")):
            # 保留最近对话，避免压缩器异常时无限携带完整长上下文。
            summary = json.dumps(
                {
                    "compaction_status": "fallback",
                    "reason": compaction_error or "压缩模型未返回有效正文",
                    "recent_messages": recent_complete_exchanges(messages),
                },
                ensure_ascii=False,
                default=str,
            )
        anchors = {"original_input": messages[0].get("content") if messages and messages[0].get("role") == "user" else None}
        for message in messages:
            if message.get("role") == "user":
                try:
                    payload = json.loads(message.get("content", ""))
                    if isinstance(payload, dict) and HISTORY_MARKER in payload:
                        anchors[HISTORY_MARKER] = payload[HISTORY_MARKER]
                    if isinstance(payload, dict) and "original_input" in payload:
                        anchors["original_input"] = payload["original_input"]
                except (ValueError, TypeError):
                    pass
        compacted = [{"role": "user", "content": json.dumps({"compacted_context": summary, **anchors,
            "instruction": "基于压缩上下文继续当前任务；保留工具证据中的版本、阻塞及完整日志引用，不将旧版本验证当作当前版本通过。"}, ensure_ascii=False)}]
        validate_tool_pairs(compacted)
        if getattr(self, "trace", None):
            self.trace.record(
                task_id,
                "context_compacted",
                context={"node_id": node_id, "estimated_tokens_before": estimated_tokens, "threshold_tokens": threshold, "message_count_before": len(messages)},
                output={"status": "compacted", "estimated_tokens_after": self._estimate_context_tokens(system, compacted), "summary_chars": len(summary)},
                node_id=node_id,
            )
        return compacted, compaction_usage

    @staticmethod
    def _estimate_context_tokens(system: str, messages: list[dict[str, Any]]) -> int:
        """无 provider tokenizer 时使用保守的字符估算，供压缩阈值控制。"""
        encoded = system + json.dumps(messages, ensure_ascii=False, default=str)
        return max(1, (len(encoded) + 3) // 4)

    @staticmethod
    def _compact_tool_results(results: list[dict[str, Any]], used_chars: int) -> tuple[list[dict[str, Any]], int]:
        """限制回填模型的工具文本总量；完整结果仍由工具调用轨迹保留。"""
        compacted: list[dict[str, Any]] = []
        for result in results:
            remaining = max(0, TOOL_FEEDBACK_TOTAL_CHARS - used_chars)
            limit = min(TOOL_FEEDBACK_RESULT_CHARS, remaining)
            item = AgentExecutor._compact_tool_result(result, limit)
            used_chars += len(json.dumps(item, ensure_ascii=False, default=str))
            compacted.append(item)
        return compacted, used_chars

    @staticmethod
    def _compact_tool_result(result: dict[str, Any], limit: int) -> dict[str, Any]:
        """对超长工具结果生成可审计摘要，避免模型上下文被日志淹没。"""
        encoded = json.dumps(result, ensure_ascii=False, default=str)
        if len(encoded) <= limit or result.get("history_kind") == "file_read" and len(encoded) <= 16000:
            return result
        preview_limit = max(0, limit - 500)
        preview = encoded[:preview_limit]
        if preview_limit and len(encoded) > preview_limit:
            preview += "\n...[工具结果已截断；完整结果保存在决策轨迹中]"
        return {
            "tool_call_id": result.get("tool_call_id"),
            "model_tool_call_id": result.get("model_tool_call_id"),
            "status": result.get("status"),
            "data_preview": preview,
            "error": result.get("error"),
            "artifacts": result.get("artifacts", []),
            "truncated": True,
            "usage": result.get("usage"),
            **{key: result[key] for key in ("history_record_id", "history_kind", "decision_evidence", "full_log_path", "process_log_path", "file_versions", "read_version_is_current", "archive_error", "recovery_guidance") if key in result},
        }

    @staticmethod
    def _system_prompt(role_name: str, prompt: str, tools: list[str], *, is_bigcodebench: bool = False, is_swebench: bool = False, is_code_benchmark: bool = False, collaborative_worker: bool = False) -> str:
        """为获授权的写文件 Agent 追加运行时产物交付契约。"""
        if role_name == "Programmer" and is_swebench:
            base = programmer_swebench_prompt(prompt, tools)
        elif role_name == "Programmer" and (is_bigcodebench or is_code_benchmark):
            base = programmer_code_benchmark_prompt(prompt, tools)
        elif "file_write" not in tools:
            base = prompt
        else:
            base = prompt + f"""

Runtime delivery contract for this invocation:
- The runtime has authorized these tools: {", ".join(tools)}. This runtime contract overrides any generic statement in the role template that tools are unavailable.
- 选择题、数学题和普通问答默认将完整答案与必要依据写入 `solution.md`；独立编程题按指定语言和接口写入 `solution.py`、`solution.cpp`、`solution.c` 等对应源文件；工程任务直接完成所需代码修改和成果。用户或评测协议指定路径时优先遵循。
- Parallel 候选使用本节点独立工作区下的同名文件；禁止覆盖父工作区或其他候选的文件。实际产物登记到 `task_output.artifacts`，实际检查记录到 `task_output.verification`。
- Do not put the full file contents in `task_output`. Use `task_output` for a concise delivery explanation: artifact purpose, public entry points, actual verification, and limitations.
- 在 task_output.artifacts 中登记真实存在的交付文件，提供 path、version、summary；工具写入日志不替代正式产物登记。
- A tool response with `status: "failed"`, `timeout`, or `denied` is failure evidence even when its transport request completed. In particular, a nonzero `data.returncode` means the command failed: do not claim that its write, readback assertion, test, or any later step succeeded.
"""
        if "file_patch" in tools:
            from virgo.tools.patch import PATCH_GUIDANCE
            base += "\n补丁工具契约：" + PATCH_GUIDANCE
        return f"当前角色：{role_name}\n" + base + """

工作状态交付契约：
- 仅输出角色约定的 JSON；task_output 包含 confirmed、possible_issue、artifacts、verification、remaining 五个数组。不要输出旧外层 task_phase、task_description、status、CoT_summary、verification、artifacts、possible_issue；运行时元数据和 pervious_artifacts 由程序维护。
- 输出当前职责范围的完整累计状态；完成的待办和已解决疑点应移除，未处理的有效内容应保留。
- 产物登记在 task_output.artifacts；实际检查登记在 task_output.verification，并关联真实版本和证据。
- 输入 state 中的多来源条目保留来源标识，不将同级候选合并为自己的已证实结论。
- SpecialPlanner 使用其独立派发协议，保留 tasks 等原定字段。
"""

    @staticmethod
    def _ensure_collaboration_handoff(output: AgentOutput) -> None:
        """接续信息统一由五字段状态承载。"""
        output.task_output = validate_state(output.task_output)

    @staticmethod
    def _sum_usage(first: Usage, second: Usage) -> Usage:
        return Usage(
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            cost=first.cost + second.cost,
            calls=first.calls + second.calls,
            duration_ms=first.duration_ms + second.duration_ms,
        )

    @staticmethod
    def tool_followup_messages(response: ModelResponse, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按 OpenAI 工具协议拼接 assistant 调用与对应 tool 响应。"""
        if len(response.tool_calls) != len(tool_results):
            raise ValueError("工具调用数量与工具结果数量不一致")
        raw_message = response.raw.get("choices", [{}])[0].get("message") if response.raw else None
        assistant_message = dict(raw_message) if isinstance(raw_message, dict) else {}
        assistant_message["role"] = "assistant"
        assistant_message.setdefault("content", response.content or None)
        assistant_message.setdefault("tool_calls", response.tool_calls)
        followups: list[dict[str, Any]] = [assistant_message]
        for call, result in zip(response.tool_calls, tool_results, strict=True):
            tool_call_id = call.get("id")
            if not tool_call_id:
                raise ValueError("模型工具调用缺少 id，无法构造 tool_call_id")
            followups.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call_id),
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        return followups

    @staticmethod
    def _parse_output(response: ModelResponse, role_name: str, assignment: str) -> AgentOutput:
        try:
            if response.raw.get("_virgo_invalid_final_output"):
                details = response.raw.get("_virgo_protocol_validation") or {}
                raise StateProtocolError(details.get("stage", "structure"), details.get("errors", ["协议纠正后仍未获得有效最终交付。 "]))
            data, state = AgentExecutor._validate_final_response(response, role_name)
        except StateProtocolError as exc:
            return AgentOutput(empty_state(), status="failed",
                error={"kind": "protocol", "message": "最终交付未通过协议接收检查。", "details": exc.as_dict()},
                extra={"protocol_validation": exc.as_dict()})
        if state is None:  # SpecialPlanner 的派发输出由 DAG 结构校验器处理。
            state = empty_state()
        legacy = {"task_phase", "task_description", "status", "task_output", "CoT_summary", "cot_summary",
                  "verification", "artifacts", "possible_issue", "progress", "remaining_work", "next_action",
                  "runtime", "protocol_validation", "pervious_artifacts"}
        extra = {key: value for key, value in data.items() if key not in legacy}
        extra["protocol_validation"] = {"accepted": True, "stage": "structure", "errors": [], "answer_correctness_checked": False}
        # 已取得符合协议的输出，仅表示节点调用完成，不裁决任务或答案是否正确。
        return AgentOutput(state, extra=extra)

    @staticmethod
    def _unwrap_json_fence(content: str) -> str:
        """移除模型仅用作 JSON 包装的 Markdown 围栏。"""
        match = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", content, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else content
