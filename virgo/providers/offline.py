"""离线演示模型。

离线模式用于验证调度系统本身，不声称具备真实模型的推理质量。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from virgo.types import ModelResponse, Usage


class OfflineProvider:
    """按请求中的任务和角色生成确定性、结构化的演示响应。"""

    def complete(self, *, system: str, messages: list[dict[str, str]], model: str, temperature: float = 0.2, response_format: str | None = "json_object", tools: list[dict[str, Any]] | None = None) -> ModelResponse:
        started = time.perf_counter()
        user_text = "\n".join(message.get("content", "") for message in messages)
        content = self._respond(system, user_text)
        input_tokens = max(1, len((system + user_text)) // 4)
        output_tokens = max(1, len(content) // 4)
        return ModelResponse(
            content=json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else content,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens, duration_ms=int((time.perf_counter() - started) * 1000)),
        )

    def _respond(self, system: str, text: str) -> dict[str, Any]:
        if "You are the initial context builder for a multi-agent system." in system:
            return self._initial_context(text)
        if "协作模式路由器" in system:
            return {"mode": self._pick_mode(text), "reason": "离线演示路由：依据任务复杂度和关键词选择可执行模式。"}
        if "角色路由器" in system:
            try:
                payload = json.loads(text)
                count = max(1, int(payload.get("count", 1)))
                candidates = [str(role) for role in payload.get("candidate_roles", [])]
                first = self._pick_role(text)
                if count == 1 and payload.get("phase") == "summary" and "Synthesizer" in candidates:
                    selected = ["Synthesizer"]
                else:
                    selected = [first] if first in candidates else []
                for candidate in candidates:
                    if len(selected) >= count:
                        break
                    if candidate not in selected and candidate != "ChainStop":
                        selected.append(candidate)
                return {"roles": selected[:count], "reason": "离线演示路由：依据任务阶段和关键词选择角色。"}
            except (json.JSONDecodeError, TypeError, ValueError):
                return {"roles": [self._pick_role(text)], "reason": "离线演示路由：依据任务关键词选择角色。"}
        if "LLM" in system and "选择路由器" in system:
            return {"model_id": "deepseek-v4.1-flash", "reason": "离线演示选择当前模型池中的 DeepSeek-V4.1-Flash。"}
        return self._agent_output(system, text)

    def _initial_context(self, text: str) -> dict[str, Any]:
        task = text.split("原始任务：", 1)[-1].strip()
        task_type = "Programming" if any(k in task.lower() for k in ("代码", "系统", "实现", "python", "api")) else "QuestionAnswering"
        difficulty = min(100, 35 + len(task) // 20)
        return {
            "task_description": task[:500],
            "tasktype": task_type,
            "domains": "ComputerScience" if task_type == "Programming" else "General",
            "difficulty": difficulty,
            "estimated_duration_minutes": ">60 minutes" if difficulty > 60 else ">15 & <30 minutes",
            "recommended_budget": "medium" if difficulty > 60 else "low",
            "decompose": "recommend" if difficulty > 50 else "not recommend",
            "constraints": "遵循用户明确要求；离线演示模式未执行外部工具。",
        }

    @staticmethod
    def _pick_mode(text: str) -> str:
        try:
            payload = json.loads(text)
            candidates = payload.get("candidate_modes", [])
            target = payload.get("subtask") or payload.get("task") or payload
            target_text = json.dumps(target, ensure_ascii=False, default=str)
        except json.JSONDecodeError:
            candidates = []
            target_text = text
        if "DAG" in candidates and ("拆分" in target_text or "有向无环" in target_text):
            return "DAG"
        if "FullMesh" in candidates and any(k in target_text for k in ("综合", "讨论", "多方")):
            return "FullMesh"
        if "Chain" in candidates and any(k in target_text for k in ("审查", "顺序", "链式")):
            return "Chain"
        if "FullMesh" in candidates and ("讨论" in target_text or "全连接" in target_text):
            return "FullMesh"
        if "Parallel" in candidates and ("多个" in target_text or "并行" in target_text or "分析" in target_text and "比较" in target_text):
            return "Parallel"
        return "Single"

    @staticmethod
    def _pick_role(text: str) -> str:
        try:
            payload = json.loads(text)
            candidates = payload.get("candidate_roles", [])
            if "ChainStop" in candidates and (payload.get("previous_result") or payload.get("previous_role")):
                return "ChainStop"
            if "FullMeshStop" in candidates and len(payload.get("selected_roles", [])) >= 3:
                return "FullMeshStop"
        except json.JSONDecodeError:
            pass
        if any(k in text for k in ("代码", "实现", "系统", "接口")):
            return "Programmer"
        if any(k in text for k in ("数学", "证明", "计算")):
            return "MathSpecialist"
        if any(k in text for k in ("检索", "资料", "来源")):
            return "Researcher"
        return "DomainSpecialist"

    @staticmethod
    def _agent_output(system: str, text: str) -> dict[str, Any]:
        role = OfflineProvider._role_name(system)
        assignment = text[-1200:]
        if role == "SpecialPlanner":
            return {'task_output': {'confirmed': ["已生成两个独立子任务和一个依赖汇总任务。"], 'possible_issue': ["当前使用离线演示规划器。"], 'artifacts': [], 'verification': [], 'remaining': []},
                'plan': [
                    {"node_id": "dag:T1", "description": "完成任务的主要实现或分析工作。", "dependencies": [], "required_capability": "Programming", "initial_context": {"task_description": "完成任务的主要实现或分析工作。", "tasktype": "Programming", "domains": "ComputerScience", "difficulty": 45, "estimated_duration_minutes": ">15 & <30 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"}, "expected_output": "一个可供后续汇总使用的主结果。", "acceptance_criteria": ["输出主任务结果"]},
                    {"node_id": "dag:T2", "description": "独立审查任务要求和潜在问题。", "dependencies": [], "required_capability": "Review", "initial_context": {"task_description": "审查任务要求、实现结果和潜在问题。", "tasktype": "Writing,Programming", "domains": "ComputerScience", "difficulty": 40, "estimated_duration_minutes": ">15 & <30 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"}, "expected_output": "审查意见和改进建议。", "acceptance_criteria": ["列出审查结论"]},
                    {"node_id": "dag:T3", "description": "依赖前两个子任务并形成最终综合结果。", "dependencies": ["dag:T1", "dag:T2"], "required_capability": "Synthesis", "initial_context": {"task_description": "综合上游结果，形成最终交付物。", "tasktype": "Writing,QuestionAnswering", "domains": "ComputerScience", "difficulty": 50, "estimated_duration_minutes": ">30 & <45 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"}, "expected_output": "最终综合结论。", "acceptance_criteria": ["覆盖上游结果并给出最终结论"]}
                ]}
        if role == "Coordinator":
            context = {"task_description": "完成一个树形分支子任务。", "tasktype": "Programming", "domains": "ComputerScience", "difficulty": 40, "estimated_duration_minutes": ">15 & <30 minutes", "recommended_budget": "low", "decompose": "not recommend", "constraints": "None"}
            return {'task_output': {'confirmed': ["已完成树形分支规划。"], 'possible_issue': ["当前使用离线演示协调器。"], 'artifacts': [], 'verification': [], 'remaining': []},
                'assignments': [
                    {"subtask_id": "tree:T1", "description": "完成任务的主要实现工作。", "initial_context": context, "acceptance_criteria": ["输出主要实现结果"], "depends_on": []},
                    {"subtask_id": "tree:T2", "description": "审查实现方案和边界条件。", "initial_context": {**context, "task_description": "审查实现方案和边界条件。", "tasktype": "Programming,Writing"}, "acceptance_criteria": ["列出审查结论"], "depends_on": []},
                    {"subtask_id": "tree:T3", "description": "综合前两个分支并提出改进建议。", "initial_context": {**context, "task_description": "综合分支输入并提出改进建议。", "tasktype": "Writing,QuestionAnswering"}, "acceptance_criteria": ["形成分支级结论"], "depends_on": []},
                ]}
        if role == "SubtaskCoordinator":
            return {'task_output': {'confirmed': ["已提交叶节点执行请求。"], 'possible_issue': [], 'artifacts': [], 'verification': [], 'remaining': []},
                'execution_request': {"request_type": "route_subtask", "limits": {"allowed_modes": ["Single", "Chain", "FullMesh"], "max_agents": 3, "allow_further_nesting": False}}}
        return {'task_output': {'confirmed': [f"离线演示结果：已由 {role or '执行智能体'} 处理当前分配。\n\n处理摘要：{assignment[:500]}"], 'possible_issue': ["当前使用离线演示模型，结果不代表真实本地模型能力。"], 'artifacts': [], 'verification': [], 'remaining': []}}

    @staticmethod
    def _role_name(system: str) -> str:
        marker = re.search(r"当前角色：(\w+)", system)
        if marker:
            return marker.group(1)
        chinese_markers = {"SubtaskCoordinator": "你是 Tree 第二层分支协调者", "Coordinator": "你是 Tree 根协调者",
            "SpecialPlanner": "你是 DAG 规划节点", "Programmer": "实现、修改、调试代码", "Synthesizer": "比较、选择或整合候选成果",
            "MathSpecialist": "完成数学建模、推导和计算", "AlgorithmDesigner": "设计符合输入规模和约束的算法",
            "Researcher": "围绕当前问题查找和核对信息", "DataAnalyst": "使用可获得的数据完成检查",
            "Reviewer": "对照要求审查指定候选及版本", "DomainSpecialist": "针对当前问题运用相应领域知识"}
        for name, marker in chinese_markers.items():
            if marker in system:
                return name
        role_markers = {
            "SubtaskCoordinator": "You are a branch coordinator",
            "MathSpecialist": "You are the mathematics specialist",
            "AlgorithmDesigner": "You are the algorithm designer",
            "Researcher": "You are the researcher for a multi-agent system",
            "Programmer": "You are the programmer for a multi-agent system",
            "DataAnalyst": "You are the data analyst for a multi-agent system",
            "ToolOperator": "You are the tool operator",
            "Reviewer": "You are the reviewer",
            "Synthesizer": "You are the synthesizer",
            "ReportWriter": "You are the report writer",
            "SpecialPlanner": "You are the task planner of a multi-agent system",
            "DomainSpecialist": "You are a specialist in",
            "Coordinator": "You are the overall coordinator",
        }
        for name, marker in role_markers.items():
            if marker in system:
                return name
        return "Agent"
