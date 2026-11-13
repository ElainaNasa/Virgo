"""角色注册表。

角色的完整系统提示词作为包内静态数据保存在 ``prompt_data.py``。
本模块只登记角色标识、执行阶段与工具能力，避免维护第二份被改写的提示词。
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompt_catalog import get_prompt


DOMAIN_SPECIALIST_TEMPLATE = "DomainSpecialist"


def role_instance(role: str, domain: str | None = None) -> str:
    """返回可审计的角色实例标识，底层权限仍由模板角色决定。"""
    return f"{DOMAIN_SPECIALIST_TEMPLATE}[{domain}]" if role == DOMAIN_SPECIALIST_TEMPLATE and domain else role


def parse_role_instance(value: str) -> tuple[str, str | None]:
    """将领域实例标识还原为稳定模板 ID 与单领域上下文。"""
    prefix = f"{DOMAIN_SPECIALIST_TEMPLATE}["
    if value.startswith(prefix) and value.endswith("]"):
        domain = value[len(prefix):-1].strip()
        if domain:
            return DOMAIN_SPECIALIST_TEMPLATE, domain
    return value, None


def expand_domain_specialists(roles: list[str], domains: str) -> list[str]:
    """将一次模板选择展开为每个领域一个独立专家实例。"""
    values = [item.strip() for item in domains.split(",") if item.strip() and item.strip().lower() != "general"]
    expanded: list[str] = []
    for role in roles:
        expanded.extend(role_instance(role, domain) for domain in values) if role == DOMAIN_SPECIALIST_TEMPLATE and values else expanded.append(role)
    return list(dict.fromkeys(expanded))


@dataclass(frozen=True)
class RoleSpec:
    name: str
    task_phase: str
    tools: tuple[str, ...] = ()
    prompt_name: str | None = None

    def system_prompt(self, domain: str = "") -> str:
        return get_prompt(self.prompt_name or self.name, domain=domain or None)


WORKER_TOOLS = (
    "web_search", "web_fetch",
    "directory_list", "file_stat", "file_search", "file_read", "file_read_lines", "file_write", "file_patch",
    "environment_inspect", "python_execute", "shell_execute",
)


ROLE_SPECS: dict[str, RoleSpec] = {
    "InitialContextBuilder": RoleSpec("InitialContextBuilder", "任务初始上下文构建"),
    "MathSpecialist": RoleSpec("MathSpecialist", "数学建模与推导", WORKER_TOOLS),
    "AlgorithmDesigner": RoleSpec("AlgorithmDesigner", "算法设计", WORKER_TOOLS),
    "Programmer": RoleSpec("Programmer", "代码实现", WORKER_TOOLS),
    "DataAnalyst": RoleSpec("DataAnalyst", "数据分析", WORKER_TOOLS),
    "Researcher": RoleSpec("Researcher", "信息检索", WORKER_TOOLS),
    "ToolOperator": RoleSpec("ToolOperator", "工具执行", WORKER_TOOLS),
    "DomainSpecialist": RoleSpec("DomainSpecialist", "领域分析", WORKER_TOOLS),
    "Coordinator": RoleSpec("Coordinator", "全局协调", WORKER_TOOLS),
    "SubtaskCoordinator": RoleSpec("SubtaskCoordinator", "分支协调", WORKER_TOOLS),
    "Reviewer": RoleSpec("Reviewer", "结果审查", WORKER_TOOLS),
    "Synthesizer": RoleSpec("Synthesizer", "结果综合", WORKER_TOOLS),
    "ReportWriter": RoleSpec("ReportWriter", "报告撰写", WORKER_TOOLS),
    "SpecialPlanner": RoleSpec("SpecialPlanner", "DAG 规划", WORKER_TOOLS),
}


def get_role(name: str) -> RoleSpec:
    name, _ = parse_role_instance(name)
    if name not in ROLE_SPECS:
        raise KeyError(f"未知角色：{name}")
    return ROLE_SPECS[name]


def role_names() -> list[str]:
    return list(ROLE_SPECS)
