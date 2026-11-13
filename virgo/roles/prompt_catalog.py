"""包内固化角色提示词的访问接口。

运行时只读取 :mod:`virgo.roles.prompt_data`，不依赖工作区中的 ``需求.md``。
需求文档是生成提示词数据时的设计输入，不是部署依赖。
"""

from __future__ import annotations

from .prompt_data import ROLE_PROMPTS, build_role_prompt


# 保留编号仅为兼容既有的测试和调用方；运行时不再用编号解析需求文档。
ROLE_NUMBERS = {name: index for index, name in enumerate(ROLE_PROMPTS, start=1)}


def get_prompt(role_name: str, *, domain: str | None = None) -> str:
    """返回包内固化的完整角色系统提示词。"""
    try:
        prompt = build_role_prompt(role_name, "Single")
    except KeyError as exc:
        raise KeyError(f"未注册角色提示词：{role_name}") from exc
    if domain and role_name == "DomainSpecialist":
        prompt += f"\n当前专业领域：{domain}"
    return prompt


def clear_cache() -> None:
    """兼容旧调用；静态提示词不使用缓存。"""
