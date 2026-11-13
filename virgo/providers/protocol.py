"""兼容模型响应中的协议残片识别。"""

from __future__ import annotations


_DSML_MARKERS = ("<｜｜DSML｜｜", "</｜｜DSML｜｜", "<tool_call>", "</tool_call>", "<function=")


def is_dsml_tool_protocol_residue(content: str) -> bool:
    """判断正文是否含有被错误当成文本的 DSML 工具调用标记。"""
    return any(marker in content for marker in _DSML_MARKERS)
