"""内置工具注册表。具体能力按领域拆分在相邻模块中。"""

from __future__ import annotations

from virgo.tools.environment import environment_inspect
from virgo.tools.filesystem import directory_list, file_read, file_read_lines, file_search, file_stat, file_write
from virgo.tools.patch import file_patch, PATCH_GUIDANCE, PATCH_EXAMPLE
from virgo.tools.registry import ToolRegistry, ToolSpec
from virgo.tools.sandbox import python_execute, shell_execute
from virgo.tools.web import web_fetch, web_search


def register_builtin_tools(registry: ToolRegistry) -> ToolRegistry:
    text = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
        "additionalProperties": False,
    }
    path = {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1}},
        "required": ["path"],
        "additionalProperties": False,
    }
    timeout = {"timeout_seconds": {"type": "number", "minimum": 1, "maximum": 120}}
    registry.register(ToolSpec("web_search", "web_search", "检索网页标题、链接和摘要。", text, web_search, True))
    registry.register(ToolSpec("web_fetch", "web_fetch", "读取指定网页正文并记录来源链接。", {"type": "object", "properties": {"url": {"type": "string", "minLength": 1}}, "required": ["url"], "additionalProperties": False}, web_fetch, True))
    registry.register(ToolSpec("directory_list", "directory_list", "枚举任务工作目录中的文件和子目录；读取目录前应使用此工具。", {"type": "object", "properties": {"path": {"type": "string", "default": "."}, "recursive": {"type": "boolean"}, "include_hidden": {"type": "boolean"}, "max_entries": {"type": "integer", "minimum": 1, "maximum": 500}}, "additionalProperties": False}, directory_list, True))
    registry.register(ToolSpec("file_stat", "file_stat", "检查工作区路径是否存在，以及其文件类型、大小和 MIME 类型。", path, file_stat, True))
    registry.register(ToolSpec("file_search", "file_search", "按文件名或内容搜索任务工作目录；查询 `*` 可枚举可见文件。", {"type": "object", "properties": {**text["properties"], "include_hidden": {"type": "boolean"}}, "required": ["query"], "additionalProperties": False}, file_search, True))
    registry.register(ToolSpec("file_read", "file_read", "按字符分页读取文本。offset_chars 与 limit_chars 的单位都是字符；首次调用省略 offset_chars，后续使用返回的 next_offset。按行定位源码请使用 file_read_lines。", {"type": "object", "properties": {**path["properties"], "offset_chars": {"type": "integer", "minimum": 0}, "limit_chars": {"type": "integer", "minimum": 1, "maximum": 10000}}, "required": ["path"], "additionalProperties": False}, file_read, True))
    registry.register(ToolSpec("file_read_lines", "file_read_lines", "按 1 起始行号读取文本闭区间 [start_line, end_line]；用于源码定位、审阅和生成最小补丁。", {"type": "object", "properties": {**path["properties"], "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1, "maximum": 100000}}, "required": ["path", "start_line", "end_line"], "additionalProperties": False}, file_read_lines, True))
    registry.register(ToolSpec("file_write", "file_write", "在任务工作目录创建或更新文本产物。", {"type": "object", "properties": {"path": {"type": "string", "minLength": 1}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}, file_write, False))
    registry.register(ToolSpec("file_patch", "file_patch", "在任务工作目录先校验再整体应用 Git unified diff；无需 Git 仓库，不操作暂存区。" + PATCH_GUIDANCE + "\n格式示例：\n" + PATCH_EXAMPLE, {"type": "object", "properties": {"patch": {"type": "string", "minLength": 20}}, "required": ["patch"], "additionalProperties": False}, file_patch, False, timeout_seconds=30))
    registry.register(ToolSpec("environment_inspect", "environment_inspect", "检查隔离代码沙箱的 Python 版本、平台和指定依赖是否可用。", {"type": "object", "properties": {"packages": {"type": "array", "items": {"type": "string"}, "maxItems": 100}, **timeout}, "additionalProperties": False}, environment_inspect, True))
    registry.register(ToolSpec("python_execute", "python_execute", "在禁网 Docker 沙箱中运行 Python。容器根文件系统只读，但当前目录和 VIRGO_WORKSPACE 是 Agent 正在修改的同一可写任务工作区，用于真实验证。", {"type": "object", "properties": {"code": {"type": "string", "minLength": 1}, **timeout}, "required": ["code"], "additionalProperties": False}, python_execute, True, timeout_seconds=125))
    registry.register(ToolSpec("shell_execute", "shell_execute", "在禁网 Docker 沙箱中运行测试、编译或检查命令。容器根文件系统只读，但当前目录和 VIRGO_WORKSPACE 是 Agent 正在修改的同一可写任务工作区，用于真实验证。", {"type": "object", "properties": {"command": {"type": "string", "minLength": 1}, **timeout}, "required": ["command"], "additionalProperties": False}, shell_execute, True, timeout_seconds=125))
    return registry
