"""工具结果的确定性诊断摘要与完整证据归档。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from virgo.tools.filesystem import safe_path


PROCESS_TOOLS = {'shell_execute', 'python_execute', 'environment_inspect'}


def process_evidence(tool: str, arguments: dict, data: Any, error=None) -> dict:
    """从未截断输出提取失败信息，不将成功传输当作检查通过。"""
    if tool not in PROCESS_TOOLS or not isinstance(data, dict):
        return {}
    stdout, stderr = str(data.get('stdout') or ''), str(data.get('stderr') or '')
    lines = (stdout + '\n' + stderr).splitlines()
    failures, stack = [], []
    for index, line in enumerate(lines):
        if re.search(r'^(?:FAILED\s|FAIL:|ERROR:)|\b\d+\s+-\s+.+\(Failed\)', line.strip()):
            failures.append(line[:800])
        if re.search(r'Traceback|^\s*File ".+", line |^E\s|\b(?:AssertionError|[\w.]+Error|Exception):|\bERROR collecting\b|(?:fatal )?error:|docker.*(?:socket|daemon)|permission denied|cannot connect', line, re.I):
            stack.extend(lines[max(0, index - 1):min(len(lines), index + 3)])
    stack = list(dict.fromkeys(line[:800] for line in stack if line.strip()))
    command = str(arguments.get('command') or arguments.get('code') or '')
    is_check = bool(failures or re.search(r'\b(?:pytest|unittest|ctest|tox|nox|rspec)\b|\b(?:cargo|go|npm|pnpm|yarn|mvn|gradle|make|dotnet)\s+test\b', command))
    return {
        'kind': 'check' if is_check else 'process',
        'command': arguments.get('command') if tool == 'shell_execute' else 'python -c' if tool == 'python_execute' else '环境依赖检查',
        'code_preview': str(arguments.get('code', ''))[:600] if tool == 'python_execute' else None,
        'returncode': data.get('returncode'), 'timed_out': bool(data.get('timed_out')),
        'failed_tests': list(dict.fromkeys(failures))[:50], 'failed_test_lines_total': len(failures),
        'key_stack': stack[-24:], 'output_head': [line[:800] for line in lines[:8]], 'output_tail': [line[:800] for line in lines[-8:]],
        **({'inspection': data['inspection']} if isinstance(data.get('inspection'), dict) else {}),
        'error': error,
    }


def archive_tool_result(workspace, node_id, call_id, tool, arguments, result, raw_data):
    """将完整输入输出写入节点自己的目录，摘要只引用真实存在的日志。"""
    node_name = re.sub(r'[^A-Za-z0-9_-]', '_', str(node_id))
    directory = safe_path(str(workspace), f'.virgo_tool_logs/{node_name}')
    directory.mkdir(parents=True, exist_ok=True)
    path = safe_path(str(workspace), f'.virgo_tool_logs/{node_name}/{call_id}.json')
    path.write_text(json.dumps({'tool': tool, 'arguments': arguments,
        'result': {**result, 'data': raw_data if raw_data is not None else result.get('data')}},
        ensure_ascii=False, default=str, indent=2), encoding='utf-8')
    references = {'full_log_path': str(path)}
    if tool in PROCESS_TOOLS and isinstance(raw_data, dict):
        process_log = path.with_suffix('.log')
        process_log.write_text(
            '执行参数：' + json.dumps(arguments, ensure_ascii=False) + '\n'
            + '返回码：' + str(raw_data.get('returncode')) + '\n'
            + 'stdout:\n' + str(raw_data.get('stdout') or '')
            + '\nstderr:\n' + str(raw_data.get('stderr') or ''), encoding='utf-8')
        references['process_log_path'] = str(process_log)
    return references
