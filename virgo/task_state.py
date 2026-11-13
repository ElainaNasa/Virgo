"""工作交付的五字段累计状态及历史格式迁移。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from virgo.roles.prompt_data import TASK_OUTPUT_TEMPLATE


def empty_state() -> dict[str, list]:
    """每次返回独立数组，避免节点间共享可变模板。"""
    return deepcopy(TASK_OUTPUT_TEMPLATE)


def is_task_state(value: Any) -> bool:
    from virgo.state_protocol import StateProtocolError, validate_state
    try:
        validate_state(value)
        return True
    except StateProtocolError:
        return False


def normalize_state(value: Any, *, artifacts=None, verification=None, issues=None) -> dict[str, list]:
    """兼容历史文本；新状态保留结构，不转成 Python 字符串。"""
    if is_task_state(value):
        return deepcopy(value)
    if isinstance(value, dict):
        # 仅用于历史快照读取，不作为新模型输出的接收入口。
        migrated = deepcopy(value)
        for check in migrated.get('verification', []) if isinstance(migrated.get('verification'), list) else []:
            if isinstance(check, dict):
                for key in ('method', 'result_summary', 'evidence'):
                    if check.get(key) is None:
                        check[key] = ''
        if is_task_state(migrated):
            return migrated
    state = empty_state()
    if isinstance(value, str) and value.strip():
        state['confirmed'] = [value]
    state['possible_issue'] = list(issues or [])
    import re
    for item in artifacts or []:
        if not isinstance(item, dict):
            continue
        location = item.get('path') or item.get('location')
        if not isinstance(location, str):
            continue
        digest = re.search(r'sha256=([0-9a-f]{64})', str(item.get('description', '')))
        state['artifacts'].append({'path': location, 'version': item.get('version') or (digest.group(1) if digest else None),
                                   'summary': item.get('summary', item.get('description', '登记产物'))})
    if isinstance(verification, list):
        state['verification'] = deepcopy(verification)
    elif isinstance(verification, dict) and verification.get('status') not in {None, 'not_run'}:
        state['verification'] = [{
            'target': '历史节点交付', 'version': None,
            'status': verification.get('status') if verification.get('status') in {'passed', 'failed'} else 'inconclusive',
            'method': str(verification.get('method') or ''), 'result_summary': str(verification.get('result') or ''),
            'evidence': '',
        }]
    return state


def combine_states(records: list[dict[str, Any]]) -> dict[str, list]:
    """单前序原样交接；多前序保留来源，不替模型裁决冲突。"""
    if len(records) == 1:
        return deepcopy(records[0]['state'])
    state = empty_state()
    for record in records:
        source = f"{record['node_id']}/{record['result_id']}"
        for key, items in record['state'].items():
            for item in items:
                if isinstance(item, str):
                    state[key].append(f'[{source}] {item}')
                elif isinstance(item, dict):
                    state[key].append({**deepcopy(item), 'source_result_id': record['result_id']})
    return state
