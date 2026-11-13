"""工作状态的严格接收协议；只检查格式和文件可访问性，不判断答案正确。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator


class ArtifactState(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    path: Annotated[StrictStr, Field(min_length=1)]
    version: StrictStr | None
    summary: StrictStr
    source_result_id: StrictStr | None = None

    @field_validator('path')
    @classmethod
    def nonblank_path(cls, value):
        if not value.strip():
            raise ValueError('path 不能为空白字符串')
        return value


class VerificationState(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    target: StrictStr
    version: StrictStr | None
    status: Literal['passed', 'failed', 'inconclusive']
    method: StrictStr
    result_summary: StrictStr
    evidence: StrictStr
    source_result_id: StrictStr | None = None


class TaskState(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    confirmed: list[StrictStr]
    possible_issue: list[StrictStr]
    artifacts: list[ArtifactState]
    verification: list[VerificationState]
    remaining: list[StrictStr]


class StateProtocolError(ValueError):
    def __init__(self, stage, errors):
        self.stage = stage
        self.errors = list(errors)
        super().__init__('；'.join(self.errors))

    def as_dict(self):
        return {'accepted': False, 'stage': self.stage, 'errors': self.errors,
                'answer_correctness_checked': False}


def validate_state(value):
    try:
        return TaskState.model_validate(value).model_dump(exclude_unset=True)
    except ValidationError as exc:
        errors = ['.'.join(map(str, item['loc'])) + ': ' + item['msg'] for item in exc.errors(include_url=False)]
        raise StateProtocolError('structure', errors) from exc


def validate_artifacts(state, workspace):
    """整体拒绝带无效引用的新快照；不静默删掉引用后接收其他字段。"""
    state = copy.deepcopy(state)
    if workspace is None:
        if state['artifacts']:
            raise StateProtocolError('artifacts', ['无法确定允许的工作区，不能接收产物引用。'])
        return state
    root = Path(workspace).resolve()
    errors = []
    for index, artifact in enumerate(state['artifacts']):
        try:
            path = (root / artifact['path'].removeprefix('workspace://')).resolve()
            if root not in path.parents:
                raise PermissionError('文件位于允许工作区之外')
            if not path.is_file():
                raise FileNotFoundError('文件不存在或不是普通文件')
            with path.open('rb') as stream:
                stream.read(1)
            artifact['path'] = str(path)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f'artifacts.{index}.path: {type(exc).__name__}: {exc}')
    if errors:
        raise StateProtocolError('artifacts', errors)
    return state


def parse_output_object(content):
    if not isinstance(content, str):
        raise StateProtocolError('json', ['响应正文必须是字符串。'])
    if not content.strip():
        raise StateProtocolError('json', ['响应正文为空。'])
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'JSON 字段重复：{key}')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f'JSON 不允许常量：{value}')
    try:
        data = json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise StateProtocolError('json', [f'JSON 解析失败：{exc}']) from exc
    if not isinstance(data, dict):
        raise StateProtocolError('json', ['响应必须解析为 JSON 对象，不能是数组、标量或 null。'])
    return data


def validate_output(content, *, role=None, workspace=None, check_artifacts=False):
    data = parse_output_object(content)
    if role == 'SpecialPlanner':
        tasks = data.get('tasks', data.get('plan'))
        if not isinstance(tasks, list) or not tasks or not all(isinstance(item, dict) for item in tasks):
            raise StateProtocolError('structure', ['SpecialPlanner 必须返回非空 tasks/plan 对象数组。'])
        # 派发协议由 DAG 规划校验器继续检查，不伪造工作成果状态。
        return data, None
    if 'task_output' not in data:
        raise StateProtocolError('structure', ['缺少 task_output 对象。'])
    state = validate_state(data['task_output'])
    if check_artifacts:
        state = validate_artifacts(state, workspace)
    return data, state
