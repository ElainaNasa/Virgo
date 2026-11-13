"""从最终有效状态读取正式交付，再构造可追溯的评测提交。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from virgo.state_protocol import StateProtocolError, validate_state, validate_artifacts


INTERNAL_DIRECTORIES = {'.git', '.virgo_tool_logs', '.candidate_inputs', '.virgo_parallel', '__pycache__', '.pytest_cache'}


class SubmissionError(ValueError):
    """缺失或失效的正式交付属于提交预检失败，不是评测正确性结论。"""


def final_state(run):
    snapshot = getattr(run, 'snapshot', None)
    state = snapshot.get('state') if isinstance(snapshot, dict) else None
    if not isinstance(state, dict):
        raise SubmissionError('任务没有最终结构化状态，不能从回复文字或旧产物列表猜测提交。')
    status = (snapshot.get('execution_state') or {}).get('status') or getattr(run, 'status', None)
    if status in {'failed', 'blocked', 'cancelled'}:
        raise SubmissionError(f'任务最终状态为 {status}，不能将失败阶段遗留文件作为正式提交。')
    try:
        return validate_state(state)
    except StateProtocolError as exc:
        raise SubmissionError(f'最终状态不符合交付协议：{exc}') from exc


def audit_output(run):
    """原始任务结果仅作审计保存，不能被答案提取器当作正式答案。"""
    return json.dumps(getattr(run, 'output', None), ensure_ascii=False, indent=2, default=str)


@dataclass(frozen=True)
class Submission:
    path: Path
    raw: bytes
    kind: str
    version: str | None
    summary: str

    @property
    def text(self):
        try:
            return self.raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise SubmissionError(f'交付不是可读取的 UTF-8 文本：{self.path}') from exc

    @property
    def sha256(self):
        return hashlib.sha256(self.raw).hexdigest()

    def metadata(self):
        return {'kind': self.kind, 'path': str(self.path), 'version': self.version,
                'sha256': self.sha256, 'bytes': len(self.raw), 'summary': self.summary}

    def persist(self, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        name = self.path.name
        (directory / name).write_bytes(self.raw)
        metadata = {**self.metadata(), 'submitted_file': name, 'submitted_path': str(directory / name),
                    'manifest_path': str(directory / 'submission.json')}
        (directory / 'submission.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return metadata


def load_final_artifact(run, workspace, required_path='solution.md', *, kind='text'):
    """严格选取最终状态登记的正式路径，不回退到候选、旧字段或现存文件。"""
    state = final_state(run)
    if not isinstance(workspace, (str, Path)):
        raise SubmissionError('无法确定正式交付的工作区。')
    root = Path(workspace).resolve()
    try:
        expected = (root / required_path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SubmissionError(f'正式交付路径无法解析：{exc}') from exc
    if root not in expected.parents or any(part in INTERNAL_DIRECTORIES for part in expected.relative_to(root).parts):
        raise SubmissionError(f'指定交付路径不是允许的正式文件：{required_path}')
    candidates = []
    for artifact in state['artifacts']:
        path = (root / artifact['path'].removeprefix('workspace://')).resolve()
        if path == expected:
            candidates.append(artifact)
    if len(candidates) != 1:
        raise SubmissionError(f'最终状态必须唯一登记 {required_path}，实际找到 {len(candidates)} 项。')
    artifact = candidates[0]
    try:
        if not expected.is_file():
            raise OSError('文件不存在或不是普通文件')
        raw = expected.read_bytes()
    except OSError as exc:
        raise SubmissionError(f'正式交付无法读取：{expected}: {exc}') from exc
    version = artifact['version']
    checksum = version.removeprefix('sha256:') if isinstance(version, str) else None
    if checksum and re.fullmatch(r'[0-9a-fA-F]{64}', checksum) and hashlib.sha256(raw).hexdigest() != checksum.lower():
        raise SubmissionError(f'正式文件与最终状态登记的版本哈希不一致：{expected}')
    if not raw.strip():
        raise SubmissionError(f'正式交付为空：{expected}')
    result = Submission(expected, raw, kind, version, artifact['summary'])
    result.text  # 在提交前确认编码，不能把解码异常伪装成评测器错误。
    return result


def load_saved_submission(directory):
    """断点续评使用已冻结的正式提交，不重新读取可能变化的工作区文件。"""
    directory = Path(directory).resolve()
    try:
        metadata = json.loads((directory / 'submission.json').read_text(encoding='utf-8'))
        if metadata.get('kind') not in {'text', 'code', 'structured'}:
            raise SubmissionError('已保存提交类型不合法')
        if not isinstance(metadata.get('submitted_file'), str) or Path(metadata['submitted_file']).name != metadata['submitted_file']:
            raise SubmissionError('已保存提交文件名不合法')
        path = (directory / metadata['submitted_file']).resolve()
        if directory not in path.parents or not path.is_file():
            raise SubmissionError('已保存提交路径越界或不存在')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata['sha256']:
            raise SubmissionError('已保存的提交内容与哈希不一致')
        result = Submission(path, raw, metadata['kind'], metadata.get('version'), metadata.get('summary', ''))
        result.text
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SubmissionError(f'无法恢复已冻结的正式提交：{exc}') from exc


@dataclass(frozen=True)
class EngineeringSubmission:
    patch: str
    base_commit: str
    artifacts: list[dict]
    changed_paths: list[str]

    def persist(self, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'model.patch').write_text(self.patch, encoding='utf-8')
        metadata = {'kind': 'patch', 'base_commit': self.base_commit,
                    'sha256': hashlib.sha256(self.patch.encode('utf-8')).hexdigest(),
                    'bytes': len(self.patch.encode('utf-8')), 'artifacts': self.artifacts, 'changed_paths': self.changed_paths, 'submitted_file': 'model.patch',
                    'submitted_path': str(directory / 'model.patch'), 'manifest_path': str(directory / 'submission.json')}
        (directory / 'submission.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return metadata


def load_engineering_submission(run, workspace, base_commit, *, excluded_paths=()):
    from virgo.benchmarks.swebench import export_patch
    state = final_state(run)
    try:
        state = validate_artifacts(state, workspace)
    except StateProtocolError as exc:
        raise SubmissionError(f'工程交付登记不可接收：{exc}') from exc
    # 工程以实际工作区为准，不能按产物列表裁剪并漏掉删除或新增的源码。
    patch, changed_paths = export_patch(Path(workspace), base_commit, excluded_paths=excluded_paths, include_paths=True)
    if not patch.strip():
        raise SubmissionError('正式工程交付没有可提交的源码差异。')
    return EngineeringSubmission(patch, base_commit, state['artifacts'], changed_paths)
