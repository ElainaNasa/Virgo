"""Parallel 候选的独立副本、汇总输入导出和受控结果发布。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


# 调度状态、其他候选与导入快照不能进入候选的共同基线。
_RESERVED = {'.virgo_parallel', '.candidate_inputs', '__pycache__', '.pytest_cache'}


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def copy_workspace(source: Path, target: Path, *, excluded=()) -> None:
    """复制实际字节，不用硬链接；内部符号链接重定位，外部链接不导入。"""
    source = source.resolve()
    target.mkdir(parents=True, exist_ok=False)
    excluded = tuple(Path(path).resolve() for path in excluded)

    def copy_dir(current, destination):
        for entry in sorted(current.iterdir()):
            if entry.name in _RESERVED or any(_inside(entry.absolute(), path) for path in excluded):
                continue
            output = destination / entry.name
            if entry.is_symlink():
                resolved = entry.resolve()
                if _inside(resolved, source) and not any(_inside(resolved, path) for path in excluded):
                    relative = resolved.relative_to(source)
                    output.symlink_to(os.path.relpath(target / relative, output.parent), target_is_directory=resolved.is_dir())
            elif entry.is_dir():
                output.mkdir()
                copy_dir(entry, output)
            elif entry.is_file():
                # 外部 Git worktree 指针会泄露宿主目录，不能复制到隔离工作区。
                if entry == source / '.git':
                    continue
                shutil.copy2(entry, output)

    copy_dir(source, target)


def file_manifest(root: Path) -> dict[str, str]:
    """只跟踪可发布的普通文件，拒绝通过链接发布到其他工作区。"""
    manifest = {}
    for current, directories, files in os.walk(root, followlinks=False):
        current = Path(current)
        directories[:] = [name for name in directories if name not in _RESERVED | {'.git'}
                          and not (current / name).is_symlink()]
        for name in files:
            path = current / name
            if not path.is_symlink() and path.is_file() and name != '.git':
                manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


class ParallelWorkspaces:
    def __init__(self, context, task_id, mode_id, source, count):
        self.context = context
        self.task_id = task_id
        self.mode_id = mode_id
        self.source = Path(source).resolve()
        self.root = Path(context.store.state_dir).resolve() / '.virgo_parallel' / task_id / mode_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.baseline = self.root / 'baseline'
        excluded = [self.root]
        state_dir = Path(context.store.state_dir).resolve()
        if state_dir != self.source and _inside(state_dir, self.source):
            excluded.append(state_dir)
        excluded.extend(path for path in getattr(context, 'workspace_excluded_paths', [])
                        if Path(path).resolve() != self.source)
        with context.workspace_publish_lock:
            copy_workspace(self.source, self.baseline, excluded=excluded)
            self.baseline_manifest = file_manifest(self.baseline)
        self.candidates = []
        for index in range(count):
            candidate = self.root / f'candidate_{index}'
            copy_workspace(self.baseline, candidate)
            self.candidates.append(candidate)
        self.summary = self.root / 'summary'

    def binding(self, index):
        return {'workspace': str(self.candidates[index]), 'workspace_source': str(self.source),
                'workspace_kind': 'parallel_candidate',
                'input_path_map': {str(self.source): str(self.candidates[index])}}

    def prepare_summary(self):
        """汇总仅接收候选产物的副本，不挂载候选原目录。"""
        copy_workspace(self.baseline, self.summary)
        exports = self.summary / '.candidate_inputs'
        exports.mkdir()
        mappings = {str(self.source): str(self.summary)}
        snapshot = self.context.store.get(self.task_id)
        for index, candidate in enumerate(self.candidates):
            destination = exports / f'candidate_{index}'
            destination.mkdir()
            current = file_manifest(candidate)
            changed = {name for name, digest in current.items() if self.baseline_manifest.get(name) != digest}
            # 即使文件没有变化，正式登记的产物和检查证据也必须可读取。
            for node in snapshot['execution_state']['nodes']:
                if node.get('workspace') != str(candidate):
                    continue
                for rid in node['output_result_ids']:
                    state = snapshot['agent_outputs'][rid]['output']['task_output']
                    references = [item.get('path') for item in state['artifacts']]
                    references += [item.get('evidence') for item in state['verification']]
                    for reference in references:
                        if not isinstance(reference, str):
                            continue
                        path = (candidate / reference.removeprefix('workspace://')).resolve()
                        if _inside(path, candidate) and path.is_file() and not path.is_symlink():
                            changed.add(str(path.relative_to(candidate)))
            for name in sorted(changed):
                path = candidate / name
                if path.is_symlink() or not path.is_file() or not _inside(path.resolve(), candidate):
                    continue
                output = destination / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
            (exports / f'candidate_{index}.manifest.json').write_text(json.dumps({
                'changed': sorted(changed), 'deleted': sorted(set(self.baseline_manifest) - set(current)),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            mappings[str(candidate)] = str(destination)
        return {'workspace': str(self.summary), 'workspace_source': str(self.source),
                'workspace_kind': 'parallel_summary', 'input_path_map': mappings}

    def publish(self):
        """同一上下文内串行检查与发布，避免多个嵌套 Parallel 同时覆盖。"""
        with self.context.workspace_publish_lock:
            return self._publish_locked()

    def _publish_locked(self):
        """仅发布汇总实际修改；父工作区发生冲突时先拒绝整批发布。"""
        current = file_manifest(self.summary)
        changed = [name for name, digest in current.items() if self.baseline_manifest.get(name) != digest]
        deleted = sorted(set(self.baseline_manifest) - set(current))
        exported = self.source / '.candidate_inputs' / self.mode_id
        if not _inside(exported.resolve(), self.source) or (self.source / '.candidate_inputs').is_symlink():
            raise PermissionError('候选导出路径越出父工作区')
        if exported.exists():
            raise FileExistsError(f'候选导出目录已存在：{exported}')
        imports = self.summary / '.candidate_inputs'
        if imports.is_symlink() or not imports.is_dir():
            raise PermissionError('候选导入目录被替换或删除')
        for name in [*changed, *deleted]:
            destination = self.source / name
            if not _inside(destination.resolve(), self.source) or destination.is_symlink():
                raise PermissionError(f'汇总产物发布路径越界：{name}')
            actual = hashlib.sha256(destination.read_bytes()).hexdigest() if destination.is_file() else None
            if actual != self.baseline_manifest.get(name):
                raise RuntimeError(f'父工作区文件已变化，拒绝覆盖：{name}')
        exported.parent.mkdir(parents=True, exist_ok=True)
        copy_workspace(imports, exported)
        for name in changed:
            destination = self.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + f'.{self.mode_id}.tmp')
            shutil.copy2(self.summary / name, temporary)
            os.replace(temporary, destination)
        for name in deleted:
            (self.source / name).unlink()
        return {'changed': changed, 'deleted': deleted,
                'path_map': {str(self.summary): str(self.source),
                             str(self.summary / '.candidate_inputs'): str(exported)}}
