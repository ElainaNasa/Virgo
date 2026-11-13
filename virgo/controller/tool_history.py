"""每个节点独立维护工具活动上下文；不修改输入 state 或原始审计记录。"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import shlex
import uuid
from pathlib import Path

from virgo.tools.evidence import PROCESS_TOOLS, process_evidence
from virgo.tools.filesystem import safe_path


HISTORY_MARKER = 'active_tool_history'
MAX_SNAPSHOT_BYTES = 2_000_000


def validate_tool_pairs(messages):
    """每个调用必须恰有一个返回；完整交换之间允许普通消息。"""
    pending = set()
    for message in messages:
        if message.get('role') == 'tool':
            identifier = str(message.get('tool_call_id', ''))
            if identifier not in pending:
                raise ValueError(f'工具返回缺少对应调用或重复返回：{identifier}')
            pending.remove(identifier)
        else:
            if pending:
                raise ValueError('上一组工具调用尚未全部返回')
            if message.get('tool_calls'):
                ids = [str(call.get('id', '')) for call in message['tool_calls']]
                if not all(ids) or len(ids) != len(set(ids)):
                    raise ValueError('工具调用 ID 缺失或重复')
                pending = set(ids)
    if pending:
        raise ValueError('存在尚未返回的工具调用')


def recent_complete_exchanges(messages, limit=8):
    """按完整交换选取最近记录，避免后备摘要截断某个工具批次。"""
    validate_tool_pairs(messages)
    groups, index = [], 0
    while index < len(messages):
        count = len(messages[index].get('tool_calls', [])) if messages[index].get('role') == 'assistant' else 0
        groups.append(messages[index:index + count + 1])
        index += count + 1
    selected, size = [], 0
    for group in reversed(groups):
        if selected and size + len(group) > limit:
            break
        selected.insert(0, group)
        size += len(group)
    return [message for group in selected for message in group]


class NodeToolHistory:
    def __init__(self, node_id, workspace=None):
        self.node_id = node_id
        self.workspace = Path(workspace).resolve() if workspace else None
        self.records = {}
        self.observed_paths = set()
        self.file_changes = []
        self.file_origins = {}
        self.blockers = {}
        self.current_versions = {}
        self.last_snapshots = {}
        self.session_id = uuid.uuid4().hex[:12]

    def _path(self, value):
        if not self.workspace or not isinstance(value, str):
            return None
        try:
            return safe_path(str(self.workspace), value.removeprefix('workspace://'))
        except (OSError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _snapshot(path):
        try:
            if not path.is_file():
                return {'version': None, 'exists': False, 'body': None}
            digest = hashlib.sha256()
            body = []
            keep_body = path.stat().st_size <= MAX_SNAPSHOT_BYTES
            with path.open('rb') as stream:
                while chunk := stream.read(65536):
                    digest.update(chunk)
                    if keep_body:
                        body.append(chunk)
            return {'version': digest.hexdigest(), 'exists': True,
                    'body': b''.join(body) if keep_body else None}
        except OSError:
            return None

    def _shell_read_paths(self, tool, arguments):
        if tool != 'shell_execute':
            return []
        command = str(arguments.get('command', ''))
        if any(mark in command for mark in (';', '|', '>', '<', '&', '$', '`', '\n')):
            return []
        try:
            words = shlex.split(command)
        except ValueError:
            return []
        if not words or words[0] not in {'cat', 'head', 'tail', 'sed'}:
            return []
        paths = []
        for value in words[1:]:
            if value.startswith('-') or value.isdigit():
                continue
            if value.startswith('/testbed/'):
                value = value[len('/testbed/'):]
            path = self._path(value)
            if path and path.is_file():
                paths.append(path)
        return paths

    def before_call(self, tool, arguments):
        targets = set(self.observed_paths) if tool in PROCESS_TOOLS | {'file_write', 'file_patch'} else set()
        targets.update(self._shell_read_paths(tool, arguments))
        if tool in {'file_read', 'file_read_lines', 'file_write', 'file_stat'}:
            path = self._path(arguments.get('path'))
            if path:
                targets.add(path)
        if tool == 'file_patch':
            for name in re.findall(r'^\+\+\+ b/(.+)$', str(arguments.get('patch', '')), re.M):
                path = self._path(name.strip('"'))
                if path:
                    targets.add(path)
        return {path: snapshot for path in targets if (snapshot := self._snapshot(path)) is not None}

    def observe(self, call, result, before):
        """保存调用身份，依据真实文件版本变化淘汰旧正文。"""
        function = call.get('function', call)
        tool = function.get('name')
        arguments = function.get('arguments', {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result = copy.deepcopy(result)
        key = f'{len(self.records) + 1}:{call["id"]}'
        record = {'id': key, 'model_tool_call_id': str(call['id']), 'tool_call_id': result.get('tool_call_id'),
                  'tool': tool, 'status': result.get('status'), 'arguments': arguments,
                  'full_log_path': result.get('full_log_path'), 'retired': None, 'file_versions': {}}
        self.records[key] = record
        result['history_record_id'] = key
        after_versions = {}
        for path, old in before.items():
            new = self._snapshot(path)
            if new is None:
                continue
            previous_snapshot = self.last_snapshots.get(path, old)
            self.current_versions[str(path)] = new['version']
            after_versions[str(path)] = new['version']
            if previous_snapshot['version'] != new['version'] or previous_snapshot['exists'] != new['exists']:
                change = self._change(path, previous_snapshot, new, record)
                change['change_source'] = 'during_tool' if old['version'] != new['version'] else 'observed_between_calls'
                self.file_changes.append(change)
            self.last_snapshots[path] = new
            for previous in self.records.values():
                if previous is record or previous['status'] != 'succeeded':
                    continue
                if previous['file_versions'] and all(
                    name in self.current_versions and version != self.current_versions[name]
                    for name, version in previous['file_versions'].items()):
                    previous['retired'] = '文件版本已被后续实际变更替代'
            if tool in {'file_read', 'file_read_lines', 'file_write', 'file_patch'}:
                target = self._path(arguments.get('path'))
                if target == path or tool == 'file_patch' and old['version'] != new['version']:
                    self.observed_paths.add(path)
                    record['file_versions'][str(path)] = new['version']
        for path in self._shell_read_paths(tool, arguments):
            self.observed_paths.add(path)
            record['file_versions'][str(path)] = after_versions.get(str(path))
        if tool in {'file_read', 'file_read_lines'}:
            result['history_kind'] = 'file_read'
            data = result.get('data')
            if isinstance(data, dict) and result.get('status') == 'succeeded':
                target = self._path(arguments.get('path'))
                if target and isinstance(data.get('sha256'), str):
                    record['file_versions'][str(target)] = data['sha256']
                    result['read_version_is_current'] = self.current_versions.get(str(target)) == data['sha256']
                start = data.get('offset', arguments.get('offset_chars', 0)) if tool == 'file_read' else data.get('start_line', arguments.get('start_line', 1))
                stop = start + len(str(data.get('content', ''))) if tool == 'file_read' else data.get('end_line', arguments.get('end_line', start)) + 1
                record['read_scope'] = [tool, start, stop]
                for previous in self.records.values():
                    scope = previous.get('read_scope')
                    if previous is not record and previous['status'] == 'succeeded' and scope and scope[0] == tool and start <= scope[1] and stop >= scope[2] and record['file_versions'] and previous['file_versions'] == record['file_versions']:
                        previous['retired'] = '后续同版本读取已覆盖原读取范围'
            if record['file_versions']:
                result['file_versions'] = dict(record['file_versions'])
        evidence = result.get('decision_evidence') or process_evidence(tool, arguments, result.get('data'), result.get('error'))
        if evidence:
            evidence = copy.deepcopy(evidence)
            evidence['full_log_path'] = result.get('full_log_path')
            evidence['process_log_path'] = result.get('process_log_path')
            evidence['observed_versions_before'] = {str(path): snapshot['version'] for path, snapshot in before.items()}
            evidence['observed_versions_after'] = after_versions
            evidence['versions_changed_during_call'] = evidence['observed_versions_before'] != after_versions
            evidence['version_scope'] = '仅覆盖已观察文件，未声明测试覆盖所有文件'
            record['process_evidence'] = evidence
            # 保留结构化诊断，而不是盲目截取测试输出开头。
            if evidence.get('kind') == 'check' or 'inspection' in evidence or result.get('status') != 'succeeded':
                result.pop('data', None)
            result['decision_evidence'] = evidence
            result['history_kind'] = 'file_read' if self._shell_read_paths(tool, arguments) else 'process'
        failure = self._failure_key(record, result)
        if failure:
            group = self.blockers.setdefault(failure, {'fact': self._error_text(result)[:1200], 'active': True, 'attempts': []})
            group['active'] = True
            group.pop('resolved_by', None)
            group['attempts'].append(record['id'])
            if len(group['attempts']) > 1:
                for identifier in group['attempts']:
                    self.records[identifier]['retired'] = '重复阻塞已合并，逐次记录保留在活动证据中'
        elif result.get('status') == 'succeeded' and (tool not in PROCESS_TOOLS or evidence.get('returncode') == 0):
            for name, group in self.blockers.items():
                if group['active'] and (name.startswith('docker_socket:') and tool in PROCESS_TOOLS or any(
                    self.records[item]['tool'] == tool and self.records[item]['arguments'] == arguments
                    for item in group['attempts'])):
                    group['active'] = False
                    group['resolved_by'] = record['id']
                    group.setdefault('resolutions', []).append(record['id'])
                    for identifier in group['attempts']:
                        self.records[identifier]['retired'] = '后续沙箱执行成功，原连接阻塞已解除'
        return result

    @staticmethod
    def _error_text(result):
        evidence = result.get('decision_evidence') or {}
        return json.dumps({'error': result.get('error'), 'key_stack': evidence.get('key_stack'),
                           'tail': evidence.get('output_tail')}, ensure_ascii=False, sort_keys=True)

    def _failure_key(self, record, result):
        if result.get('status') not in {'failed', 'timeout', 'denied', 'unavailable'}:
            return None
        error = self._error_text(result)
        lower = error.lower()
        if ('docker' in lower and any(word in lower for word in ('socket', 'daemon', 'docker.sock'))
                and any(word in lower for word in ('permission denied', 'cannot connect', 'connection refused'))):
            reason = 'permission_denied' if 'permission denied' in lower else 'connection_unavailable'
            socket = re.search(r'(?:unix://)?[^\s"\\]+docker\.sock', lower)
            return 'docker_socket:' + reason + ':' + (socket.group(0) if socket else 'default')
        return 'identical:' + hashlib.sha256((str(record['tool']) + json.dumps(record['arguments'], sort_keys=True) + error).encode()).hexdigest()

    @staticmethod
    def _text_diff(before, after, path):
        # 先剥离共同前后缀，避免重复源码触发 difflib 的自动噪声启发式。
        first, second = before.decode('utf-8').splitlines(True), after.decode('utf-8').splitlines(True)
        prefix = 0
        while prefix < min(len(first), len(second)) and first[prefix] == second[prefix]:
            prefix += 1
        suffix = 0
        while suffix < min(len(first), len(second)) - prefix and first[-suffix - 1] == second[-suffix - 1]:
            suffix += 1
        start = max(0, prefix - 3)
        end_first = len(first) - max(0, suffix - 3)
        end_second = len(second) - max(0, suffix - 3)
        diff = ''.join(difflib.unified_diff(first[start:end_first], second[start:end_second],
            fromfile=str(path) + ':before', tofile=str(path) + ':after', n=3))
        return re.sub(r'^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@',
            lambda match: f'@@ -{int(match[1]) + start}{match[2] or ""} +{int(match[3]) + start}{match[4] or ""} @@', diff, flags=re.M)

    def _change(self, path, old, new, record):
        change = {'path': str(path), 'before_version': old['version'], 'after_version': new['version'],
                  'changed_by': record['id'], 'full_log_path': record['full_log_path']}
        origin = self.file_origins.setdefault(str(path), old)
        change['original_version'] = origin['version']
        change['diff_from_version'] = origin['version']
        if origin['body'] is not None and new['body'] is not None:
            try:
                diff = self._text_diff(origin['body'], new['body'], path)
                change['diff_preview'] = diff if len(diff) <= 3500 else diff[:1700] + '\n...[差异节选，完整内容见 full_diff_path]...\n' + diff[-1700:]
                change['diff_truncated'] = len(diff) > 3500
                if self.workspace:
                    name = re.sub(r'[^A-Za-z0-9_-]', '_', self.node_id)
                    directory = safe_path(str(self.workspace), f'.virgo_tool_logs/{name}')
                    directory.mkdir(parents=True, exist_ok=True)
                    log = directory / f'change_{self.session_id}_{len(self.file_changes) + 1}.diff'
                    log.write_text(diff, encoding='utf-8')
                    change['full_diff_path'] = str(log)
            except (UnicodeError, OSError):
                change['diff_note'] = '无法生成文本差异；保留实际版本和原始工具证据。'
        else:
            change['diff_note'] = '文件新增、删除或内容过大；保留版本变化与原始工具证据。'
        return change

    def _current_evidence(self, record):
        evidence = copy.deepcopy(record['process_evidence'])
        versions = evidence.get('observed_versions_after', {})
        evidence['applies_to_current_observed_versions'] = (all(self.current_versions.get(path) == version
            for path, version in versions.items()) if versions else None)
        return evidence

    def summary(self):
        def reference(record):
            return {key: record.get(key) for key in ('id', 'model_tool_call_id', 'tool_call_id', 'tool', 'status', 'full_log_path')}
        blockers = []
        for group in self.blockers.values():
            if len(group['attempts']) < 2 and group['active']:
                continue
            attempts = []
            for identifier in group['attempts']:
                record = self.records[identifier]
                attempt = reference(record)
                evidence = record.get('process_evidence', {})
                attempt.update(command=evidence.get('command'), returncode=evidence.get('returncode'),
                               process_log_path=evidence.get('process_log_path'))
                attempts.append(attempt)
            blockers.append({'fact': group['fact'], 'active': group['active'], 'count': len(attempts),
                             'attempts': attempts, 'resolved_by': group.get('resolved_by'), 'resolution_history': group.get('resolutions', [])})
        latest_changes = {}
        transitions = {}
        for change in self.file_changes:
            transitions.setdefault(change['path'], []).append({key: change.get(key) for key in
                ('before_version', 'after_version', 'changed_by', 'full_diff_path', 'full_log_path')})
            latest_changes[change['path']] = {**copy.deepcopy(change), 'transitions': transitions[change['path']]}
        grouped = {identifier for group in self.blockers.values()
                   if len(group['attempts']) > 1 or not group['active'] for identifier in group['attempts']}
        return {'scope': '当前节点工具证据，不是 task.state 的更新',
                'file_changes': list(latest_changes.values()), 'blockers': blockers,
                'file_observations': [{**reference(record), 'versions': dict(record['file_versions']),
                                       'read_scope': record.get('read_scope')} for record in self.records.values()
                                      if record['file_versions'] and record['retired'] is None],
                'checks': [{**reference(record), **self._current_evidence(record)} for record in self.records.values()
                           if record.get('process_evidence', {}).get('kind') == 'check' and record['retired'] is None],
                'executions': [{**reference(record), **self._current_evidence(record)} for record in self.records.values()
                               if record.get('process_evidence', {}).get('kind') == 'process' and record['retired'] is None],
                'retired_calls': [{**reference(record), 'reason': record['retired']} for record in self.records.values()
                                  if record['retired'] and record['id'] not in grouped]}

    def compact(self, messages):
        """成对移除过期调用与返回，混合批次保留仍有效的调用身份。"""
        validate_tool_pairs(messages)
        messages = [message for message in messages if not self._is_summary(message)]
        keep, index = [], 0
        while index < len(messages):
            message = messages[index]
            if message.get('role') != 'assistant' or not message.get('tool_calls'):
                keep.append(message)
                index += 1
                continue
            count = len(message['tool_calls'])
            replies = messages[index + 1:index + count + 1]
            retained_ids, retained_replies = set(), []
            for reply in replies:
                try:
                    key = json.loads(reply['content']).get('history_record_id')
                except (ValueError, TypeError, AttributeError):
                    key = None
                if key not in self.records or self.records[key]['retired'] is None:
                    retained_ids.add(str(reply['tool_call_id']))
                    if key in self.records and self.records[key].get('process_evidence'):
                        payload = json.loads(reply['content'])
                        payload['decision_evidence'] = self._current_evidence(self.records[key])
                        reply = {**reply, 'content': json.dumps(payload, ensure_ascii=False)}
                    retained_replies.append(reply)
            if retained_ids:
                assistant = {**message, 'tool_calls': [call for call in message['tool_calls'] if str(call['id']) in retained_ids]}
                keep.extend([assistant, *retained_replies])
            index += count + 1
        if self.records:
            summary = {'role': 'user', 'content': json.dumps({HISTORY_MARKER: self.summary()}, ensure_ascii=False)}
            keep.insert(1 if keep and keep[0].get('role') == 'user' else 0, summary)
        validate_tool_pairs(keep)
        return keep

    @staticmethod
    def _is_summary(message):
        if message.get('role') != 'user':
            return False
        try:
            payload = json.loads(message.get('content', ''))
            return isinstance(payload, dict) and set(payload) == {HISTORY_MARKER}
        except (ValueError, TypeError):
            return False
