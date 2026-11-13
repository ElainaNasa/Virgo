"""从运行快照统计已落实的拓扑、角色和模型选择，不把API重试计为新选择。"""
import json
from collections import Counter
from pathlib import Path

KINDS = ('topology', 'role', 'model')


def collect_selections(record):
    result = {kind: {} for kind in KINDS}
    path = record.get('task_snapshot')
    if not path:
        return {**result, 'collection_status': 'unavailable', 'reason': '未创建任务快照'}
    try:
        snapshot = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return {**result, 'collection_status': 'unavailable', 'reason': type(exc).__name__}
    counts = {kind: Counter() for kind in KINDS}
    seen = set()
    for mode in snapshot.get('mode_state', {}).get('instances', []):
        identity = mode.get('mode_instance_id')
        if not identity or identity in seen:
            continue
        seen.add(identity)
        name = mode.get('execution_mode') or mode.get('mode')
        if name:
            counts['topology'][name] += 1
    seen = set()
    for assignment in snapshot.get('agent_assignments', []):
        identity = assignment.get('assignment_id')
        if not identity or identity in seen:
            continue
        seen.add(identity)
        role = assignment.get('role', {})
        name = role.get('name')
        if name:
            if name == 'DomainSpecialist' and role.get('domain'):
                name += '[' + str(role['domain']) + ']'
            counts['role'][name] += 1
        model = assignment.get('llm', {}).get('model_id')
        if model and model != 'Stop':
            counts['model'][model] += 1
    return {**{kind: dict(counts[kind]) for kind in KINDS}, 'collection_status': 'collected'}


def aggregate_selections(records):
    counts = {kind: Counter() for kind in KINDS}
    unavailable = []
    for record in records:
        stats = record.get('selection_statistics') or collect_selections(record)
        if stats.get('collection_status') != 'collected':
            unavailable.append(record.get('sample_id'))
            continue
        for kind in KINDS:
            counts[kind].update(stats.get(kind, {}))
    distributions = {}
    for kind in KINDS:
        total = sum(counts[kind].values())
        distributions[kind] = {'total': total, 'counts': dict(sorted(counts[kind].items())),
                               'percentages': {name: round(count / total * 100, 4) for name,count in sorted(counts[kind].items())} if total else {}}
    return {**distributions, 'task_count': len(records), 'collected_task_count': len(records)-len(unavailable),
            'unavailable_samples': unavailable,
            'counting_rule': '拓扑按实际mode_instance_id计数，使用execution_mode；角色/模型按assignment_id计数，含嵌套、汇总及恢复分配，排除Stop和HTTP重试；不代表API调用次数。'}


def print_selections(stats):
    for kind,label in [('topology','协作拓扑选择'),('role','角色选择'),('model','模型选择')]:
        group=stats[kind]
        parts=[f'{name}: {count} ({group["percentages"][name]:.2f}%)' for name,count in group['counts'].items()]
        print(f'{label}分布（共{group["total"]}次）：' + ('；'.join(parts) or '无可用记录'), flush=True)
    if stats['unavailable_samples']:
        print(f'注意：{len(stats["unavailable_samples"])}题缺少可读快照，未纳入选择分布。', flush=True)
