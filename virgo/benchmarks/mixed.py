"""四数据集混合题库：可复现抽样、无答案泄露的任务构造及原评分规则分发。"""

from virgo.benchmarks.submission import load_final_artifact, SubmissionError, audit_output
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from scripts import evaluate_aime as aime
from scripts import evaluate_math_sample as math
from virgo.benchmarks.bigcodebench import compile_check
from virgo.benchmarks.livecodebench import load_examples, task_instruction, run_official_test

DATASETS = ('aime', 'math', 'mmlu_pro', 'livecodebench')


def balanced_sample(rows, count, rng):
    """按领域轮流分配配额，余数随机分配，容量不足时重分配。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row['category']].append(row)
    categories = sorted(groups)
    rng.shuffle(categories)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(count, len(rows)):
        for category in categories:
            if groups[category] and len(selected) < count:
                selected.append(groups[category].pop())
    return selected


def build_pool(data_dir, per_dataset=120, seed=77):
    import pyarrow.parquet as pq
    root = Path(data_dir).resolve()
    source_files = sorted((root / 'aime').glob('*/*.jsonl'))
    all_rows = {'aime': [dict(row, source_file=str(path.relative_to(root)), source_index=i, exam='I' if i < 15 else 'II', problem_number=i % 15 + 1) for path in source_files for i, row in enumerate(aime.load_dataset(str(path)))]}
    math_files = sorted((root / 'math/test').glob('*.parquet'))
    all_rows['math'] = [r for r in math.load_examples(root / 'math/test') if r['level'] == 'Level 5']
    mmlu_files = sorted((root / 'mmlu_pro/data').glob('test-*.parquet'))
    all_rows['mmlu_pro'] = [dict(row, source_file=path.name, source_index=i) for path in mmlu_files for i, row in enumerate(pq.read_table(path).to_pylist())]
    lcb_file = root / 'livecodebench/test6.jsonl'
    all_rows['livecodebench'] = load_examples(lcb_file)
    source_files += math_files + mmlu_files + [lcb_file]
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    pool = []
    for dataset in DATASETS:
        rows = all_rows[dataset]
        if not rows:
            raise ValueError(f'{dataset} 筛选后为空')
        rng = random.Random(f'{seed}:{dataset}')
        selected = balanced_sample(rows, per_dataset, rng) if dataset == 'mmlu_pro' else rng.sample(rows, min(per_dataset, len(rows)))
        for row in selected:
            if dataset == 'aime':
                identity = f"{row['source_file']}:{row['source_index']}"
            elif dataset in {'math', 'mmlu_pro'}:
                identity = f"{row['source_file']}:{row['source_index']}"
            else:
                identity = f"{row['platform']}:{row['question_id']}"
            pool.append({'sample_id': f'{dataset}:{identity}', 'dataset': dataset, 'example': row})
    if len({row['sample_id'] for row in pool}) != len(pool):
        raise ValueError('混合题库存在重复题目标识')
    return pool, {'pool_seed': seed, 'per_dataset_target': per_dataset, 'available': {k: len(v) for k,v in all_rows.items()}, 'source_sha256': manifest}


def instruction(sample):
    """只向系统发送题面及输出协议，不传标准答案、参考解或私有测试。"""
    row, dataset = sample['example'], sample['dataset']
    if dataset == 'livecodebench':
        return task_instruction(row)
    if dataset == 'mmlu_pro':
        labels = 'ABCDEFGHIJ'
        options = '\n'.join(f'({labels[i]}): {v}' for i,v in enumerate(row['options']))
        return f"完成以下 MMLU-Pro 选择题。\n领域：{row['category']}\n{row['question']}\n选项：\n{options}\n请独立推理，最终一行严格写为 The answer is X，其中 X 为所选选项字母。"
    if dataset == 'aime':
        return f"AIME {row['year']} {row['exam']} 第 {row['problem_number']} 题\n\n{row['problem']}\n\n请给出推理，最终一行将答案写成 \\boxed{{0到999之间的整数}}。"
    if dataset == 'math':
        return row['problem'] + '\n\nFor benchmark scoring, end the final answer with \\boxed{your final answer}.'
    raise ValueError(f'未知数据集 {dataset}')


def grade(sample, run, workspace, system, case_dir, *, backend='local', timeout=180, submission=None):
    dataset, row = sample['dataset'], sample['example']
    try:
        submission = submission or load_final_artifact(run, workspace,
            'solution.py' if dataset == 'livecodebench' else 'solution.md',
            kind='code' if dataset == 'livecodebench' else 'text')
        metadata = submission.persist(case_dir) if case_dir is not None else submission.metadata()
    except SubmissionError as exc:
        return {'validator': dataset, 'status': 'failed_preflight', 'correct': False,
                'prediction': None, 'reason': str(exc)}
    output = submission.text
    if dataset == 'aime':
        prediction = aime.extract_prediction(output)
        expected = aime.normalize_answer(row['answer'])
        return {'submission': metadata, 'validator': 'evaluate_aime.extract_prediction/normalize_answer', 'prediction': prediction, 'reference': expected, 'correct': prediction is not None and prediction == expected}
    if dataset == 'math':
        expected = math.remove_boxed(math.last_boxed_only_string(row['solution']))
        value = math.score_result({'output': output, 'reference_answer': expected})
        return {'submission': metadata, 'validator': 'evaluate_math_sample.score_result', 'prediction': value['prediction'], 'reference': expected, 'correct': value['correct'], 'normalized_exact_match': value['normalized_exact_match'], 'comparison_method': value['comparison_method']}
    if dataset == 'mmlu_pro':
        # 沿用原脚本 get_prediction 的提取正则与字母精确比较；无法提取不随机猜测。
        match = re.search(r'answer is \(?([ABCDEFGHIJ])\)?', output)
        prediction = match.group(1) if match else None
        valid = 'ABCDEFGHIJ'[:len(row['options'])]
        return {'submission': metadata, 'validator': 'MMLU-Pro A-J exact match (no random fallback)', 'prediction': prediction, 'reference': row['answer'], 'correct': prediction in valid and prediction == row['answer'] if prediction else False}
    if dataset == 'livecodebench':
        artifact = (submission.path, submission.text)
        check = compile_check(artifact[1] if artifact else None, 'solution.py')
        if check['status'] != 'passed':
            return {'submission': metadata, 'validator': 'LiveCodeBench official', 'status': 'failed_preflight', 'compile_check': check, 'correct': False}
        official = run_official_test(artifact[1], row, case_dir, backend=backend, timeout=timeout)
        return {'submission': metadata, 'validator': 'LiveCodeBench official', 'artifact': str(artifact[0]), 'compile_check': check, 'official_test': official, 'status': official['status'], 'correct': official['status'] == 'passed'}
    raise ValueError(f'未知数据集 {dataset}')
