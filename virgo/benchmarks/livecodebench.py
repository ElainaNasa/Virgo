"""LiveCodeBench 数据选择和隔离的官方执行器适配。"""
import json
import re
import hashlib
import random
import subprocess
import sys
from pathlib import Path


def load_examples(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ('question_id', 'question_content', 'public_test_cases', 'private_test_cases', 'metadata'):
                if key not in row:
                    raise ValueError(f"数据缺少字段 {key}")
            identity = (row.get('platform'), row['question_id'])
            if identity in seen:
                raise ValueError(f"重复题目 {identity}")
            seen.add(identity)
            rows.append({**row, 'q_idx': len(rows) + 1})
    if not rows:
        raise ValueError('数据集为空')
    return rows


def select_examples(rows, *, mode='random', count=None, start_index=1, seed=0):
    if not 1 <= start_index <= len(rows):
        raise ValueError('start-index 超出数据范围（从1开始）')
    if count is not None and count < 1:
        raise ValueError('count 必须为正数')
    eligible = rows[start_index - 1:]
    if mode == 'all':
        if count is not None:
            raise ValueError('all 模式不接受 count')
        return eligible
    n = count if count is not None else (10 if mode == 'random' else len(eligible))
    if n > len(eligible):
        raise ValueError('count 超出起始序号之后的可用题数')
    if mode == 'random':
        return random.Random(seed).sample(eligible, n)
    if mode == 'sequential':
        return eligible[:n]
    raise ValueError('未知抽取模式')


def task_instruction(row):
    return ('完成以下 LiveCodeBench 编程任务。使用 file_write 将完整 Python 程序写入工作区 solution.py。'
            '标准输入输出题使用 stdin/stdout；函数题严格遵守 starter_code 的类与函数签名。'
            '最终 task_output 说明交付产物和验证情况，不重复粘贴代码。\n\n'
            f"题目：{row.get('question_title', '')}\n{row['question_content']}\n\n"
            f"starter_code：\n{row.get('starter_code', '')}")



def output_comparison(row):
    """仅从题面输出段明确声明的绝对或相对误差提取容差，默认严格比较。"""
    text = row.get('question_content', '')
    section = re.search(r'(?:^|\n)Output\s*\n(.*?)(?=\nConstraints|\nSample Input|\Z)', text, re.S | re.I)
    if not section:
        return {'mode': 'exact'}
    match = re.search(r'absolute or relative error.{0,160}?(?:10\s*\^\s*\{?\s*-\s*(\d{1,2})\s*\}?|1[eE]-(\d{1,2}))', section.group(1), re.S | re.I)
    if not match:
        return {'mode': 'exact'}
    exponent = int(match.group(1) or match.group(2))
    if not 1 <= exponent <= 30:
        return {'mode': 'exact'}
    return {'mode': 'absolute_or_relative', 'tolerance': f'1e-{exponent}', 'source': 'question_content.Output'}

# 官方执行器在独立进程运行；本地与 Docker 共用同一入口。
HARNESS = r'''
import base64, json, pickle, zlib
from pathlib import Path
case_dir = Path(__file__).resolve().parent
row = json.loads((case_dir / 'problem.json').read_text())
comparison = json.loads((case_dir / 'comparison.json').read_text())
if comparison['mode'] == 'absolute_or_relative':
    from official_testing_util import run_test
else:
    from lcb_runner.evaluation.testing_util import run_test
public = json.loads(row['public_test_cases'])
private = row['private_test_cases']
try:
    private = json.loads(private)
except (ValueError, TypeError):
    private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(private))))
cases = public + private
metadata = json.loads(row['metadata'])
sample = {'input_output': json.dumps({'inputs': [x['input'] for x in cases],
    'outputs': [x['output'] for x in cases], 'fn_name': metadata.get('func_name')})}
if comparison['mode'] == 'absolute_or_relative':
    sample['float_tolerance'] = comparison['tolerance']
results, metadata = run_test(sample, test=(case_dir / 'solution.py').read_text(), debug=False, timeout=6)
results = [x.item() if hasattr(x, 'item') else x for x in results]
print('VIRGO_LCB_RESULT=' + json.dumps({'status': 'passed' if results and all(x == True for x in results) else 'failed',
    'results': results, 'metadata': metadata, 'test_count': len(cases), 'comparison': comparison}))
'''


def run_official_test(code, row, case_dir, *, backend="local", image="virgo-livecodebench:local", timeout=180):
    if backend not in {"local", "docker"}:
        raise ValueError("backend 仅支持 local 或 docker")
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / 'problem.json').write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
    (case_dir / 'solution.py').write_text(code, encoding='utf-8')
    comparison = output_comparison(row)
    if comparison['mode'] == 'absolute_or_relative':
        # 固化修补后的官方执行器到本次用例目录，本地与已有 Docker 镜像一致生效。
        source = Path(__file__).resolve().parents[2] / 'data/livecodebench/LiveCodeBench/lcb_runner/evaluation/testing_util.py'
        code_source = source.read_bytes()
        (case_dir / 'official_testing_util.py').write_bytes(code_source)
        comparison['executor_sha256'] = hashlib.sha256(code_source).hexdigest()
    (case_dir / 'comparison.json').write_text(json.dumps(comparison), encoding='utf-8')
    (case_dir / 'runner.py').write_text(HARNESS, encoding='utf-8')
    command = ['docker', 'run', '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,size=512m', '--memory', '2g', '--cpus', '1',
        '--pids-limit', '128', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--user', '65534:65534', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '-v', f'{case_dir.resolve()}:/case:ro', image, 'python', '/case/runner.py']
    if backend == 'local':
        command = [sys.executable, str((case_dir / 'runner.py').resolve())]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              cwd=case_dir.resolve())
        stdout, stderr = done.stdout, done.stderr
        lines = [line.removeprefix('VIRGO_LCB_RESULT=') for line in stdout.splitlines()
                 if line.startswith('VIRGO_LCB_RESULT=')]
        value = json.loads(lines[-1]) if done.returncode == 0 and lines else {
            'status': 'infrastructure_failed', 'reason': f'官方执行器异常，退出码 {done.returncode}'}
    except subprocess.TimeoutExpired:
        stdout, stderr = '', '整题执行进程超时'
        value = {'status': 'timeout', 'reason': stderr}
    except OSError as exc:
        stdout, stderr = '', str(exc)
        value = {'status': 'infrastructure_failed', 'reason': stderr}
    (case_dir / 'stdout.txt').write_text(stdout, encoding='utf-8')
    (case_dir / 'stderr.txt').write_text(stderr, encoding='utf-8')
    return {**value, 'environment': backend, 'command': command,
            'stdout_path': str(case_dir / 'stdout.txt'), 'stderr_path': str(case_dir / 'stderr.txt')}
