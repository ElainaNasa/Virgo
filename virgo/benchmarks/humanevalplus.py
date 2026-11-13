"""HumanEval+ 的 Agent 任务适配与 EvalPlus 样本格式转换。"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Any


def load_problems() -> dict[str, dict[str, Any]]:
    """延迟导入 EvalPlus，避免普通 Virgo 安装强制拉取基准依赖。"""
    try:
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise RuntimeError("缺少 EvalPlus；请安装 `./.venv/bin/pip install 'evalplus'`。") from exc
    return dict(get_human_eval_plus())


def prepare_evalplus_cache(destination: Path) -> Path:
    """为禁网评分容器准备公开数据集缓存，避免容器运行时联网下载。"""
    try:
        from evalplus.data.utils import CACHE_DIR
    except ImportError as exc:
        raise RuntimeError("缺少 EvalPlus；请安装 `./.venv/bin/pip install 'evalplus'`。") from exc
    source_dir = Path(CACHE_DIR)
    datasets = sorted(source_dir.glob("HumanEvalPlus-*.jsonl"))
    if not datasets:
        raise RuntimeError(f"宿主 EvalPlus 缓存中没有 HumanEvalPlus 数据：{source_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        target = destination / dataset.name
        if not target.is_file() or target.stat().st_size != dataset.stat().st_size:
            shutil.copy2(dataset, target)
    return destination


def seed_solution(workspace: Path, problem: dict[str, Any]) -> Path:
    """将 EvalPlus 给定的 Python 前缀原样预置到题目独立工作区。"""
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "solution.py"
    path.write_text(str(problem["prompt"]), encoding="utf-8")
    return path


def task_instruction(task_id: str, problem: dict[str, Any]) -> str:
    return f"""完成以下 HumanEval+ Python 函数实现任务。

任务标识：{task_id}
题目给定的 Python 函数签名和 docstring 已由运行时原样写入任务工作区根目录的 `solution.py`。先使用 `file_read` 读取该文件，再补全函数体。修改时仅更新同一相对路径 `solution.py`；不得改写既有函数签名、import 或 docstring，不得加入 Markdown 围栏、测试代码、解释文本或题外入口。仅使用 Python 标准库。

验证函数行为时使用 `python -m doctest -v solution.py`，或通过 `python_execute` 导入函数并独立执行断言；不要向 `solution.py` 添加测试代码。doctest 没有发现示例时不代表已验证函数行为，应补充独立断言。
"""


def completion_from_source(source: str, prompt: str) -> str:
    """将完整源码转换为 EvalPlus completion，并容忍无语义的文件头格式变化。"""
    if source.startswith(prompt):
        completion = source[len(prompt):]
    else:
        completion = _completion_from_matching_function(source, prompt)
    if not completion.strip():
        raise ValueError("solution.py 未包含函数实现")
    return completion


def _completion_from_matching_function(source: str, prompt: str) -> str:
    """在文件头空行或 import 格式变化后，按函数契约提取真正的函数体。"""
    try:
        expected = ast.parse(prompt)
        actual = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"solution.py 不是可解析的 Python：{exc.msg}") from exc
    expected_function = next((node for node in expected.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    actual_function = next((node for node in actual.body if isinstance(node, type(expected_function)) and node.name == getattr(expected_function, "name", None)), None)
    if expected_function is None or actual_function is None:
        raise ValueError("solution.py 缺少题目要求的入口函数")
    if ast.dump(expected_function.args, include_attributes=False) != ast.dump(actual_function.args, include_attributes=False):
        raise ValueError("solution.py 的入口函数参数与题目不一致")
    doc_node = actual_function.body[0] if actual_function.body and isinstance(actual_function.body[0], ast.Expr) else None
    if not isinstance(doc_node, ast.Expr) or not isinstance(doc_node.value, ast.Constant) or not isinstance(doc_node.value.value, str) or not hasattr(doc_node, "end_lineno"):
        raise ValueError("solution.py 无法定位入口函数 docstring 结束位置")
    lines = source.splitlines(keepends=True)
    return "".join(lines[doc_node.end_lineno:])


def sample(task_id: str, source: str, problem: dict[str, Any]) -> dict[str, str]:
    return {"task_id": task_id, "completion": completion_from_source(source, str(problem["prompt"]))}
