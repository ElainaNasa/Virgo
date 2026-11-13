"""BigCodeBench-Hard 数据适配与非执行性代码检查。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


BIGCODEBENCH_HARD_VERSION = "v0.1.4"
BIGCODEBENCH_HARD_SHA256 = "73a4270b43feb81abefad7bb1b592937c768ea45c3263bedc1cabd30a9a6ce79"
REQUIRED_COLUMNS = {
    "task_id",
    "complete_prompt",
    "test",
    "entry_point",
    "libs",
    "q_idx",
}


def sha256_file(path: Path) -> str:
    """计算数据文件摘要，确保评测数据可追溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bigcodebench_examples(path: Path) -> list[dict[str, Any]]:
    """读取运行 BigCodeBench-Hard 所需的公开字段。"""
    table = pq.read_table(path)
    missing = REQUIRED_COLUMNS - set(table.column_names)
    if missing:
        raise ValueError(f"BigCodeBench 数据缺少字段：{sorted(missing)}")
    return table.to_pylist()


def task_instruction(example: dict[str, Any]) -> str:
    """将原始基准题面转换为系统可执行的代码交付任务。"""
    return (
        "完成以下 BigCodeBench-Hard 编程任务。严格实现题目指定的函数签名、行为、异常语义和返回值。"
        "使用 `file_write` 将完整、可导入的 Python 实现写入任务工作区的 `solution.py`。"
        "最终 `task_output` 只说明实现范围、入口、实际验证和限制，不要重复粘贴源代码或输出测试代码；"
        "系统会登记实际产生的 artifact。\n\n"
        f"任务标识：{example['task_id']}\n"
        f"目标入口：{example['entry_point']}\n"
        f"允许依赖：{example['libs']}\n\n"
        "原始题目：\n"
        f"{example['complete_prompt']}"
    )


def extract_python_code(output: str) -> str | None:
    """兼容历史文本的独立提取工具；正式评测不使用文本猜测代码。"""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        return max((block.strip() for block in blocks), key=len, default="") or None
    candidate = output.strip()
    if re.search(r"(?m)^(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+\s*\()", candidate):
        return candidate
    return None


def artifact_python_code(artifacts: list[dict[str, Any]], workspace: Path, *, pervious_artifacts: list[dict[str, Any]] | None = None, tool_trace: list[dict[str, Any]] | None = None, required_path: str | None = None) -> tuple[Path, str] | None:
    """兼容旧产物清单的独立读取工具；正式评测使用 load_final_artifact。"""
    root = workspace.resolve()
    recovered = []
    for event in tool_trace or []:
        context = event.get("context") or {}
        output = event.get("output") or {}
        if event.get("event_type") == "tool_call" and context.get("tool_id") in {"file_write", "file_patch"} and output.get("status") == "succeeded":
            recovered.extend(output.get("artifacts") or [])
    candidates = []
    for artifact in reversed([*recovered, *(pervious_artifacts or []), *artifacts]):
        if isinstance(artifact.get("path"), str):
            path = (root / artifact["path"].removeprefix("workspace://")).resolve()
        else:
            if artifact.get("type") != "code" or not isinstance(artifact.get("location"), str):
                continue
            location = artifact["location"]
            if not location.startswith("workspace://"):
                continue
            path = (root / location.removeprefix("workspace://")).resolve()
        if root not in path.parents or path.suffix != ".py" or not path.is_file():
            continue
        if required_path is not None and path != (root / required_path).resolve():
            continue
        candidates.append(path)
    # 登记顺序不代表交付优先级，标准解答必须优先于验证脚本。
    candidates.sort(key=lambda path: path != root / "solution.py")
    if candidates:
        path = candidates[0]
        return path, path.read_text(encoding="utf-8")
    return None


def compile_check(code: str | None, filename: str) -> dict[str, Any]:
    """只编译生成代码，不在主机上执行模型输出。"""
    if not code:
        return {"status": "not_run", "reason": "未从模型输出提取到 Python 代码"}
    try:
        compile(code, filename, "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "passed", "reason": "Python 编译通过；未在主机执行生成代码"}
