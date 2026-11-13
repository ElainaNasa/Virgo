"""SWE-bench 本地数据适配与补丁导出。

只读取任务执行所需的公开字段。参考补丁、测试补丁及测试选择字段不进入
Virgo 的任务提示词或工作区。
"""

from __future__ import annotations

import subprocess
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


SAFE_COLUMNS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "difficulty",
    "version",
    "environment_setup_commit",
)


def load_instance(dataset_path: Path, instance_id: str) -> dict[str, Any]:
    """读取一个实例的公开任务字段，拒绝访问评测答案相关列。"""
    table = pq.read_table(dataset_path, columns=list(SAFE_COLUMNS))
    selected = table.filter(pc.equal(table["instance_id"], instance_id)).to_pylist()
    if len(selected) != 1:
        raise ValueError(f"数据集内未找到唯一实例：{instance_id}")
    return selected[0]


def task_instruction(instance: dict[str, Any]) -> str:
    """构造交给 Virgo 的源码修复任务，不携带测试或参考实现。"""
    return f"""完成以下 SWE-bench 软件修复任务。

任务标识：{instance['instance_id']}
目标仓库：{instance['repo']}
基线提交：{instance['base_commit']}
目标版本：{instance['version']}
SWE-bench 基准标注难度：{instance.get('difficulty') or '未标注'}

当前工作区已经是目标仓库在基线提交的 checkout。请分析问题、定位相关源代码，并直接修改必要的源码文件以修复问题。不要读取或修改 .git、基准数据、测试补丁、参考补丁或任务元数据；不要新增 solution.py 或 patch 文件。最终评测器会从 git diff 导出补丁并运行隐藏测试。

用户报告的问题：
{instance['problem_statement']}
"""


def checkout_base_repo(instance: dict[str, Any], workspace: Path) -> None:
    """在独立目录检出指定基线提交，避免污染主工作区。"""
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"SWE-bench 工作区已存在且非空：{workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    remote = f"https://github.com/{instance['repo']}.git"
    _run(["git", "clone", "--no-checkout", remote, str(workspace)])
    _run(["git", "-C", str(workspace), "checkout", "--detach", str(instance["base_commit"])])


def export_patch(workspace: Path, base_commit: str, *, excluded_paths=(), include_paths=False):
    """导出工作区相对基线的完整二进制安全 diff。"""
    # 临时索引收集所有非忽略的新文件，不修改智能体checkout原有的索引。
    with tempfile.TemporaryDirectory(prefix="virgo-patch-index-") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        def git(*args):
            return subprocess.run(["git", "-C", str(workspace), *args], env=env,
                                  text=True, capture_output=True, check=True)
        from virgo.benchmarks.submission import INTERNAL_DIRECTORIES
        root = workspace.resolve()
        pathspec = ["."]
        for name in sorted(INTERNAL_DIRECTORIES - {".git"}):
            pathspec.extend([f":(exclude,glob){name}/**", f":(exclude,glob)**/{name}/**"])
        for excluded in excluded_paths:
            path = Path(excluded).resolve()
            if root in path.parents:
                relative = path.relative_to(root).as_posix()
                pathspec.extend([f":(exclude,literal){relative}", f":(exclude,glob){relative}/**"])
        git("read-tree", base_commit)
        git("add", "-A", "--", *pathspec)
        patch = git("diff", "--cached", "--binary", base_commit, "--").stdout
        if include_paths:
            paths = git("diff", "--cached", "--name-only", "-z", base_commit, "--").stdout
            return patch, [name for name in paths.split("\0") if name]
        return patch


def prediction(instance_id: str, patch: str) -> dict[str, str]:
    if not patch.strip():
        raise ValueError("Virgo 未产生源码差异，不能提交空补丁")
    return {"instance_id": instance_id, "model_name_or_path": "virgo", "model_patch": patch}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True)
