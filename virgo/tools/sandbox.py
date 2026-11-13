"""在隔离 Docker 容器中执行验证命令。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_SANDBOX_IMAGE = "terryzho/bigcodebench-evaluate:latest"
# SWE-bench 镜像将项目依赖和默认工作目录设为 /testbed。验证时以宿主
# 工作区覆盖该路径，保证 Agent 修改、git diff 与测试读取的是同一 checkout。
SANDBOX_WORKSPACE = "/testbed"


def sandbox_image() -> str:
    return os.getenv("VIRGO_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)


def sandbox_command(workspace: str, command: list[str], *, image: str | None = None) -> list[str]:
    """构造禁网、根文件系统只读且使用唯一可写工作区副本的容器命令。"""
    if not command:
        raise ValueError("沙箱命令不能为空")
    root = Path(workspace).resolve()
    return [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=512m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--memory", "2g", "--cpus", "2",
        "--env", "HOME=/tmp", "--env", "MPLCONFIGDIR=/tmp/matplotlib",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", f"VIRGO_WORKSPACE={SANDBOX_WORKSPACE}",
        # 验证命令必须看到 Agent 实际修改的 checkout。仅该 bind mount 可写，
        # 以允许测试创建临时文件、数据库或构建输出；根文件系统保持只读。
        "--volume", f"{root}:{SANDBOX_WORKSPACE}:rw", "--workdir", SANDBOX_WORKSPACE,
        "--entrypoint", command[0],
        image or sandbox_image(),
        *command[1:],
    ]


def run_sandbox_command(workspace: str, command: list[str], timeout_seconds: float) -> dict[str, Any]:
    """执行容器命令并只返回可传递给模型的运行结果。"""
    try:
        completed = subprocess.run(
            sandbox_command(workspace, command),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        error = TimeoutError(f"沙箱运行超过 {timeout_seconds:g} 秒")
        error.stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        error.stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        raise error from exc
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "environment": f"docker:{sandbox_image()}",
    }


def python_execute(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    return run_sandbox_command(
        workspace,
        ["python", "-c", str(args["code"])],
        float(args.get("timeout_seconds", 20)),
    )


def shell_execute(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    return run_sandbox_command(
        workspace,
        ["sh", "-lc", str(args["command"])],
        float(args.get("timeout_seconds", 20)),
    )
