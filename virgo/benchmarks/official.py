"""BigCodeBench 官方公开测试的受限 Docker 执行器。"""

from __future__ import annotations

import ast
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_BIGCODEBENCH_IMAGE = "terryzho/bigcodebench-evaluate:latest"
NETWORK_POLICIES = frozenset({"none", "bridge"})
NETWORK_CALLS = frozenset(
    {
        "urlopen",
        "urlretrieve",
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
        "requests.get",
        "requests.post",
        "requests.request",
        "httpx.get",
        "httpx.post",
        "socket.create_connection",
        "ftplib.FTP",
    }
)


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def public_test_requires_network(test_source: str) -> bool:
    """仅在公开测试实际调用网络 API 时启用联网评测。

    URL 字符串或 mock 的目标名称不构成联网需求，因此像 URL 被 mock 的下载测试
    仍会保持禁网。
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return False
    return any(
        _dotted_name(node.func) in NETWORK_CALLS
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def resolve_network_policy(test_source: str, requested_policy: str = "auto") -> str:
    """解析评测网络策略；auto 只为公开测试的实际联网调用打开 bridge。"""
    if requested_policy == "auto":
        return "bridge" if public_test_requires_network(test_source) else "none"
    if requested_policy not in NETWORK_POLICIES:
        raise ValueError(f"未知评测网络策略：{requested_policy}")
    return requested_policy


def test_driver_source(test_source: str) -> str:
    """让候选实现与公开 unittest 用例共享同一全局命名空间。"""
    return f'''from pathlib import Path
import sys
import types
import unittest

solution_path = Path("solution.py")
module = types.ModuleType("__solution__")
module.__file__ = str(solution_path.resolve())
sys.modules["__solution__"] = module
namespace = module.__dict__
exec(compile(solution_path.read_text(encoding="utf-8"), str(solution_path), "exec"), namespace, namespace)
TEST_SOURCE = {test_source!r}
exec(compile(TEST_SOURCE, "benchmark_test.py", "exec"), namespace, namespace)
case = namespace.get("TestCases")
if case is None:
    raise RuntimeError("公开测试未定义 TestCases")
suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def docker_command(case_dir: Path, image: str, *, network_policy: str = "none") -> list[str]:
    """在只读根文件系统和资源上限下执行单题测试。"""
    if network_policy not in NETWORK_POLICIES:
        raise ValueError(f"未知评测网络策略：{network_policy}")
    return [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "--network", network_policy, "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=1g",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--memory", "3g", "--cpus", "2",
        "--env", "HOME=/tmp", "--env", "MPLCONFIGDIR=/tmp/matplotlib",
        "--volume", f"{case_dir.resolve()}:/work:rw", "--workdir", "/work",
        "--entrypoint", "python", image, "test_driver.py",
    ]


def run_official_test(
    code: str,
    test_source: str,
    case_dir: Path,
    *,
    image: str = DEFAULT_BIGCODEBENCH_IMAGE,
    timeout: float = 120.0,
    network_policy: str = "none",
) -> dict[str, Any]:
    """运行一次公开测试，并将 stdout/stderr 留在 case_dir 供失败轨迹引用。"""
    case_dir.mkdir(parents=True, exist_ok=False)
    case_dir.chmod(0o777)
    (case_dir / "solution.py").write_text(code + "\n", encoding="utf-8")
    (case_dir / "test_driver.py").write_text(test_driver_source(test_source), encoding="utf-8")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            docker_command(case_dir, image, network_policy=network_policy),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "passed" if completed.returncode == 0 else "failed"
        reason = "公开单元测试通过" if status == "passed" else f"公开单元测试退出码：{completed.returncode}"
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        reason = f"容器运行超过 {timeout:g} 秒"
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    (case_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (case_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    return {
        "status": status,
        "reason": reason,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "environment": f"docker:{image};network={network_policy}",
        "network_policy": network_policy,
        "stdout_path": str(case_dir / "stdout.txt"),
        "stderr_path": str(case_dir / "stderr.txt"),
        "evidence": (stderr or stdout)[-4000:] or reason,
    }
