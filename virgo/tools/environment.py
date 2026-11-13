"""沙箱运行环境与依赖可用性检查。"""

from __future__ import annotations

import json
from typing import Any

from virgo.tools.sandbox import run_sandbox_command


def environment_inspect(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    """查询实际代码执行沙箱中的 Python 与指定依赖版本。"""
    packages = [str(item) for item in args.get("packages", [])][:100]
    script = """import importlib.metadata as metadata
import json
import platform
import sys

requested = json.loads(sys.argv[1])
available = []
for name in requested:
    try:
        available.append({\"name\": name, \"installed\": True, \"version\": metadata.version(name)})
    except metadata.PackageNotFoundError:
        available.append({\"name\": name, \"installed\": False, \"version\": None})
if not requested:
    available = [
        {\"name\": distribution.metadata[\"Name\"] or distribution.metadata[\"Summary\"], \"version\": distribution.version}
        for distribution in sorted(metadata.distributions(), key=lambda item: (item.metadata[\"Name\"] or \"\").lower())[:100]
    ]
print(json.dumps({\"python_version\": sys.version.split()[0], \"platform\": platform.platform(), \"packages\": available}, ensure_ascii=False))
"""
    result = run_sandbox_command(
        workspace,
        ["python", "-c", script, json.dumps(packages, ensure_ascii=False)],
        float(args.get("timeout_seconds", 20)),
    )
    if result["returncode"] == 0:
        try:
            result["inspection"] = json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
    return result
