"""工作区内的受控 Git unified diff 应用工具。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from virgo.tools.filesystem import safe_path


PATCH_EXAMPLE = "diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
PATCH_GUIDANCE = (
    "提交裸 Git unified diff，不要加 Markdown 围栏或 *** Begin Patch。"
    "必须包含 diff --git、---、+++ 和正确的 @@ hunk；上下文行以一个空格开头，"
    "删除行以 - 开头，新增行以 + 开头，hunk 行数须与正文一致，末尾保留换行。"
    "先读取目标文件核对实际内容；不要原样重试失败补丁。"
    "连续失败时停止同类尝试，检查文件及错误，再重建最小补丁；"
    "仅在完整读取目标文件且确认未截断时，才考虑 file_write，禁止用局部内容覆盖完整文件。"
)


class PatchError(ValueError):
    """携带阶段、退出码及文件是否可能已改变的补丁错误。"""

    def __init__(self, stage: str, detail: str, returncode: int | None = None):
        super().__init__(f"补丁{stage}失败：{detail}\n{PATCH_GUIDANCE}")
        self.details = {"code": "patch_error", "stage": stage, "returncode": returncode,
                        "message": str(self), "retryable": False,
                        "files_may_have_changed": stage == "apply",
                        "recovery": PATCH_GUIDANCE, "format_example": PATCH_EXAMPLE}


def file_patch(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    """先检查再整体应用，不依赖 Git 仓库或暂存区，不使用 --reject。"""
    patch = str(args["patch"])
    root = Path(workspace).resolve()
    if not patch.startswith("diff --git "):
        raise PatchError("format", "patch 必须是以 'diff --git ' 开头的 unified diff")

    def run(stage: str, options: list[str]):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "apply", *options, "-"],
                input=patch, text=True, capture_output=True, check=False, timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PatchError(stage, str(exc)) from exc
        if result.returncode:
            raise PatchError(stage, result.stderr.strip()[:2000], result.returncode)
        return result

    # 从补丁本身取得目标，不再用 git diff 猜测此次改动；普通目录也可应用。
    stats = run("parse", ["--numstat", "-z"])
    changed = []
    for record in stats.stdout.split("\0"):
        if not record:
            continue
        fields = record.split("\t", 2)
        if len(fields) != 3 or not fields[2]:
            raise PatchError("parse", "无法解析补丁目标路径")
        path = fields[2]
        safe_path(str(root), path)
        if Path(path).is_absolute() or ".git" in Path(path).parts:
            raise PatchError("parse", "不允许绝对路径或修改 .git 元数据")
        changed.append(path)
    if not changed:
        raise PatchError("parse", "补丁没有可应用的文件修改")
    run("check", ["--check", "--whitespace=nowarn"])
    run("apply", ["--whitespace=nowarn"])
    return {"changed_paths": list(dict.fromkeys(changed)), "patch_chars": len(patch)}
