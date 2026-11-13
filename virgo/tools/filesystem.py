"""任务工作区内的受限文件工具。"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any


_HIDDEN_DIRECTORIES = {".git", "__pycache__", "var"}


def safe_path(workspace: str, relative: str) -> Path:
    """解析工作区相对路径，并阻止越界访问。"""
    root = Path(workspace).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise PermissionError("文件路径超出任务工作目录")
    return path


def _relative(root: Path, path: Path) -> str:
    value = str(path.relative_to(root))
    return "." if not value else value


def _visible(path: Path, include_hidden: bool) -> bool:
    return include_hidden or not any(part.startswith(".") or part in _HIDDEN_DIRECTORIES for part in path.parts)


def directory_list(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    """枚举目录，替代把目录误传给 file_read 的做法。"""
    root = Path(workspace).resolve()
    relative = str(args.get("path", "."))
    directory = safe_path(workspace, relative)
    if not directory.exists():
        return {"path": _relative(root, directory), "exists": False, "entries": []}
    if not directory.is_dir():
        raise NotADirectoryError(f"不是目录：{relative}")

    recursive = bool(args.get("recursive", False))
    include_hidden = bool(args.get("include_hidden", False))
    limit = max(1, min(int(args.get("max_entries", 100)), 500))
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    entries: list[dict[str, Any]] = []
    truncated = False
    for path in sorted(candidates, key=lambda item: str(item).lower()):
        if root not in path.resolve().parents or not _visible(path.relative_to(root), include_hidden):
            continue
        if len(entries) >= limit:
            truncated = True
            break
        kind = "directory" if path.is_dir() else "file" if path.is_file() else "other"
        entries.append(
            {
                "path": _relative(root, path),
                "type": kind,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return {
        "path": _relative(root, directory),
        "exists": True,
        "recursive": recursive,
        "entries": entries,
        "truncated": truncated,
    }


def file_stat(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    """返回文件或目录状态，供读取前检查。"""
    root = Path(workspace).resolve()
    relative = str(args["path"])
    path = safe_path(workspace, relative)
    if not path.exists():
        return {"path": _relative(root, path), "exists": False, "type": None, "bytes": None}
    kind = "directory" if path.is_dir() else "file" if path.is_file() else "other"
    return {
        "path": _relative(root, path),
        "exists": True,
        "type": kind,
        "bytes": path.stat().st_size if path.is_file() else None,
        "mime_type": mimetypes.guess_type(path.name)[0] if path.is_file() else None,
    }


def file_search(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    """按名称或文本内容检索文件；`*` 可枚举可见文件。"""
    root = Path(workspace).resolve()
    query = str(args["query"]).lower()
    include_hidden = bool(args.get("include_hidden", False))
    matches: list[str] = []
    for path in root.rglob("*"):
        if root not in path.resolve().parents:
            continue
        if not path.is_file() or not _visible(path.relative_to(root), include_hidden):
            continue
        relative = _relative(root, path)
        if query in {"*", "**"} or query in path.name.lower():
            matches.append(relative)
            continue
        try:
            if query in path.read_text(encoding="utf-8", errors="ignore").lower():
                matches.append(relative)
        except OSError:
            continue
    return {"matches": matches[:100], "truncated": len(matches) > 100}


def file_read(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    """按字符分页读取文本；偏移与长度均以字符为单位。"""
    path = safe_path(workspace, str(args["path"]))
    if path.is_dir():
        raise IsADirectoryError("目标是目录；请使用 directory_list 枚举目录内容")
    raw = path.read_bytes()
    content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    offset = int(args.get("offset_chars", 0))
    limit = int(args.get("limit_chars", 8_000))
    if offset < 0:
        raise ValueError("offset 不能小于 0")
    if limit < 1 or limit > 10_000:
        raise ValueError("limit 必须在 1 到 10000 之间")
    if offset > len(content):
        raise ValueError("offset 超出文件末尾")

    next_offset = min(len(content), offset + limit)
    eof = next_offset == len(content)
    return {
        "path": str(path.relative_to(Path(workspace).resolve())),
        "content": content[offset:next_offset],
        "offset": offset,
        "next_offset": None if eof else next_offset,
        "eof": eof,
        "total_chars": len(content),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def file_read_lines(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    """按 1 起始行号读取闭区间，供源码定位和小范围编辑使用。"""
    path = safe_path(workspace, str(args["path"]))
    if path.is_dir():
        raise IsADirectoryError("目标是目录；请使用 directory_list 枚举目录内容")
    start_line = int(args["start_line"])
    end_line = int(args["end_line"])
    if start_line < 1 or end_line < start_line:
        raise ValueError("行范围必须满足 1 <= start_line <= end_line")
    raw = path.read_bytes()
    lines = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    if start_line > len(lines):
        raise ValueError("start_line 超出文件末尾")
    actual_end = min(end_line, len(lines))
    return {
        "path": str(path.relative_to(Path(workspace).resolve())),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": "".join(lines[start_line - 1:actual_end]),
        "start_line": start_line,
        "end_line": actual_end,
        "eof": actual_end == len(lines),
        "total_lines": len(lines),
    }


def file_write(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    path = safe_path(workspace, str(args["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args["content"]), encoding="utf-8")
    content = path.read_bytes()
    relative_path = str(path.relative_to(Path(workspace).resolve()))
    return {
        "path": relative_path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "mime_type": mimetypes.guess_type(path.name)[0] or "text/plain",
    }
