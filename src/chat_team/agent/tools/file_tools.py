"""Filesystem tools sandboxed to the session's cwd."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolError


def _resolve_under(cwd: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise ToolError("path must be a non-empty string")
    if os.path.isabs(rel) or ".." in Path(rel).parts:
        raise ToolError("absolute paths and '..' are not allowed; use relative paths under the workspace")
    candidate = (cwd / rel).resolve()
    cwd_resolved = cwd.resolve()
    try:
        common = os.path.commonpath([str(candidate), str(cwd_resolved)])
    except ValueError:
        raise ToolError("path escapes the workspace")
    if common != str(cwd_resolved):
        raise ToolError("path escapes the workspace")
    return candidate


class ReadFileTool(Tool):
    name = "read_file"
    description = "读取当前工作目录下的文件内容(UTF-8 文本)。仅接受相对路径,不允许 ../ 或绝对路径。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对当前工作目录的文件路径"},
        },
        "required": ["path"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        rel = kwargs.get("path", "")
        target = _resolve_under(ctx.cwd, rel)
        if not target.exists():
            raise ToolError(f"file not found: {rel}")
        if not target.is_file():
            raise ToolError(f"not a regular file: {rel}")
        max_bytes = ctx.settings.tools.file_read_max_bytes
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolError(f"file too large ({size} bytes > {max_bytes}); refusing to read")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError("file is not valid UTF-8 text")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "在当前工作目录下写入文件(UTF-8 文本)。已存在则覆盖,父目录会自动创建。"
        "仅接受相对路径,不允许 ../ 或绝对路径。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对当前工作目录的目标路径"},
            "content": {"type": "string", "description": "要写入的完整内容"},
        },
        "required": ["path", "content"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        rel = kwargs.get("path", "")
        content = kwargs.get("content", "")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        max_bytes = ctx.settings.tools.file_write_max_bytes
        encoded = content.encode("utf-8")
        if len(encoded) > max_bytes:
            raise ToolError(f"content too large ({len(encoded)} bytes > {max_bytes})")
        target = _resolve_under(ctx.cwd, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp + rename
        import tempfile
        with tempfile.NamedTemporaryFile(
            "wb", dir=str(target.parent),
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as tmp:
            tmp.write(encoded)
            tmp_path = tmp.name
        os.replace(tmp_path, target)
        return f"wrote {len(encoded)} bytes to {rel}"


class ListDirTool(Tool):
    name = "list_dir"
    description = "列出工作目录下某子目录内的条目。不传 path 则列出根目录。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对当前工作目录的目录路径,可选,默认 '.'"},
        },
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        rel = kwargs.get("path") or "."
        target = ctx.cwd if rel == "." else _resolve_under(ctx.cwd, rel)
        if not target.exists():
            raise ToolError(f"path not found: {rel}")
        if not target.is_dir():
            raise ToolError(f"not a directory: {rel}")
        entries = []
        for child in sorted(target.iterdir()):
            if child.name.startswith(".chat_team"):
                continue                                 # hide internal session metadata
            kind = "dir" if child.is_dir() else "file"
            size = "" if child.is_dir() else f" ({child.stat().st_size} bytes)"
            entries.append(f"{kind}\t{child.name}{size}")
        return "\n".join(entries) if entries else "(empty)"
