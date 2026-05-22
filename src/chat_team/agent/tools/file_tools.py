"""Filesystem tools sandboxed to the session's cwd."""
from __future__ import annotations

import os
import re
import tempfile
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
    relative = candidate.relative_to(cwd_resolved)
    if relative.parts and relative.parts[0].startswith(".chat_team"):
        raise ToolError("path is reserved for internal session metadata")
    return candidate


def _atomic_write(target: Path, encoded: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=str(target.parent),
        prefix=f".{target.name}.", suffix=".tmp", delete=False,
    ) as tmp:
        tmp.write(encoded)
        tmp_path = tmp.name
    os.replace(tmp_path, target)


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "读取当前工作目录下的文件内容(UTF-8 文本)。仅接受相对路径,不允许 ../ 或绝对路径。"
        "可选 offset/limit 按行切片(0 索引)。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对当前工作目录的文件路径"},
            "offset": {"type": "integer", "description": "可选:起始行号(0 索引,默认 0)"},
            "limit": {"type": "integer", "description": "可选:返回行数上限,默认全部"},
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
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError("file is not valid UTF-8 text")
        offset_arg = kwargs.get("offset")
        limit_arg = kwargs.get("limit")
        if offset_arg is None and limit_arg is None:
            return text
        offset = 0 if offset_arg is None else int(offset_arg)
        if offset < 0:
            raise ToolError("offset must be >= 0")
        if limit_arg is not None:
            limit = int(limit_arg)
            if limit < 0:
                raise ToolError("limit must be >= 0")
        else:
            limit = None
        lines = text.splitlines(keepends=True)
        total = len(lines)
        end = total if limit is None else min(offset + limit, total)
        chunk = "".join(lines[offset:end])
        if offset == 0 and end == total:
            return chunk
        if chunk and not chunk.endswith("\n"):
            chunk += "\n"
        return f"{chunk}[shown lines {offset}..{end} of {total}]"


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
        _atomic_write(target, encoded)
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


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "对工作目录下的文件做精确字符串替换。匹配 0 次或多次(未传 replace_all)均报错。"
        "改一两处长文件时优先使用,避免整文件重写。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的文件路径"},
            "old": {"type": "string", "description": "被替换的原始字符串(必须精确匹配)"},
            "new": {"type": "string", "description": "替换后的字符串"},
            "replace_all": {"type": "boolean", "description": "可选:为 true 时替换所有匹配,默认 false"},
        },
        "required": ["path", "old", "new"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        rel = kwargs.get("path", "")
        old = kwargs.get("old", "")
        new = kwargs.get("new", "")
        replace_all = bool(kwargs.get("replace_all", False))
        if not isinstance(old, str) or old == "":
            raise ToolError("old must be a non-empty string")
        if not isinstance(new, str):
            raise ToolError("new must be a string")
        target = _resolve_under(ctx.cwd, rel)
        if not target.exists() or not target.is_file():
            raise ToolError(f"file not found: {rel}")
        max_read = ctx.settings.tools.file_read_max_bytes
        if target.stat().st_size > max_read:
            raise ToolError(f"file too large to edit (> {max_read} bytes)")
        try:
            src = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError("file is not valid UTF-8 text")
        count = src.count(old)
        if count == 0:
            raise ToolError(f"old not found in {rel}")
        if count > 1 and not replace_all:
            raise ToolError(
                f"{count} matches in {rel}; pass replace_all=true or extend old with surrounding context"
            )
        result = src.replace(old, new) if replace_all else src.replace(old, new, 1)
        encoded = result.encode("utf-8")
        max_write = ctx.settings.tools.file_write_max_bytes
        if len(encoded) > max_write:
            raise ToolError(f"resulting content too large ({len(encoded)} bytes > {max_write})")
        _atomic_write(target, encoded)
        replacements = count if replace_all else 1
        return f"edited {rel}: {replacements} replacement(s)"


def _is_internal(rel: Path) -> bool:
    return bool(rel.parts) and rel.parts[0].startswith(".chat_team")


class GlobTool(Tool):
    name = "glob"
    description = (
        "在工作目录下按文件名模式匹配(如 '**/*.py')。返回相对路径,排除 .chat_team 内部文件。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "glob 模式,例如 '**/*.md'"},
            "base": {"type": "string", "description": "可选:起始目录,默认 '.'"},
            "max_results": {"type": "integer", "description": "可选:返回上限,默认 200"},
        },
        "required": ["pattern"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        pattern = kwargs.get("pattern", "")
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty string")
        base = kwargs.get("base") or "."
        max_results = int(kwargs.get("max_results") or 200)
        if max_results <= 0:
            raise ToolError("max_results must be > 0")
        start = ctx.cwd if base == "." else _resolve_under(ctx.cwd, base)
        if not start.is_dir():
            raise ToolError(f"not a directory: {base}")
        cwd_resolved = ctx.cwd.resolve()
        results: list[str] = []
        total = 0
        for p in sorted(start.glob(pattern)):
            try:
                rel = p.resolve().relative_to(cwd_resolved)
            except ValueError:
                continue
            if _is_internal(rel):
                continue
            total += 1
            if len(results) < max_results:
                kind = "dir" if p.is_dir() else "file"
                results.append(f"{kind}\t{rel}")
        if not results:
            return "(no matches)"
        out = "\n".join(results)
        if total > max_results:
            out += f"\n[truncated, showing {max_results} of {total} matches]"
        return out


class GrepTool(Tool):
    name = "grep"
    description = (
        "在工作目录下用 Python 正则搜索文件内容。返回 path:lineno:line。"
        "二进制/非 UTF-8 文件会被跳过,.chat_team 内部文件不参与搜索。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python 正则"},
            "path": {"type": "string", "description": "可选:起点目录,默认 '.'"},
            "glob": {"type": "string", "description": "可选:文件名 glob 过滤,默认 '*'"},
            "ignore_case": {"type": "boolean", "description": "可选:忽略大小写,默认 false"},
            "max_results": {"type": "integer", "description": "可选:返回行数上限,默认 200"},
        },
        "required": ["pattern"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        pattern = kwargs.get("pattern", "")
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty string")
        path = kwargs.get("path") or "."
        file_glob = kwargs.get("glob") or "*"
        ignore_case = bool(kwargs.get("ignore_case", False))
        max_results = int(kwargs.get("max_results") or 200)
        if max_results <= 0:
            raise ToolError("max_results must be > 0")
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}")
        start = ctx.cwd if path == "." else _resolve_under(ctx.cwd, path)
        if not start.is_dir():
            raise ToolError(f"not a directory: {path}")
        cwd_resolved = ctx.cwd.resolve()
        max_bytes = ctx.settings.tools.file_read_max_bytes
        results: list[str] = []
        truncated = False
        for p in sorted(start.rglob(file_glob)):
            if not p.is_file():
                continue
            try:
                rel = p.resolve().relative_to(cwd_resolved)
            except ValueError:
                continue
            if _is_internal(rel):
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{lineno}:{line}")
                    if len(results) >= max_results:
                        truncated = True
                        break
            if truncated:
                break
        if not results:
            return "(no matches)"
        out = "\n".join(results)
        if truncated:
            out += f"\n[truncated at {max_results}]"
        return out
