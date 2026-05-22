"""Read/write/delete tools for the session-level shared notebook."""
from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolError


class NotebookReadTool(Tool):
    name = "notebook_read"
    description = "读取团队会话记事本。不传 key 时返回全部条目;传 key 时只返回该条目内容。"
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "可选;具体记事本条目键名"},
        },
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        key = kwargs.get("key")
        notebook = ctx.session.notebook
        if key:
            entry = notebook.read(key)
            if entry is None:
                raise ToolError(f"notebook key not found: {key}")
            return entry
        items = notebook.dump()
        if not items:
            return "(notebook is empty)"
        return "\n\n".join(f"## {k}\n{v}" for k, v in items.items())


class NotebookWriteTool(Tool):
    name = "notebook_write"
    description = (
        "把跨员工应共享的'长期事实'写入团队会话记事本(用户名/目标/关键决策等)。"
        "已存在的键会被覆盖。总容量软上限 4KB,超出时会返回错误,这时请把多个相关条目合并精简。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "条目键名,简短英文/拼音"},
            "value": {"type": "string", "description": "条目内容,可多行"},
        },
        "required": ["key", "value"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        key = kwargs.get("key")
        value = kwargs.get("value")
        if not key or not isinstance(key, str):
            raise ToolError("key must be a non-empty string")
        if not isinstance(value, str):
            raise ToolError("value must be a string")
        ctx.session.notebook.write(key, value)
        return f"notebook[{key}] saved"


class NotebookDeleteTool(Tool):
    name = "notebook_delete"
    description = "从团队会话记事本删除某个条目。"
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "要删除的条目键名"},
        },
        "required": ["key"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        key = kwargs.get("key")
        if not key or not isinstance(key, str):
            raise ToolError("key must be a non-empty string")
        if not ctx.session.notebook.delete(key):
            raise ToolError(f"notebook key not found: {key}")
        return f"notebook[{key}] removed"
