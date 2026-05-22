"""Media-reply tools: let an agent push images/files back to the WeCom user.

Both tools take a relative path under the session's cwd, validate the file
(existence, size, extension as relevant), and call the StreamHandle's
``send_image`` / ``send_file`` — which on WeCom uploads the bytes via the
3-step ``aibot_upload_media_*`` flow and emits an ``aibot_respond_msg``
media frame on the same ``req_id`` as the in-flight text reply.
"""
from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolError
from .file_tools import _resolve_under

IMAGE_EXTS = {"png", "jpg", "jpeg", "gif"}
IMAGE_MAX_BYTES = 10 * 1024 * 1024
FILE_MAX_BYTES = 20 * 1024 * 1024


class SendImageTool(Tool):
    name = "send_image"
    description = (
        "把工作目录下的一张图片直接发给当前用户(WeCom)。"
        "支持 png/jpg/jpeg/gif,大小 ≤10MB。仅接受相对路径,不允许 ../ 或绝对路径。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对当前工作目录的图片路径"},
            "filename": {"type": "string", "description": "可选,展示给用户的文件名(默认用文件本名)"},
        },
        "required": ["path"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        rel = kwargs.get("path", "")
        filename = kwargs.get("filename")
        if filename is not None and not isinstance(filename, str):
            raise ToolError("filename must be a string")
        target = _resolve_under(ctx.cwd, rel)
        if not target.exists():
            raise ToolError(f"file not found: {rel}")
        if not target.is_file():
            raise ToolError(f"not a regular file: {rel}")
        ext = target.suffix.lstrip(".").lower()
        if ext not in IMAGE_EXTS:
            raise ToolError(
                f"unsupported image extension '.{ext}'; allowed: {sorted(IMAGE_EXTS)}"
            )
        size = target.stat().st_size
        if size > IMAGE_MAX_BYTES:
            raise ToolError(f"image too large ({size} bytes > {IMAGE_MAX_BYTES})")
        if size < 5:
            raise ToolError(f"image too small ({size} bytes < 5)")
        if ctx.stream is None:
            raise ToolError("当前会话不支持媒体回传")
        try:
            await ctx.stream.send_image(target, filename=filename)
        except NotImplementedError as err:
            raise ToolError(str(err))
        except RuntimeError as err:
            raise ToolError(f"send_image failed: {err}")
        return f"已发送图片: {filename or target.name} ({size} bytes)"


class SendFileTool(Tool):
    name = "send_file"
    description = (
        "把工作目录下的任意文件直接发给当前用户(WeCom)。"
        "大小 ≤20MB。仅接受相对路径,不允许 ../ 或绝对路径。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对当前工作目录的文件路径"},
            "filename": {"type": "string", "description": "可选,展示给用户的文件名(默认用文件本名)"},
        },
        "required": ["path"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        rel = kwargs.get("path", "")
        filename = kwargs.get("filename")
        if filename is not None and not isinstance(filename, str):
            raise ToolError("filename must be a string")
        target = _resolve_under(ctx.cwd, rel)
        if not target.exists():
            raise ToolError(f"file not found: {rel}")
        if not target.is_file():
            raise ToolError(f"not a regular file: {rel}")
        size = target.stat().st_size
        if size > FILE_MAX_BYTES:
            raise ToolError(f"file too large ({size} bytes > {FILE_MAX_BYTES})")
        if size < 5:
            raise ToolError(f"file too small ({size} bytes < 5)")
        if ctx.stream is None:
            raise ToolError("当前会话不支持媒体回传")
        try:
            await ctx.stream.send_file(target, filename=filename)
        except NotImplementedError as err:
            raise ToolError(str(err))
        except RuntimeError as err:
            raise ToolError(f"send_file failed: {err}")
        return f"已发送文件: {filename or target.name} ({size} bytes)"
