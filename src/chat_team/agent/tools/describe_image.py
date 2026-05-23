"""Vision-as-tool: turn an image into text via an LLM call.

Two callers:

* The eager vision shim (``runtime.vision_shim``) calls :func:`describe_images`
  on inbound user-message images so ``agent.history`` only ever sees a text
  rendering. Same image + prompt + detail + model is OCR'd at most once per
  process thanks to :class:`ImageDescriptionCache`.

* The agent itself can call :class:`DescribeImageTool` to re-query an image
  with a different prompt (e.g. "把图 2 的表格转成 Markdown"), a higher
  detail, or simply because the eager pass missed something.

The function is deliberately **independent of ToolContext** so the shim can
use it without faking a tool invocation. The tool wrapper is a thin shell
that handles sandbox validation and pulls the LLM provider from ``ctx.llm``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from ...llm.base import ChatMessage, CompletionRequest, LLMProvider
from ...llm.image_cache import MAX_INLINE_BYTES
from ...llm.image_description_cache import ImageDescriptionCache, default_cache
from .base import Tool, ToolContext, ToolError
from .file_tools import _resolve_under

log = logging.getLogger(__name__)

DEFAULT_TOOL_PROMPT = (
    "请详细描述这张图片的内容,包括其中的可见文字(若有)以及画面要素。"
    "对文字部分尽量原样转写,不要总结、不要翻译。"
)


def _read_failure_reason(abs_path: str) -> str | None:
    """Returns a short reason string if the file can't / shouldn't be sent
    to the vision model; ``None`` if the file is fine."""
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        return "文件不存在或无法读取"
    if size <= 0:
        return "文件为空"
    if size > MAX_INLINE_BYTES:
        return f"图片过大 ({size} 字节,上限 {MAX_INLINE_BYTES})"
    return None


async def _describe_one(
    abs_path: str,
    *,
    prompt: str,
    detail: str,
    llm: LLMProvider,
    model: str,
    image_base_dir: str,
    cache: ImageDescriptionCache,
    session_id: str | None = None,
    role_name: str | None = None,
    debug_log_dir: Path | None = None,
) -> str:
    fail = _read_failure_reason(abs_path)
    if fail is not None:
        return f"[读取失败:{fail}]"

    cached = cache.get(abs_path, detail=detail, model=model, prompt=prompt)
    if cached is not None:
        return cached

    request = CompletionRequest(
        messages=[
            ChatMessage(role="user", content=[
                {"type": "text", "text": prompt},
                {"type": "image", "path": abs_path},
            ]),
        ],
        model=model,
        temperature=0.0,
        image_detail=detail,
        image_base_dir=image_base_dir,
        session_id=session_id,
        role_name=role_name,
        call_kind="vision",
        debug_log_dir=debug_log_dir,
    )
    try:
        resp = await llm.complete(request)
    except Exception as err:                                  # noqa: BLE001
        log.exception("describe_image LLM call failed for %s", abs_path)
        return f"[读取失败:视觉模型调用异常 {type(err).__name__}]"
    text = (resp.message.content or "").strip()
    if not text:
        return "[读取失败:视觉模型返回空内容]"
    cache.put(abs_path, text, detail=detail, model=model, prompt=prompt)
    return text


async def describe_images(
    paths: list[str],
    *,
    prompt: str,
    detail: str,
    llm: LLMProvider,
    model: str,
    image_base_dir: str,
    cache: ImageDescriptionCache | None = None,
    session_id: str | None = None,
    role_name: str | None = None,
    debug_log_dir: Path | None = None,
) -> list[str]:
    """Describe a batch of images, returning one description per input path
    in the same order. Each image is its own concurrent LLM call so a single
    bad image doesn't poison the rest, and the cache makes repeats free.

    ``paths`` are expected to be **absolute** and already sandbox-validated.
    """
    if not paths:
        return []
    cache = cache or default_cache()
    coros = [
        _describe_one(
            p,
            prompt=prompt,
            detail=detail,
            llm=llm,
            model=model,
            image_base_dir=image_base_dir,
            cache=cache,
            session_id=session_id,
            role_name=role_name,
            debug_log_dir=debug_log_dir,
        )
        for p in paths
    ]
    return await asyncio.gather(*coros)


class DescribeImageTool(Tool):
    name = "describe_image"
    description = (
        "对工作区里的图片调用视觉模型生成文字描述/OCR 结果。"
        "适合在 agent 已经看到 [图:path] 占位但默认摘要不够,"
        "想要换一个 prompt 再问、或者用更高的 detail 重新扫描时使用。"
        "只接受相对当前工作目录的路径(例如 ./inbox/xxx.jpg);一次最多 4 张。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1 到 4 个相对工作目录的图片路径",
                "minItems": 1,
                "maxItems": 4,
            },
            "prompt": {
                "type": "string",
                "description": (
                    "可选;描述图片时给视觉模型的指令。默认让模型详细描述图片"
                    "并原样转写文字。"
                ),
            },
            "detail": {
                "type": "string",
                "enum": ["low", "high", "auto"],
                "description": "可选;视觉细节级别,默认 high。",
            },
        },
        "required": ["paths"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        if ctx.llm is None:
            raise ToolError("describe_image requires an LLM provider in ToolContext")
        raw_paths = kwargs.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ToolError("paths must be a non-empty list of strings")
        if len(raw_paths) > 4:
            raise ToolError("paths supports at most 4 images per call")
        rel_paths: list[str] = []
        abs_paths: list[str] = []
        for rel in raw_paths:
            if not isinstance(rel, str):
                raise ToolError("paths entries must be strings")
            target = _resolve_under(ctx.cwd, rel)        # raises ToolError on escape
            rel_paths.append(rel)
            abs_paths.append(str(target))

        prompt = kwargs.get("prompt") or DEFAULT_TOOL_PROMPT
        if not isinstance(prompt, str):
            raise ToolError("prompt must be a string")
        detail = kwargs.get("detail") or "high"
        if detail not in ("low", "high", "auto"):
            raise ToolError("detail must be one of low|high|auto")

        model = ctx.settings.llm.default_vision_model or ctx.settings.llm.default_model

        descriptions = await describe_images(
            abs_paths,
            prompt=prompt,
            detail=detail,
            llm=ctx.llm,
            model=model,
            image_base_dir=str(ctx.cwd),
            session_id=ctx.session.session_id,
            role_name=ctx.session.current_role,
            debug_log_dir=ctx.cwd / ".chat_team" / "llm",
        )
        sections = [
            f"[图:{rel}]\n{desc}"
            for rel, desc in zip(rel_paths, descriptions)
        ]
        return "\n\n".join(sections)
