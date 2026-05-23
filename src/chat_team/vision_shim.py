"""Eager vision shim: turn inbound image blocks into text BEFORE they reach
``agent.history``.

Sits between :class:`Dispatcher` and :meth:`Agent.handle`. When the role's
``vision_strategy`` is ``"tool"`` (default), each image content block is
replaced with ``[图:<rel>]\\n<description>`` text via
:func:`describe_images`. The whole user message is then handed to the agent
as a single string, so:

* ``agent.history`` never carries list content for these turns
* ``compactor.count_tokens`` sees real text and counts accurately
* The OpenAI provider never re-base64s the same image across turns

When the role opts into ``vision_strategy: "direct"``, the shim is a
no-op and the original list of content blocks is forwarded — the legacy
direct-vision path stays intact for high-fidelity multi-turn visual chat.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .adapters.base import ContentBlock, blocks_to_text
from .agent.tools.describe_image import describe_images

if TYPE_CHECKING:
    from .config import Settings
    from .llm.base import LLMProvider
    from .roles.config import Role

log = logging.getLogger(__name__)


DEFAULT_OCR_PROMPT = (
    "请处理这张图片:\n"
    "1. 如果图中含有可见文字(印刷体、手写、屏幕截图、表格等),"
    "请完整提取所有文字,尽量保留原始排版顺序,不要总结、不要翻译。\n"
    "2. 如果图中没有文字内容,请用一两句话简要描述图片画面。\n"
    "直接输出结果,不要添加任何前后缀说明。"
)

VALID_STRATEGIES = ("tool", "direct")


def resolve_vision_strategy(role: "Role", settings: "Settings") -> str:
    """Return ``"tool"`` or ``"direct"``. Falls back to settings default
    when the role doesn't specify, and warns + defaults safely on bad
    values rather than raising mid-turn."""
    val = role.llm.vision_strategy
    if val and val in VALID_STRATEGIES:
        return val
    if val:
        log.warning(
            "invalid role.llm.vision_strategy=%r on role %s; using settings default",
            val, role.name,
        )
    default = settings.llm.default_vision_strategy
    if default in VALID_STRATEGIES:
        return default
    log.warning(
        "invalid settings.llm.default_vision_strategy=%r; falling back to 'tool'",
        default,
    )
    return "tool"


def _absolute(rel: str, cwd: Path) -> str:
    if os.path.isabs(rel):
        return rel
    return str((cwd / rel).resolve())


async def apply_vision_strategy(
    content: str | list[ContentBlock],
    *,
    role: "Role",
    settings: "Settings",
    llm: "LLMProvider",
    cwd: Path,
    session_id: str | None = None,
) -> str | list[ContentBlock]:
    """Apply the role's vision strategy to an inbound user message content.

    Tool mode returns a flat string with ``[图:<rel>]\\n<description>``
    sections in place of every image block. Direct mode returns the input
    unchanged (so the OpenAI provider can expand images at request time).
    """
    strategy = resolve_vision_strategy(role, settings)
    if strategy == "direct":
        return content
    # tool mode
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content                                       # defensive
    image_indices = [i for i, b in enumerate(content) if b.get("type") == "image"]
    if not image_indices:
        # Pure-text list — flatten to string for clean string-only history.
        return blocks_to_text(content)

    rel_paths = [content[i].get("path") or "" for i in image_indices]
    abs_paths = [_absolute(rel, cwd) for rel in rel_paths]

    prompt = settings.llm.default_eager_prompt or DEFAULT_OCR_PROMPT
    detail = settings.llm.default_eager_detail or "high"
    model = settings.llm.default_vision_model or settings.llm.default_model

    descriptions = await describe_images(
        abs_paths,
        prompt=prompt,
        detail=detail,
        llm=llm,
        model=model,
        image_base_dir=str(cwd),
        session_id=session_id,
        role_name=role.name,
        debug_log_dir=cwd / ".chat_team" / "llm",
    )
    desc_by_index = dict(zip(image_indices, descriptions))

    parts: list[str] = []
    for i, block in enumerate(content):
        btype = block.get("type")
        if btype == "text":
            text = (block.get("text") or "").strip()
            if text:
                parts.append(text)
        elif btype == "image":
            rel = rel_paths[image_indices.index(i)]
            desc = desc_by_index[i]
            parts.append(f"[图:{rel}]\n{desc}")
        else:
            parts.append(f"[未支持:{btype}]")
    return "\n\n".join(p for p in parts if p)
