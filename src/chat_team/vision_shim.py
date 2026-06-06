"""Vision strategy shim for inbound user messages.

Sits between :class:`Dispatcher` and :meth:`Agent.handle`. When the role's
``vision_strategy`` is ``"tool"`` (default), each image content block is
replaced with a plain placeholder text ``[图:<rel>]`` and no OCR call is made.
The whole user message is then handed to the agent as a single string, so:

* ``agent.history`` never carries list content for these turns
* the first response path avoids pre-vision latency

When the role opts into ``vision_strategy: "direct"``, the shim is a
no-op and the original list of content blocks is forwarded — the legacy
direct-vision path stays intact for high-fidelity multi-turn visual chat.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .adapters.base import ContentBlock, blocks_to_text

if TYPE_CHECKING:
    from .config import Settings
    from .llm.base import LLMProvider
    from .roles.config import Role

log = logging.getLogger(__name__)

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
    default = settings.llm.vision.strategy
    if default in VALID_STRATEGIES:
        return default
    log.warning(
        "invalid settings.llm.vision.strategy=%r; falling back to 'tool'",
        default,
    )
    return "tool"


def _render_tool_mode_text(content: list[ContentBlock]) -> str:
    parts: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = (block.get("text") or "").strip()
            if text:
                parts.append(text)
        elif btype == "image":
            rel = (block.get("path") or "").strip()
            parts.append(f"[图:{rel}]" if rel else "[图]")
        else:
            parts.append(f"[未支持:{btype}]")
    return "\n\n".join(p for p in parts if p)


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

    Tool mode returns a flat string with ``[图:<rel>]`` placeholders
    in place of image blocks. Direct mode returns the input
    unchanged (so the OpenAI provider can expand images at request time).
    """
    _ = (llm, cwd, session_id)  # reserved for signature stability
    strategy = resolve_vision_strategy(role, settings)
    if strategy == "direct":
        return content
    # tool mode
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content                                       # defensive
    has_image = any(b.get("type") == "image" for b in content)
    if not has_image:
        # Pure-text list — flatten to string for clean string-only history.
        return blocks_to_text(content)
    return _render_tool_mode_text(content)
