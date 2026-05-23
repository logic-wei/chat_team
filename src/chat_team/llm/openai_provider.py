"""OpenAI Chat Completion provider."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from ..adapters.base import blocks_to_text
from . import debug_logger
from .base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from .image_cache import ImageDataURICache, default_cache


def _content_as_string(content: Any) -> str:
    """Coerce ChatMessage.content to a string. Image blocks render as
    `[图:<basename>]`; used for tool / assistant / system messages."""
    if isinstance(content, list):
        return blocks_to_text(content)
    return content or ""


def _expand_user_content(
    content: Any,
    *,
    image_detail: str | None,
    image_base_dir: Path | str | None,
    cache: ImageDataURICache,
) -> str | list[dict[str, Any]]:
    """User-message content expansion. Returns a flat string when content
    is a string OR when the list contains only text blocks (so we never
    confuse the API with single-element content arrays). Otherwise returns
    the OpenAI multi-part content list with ``image_url`` data URIs."""
    if not isinstance(content, list):
        return content or ""

    has_image = any(b.get("type") == "image" for b in content)
    if not has_image:
        return blocks_to_text(content)

    detail = image_detail or "high"
    parts: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text") or ""
            if not text:
                continue
            parts.append({"type": "text", "text": text})
            continue
        if btype == "image":
            rel = block.get("path") or ""
            if not rel:
                continue
            abs_path = (
                rel
                if os.path.isabs(rel)
                else os.path.join(str(image_base_dir or "."), rel)
            )
            uri = cache.get(abs_path)
            if uri is None:
                # Defensive fallback: degrade to a text block so the rest of the
                # turn proceeds. Distinguishes missing vs. oversize via a probe.
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    size = -1
                tag = "已丢失" if size < 0 else "过大,已省略"
                parts.append({
                    "type": "text",
                    "text": f"[图:{os.path.basename(rel)}({tag})]",
                })
                continue
            parts.append({
                "type": "image_url",
                "image_url": {"url": uri, "detail": detail},
            })
            continue
        # Unknown block type: render as text placeholder.
        parts.append({"type": "text", "text": f"[未支持:{btype}]"})

    if not parts:
        return ""
    # If after fallback nothing image-shaped remains, collapse to flat string.
    if not any(p.get("type") == "image_url" for p in parts):
        return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return parts


def _to_openai_messages(
    messages: list[ChatMessage],
    *,
    image_detail: str | None = None,
    image_base_dir: Path | str | None = None,
    cache: ImageDataURICache | None = None,
) -> list[dict[str, Any]]:
    cache = cache or default_cache()
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": _content_as_string(m.content),
            })
            continue
        if m.role == "user":
            content = _expand_user_content(
                m.content,
                image_detail=image_detail,
                image_base_dir=image_base_dir,
                cache=cache,
            )
        else:
            content = _content_as_string(m.content)
        msg: dict[str, Any] = {"role": m.role, "content": content}
        if m.role == "assistant" and m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
        if m.name:
            msg["name"] = m.name
        out.append(msg)
    return out


def _to_openai_tools(specs) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters or {"type": "object", "properties": {}},
            },
        }
        for s in specs
    ]


class OpenAIChatCompletionProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        *,
        debug_log_enabled: bool = False,
    ):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._debug_log_enabled = debug_log_enabled

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        messages_payload = _to_openai_messages(
            request.messages,
            image_detail=request.image_detail,
            image_base_dir=request.image_base_dir,
        )
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages_payload,
            "temperature": request.temperature,
        }
        if request.tools:
            kwargs["tools"] = _to_openai_tools(request.tools)
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        t0 = time.monotonic()
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:                                  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000.0
            self._maybe_write_log(
                request,
                messages_payload=messages_payload,
                response_message=None,
                finish_reason=None,
                usage=None,
                latency_ms=latency_ms,
                error=repr(exc),
            )
            raise

        latency_ms = (time.monotonic() - t0) * 1000.0
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        chat_msg = ChatMessage(
            role="assistant",
            content=msg.content or "",
            tool_calls=tool_calls,
        )
        usage = None
        if getattr(completion, "usage", None) is not None:
            try:
                usage = completion.usage.model_dump()
            except Exception:                                     # noqa: BLE001
                usage = None
        self._maybe_write_log(
            request,
            messages_payload=messages_payload,
            response_message=debug_logger._serialise_response_message(msg),
            finish_reason=choice.finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            error=None,
        )
        return CompletionResponse(
            message=chat_msg,
            finish_reason=choice.finish_reason or "stop",
            raw=completion,
        )

    def _maybe_write_log(
        self,
        request: CompletionRequest,
        *,
        messages_payload: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        finish_reason: str | None,
        usage: dict[str, Any] | None,
        latency_ms: float,
        error: str | None,
    ) -> None:
        if not self._debug_log_enabled:
            return
        if request.debug_log_dir is None:
            return
        debug_logger.write_call_log(
            dir_path=request.debug_log_dir,
            session_id=request.session_id,
            role_name=request.role_name,
            call_kind=request.call_kind,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tool_names=[s.name for s in request.tools],
            messages_payload=messages_payload,
            response_message=response_message,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            error=error,
        )
