"""OpenAI Chat Completion provider."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import random
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from ..adapters.base import blocks_to_text
from . import debug_logger
from . import http_debug_logger
from .base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from .image_cache import ImageDataURICache, default_cache

log = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


@dataclass
class _HttpLogContext:
    dir_path: Path
    session_id: str | None
    role_name: str | None
    call_kind: str | None
    call_id: str
    request_index: int = 0


_http_log_ctx: contextvars.ContextVar[_HttpLogContext | None] = contextvars.ContextVar(
    "chat_team_http_log_ctx",
    default=None,
)


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
        http_debug_log_enabled: bool = False,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
    ):
        self._http_debug_log_enabled = http_debug_log_enabled
        event_hooks: dict[str, list[Any]] | None = None
        if http_debug_log_enabled:
            event_hooks = {
                "request": [self._on_http_request],
                "response": [self._on_http_response],
            }
        http_client = httpx.AsyncClient(
            timeout=request_timeout_seconds,
            event_hooks=event_hooks,
        )
        # Pass timeout into the SDK client so a hung request can't hold the
        # session lock indefinitely (the dispatcher holds it for the whole turn).
        # max_retries=0 disables the SDK's own retry layer so our outer loop
        # is the single source of truth for retry policy.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            http_client=http_client,
            timeout=request_timeout_seconds,
            max_retries=0,
        )
        self._debug_log_enabled = debug_log_enabled
        self._max_retries = max(1, int(max_retries))
        self._retry_initial_delay = max(0.0, float(retry_initial_delay))

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        token = None
        if self._http_debug_log_enabled and request.debug_log_dir is not None:
            token = _http_log_ctx.set(_HttpLogContext(
                dir_path=request.debug_log_dir.parent / "llm_http",
                session_id=request.session_id,
                role_name=request.role_name,
                call_kind=request.call_kind,
                call_id=secrets.token_hex(8),
            ))
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
        attempts = 0
        completion = None
        last_exc: Exception | None = None
        try:
            for attempt in range(self._max_retries):
                attempts = attempt + 1
                try:
                    completion = await self._client.chat.completions.create(**kwargs)
                    break
                except _RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt >= self._max_retries - 1:
                        break
                    delay = self._retry_initial_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    log.warning(
                        "LLM call failed (%s) on attempt %d/%d; retrying in %.2fs",
                        type(exc).__name__, attempts, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                except Exception as exc:                              # noqa: BLE001
                    last_exc = exc
                    break
        finally:
            if token is not None:
                _http_log_ctx.reset(token)

        if completion is None:
            latency_ms = (time.monotonic() - t0) * 1000.0
            self._maybe_write_log(
                request,
                messages_payload=messages_payload,
                response_message=None,
                finish_reason=None,
                usage=None,
                latency_ms=latency_ms,
                error=repr(last_exc) if last_exc else "unknown",
                attempts=attempts,
            )
            assert last_exc is not None
            raise last_exc

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
            attempts=attempts,
        )
        return CompletionResponse(
            message=chat_msg,
            finish_reason=choice.finish_reason or "stop",
            raw=completion,
        )

    async def _on_http_request(self, request: httpx.Request) -> None:
        ctx = _http_log_ctx.get()
        if ctx is None:
            return
        try:
            body_bytes = request.content if request.content else b""
        except Exception:                                          # noqa: BLE001
            body_bytes = None
        ctx.request_index += 1
        request.extensions["chat_team_http_request_index"] = ctx.request_index
        http_debug_logger.write_http_request_log(
            dir_path=ctx.dir_path,
            session_id=ctx.session_id,
            role_name=ctx.role_name,
            call_kind=ctx.call_kind,
            call_id=ctx.call_id,
            request_index=ctx.request_index,
            method=request.method,
            url=str(request.url),
            headers_raw=list(request.headers.raw),
            body_bytes=body_bytes,
        )

    async def _on_http_response(self, response: httpx.Response) -> None:
        ctx = _http_log_ctx.get()
        if ctx is None:
            return
        request = response.request
        try:
            body_bytes = await response.aread()
        except Exception:                                          # noqa: BLE001
            body_bytes = None
        request_index = request.extensions.get("chat_team_http_request_index")
        if not isinstance(request_index, int) or request_index <= 0:
            ctx.request_index += 1
            request_index = ctx.request_index
        http_debug_logger.write_http_response_log(
            dir_path=ctx.dir_path,
            session_id=ctx.session_id,
            role_name=ctx.role_name,
            call_kind=ctx.call_kind,
            call_id=ctx.call_id,
            request_index=request_index,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            reason_phrase=response.reason_phrase or "",
            headers_raw=list(response.headers.raw),
            body_bytes=body_bytes,
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
        attempts: int = 1,
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
            attempts=attempts,
        )
