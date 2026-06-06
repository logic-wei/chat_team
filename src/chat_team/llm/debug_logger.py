"""Per-call LLM debug log.

Writes one JSON file per `OpenAIChatCompletionProvider.complete` call to
``<workspace>/.chat_team/llm/<ts>-<seq>-<role>-<kind>.json`` so the user
can ``cat``/``jq`` exactly what was sent and returned when debugging.

Base64 image data URIs are replaced with a short ``[redacted: <mime>
<bytes> bytes]`` marker so the files stay grep-able. Per-session
monotonic ``seq`` keeps filenames sortable when the millisecond clock
collides on burst turns.

Failures here are caught and logged — debug logging never breaks the
underlying LLM call.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)
_BASE64_PADDING = re.compile(r"=+$")

_seq_lock = threading.Lock()
_seq_by_session: dict[str, int] = {}


def _next_seq(session_id: str | None) -> int:
    key = session_id or "_anon"
    with _seq_lock:
        n = _seq_by_session.get(key, 0) + 1
        _seq_by_session[key] = n
        return n


def _safe_name(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or fallback


def _redact_data_uri(uri: str, detail: Any) -> dict[str, Any]:
    m = _DATA_URI_RE.match(uri)
    if not m:
        return {"url": "[redacted: non-data-uri image_url omitted]", "detail": detail}
    mime, b64 = m.group(1), m.group(2)
    padded = _BASE64_PADDING.sub("", b64)
    approx_bytes = (len(padded) * 3) // 4
    return {
        "url": f"[redacted: {mime} {approx_bytes} bytes]",
        "detail": detail,
    }


def _redact_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    if part.get("type") != "image_url":
        return part
    img = part.get("image_url")
    if not isinstance(img, dict):
        return part
    url = img.get("url")
    if not isinstance(url, str) or not url.startswith("data:"):
        return part
    return {
        "type": "image_url",
        "image_url": _redact_data_uri(url, img.get("detail")),
    }


def redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep copy of ``messages`` with base64 image data URIs
    replaced by ``[redacted: <mime> <bytes> bytes]`` markers. Never
    mutates the input."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        msg_copy = copy.deepcopy(msg)
        content = msg_copy.get("content")
        if isinstance(content, list):
            msg_copy["content"] = [_redact_part(p) for p in content]
        out.append(msg_copy)
    return out


def _serialise_response_message(message: Any) -> dict[str, Any] | None:
    """Turn an OpenAI ``Choice.message`` into a JSON-safe dict. Returns
    ``None`` when ``message`` is None (call errored before a choice was
    available)."""
    if message is None:
        return None
    out: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
    }
    tool_calls = getattr(message, "tool_calls", None) or []
    serialised: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        serialised.append({
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": getattr(fn, "name", None) if fn else None,
                "arguments": getattr(fn, "arguments", None) if fn else None,
            },
        })
    if serialised:
        out["tool_calls"] = serialised
    return out


def write_call_log(
    *,
    dir_path: Path,
    session_id: str | None,
    role_name: str | None,
    call_kind: str | None,
    model: str,
    temperature: float,
    max_tokens: int | None,
    reasoning_effort: str | None,
    tool_names: list[str],
    messages_payload: list[dict[str, Any]],
    response_message: dict[str, Any] | None,
    finish_reason: str | None,
    usage: dict[str, Any] | None,
    latency_ms: float,
    error: str | None,
    attempts: int = 1,
) -> Path | None:
    """Write one JSON file describing a single LLM call. Returns the
    path written, or ``None`` if the write failed (the call itself
    keeps going — debug logging is best-effort)."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"
        seq = _next_seq(session_id)
        role_part = _safe_name(role_name, "unknown_role")
        kind_part = _safe_name(call_kind, "call")
        filename = f"{ts}-{seq:03d}-{role_part}-{kind_part}.json"
        path = dir_path / filename
        if path.exists():                                # paranoia: collision
            path = dir_path / f"{ts}-{seq:03d}-{role_part}-{kind_part}-{secrets.token_hex(2)}.json"
        record = {
            "session_id": session_id,
            "role_name": role_name,
            "call_kind": call_kind,
            "timestamp_utc": now.isoformat(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "tools": tool_names,
            "messages": redact_messages(messages_payload),
            "response": response_message,
            "finish_reason": finish_reason,
            "usage": usage,
            "latency_ms": round(latency_ms, 3),
            "error": error,
            "attempts": attempts,
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path
    except Exception as exc:                                # noqa: BLE001
        log.warning("debug log write failed: %s", exc)
        return None
