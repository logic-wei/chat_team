"""Per-request/response HTTP debug log for LLM calls.

Writes one JSON file per outbound HTTP request/response to the LLM provider.
This logger intentionally records full request headers and full request
body (including sensitive values) and must stay OFF in production.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

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


def _headers_as_list(raw_headers: list[tuple[bytes, bytes]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for name_b, value_b in raw_headers:
        out.append({
            "name": name_b.decode("latin-1", errors="replace"),
            "value": value_b.decode("latin-1", errors="replace"),
        })
    return out


def _serialise_body(body_bytes: bytes | None) -> dict[str, str | int | None]:
    if body_bytes is None:
        return {
            "encoding": None,
            "body": None,
            "byte_length": 0,
        }
    try:
        decoded = body_bytes.decode("utf-8")
        return {
            "encoding": "utf-8",
            "body": decoded,
            "byte_length": len(body_bytes),
        }
    except UnicodeDecodeError:
        return {
            "encoding": "base64",
            "body": base64.b64encode(body_bytes).decode("ascii"),
            "byte_length": len(body_bytes),
        }


def _build_base_record(
    *,
    now: datetime,
    session_id: str | None,
    role_name: str | None,
    call_kind: str | None,
    call_id: str,
    request_index: int,
) -> dict[str, object]:
    return {
        "timestamp_utc": now.isoformat(),
        "session_id": session_id,
        "role_name": role_name,
        "call_kind": call_kind,
        "call_id": call_id,
        "request_index": request_index,
    }


def write_http_request_log(
    *,
    dir_path: Path,
    session_id: str | None,
    role_name: str | None,
    call_kind: str | None,
    call_id: str,
    request_index: int,
    method: str,
    url: str,
    headers_raw: list[tuple[bytes, bytes]],
    body_bytes: bytes | None,
) -> Path | None:
    """Write one JSON file for one outbound HTTP request.

    Returns the written file path, or ``None`` if writing fails.
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"
        seq = _next_seq(session_id)
        role_part = _safe_name(role_name, "unknown_role")
        kind_part = _safe_name(call_kind, "call")
        filename = f"{ts}-{seq:03d}-{role_part}-{kind_part}-http.json"
        path = dir_path / filename
        if path.exists():
            path = dir_path / (
                f"{ts}-{seq:03d}-{role_part}-{kind_part}-http-"
                f"{secrets.token_hex(2)}.json"
            )
        record = {
            **_build_base_record(
                now=now,
                session_id=session_id,
                role_name=role_name,
                call_kind=call_kind,
                call_id=call_id,
                request_index=request_index,
            ),
            "direction": "request",
            "http": {
                "method": method,
                "url": url,
                "headers": _headers_as_list(headers_raw),
                **_serialise_body(body_bytes),
            },
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path
    except Exception as exc:                                  # noqa: BLE001
        log.warning("http debug log write failed: %s", exc)
        return None


def write_http_response_log(
    *,
    dir_path: Path,
    session_id: str | None,
    role_name: str | None,
    call_kind: str | None,
    call_id: str,
    request_index: int,
    method: str,
    url: str,
    status_code: int,
    reason_phrase: str,
    headers_raw: list[tuple[bytes, bytes]],
    body_bytes: bytes | None,
) -> Path | None:
    """Write one JSON file for one inbound HTTP response.

    Returns the written file path, or ``None`` if writing fails.
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"
        seq = _next_seq(session_id)
        role_part = _safe_name(role_name, "unknown_role")
        kind_part = _safe_name(call_kind, "call")
        filename = f"{ts}-{seq:03d}-{role_part}-{kind_part}-http-response.json"
        path = dir_path / filename
        if path.exists():
            path = dir_path / (
                f"{ts}-{seq:03d}-{role_part}-{kind_part}-http-response-"
                f"{secrets.token_hex(2)}.json"
            )
        record = {
            **_build_base_record(
                now=now,
                session_id=session_id,
                role_name=role_name,
                call_kind=call_kind,
                call_id=call_id,
                request_index=request_index,
            ),
            "direction": "response",
            "http": {
                "method": method,
                "url": url,
                "status_code": status_code,
                "reason_phrase": reason_phrase,
                "headers": _headers_as_list(headers_raw),
                **_serialise_body(body_bytes),
            },
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path
    except Exception as exc:                                  # noqa: BLE001
        log.warning("http debug log write failed: %s", exc)
        return None
