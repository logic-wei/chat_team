"""Smoke test for the LLM debug log helper.

* redact_messages replaces base64 image data URIs with placeholders and
  preserves the original ``detail`` value
* write_call_log creates the directory lazily, emits a sortable filename
  with a monotonically incrementing per-session seq, and writes a JSON
  record with the expected top-level keys
* The same session_id keyed twice in a row gives seq=001, seq=002
* The error path writes the file with ``error`` populated and
  ``response`` null
* Bad target dir is swallowed (returns None) instead of raising
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force a clean home so we don't read user state.
home = Path("/tmp/chat_team_smoke_debug_log")
shutil.rmtree(home, ignore_errors=True)
home.mkdir(parents=True, exist_ok=True)
os.environ["CHAT_TEAM_HOME"] = str(home)

from chat_team.llm import debug_logger
from chat_team.llm.debug_logger import (
    _seq_by_session,
    redact_messages,
    write_call_log,
)


def _reset_seq() -> None:
    _seq_by_session.clear()


def test_redact_messages_strips_base64() -> None:
    fake_png_bytes = b"\x89PNG\r\n" + b"X" * 1024            # 1030 bytes
    b64 = base64.b64encode(fake_png_bytes).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"
    messages = [
        {"role": "system", "content": "hello"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "high"},
                },
            ],
        },
        {"role": "assistant", "content": "a picture"},
    ]
    original_snapshot = json.dumps(messages)
    redacted = redact_messages(messages)
    # Input must be untouched.
    assert json.dumps(messages) == original_snapshot, "redact mutated input"
    # Output must not contain raw base64 of our fake bytes.
    assert b64 not in json.dumps(redacted), "base64 leaked through"
    redacted_img = redacted[1]["content"][1]
    assert redacted_img["type"] == "image_url"
    placeholder = redacted_img["image_url"]["url"]
    assert "image/png" in placeholder, placeholder
    assert "bytes" in placeholder, placeholder
    assert "redacted" in placeholder, placeholder
    assert redacted_img["image_url"]["detail"] == "high"
    # Non-image messages are passed through verbatim.
    assert redacted[0] == {"role": "system", "content": "hello"}
    assert redacted[2] == {"role": "assistant", "content": "a picture"}
    print("ok  redact_messages strips base64 and preserves detail")


def test_redact_messages_ignores_non_data_image_urls() -> None:
    messages = [{
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png", "detail": "low"},
        }],
    }]
    out = redact_messages(messages)
    # http URLs are left alone — only base64 data URIs are redacted.
    assert out[0]["content"][0]["image_url"]["url"] == "https://example.com/cat.png"
    print("ok  redact_messages leaves http image_url untouched")


def test_write_call_log_creates_file_and_increments_seq() -> None:
    _reset_seq()
    target = home / "ws" / ".chat_team" / "llm"
    assert not target.exists()
    payload = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "hi"},
    ]
    response_msg = {
        "role": "assistant",
        "content": "hello!",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "noop", "arguments": "{}"}}
        ],
    }
    p1 = write_call_log(
        dir_path=target,
        session_id="sid-abc",
        role_name="team_admin",
        call_kind="agent",
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=None,
        tool_names=["read_file", "write_file"],
        messages_payload=payload,
        response_message=response_msg,
        finish_reason="stop",
        usage={"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        latency_ms=42.125,
        error=None,
    )
    assert p1 is not None and p1.exists(), p1
    assert target.exists(), "dir not created lazily"
    assert "-001-team_admin-agent.json" in p1.name, p1.name

    record = json.loads(p1.read_text(encoding="utf-8"))
    expected_keys = {
        "session_id", "role_name", "call_kind", "timestamp_utc",
        "model", "temperature", "max_tokens", "tools",
        "messages", "response", "finish_reason", "usage",
        "latency_ms", "error",
    }
    assert expected_keys.issubset(record.keys()), record.keys()
    assert record["session_id"] == "sid-abc"
    assert record["call_kind"] == "agent"
    assert record["latency_ms"] == 42.125
    assert record["error"] is None
    assert record["response"]["tool_calls"][0]["function"]["name"] == "noop"
    assert record["tools"] == ["read_file", "write_file"]

    p2 = write_call_log(
        dir_path=target,
        session_id="sid-abc",
        role_name="team_admin",
        call_kind="agent",
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=None,
        tool_names=[],
        messages_payload=payload,
        response_message=response_msg,
        finish_reason="stop",
        usage=None,
        latency_ms=1.0,
        error=None,
    )
    assert p2 is not None and "-002-team_admin-agent.json" in p2.name, p2.name

    # Different session → seq counter is per-session.
    p3 = write_call_log(
        dir_path=target,
        session_id="sid-other",
        role_name="team_admin",
        call_kind="compactor",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=None,
        tool_names=[],
        messages_payload=payload,
        response_message=response_msg,
        finish_reason="stop",
        usage=None,
        latency_ms=1.0,
        error=None,
    )
    assert p3 is not None and "-001-team_admin-compactor.json" in p3.name, p3.name
    print("ok  write_call_log writes file and seq increments per session")


def test_write_call_log_error_path() -> None:
    _reset_seq()
    target = home / "ws" / ".chat_team" / "llm"
    p = write_call_log(
        dir_path=target,
        session_id="sid-err",
        role_name="team_admin",
        call_kind="agent",
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=None,
        tool_names=[],
        messages_payload=[{"role": "user", "content": "boom"}],
        response_message=None,
        finish_reason=None,
        usage=None,
        latency_ms=5.0,
        error="RateLimitError('quota exceeded')",
    )
    assert p is not None and p.exists()
    record = json.loads(p.read_text(encoding="utf-8"))
    assert record["response"] is None
    assert "RateLimitError" in record["error"]
    assert record["finish_reason"] is None
    print("ok  write_call_log error path records error and null response")


def test_write_call_log_swallow_io_failure() -> None:
    # Pointing dir_path at a path whose parent is a regular file forces
    # mkdir to fail; the function must catch and return None, never raise.
    blocker = home / "blocker"
    blocker.write_text("not-a-dir", encoding="utf-8")
    bad_dir = blocker / ".chat_team" / "llm"
    p = write_call_log(
        dir_path=bad_dir,
        session_id="sid-x",
        role_name="r",
        call_kind="agent",
        model="m",
        temperature=0.0,
        max_tokens=None,
        tool_names=[],
        messages_payload=[],
        response_message=None,
        finish_reason=None,
        usage=None,
        latency_ms=0.0,
        error=None,
    )
    assert p is None, "expected None on IO failure"
    print("ok  write_call_log swallows IO failures")


async def main() -> None:
    test_redact_messages_strips_base64()
    test_redact_messages_ignores_non_data_image_urls()
    test_write_call_log_creates_file_and_increments_seq()
    test_write_call_log_error_path()
    test_write_call_log_swallow_io_failure()
    print("PASS")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
