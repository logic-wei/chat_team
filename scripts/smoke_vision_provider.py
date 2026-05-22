"""Smoke test for OpenAI provider's vision content-block expansion.

* User message with text + image blocks → multi-part content with
  ``image_url`` data URI and ``detail`` stamped from the request.
* Pure-text list user content → collapses to a flat string.
* Missing image path → degrades to a ``[图:...(已丢失)]`` text block.
* Tool / assistant / system messages remain string content.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat_team.llm.base import ChatMessage, ToolCall
from chat_team.llm.image_cache import ImageDataURICache
from chat_team.llm.openai_provider import _to_openai_messages


_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6364f8ffff3f0000050001017a96cb6c0000000049454e"
    "44ae426082"
)


def test_user_with_image_expands_to_image_url():
    print("== test 1: user list content with image → image_url + data URI ==")
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "inbox" / "a.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(_PNG_BYTES)

        cache = ImageDataURICache()
        msgs = [
            ChatMessage(role="system", content="你是助手"),
            ChatMessage(role="user", content=[
                {"type": "text", "text": "看这张图"},
                {"type": "image", "path": "./inbox/a.png"},
                {"type": "text", "text": "这是什么?"},
            ]),
        ]
        out = _to_openai_messages(
            msgs, image_detail="high", image_base_dir=d, cache=cache,
        )
        assert out[0] == {"role": "system", "content": "你是助手"}
        assert out[1]["role"] == "user"
        parts = out[1]["content"]
        assert isinstance(parts, list), f"expected list content, got {type(parts)}"
        assert len(parts) == 3
        assert parts[0] == {"type": "text", "text": "看这张图"}
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["detail"] == "high"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert parts[2] == {"type": "text", "text": "这是什么?"}
        print("  ✓ image_url block carries data URI + detail=high")


def test_pure_text_list_collapses():
    print("== test 2: list content with only text blocks collapses to string ==")
    msgs = [ChatMessage(role="user", content=[
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ])]
    out = _to_openai_messages(msgs)
    assert isinstance(out[0]["content"], str), out[0]
    assert "hello" in out[0]["content"] and "world" in out[0]["content"]
    print("  ✓ text-only list → flat string:", repr(out[0]["content"]))


def test_missing_image_degrades_to_text():
    print("== test 3: missing image path falls back to text block ==")
    msgs = [ChatMessage(role="user", content=[
        {"type": "text", "text": "看图"},
        {"type": "image", "path": "./inbox/does-not-exist.jpg"},
    ])]
    cache = ImageDataURICache()
    out = _to_openai_messages(msgs, image_detail="high", cache=cache)
    # Only the image was missing → no image_url left → entire content collapses.
    content = out[0]["content"]
    assert isinstance(content, str)
    assert "看图" in content
    assert "[图:does-not-exist.jpg(已丢失)]" in content
    print("  ✓ missing image → degraded to text:", repr(content))


def test_tool_and_assistant_unchanged():
    print("== test 4: tool / assistant messages stay string ==")
    msgs = [
        ChatMessage(
            role="assistant", content="",
            tool_calls=[ToolCall(id="t1", name="foo", arguments={"a": 1})],
        ),
        ChatMessage(role="tool", content="ok", tool_call_id="t1"),
    ]
    out = _to_openai_messages(msgs)
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == ""
    assert out[0]["tool_calls"][0]["function"]["name"] == "foo"
    assert out[1]["role"] == "tool"
    assert out[1]["content"] == "ok"
    assert out[1]["tool_call_id"] == "t1"
    print("  ✓ tool/assistant string-content preserved")


def test_assistant_list_content_flattened():
    print("== test 5: defensive — list content on assistant is flattened ==")
    msgs = [ChatMessage(role="assistant", content=[
        {"type": "text", "text": "我看到一张图"},
        {"type": "image", "path": "./inbox/x.jpg"},
    ])]
    out = _to_openai_messages(msgs)
    content = out[0]["content"]
    assert isinstance(content, str)
    assert "我看到一张图" in content and "[图:x.jpg]" in content
    print("  ✓ assistant list-content flattened to string:", repr(content))


async def main():
    test_user_with_image_expands_to_image_url()
    test_pure_text_list_collapses()
    test_missing_image_degrades_to_text()
    test_tool_and_assistant_unchanged()
    test_assistant_list_content_flattened()
    print("\nALL VISION-PROVIDER SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
