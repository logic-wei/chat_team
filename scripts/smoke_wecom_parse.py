"""Offline smoke test for WeCom adapter parsing & write-frame assembly.

Run: ``python3 scripts/smoke_wecom_parse.py`` — does NOT touch the network.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_wecom_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.adapters.base import ChatType, blocks_to_text, coalesce_text_blocks
from chat_team.adapters.wecom import (
    WeComBotAdapter,
    WeComStreamHandle,
    _LRU,
    _MENTION_RE,
    _strip_mention_from_first_text,
)
from chat_team.config import load_settings


def test_lru():
    lru = _LRU(3)
    assert lru.add("a") is True
    assert lru.add("a") is False                 # dedup hit
    lru.add("b"); lru.add("c"); lru.add("d")     # evicts a
    assert "a" not in lru
    assert lru.add("a") is True                  # a was evicted, can re-add


def test_mention_strip():
    assert _MENTION_RE.sub("", "@RobotA hello", count=1) == "hello"
    assert _MENTION_RE.sub("", "@小管 你好啊", count=1) == "你好啊"
    assert _MENTION_RE.sub("", "no mention here", count=1) == "no mention here"


async def _parse_full(adapter, frame):
    """Mirror what `_handle_msg_callback` does up to handler dispatch:
    metadata + async block resolution (handles quote + group @-strip) + coalesce."""
    body = frame.get("body") or {}
    msgtype = body.get("msgtype") or "text"
    inbound = adapter._parse_metadata(frame)
    blocks = await adapter._resolve_inbound_blocks(
        body, msgtype, inbound.session_id, inbound.chat_type,
    )
    if blocks is None:
        inbound.content_blocks = []
        inbound.text = ""
        return inbound
    blocks = coalesce_text_blocks(blocks)
    inbound.content_blocks = blocks
    inbound.text = blocks_to_text(blocks)
    return inbound


async def test_parse_inbound_single():
    os.environ["WECOM_BOT_ID"] = "BOT123"
    os.environ["WECOM_SECRET"] = "SEC"
    settings = load_settings()                   # reload picks env
    adapter = WeComBotAdapter(settings)
    frame = {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": "rq-1"},
        "body": {
            "msgid": "m1",
            "aibotid": "BOT123",
            "chattype": "single",
            "from": {"userid": "u1"},
            "msgtype": "text",
            "text": {"content": "你好"},
        },
    }
    inbound = await _parse_full(adapter, frame)
    assert inbound is not None
    assert inbound.session_id == "wecom-single-BOT123-u1", inbound.session_id
    assert inbound.chat_type == ChatType.SINGLE
    assert inbound.text == "你好"
    assert inbound.reply_token == "rq-1"


async def test_parse_inbound_group_strips_mention():
    settings = load_settings()
    adapter = WeComBotAdapter(settings)
    frame = {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": "rq-2"},
        "body": {
            "msgid": "m2",
            "aibotid": "BOT123",
            "chatid": "CHAT_X",
            "chattype": "group",
            "from": {"userid": "u2"},
            "msgtype": "text",
            "text": {"content": "@小管 帮我看代码"},
        },
    }
    inbound = await _parse_full(adapter, frame)
    assert inbound.session_id == "wecom-group-CHAT_X"
    assert inbound.text == "帮我看代码", repr(inbound.text)


async def test_parse_inbound_voice():
    settings = load_settings()
    adapter = WeComBotAdapter(settings)
    frame = {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": "rq-3"},
        "body": {
            "msgid": "m3", "aibotid": "BOT", "chattype": "single",
            "from": {"userid": "u"}, "msgtype": "voice",
            "voice": {"content": "用文字转写后的内容"},
        },
    }
    inbound = await _parse_full(adapter, frame)
    assert inbound.text == "用文字转写后的内容"


async def test_parse_inbound_image_no_workspace():
    """No workspace_resolver wired → image save fails gracefully and
    becomes a single ``[图片下载失败]`` text block, never an image block."""
    settings = load_settings()
    adapter = WeComBotAdapter(settings)            # no resolver
    frame = {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": "rq-4"},
        "body": {
            "msgid": "m4", "aibotid": "BOT", "chattype": "single",
            "from": {"userid": "u"}, "msgtype": "image",
            "image": {"url": "x", "aeskey": "y"},
        },
    }
    inbound = await _parse_full(adapter, frame)
    assert inbound is not None
    assert inbound.content_blocks == [{"type": "text", "text": "[图片下载失败]"}], inbound.content_blocks
    assert "下载失败" in inbound.text


def test_strip_mention_first_text_block_only():
    """The strip MUST only touch the leading text block — image blocks
    before it are preserved, and trailing text blocks are unchanged."""
    blocks = [
        {"type": "text", "text": "@小管 看下这张图"},
        {"type": "image", "path": "./inbox/a.jpg"},
        {"type": "text", "text": "@小管 followup"},
    ]
    out = _strip_mention_from_first_text(blocks)
    assert out[0] == {"type": "text", "text": "看下这张图"}
    assert out[1] == {"type": "image", "path": "./inbox/a.jpg"}
    assert out[2] == {"type": "text", "text": "@小管 followup"}, out[2]


def test_dedup():
    settings = load_settings()
    adapter = WeComBotAdapter(settings)
    assert adapter._msgid_lru.add("dup") is True
    assert adapter._msgid_lru.add("dup") is False


async def test_stream_frame_shape():
    """Verify WeComStreamHandle queues correctly-shaped frames without a real ws."""
    settings = load_settings()
    adapter = WeComBotAdapter(settings)
    h = WeComStreamHandle(adapter, req_id="REQ")
    # Force send-frame regardless of throttle by zeroing the timer
    h._last_push = -1000.0
    await h._send_frame("first", finish=False)
    h._last_push = -1000.0
    await h._send_frame("done", finish=True)

    frames = []
    while not adapter._write_queue.empty():
        frames.append(adapter._write_queue.get_nowait())

    assert len(frames) == 2, len(frames)
    assert frames[0]["cmd"] == "aibot_respond_msg"
    assert frames[0]["headers"]["req_id"] == "REQ"
    assert frames[0]["body"]["msgtype"] == "stream"
    assert frames[0]["body"]["stream"]["finish"] is False
    assert frames[0]["body"]["stream"]["content"] == "first"
    assert frames[0]["body"]["stream"]["id"] == frames[1]["body"]["stream"]["id"]
    assert frames[1]["body"]["stream"]["finish"] is True
    assert frames[1]["body"]["stream"]["content"] == "done"


async def main():
    test_lru()
    test_mention_strip()
    test_strip_mention_first_text_block_only()
    await test_parse_inbound_single()
    await test_parse_inbound_group_strips_mention()
    await test_parse_inbound_voice()
    await test_parse_inbound_image_no_workspace()
    test_dedup()
    await test_stream_frame_shape()
    print("ALL WeCom UNIT TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
