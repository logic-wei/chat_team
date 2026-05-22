"""Offline smoke for the file/image upload path:
- WeComBotAdapter.upload_media (3-step init / chunk / finish)
- WeComStreamHandle.send_image / send_file (media reply frame)
- SendImageTool / SendFileTool (sandbox + size + extension validation)

Run: ``python3 scripts/smoke_send_media.py`` — does NOT touch the network.
"""
from __future__ import annotations

import asyncio
import base64
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_smoke_send_media"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.adapters.wecom import (
    UPLOAD_CHUNK_SIZE,
    WeComBotAdapter,
    WeComStreamHandle,
)
from chat_team.agent.tools.base import ToolContext, ToolError
from chat_team.agent.tools.media_tools import (
    FILE_MAX_BYTES,
    IMAGE_MAX_BYTES,
    SendFileTool,
    SendImageTool,
)
from chat_team.config import load_settings


def _make_adapter_with_fake_acks(captured: list[dict]):
    """Build a WeComBotAdapter that auto-acks every upload command on the loop.

    Each time _enqueue_write is awaited, we capture the payload and schedule
    a synthetic server ack via _dispatch_ack on the next loop tick. Non-upload
    commands (e.g. aibot_respond_msg media frames) just get captured.
    """
    settings = load_settings()
    adapter = WeComBotAdapter(settings)

    async def fake_enqueue(payload: dict) -> None:
        captured.append(payload)
        cmd = payload.get("cmd", "")
        req_id = (payload.get("headers") or {}).get("req_id") or ""
        ack: dict | None = None
        if cmd == "aibot_upload_media_init":
            ack = {
                "headers": {"req_id": req_id},
                "body": {"upload_id": "UP1"},
                "errcode": 0, "errmsg": "ok",
            }
        elif cmd == "aibot_upload_media_chunk":
            ack = {"headers": {"req_id": req_id}, "errcode": 0, "errmsg": "ok"}
        elif cmd == "aibot_upload_media_finish":
            ack = {
                "headers": {"req_id": req_id},
                "body": {"type": "image", "media_id": "MID-XYZ", "created_at": "0"},
                "errcode": 0, "errmsg": "ok",
            }
        if ack is not None:
            asyncio.get_running_loop().call_soon(adapter._dispatch_ack, ack)

    adapter._enqueue_write = fake_enqueue                # type: ignore[assignment]
    return adapter


async def test_upload_media_3step_chunking():
    captured: list[dict] = []
    adapter = _make_adapter_with_fake_acks(captured)

    # 700_000 bytes → ceil(700000 / 262144) = 3 chunks
    data = b"x" * 700_000
    media_id = await adapter.upload_media(data, kind="image", filename="a.png")
    assert media_id == "MID-XYZ", media_id

    cmds = [f["cmd"] for f in captured]
    assert cmds == [
        "aibot_upload_media_init",
        "aibot_upload_media_chunk",
        "aibot_upload_media_chunk",
        "aibot_upload_media_chunk",
        "aibot_upload_media_finish",
    ], cmds

    init = captured[0]
    assert init["body"]["type"] == "image"
    assert init["body"]["filename"] == "a.png"
    assert init["body"]["total_size"] == 700_000
    assert init["body"]["total_chunks"] == 3
    assert init["body"]["md5"] and len(init["body"]["md5"]) == 32

    for idx, frame in enumerate(captured[1:4]):
        assert frame["body"]["chunk_index"] == idx, (idx, frame)
        assert frame["body"]["upload_id"] == "UP1"
        decoded = base64.b64decode(frame["body"]["base64_data"])
        assert len(decoded) <= UPLOAD_CHUNK_SIZE
    # chunks reassemble to the original bytes
    reassembled = b"".join(
        base64.b64decode(f["body"]["base64_data"]) for f in captured[1:4]
    )
    assert reassembled == data
    assert captured[-1]["body"]["upload_id"] == "UP1"


async def test_upload_media_too_small():
    adapter = _make_adapter_with_fake_acks([])
    try:
        await adapter.upload_media(b"x", kind="file", filename="t.bin")
    except RuntimeError as err:
        assert "too small" in str(err) or "≥5" in str(err), err
    else:
        raise AssertionError("expected RuntimeError for tiny payload")


async def test_upload_media_too_large():
    adapter = _make_adapter_with_fake_acks([])
    try:
        await adapter.upload_media(b"x" * (10 * 1024 * 1024 + 1), kind="image", filename="big.png")
    except RuntimeError as err:
        assert "exceeds" in str(err), err
    else:
        raise AssertionError("expected RuntimeError for oversized image")


async def test_stream_send_image_emits_media_frame(tmp_path: Path):
    captured: list[dict] = []
    adapter = _make_adapter_with_fake_acks(captured)
    handle = WeComStreamHandle(adapter, req_id="USERREQ")

    img = tmp_path / "tiny.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"P" * 200)             # valid-ish 208 bytes
    await handle.send_image(img)

    media_frame = captured[-1]
    assert media_frame["cmd"] == "aibot_respond_msg"
    assert media_frame["headers"]["req_id"] == "USERREQ"
    assert media_frame["body"]["msgtype"] == "image"
    assert media_frame["body"]["image"]["media_id"] == "MID-XYZ"


async def test_stream_send_file_uses_filename_override(tmp_path: Path):
    captured: list[dict] = []
    adapter = _make_adapter_with_fake_acks(captured)
    handle = WeComStreamHandle(adapter, req_id="REQ-F")

    f = tmp_path / "raw.bin"
    f.write_bytes(b"hello world payload")
    await handle.send_file(f, filename="report.pdf")

    init = next(p for p in captured if p["cmd"] == "aibot_upload_media_init")
    assert init["body"]["filename"] == "report.pdf"
    assert init["body"]["type"] == "file"
    media_frame = captured[-1]
    assert media_frame["body"]["msgtype"] == "file"
    assert media_frame["body"]["file"]["media_id"] == "MID-XYZ"


# ---- Tool-level validation -------------------------------------------------


class _StubStream:
    """Minimal StreamHandle stub that captures send_image / send_file args."""

    def __init__(self):
        self.images: list[tuple[Path, str | None]] = []
        self.files: list[tuple[Path, str | None]] = []

    async def push(self, chunk: str, *, append: bool = True) -> None:  # noqa: ARG002
        return None

    async def status(self, note: str) -> None:                         # noqa: ARG002
        return None

    async def finish(self, final_text: str) -> None:                   # noqa: ARG002
        return None

    async def send_image(self, path: Path, *, filename: str | None = None) -> None:
        self.images.append((path, filename))

    async def send_file(self, path: Path, *, filename: str | None = None) -> None:
        self.files.append((path, filename))


def _make_ctx(cwd: Path, stream=None) -> ToolContext:
    settings = load_settings()
    return ToolContext(cwd=cwd, session=None, settings=settings, stream=stream)  # type: ignore[arg-type]


async def test_send_image_tool_happy(tmp_path: Path):
    img = tmp_path / "ok.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"P" * 100)
    stream = _StubStream()
    ctx = _make_ctx(tmp_path, stream=stream)
    out = await SendImageTool().run(ctx, path="ok.png")
    assert "已发送图片" in out, out
    assert len(stream.images) == 1
    sent_path, sent_name = stream.images[0]
    assert sent_path.resolve() == img.resolve()
    assert sent_name is None


async def test_send_image_tool_rejects_bad_ext(tmp_path: Path):
    img = tmp_path / "weird.bmp"
    img.write_bytes(b"BM" + b"x" * 200)
    ctx = _make_ctx(tmp_path, stream=_StubStream())
    try:
        await SendImageTool().run(ctx, path="weird.bmp")
    except ToolError as err:
        assert "extension" in str(err) or "bmp" in str(err), err
    else:
        raise AssertionError("expected ToolError on .bmp")


async def test_send_image_tool_rejects_oversize(tmp_path: Path):
    img = tmp_path / "big.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * (IMAGE_MAX_BYTES + 1))
    ctx = _make_ctx(tmp_path, stream=_StubStream())
    try:
        await SendImageTool().run(ctx, path="big.png")
    except ToolError as err:
        assert "too large" in str(err), err
    else:
        raise AssertionError("expected ToolError on oversize image")


async def test_send_image_tool_rejects_escape(tmp_path: Path):
    ctx = _make_ctx(tmp_path, stream=_StubStream())
    try:
        await SendImageTool().run(ctx, path="../escape.png")
    except ToolError as err:
        assert "absolute paths" in str(err) or "escape" in str(err), err
    else:
        raise AssertionError("expected ToolError on '..' path")


async def test_send_image_tool_rejects_no_stream(tmp_path: Path):
    img = tmp_path / "ok.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"P" * 100)
    ctx = _make_ctx(tmp_path, stream=None)
    try:
        await SendImageTool().run(ctx, path="ok.png")
    except ToolError as err:
        assert "媒体回传" in str(err), err
    else:
        raise AssertionError("expected ToolError when ctx.stream is None")


async def test_send_file_tool_happy(tmp_path: Path):
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF-1.4 dummy " + b"x" * 200)
    stream = _StubStream()
    ctx = _make_ctx(tmp_path, stream=stream)
    out = await SendFileTool().run(ctx, path="report.pdf", filename="final.pdf")
    assert "已发送文件" in out, out
    assert stream.files[0][1] == "final.pdf"


async def test_send_file_tool_rejects_oversize(tmp_path: Path):
    f = tmp_path / "huge.zip"
    f.write_bytes(b"x" * (FILE_MAX_BYTES + 1))
    ctx = _make_ctx(tmp_path, stream=_StubStream())
    try:
        await SendFileTool().run(ctx, path="huge.zip")
    except ToolError as err:
        assert "too large" in str(err), err
    else:
        raise AssertionError("expected ToolError on oversize file")


async def main():
    import tempfile
    await test_upload_media_3step_chunking()
    await test_upload_media_too_small()
    await test_upload_media_too_large()

    with tempfile.TemporaryDirectory() as td:
        await test_stream_send_image_emits_media_frame(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_stream_send_file_uses_filename_override(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_send_image_tool_happy(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_send_image_tool_rejects_bad_ext(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_send_image_tool_rejects_oversize(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_send_image_tool_rejects_escape(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_send_image_tool_rejects_no_stream(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_send_file_tool_happy(Path(td))
    with tempfile.TemporaryDirectory() as td:
        await test_send_file_tool_rejects_oversize(Path(td))

    print("ALL send_media SMOKES PASSED")


if __name__ == "__main__":
    asyncio.run(main())
