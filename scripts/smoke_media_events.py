"""Stage 7 smoke: media decrypt, media flow into workspace, event handling.

* AES-256-CBC PKCS#7 round-trip with a known key.
* sniff_extension recognises common magic bytes.
* WeComBotAdapter._handle_msg_callback routes an image/file/video frame
  through download+decrypt (stubbed) and writes the plaintext to
  ``<cwd>/inbox/``; the resolved text is fed to the handler.
* Mixed payload (text + image) yields concatenated text with the saved
  image path inline.
* enter_chat event triggers an aibot_respond_welcome_msg frame on the
  write queue; disconnected_event triggers stop.
"""
from __future__ import annotations

import asyncio
import base64
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_media_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from chat_team.adapters import wecom_media
from chat_team.adapters.wecom import WeComBotAdapter
from chat_team.config import load_settings


# --- helpers ---------------------------------------------------------------

def encrypt_for_test(plaintext: bytes, aeskey_b64: str) -> bytes:
    """Mirror of wecom_media.decrypt — used to fabricate test ciphertext."""
    raw = base64.b64decode(aeskey_b64 + "=" * (-len(aeskey_b64) % 4))[:32]
    iv = raw[:16]
    pad = 16 - (len(plaintext) % 16)            # PKCS#7 with block size 16
    padded = plaintext + bytes([pad]) * pad
    cipher = Cipher(algorithms.AES(raw), modes.CBC(iv))
    enc = cipher.encryptor()
    return enc.update(padded) + enc.finalize()


def b64_aeskey() -> str:
    # 32 random bytes → 43-char base64 (drop trailing '=')
    return base64.b64encode(os.urandom(32)).decode()[:43]


# --- 1. decrypt round-trip --------------------------------------------------

async def test_decrypt_round_trip():
    print("== test 1: AES-256-CBC PKCS#7 round-trip ==")
    aeskey = b64_aeskey()
    plain = b"hello chat_team \xff\xd8\xff" + b"x" * 100
    cipher = encrypt_for_test(plain, aeskey)
    out = wecom_media.decrypt(cipher, aeskey)
    assert out == plain, (out, plain)
    print(f"  ok — round-trip {len(plain)} bytes")


# --- 2. sniff_extension -----------------------------------------------------

async def test_sniff_extension():
    print("== test 2: sniff_extension magic bytes ==")
    cases = [
        (b"\xff\xd8\xff\xe0xxx", "image", "jpg"),
        (b"\x89PNG\r\n\x1a\nfoo", "image", "png"),
        (b"GIF89aXX", "image", "gif"),
        (b"%PDF-1.4 hello", "file", "pdf"),
        (b"PK\x03\x04abcd", "file", "zip"),
        (b"\x00\x00\x00 ftypisom....rest", "video", "mp4"),
        (b"random bytes", "image", "jpg"),    # fallback
        (b"random bytes", "file", "bin"),     # fallback
    ]
    for data, mt, expected in cases:
        got = wecom_media.sniff_extension(data, mt)
        assert got == expected, f"{mt} {data[:8]!r}: expected {expected}, got {got}"
    print("  ok — all magic bytes resolved")


# --- 3. adapter media flow (image, single chat) ----------------------------

async def test_adapter_image_flow():
    print("== test 3: image frame → inbox/<file>.jpg, handler sees text ==")
    settings = load_settings()
    workspace_root = settings.workspace_root
    workspace_root.mkdir(parents=True, exist_ok=True)

    aeskey = b64_aeskey()
    fake_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"y" * 200
    cipher = encrypt_for_test(fake_jpg, aeskey)

    async def fake_fetch(url: str) -> bytes:
        assert url == "https://example.com/img"
        return cipher

    # Patch download_and_decrypt's default fetch by monkey-patching the module.
    real_dl = wecom_media.download_and_decrypt

    async def patched_dl(url, key, *, max_bytes=wecom_media.DEFAULT_MAX_MEDIA_BYTES, fetch=None):
        return await real_dl(url, key, max_bytes=max_bytes, fetch=fake_fetch)

    wecom_media.download_and_decrypt = patched_dl

    captured: list[tuple] = []

    async def handler(inbound, stream):
        captured.append((inbound.session_id, inbound.text))

    settings.env["WECOM_BOT_ID"] = "BOT"
    settings.env["WECOM_SECRET"] = "S"

    def workspace_for(sid: str) -> Path:
        return workspace_root / sid

    adapter = WeComBotAdapter(settings, workspace_resolver=workspace_for)
    adapter.set_handler(handler)

    # Capture writes instead of opening a socket
    sent: list[dict] = []

    async def fake_enqueue(payload):
        sent.append(payload)

    adapter._enqueue_write = fake_enqueue            # noqa: SLF001

    frame = {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": "rq1"},
        "body": {
            "msgid": "M1", "aibotid": "BOT", "chattype": "single",
            "from": {"userid": "U1"}, "msgtype": "image",
            "image": {"url": "https://example.com/img", "aeskey": aeskey},
        },
    }
    await adapter._handle_msg_callback(frame)        # noqa: SLF001

    wecom_media.download_and_decrypt = real_dl

    assert captured, "handler was never invoked"
    sid, text = captured[0]
    assert sid == "wecom-single-BOT-U1"
    assert "用户发来 image" in text and "./inbox/" in text and ".jpg" in text
    print("  text:", text)

    inbox = workspace_for(sid) / "inbox"
    files = list(inbox.glob("*.jpg"))
    assert files, "no image written to inbox"
    assert files[0].read_bytes() == fake_jpg
    print(f"  ✓ wrote {files[0].name} ({files[0].stat().st_size} bytes)")
    # First write to queue must be the initial 思考中 stream frame
    assert sent and sent[0]["body"]["stream"]["content"] == "思考中…"


# --- 4. mixed payload (text + image) ---------------------------------------

async def test_adapter_mixed_flow():
    print("== test 4: mixed (text + image) handler sees joined text ==")
    settings = load_settings()
    aeskey = b64_aeskey()
    fake_png = b"\x89PNG\r\n\x1a\n" + b"z" * 50
    cipher = encrypt_for_test(fake_png, aeskey)

    real_dl = wecom_media.download_and_decrypt

    async def patched_dl(url, key, *, max_bytes=wecom_media.DEFAULT_MAX_MEDIA_BYTES, fetch=None):
        async def fake_fetch(_url):
            return cipher
        return await real_dl(url, key, max_bytes=max_bytes, fetch=fake_fetch)
    wecom_media.download_and_decrypt = patched_dl

    captured: list = []

    async def handler(inbound, stream):
        captured.append(inbound.text)

    def workspace_for(sid: str) -> Path:
        return settings.workspace_root / sid

    settings.env.update({"WECOM_BOT_ID": "BOT", "WECOM_SECRET": "S"})
    adapter = WeComBotAdapter(settings, workspace_resolver=workspace_for)
    adapter.set_handler(handler)

    async def fake_enqueue(payload):
        pass
    adapter._enqueue_write = fake_enqueue

    frame = {
        "cmd": "aibot_msg_callback", "headers": {"req_id": "rq2"},
        "body": {
            "msgid": "M2", "aibotid": "BOT", "chattype": "group",
            "chatid": "G1", "from": {"userid": "U1"}, "msgtype": "mixed",
            "mixed": {"msg_item": [
                {"msgtype": "text", "text": {"content": "@bot 看下这张图"}},
                {"msgtype": "image", "image": {"url": "u", "aeskey": aeskey}},
            ]},
        },
    }
    await adapter._handle_msg_callback(frame)

    wecom_media.download_and_decrypt = real_dl

    assert captured, "handler not invoked"
    text = captured[0]
    print("  text:", text)
    # @bot prefix stripped, image placeholder appended
    assert "@bot" not in text
    assert "看下这张图" in text
    assert "./inbox/" in text and ".png" in text


# --- 5. event flow: enter_chat + disconnected_event ------------------------

async def test_event_flow():
    print("== test 5: enter_chat → welcome frame; disconnected → stop ==")
    settings = load_settings()
    settings.env.update({"WECOM_BOT_ID": "BOT", "WECOM_SECRET": "S"})

    adapter = WeComBotAdapter(settings)
    adapter.set_handler(lambda *a, **k: asyncio.sleep(0))

    sent: list[dict] = []
    async def fake_enqueue(payload):
        sent.append(payload)
    adapter._enqueue_write = fake_enqueue

    enter_frame = {
        "cmd": "aibot_event_callback", "headers": {"req_id": "ev1"},
        "body": {
            "msgid": "E1", "aibotid": "BOT", "chattype": "single",
            "from": {"userid": "U1"}, "msgtype": "event",
            "event": {"eventtype": "enter_chat"},
        },
    }
    await adapter._handle_event_callback(enter_frame)
    assert sent, "no welcome frame queued"
    welcome = sent[0]
    assert welcome["cmd"] == "aibot_respond_welcome_msg"
    assert welcome["headers"]["req_id"] == "ev1"
    body = welcome["body"]
    assert body["msgtype"] == "text"
    assert body["text"]["content"]
    print("  welcome payload:", body["text"]["content"][:40], "...")

    # disconnected event sets stop flag
    assert not adapter._stop.is_set()
    disc_frame = {
        "cmd": "aibot_event_callback", "headers": {"req_id": "ev2"},
        "body": {
            "msgid": "E2", "aibotid": "BOT", "msgtype": "event",
            "event": {"eventtype": "disconnected_event"},
        },
    }
    await adapter._handle_event_callback(disc_frame)
    assert adapter._stop.is_set()
    print("  ✓ disconnected_event raised stop flag")


async def main():
    await test_decrypt_round_trip()
    await test_sniff_extension()
    await test_adapter_image_flow()
    await test_adapter_mixed_flow()
    await test_event_flow()
    print("\nALL STAGE-7 SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
