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
    print("== test 3: image frame → inbox/<file>.jpg + image content_block ==")
    settings = load_settings()
    workspace_root = settings.workspace_root
    workspace_root.mkdir(parents=True, exist_ok=True)

    aeskey = b64_aeskey()
    fake_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"y" * 200
    cipher = encrypt_for_test(fake_jpg, aeskey)

    async def fake_fetch(url: str) -> bytes:
        assert url == "https://example.com/img"
        return cipher

    real_dl = wecom_media.download_and_decrypt

    async def patched_dl(url, key, *, max_bytes=wecom_media.DEFAULT_MAX_MEDIA_BYTES, fetch=None):
        return await real_dl(url, key, max_bytes=max_bytes, fetch=fake_fetch)

    wecom_media.download_and_decrypt = patched_dl

    captured: list[tuple] = []

    async def handler(inbound, stream):
        captured.append((inbound.session_id, inbound.content_blocks, inbound.text))

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
    sid, blocks, text = captured[0]
    assert sid == "wecom-single-BOT-U1"
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["path"].startswith("./inbox/") and blocks[0]["path"].endswith(".jpg")
    # text rendering uses [图:<basename>]
    assert text.startswith("[图:") and text.endswith(".jpg]")
    print("  blocks:", blocks)
    print("  text:", text)

    inbox = workspace_for(sid) / "inbox"
    files = list(inbox.glob("*.jpg"))
    assert files, "no image written to inbox"
    assert files[0].read_bytes() == fake_jpg
    print(f"  ✓ wrote {files[0].name} ({files[0].stat().st_size} bytes)")
    assert sent and sent[0]["body"]["stream"]["content"] == "思考中…"


# --- 4. mixed payload (text + image, multi-image interleaved) -------------

async def test_adapter_mixed_flow():
    print("== test 4: mixed (text+image+text+image) preserves order in blocks ==")
    settings = load_settings()
    aeskey1 = b64_aeskey()
    aeskey2 = b64_aeskey()
    fake_png = b"\x89PNG\r\n\x1a\n" + b"z" * 50
    fake_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"y" * 60
    cipher_by_url = {
        "u1": encrypt_for_test(fake_png, aeskey1),
        "u2": encrypt_for_test(fake_jpg, aeskey2),
    }

    real_dl = wecom_media.download_and_decrypt

    async def patched_dl(url, key, *, max_bytes=wecom_media.DEFAULT_MAX_MEDIA_BYTES, fetch=None):
        async def fake_fetch(u):
            return cipher_by_url[u]
        return await real_dl(url, key, max_bytes=max_bytes, fetch=fake_fetch)
    wecom_media.download_and_decrypt = patched_dl

    captured: list = []

    async def handler(inbound, stream):
        captured.append((inbound.content_blocks, inbound.text))

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
                {"msgtype": "text", "text": {"content": "@bot 看下这两张"}},
                {"msgtype": "image", "image": {"url": "u1", "aeskey": aeskey1}},
                {"msgtype": "text", "text": {"content": "对比一下"}},
                {"msgtype": "image", "image": {"url": "u2", "aeskey": aeskey2}},
                {"msgtype": "text", "text": {"content": "哪张好?"}},
            ]},
        },
    }
    await adapter._handle_msg_callback(frame)

    wecom_media.download_and_decrypt = real_dl

    assert captured, "handler not invoked"
    blocks, text = captured[0]
    print("  blocks:", blocks)
    print("  text:", text)
    # Five logical blocks; coalesce keeps them apart because they alternate.
    types = [b.get("type") for b in blocks]
    assert types == ["text", "image", "text", "image", "text"], types
    # @bot stripped from the leading text block only
    assert blocks[0]["text"] == "看下这两张", blocks[0]
    assert blocks[2]["text"] == "对比一下"
    assert blocks[4]["text"] == "哪张好?"
    assert blocks[1]["path"].endswith(".png"), blocks[1]
    assert blocks[3]["path"].endswith(".jpg"), blocks[3]
    # text rendering preserves order
    assert "看下这两张" in text and "对比一下" in text and "哪张好?" in text
    assert "@bot" not in text


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


# --- 6. quote payloads -----------------------------------------------------

def _build_quote_adapter(settings, cipher_by_url):
    """Return (adapter, restore_fn). adapter has download stubbed so each
    URL in cipher_by_url decrypts to its mapped fake bytes."""
    real_dl = wecom_media.download_and_decrypt

    async def patched_dl(url, key, *, max_bytes=wecom_media.DEFAULT_MAX_MEDIA_BYTES, fetch=None):
        async def fake_fetch(u):
            return cipher_by_url[u]
        return await real_dl(url, key, max_bytes=max_bytes, fetch=fake_fetch)

    wecom_media.download_and_decrypt = patched_dl

    def restore():
        wecom_media.download_and_decrypt = real_dl

    settings.env.update({"WECOM_BOT_ID": "BOT", "WECOM_SECRET": "S"})

    def workspace_for(sid):
        return settings.workspace_root / sid

    adapter = WeComBotAdapter(settings, workspace_resolver=workspace_for)

    async def fake_enqueue(payload):
        pass
    adapter._enqueue_write = fake_enqueue            # noqa: SLF001
    return adapter, restore


async def _run_with_handler(adapter, frame):
    captured = []

    async def handler(inbound, stream):
        captured.append((inbound.content_blocks, inbound.text))
    adapter.set_handler(handler)
    await adapter._handle_msg_callback(frame)        # noqa: SLF001
    assert captured, "handler never invoked"
    return captured[0]


async def test_quote_text():
    print("== test 6a: quote (text) on top-level text ==")
    settings = load_settings()
    adapter, restore = _build_quote_adapter(settings, cipher_by_url={})
    try:
        frame = {
            "cmd": "aibot_msg_callback", "headers": {"req_id": "rq-q1"},
            "body": {
                "msgid": "MQ1", "aibotid": "BOT", "chattype": "single",
                "from": {"userid": "U"}, "msgtype": "text",
                "text": {"content": "我同意你刚说的"},
                "quote": {"msgtype": "text", "text": {"content": "原计划周三上线"}},
            },
        }
        blocks, text = await _run_with_handler(adapter, frame)
    finally:
        restore()

    print("  blocks:", blocks)
    # all-text: coalesce merges into a single text block
    types = [b.get("type") for b in blocks]
    assert types == ["text"], types
    merged = blocks[0]["text"]
    assert merged.startswith("[引用开始]"), merged
    assert "原计划周三上线" in merged
    assert "[引用结束" in merged
    assert merged.endswith("我同意你刚说的"), merged
    # markers must precede quote content; quote content must precede current
    assert merged.index("[引用开始]") < merged.index("原计划周三上线")
    assert merged.index("原计划周三上线") < merged.index("[引用结束")
    assert merged.index("[引用结束") < merged.index("我同意你刚说的")


async def test_quote_image():
    print("== test 6b: quote (image-only) on top-level text ==")
    settings = load_settings()
    aeskey = b64_aeskey()
    fake_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"q" * 32
    adapter, restore = _build_quote_adapter(
        settings, cipher_by_url={"qu1": encrypt_for_test(fake_jpg, aeskey)},
    )
    try:
        frame = {
            "cmd": "aibot_msg_callback", "headers": {"req_id": "rq-q2"},
            "body": {
                "msgid": "MQ2", "aibotid": "BOT", "chattype": "single",
                "from": {"userid": "U"}, "msgtype": "text",
                "text": {"content": "这张图怎么解读?"},
                "quote": {"msgtype": "image", "image": {"url": "qu1", "aeskey": aeskey}},
            },
        }
        blocks, text = await _run_with_handler(adapter, frame)
    finally:
        restore()

    print("  blocks:", blocks)
    # [text(open), image, text(close), text(current)] → coalesce merges adjacent texts
    types = [b.get("type") for b in blocks]
    assert types == ["text", "image", "text"], types
    assert blocks[0]["text"] == "[引用开始]"
    assert blocks[1]["path"].endswith(".jpg")
    tail = blocks[2]["text"]
    assert tail.startswith("[引用结束"), tail
    assert tail.endswith("这张图怎么解读?"), tail


async def test_quote_mixed_multi_image_with_group_strip():
    print("== test 6c: quote (mixed multi-image) + group @bot strip on current ==")
    settings = load_settings()
    aeskey1 = b64_aeskey()
    aeskey2 = b64_aeskey()
    aeskey3 = b64_aeskey()
    png = b"\x89PNG\r\n\x1a\n" + b"a" * 30
    jpg1 = b"\xff\xd8\xff\xe0" + b"b" * 30
    jpg2 = b"\xff\xd8\xff\xe0" + b"c" * 30
    cipher_by_url = {
        "qm1": encrypt_for_test(png, aeskey1),
        "qm2": encrypt_for_test(jpg1, aeskey2),
        "cur1": encrypt_for_test(jpg2, aeskey3),
    }
    adapter, restore = _build_quote_adapter(settings, cipher_by_url=cipher_by_url)
    try:
        frame = {
            "cmd": "aibot_msg_callback", "headers": {"req_id": "rq-q3"},
            "body": {
                "msgid": "MQ3", "aibotid": "BOT", "chattype": "group", "chatid": "G2",
                "from": {"userid": "U"}, "msgtype": "mixed",
                "mixed": {"msg_item": [
                    {"msgtype": "text", "text": {"content": "@bot 看下当前的图"}},
                    {"msgtype": "image", "image": {"url": "cur1", "aeskey": aeskey3}},
                ]},
                "quote": {"msgtype": "mixed", "mixed": {"msg_item": [
                    {"msgtype": "text", "text": {"content": "对照一下:"}},
                    {"msgtype": "image", "image": {"url": "qm1", "aeskey": aeskey1}},
                    {"msgtype": "image", "image": {"url": "qm2", "aeskey": aeskey2}},
                ]}},
            },
        }
        blocks, text = await _run_with_handler(adapter, frame)
    finally:
        restore()

    print("  blocks:", blocks)
    print("  text:", text)
    # raw: [open], quote-text, qimg1, qimg2, [close], cur-text(@bot stripped), cur-img
    # coalesce merges open+quote-text and close+cur-text:
    # → text("[引用开始]\n对照一下"), image(png), image(jpg),
    #   text("[引用结束…]\n看下当前的图"), image(jpg)
    types = [b.get("type") for b in blocks]
    assert types == ["text", "image", "image", "text", "image"], types
    head = blocks[0]["text"]
    assert head.startswith("[引用开始]"), head
    assert "对照一下" in head
    assert blocks[1]["path"].endswith(".png")
    assert blocks[2]["path"].endswith(".jpg")
    mid = blocks[3]["text"]
    assert mid.startswith("[引用结束"), mid
    # @bot stripped on first text block of CURRENT side, not on quote interior
    assert mid.endswith("看下当前的图"), mid
    assert "@bot" not in mid
    assert blocks[4]["path"].endswith(".jpg")
    assert "@bot" not in text


async def main():
    await test_decrypt_round_trip()
    await test_sniff_extension()
    await test_adapter_image_flow()
    await test_adapter_mixed_flow()
    await test_event_flow()
    await test_quote_text()
    await test_quote_image()
    await test_quote_mixed_multi_image_with_group_strip()
    print("\nALL STAGE-7 SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
