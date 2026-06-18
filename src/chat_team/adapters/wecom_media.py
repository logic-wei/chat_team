"""Decrypt + download for WeCom AI Bot media (image / file / video).

Long-connection mode delivers each media payload with its own per-URL
``aeskey`` (base64). The download URL is valid for 5 minutes; the bytes
returned are AES-256-CBC encrypted with PKCS#7 padding. The IV is the
first 16 bytes of the (decoded) aeskey, per the protocol docs.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from typing import Awaitable, Callable

import aiohttp
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

log = logging.getLogger(__name__)

DEFAULT_MAX_MEDIA_BYTES = 100 * 1024 * 1024     # protocol cap is ~100MB


# ---- decrypt ---------------------------------------------------------------


def _decode_aeskey(aeskey_b64: str) -> bytes:
    pad = "=" * (-len(aeskey_b64) % 4)
    raw = base64.b64decode(aeskey_b64 + pad)
    if len(raw) < 32:
        raise ValueError(f"aeskey too short after b64 decode: {len(raw)} bytes")
    return raw[:32]                            # AES-256 wants exactly 32 bytes


def decrypt(ciphertext: bytes, aeskey_b64: str) -> bytes:
    if not ciphertext:
        return b""
    key = _decode_aeskey(aeskey_b64)
    iv = key[:16]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    plain = decryptor.update(ciphertext) + decryptor.finalize()
    if not plain:
        return plain
    pad = plain[-1]
    if not 1 <= pad <= 32 or pad > len(plain):
        raise ValueError(f"invalid PKCS#7 pad length: {pad}")
    return plain[:-pad]


# ---- download --------------------------------------------------------------


HttpGet = Callable[[str], Awaitable[bytes]]


async def _http_get(url: str) -> bytes:
    """Download one media URL with a tight per-attempt timeout + one retry.

    Why tight: WeCom download URLs expire ~5 minutes after delivery. Even
    under concurrent download, a single slow CDN node burning a 60s timeout
    eats the validity window for the rest. So:
      - per-attempt total timeout = 30s (down from 60s)
      - one immediate retry on transient failure with a fresh session
    Both attempts together stay well inside the URL lifetime.
    """
    timeout = aiohttp.ClientTimeout(total=30)
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.read()
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            last_exc = exc
            log.warning(
                "media download attempt %d failed for %s: %r",
                attempt, url[:80], exc,
            )
            # fall through to retry (attempt 2) or re-raise after attempt 2
    assert last_exc is not None
    raise last_exc


async def download_and_decrypt(
    url: str,
    aeskey_b64: str,
    *,
    max_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
    fetch: HttpGet | None = None,
) -> bytes:
    """Fetch the URL, validate size, AES-256-CBC decrypt, return plaintext.

    ``fetch`` lets callers inject a stub for tests.
    """
    fetch = fetch or _http_get
    data = await fetch(url)
    if len(data) > max_bytes:
        raise ValueError(f"media payload too large: {len(data)} > {max_bytes}")
    return decrypt(data, aeskey_b64)


# ---- magic-byte sniffer ----------------------------------------------------


def sniff_extension(data: bytes, msgtype: str) -> str:
    """Best-effort file extension. Falls back by msgtype."""
    head = data[:16] if data else b""
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if head[:5] == b"%PDF-":
        return "pdf"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4"
    if msgtype == "image":
        return "jpg"
    if msgtype == "video":
        return "mp4"
    if msgtype == "voice":
        return "amr"
    return "bin"
