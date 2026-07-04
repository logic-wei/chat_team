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


# Per-attempt timeouts and total attempt count for a single media URL.
#
# Rationale: WeCom download URLs expire ~5 minutes after delivery. Observed
# failure mode is NOT slow bulk transfer — it is the connection-setup
# phase stalling (TCP/TLS handshake against a bad CDN edge / routing flap):
# one attempt hangs for the full timeout, then the immediate retry against
# a different node succeeds. So we split the timeout budget by phase and
# give the connection phases a TIGHT ceiling while leaving body transfer
# generous headroom. Worst case below stays well inside the 300s URL
# lifetime: 20s × 6 attempts = 120s, plus the rare full-20s handover.
#
# Phase split (per attempt):
#   - connect        = 5s   # whole connection setup incl. DNS + TCP + TLS
#   - sock_connect   = 5s   # TCP connect only (after DNS resolved)
#   - sock_read      = 15s  # max gap between body bytes once connected
#   - total          = 20s  # hard ceiling for one whole attempt
# The tight ``connect`` / ``sock_connect`` is what kills the "stuck for
# ~20s then retry succeeds" pathology: a hung connection now fails in 5s
# and the retry immediately hits a fresh node, instead of burning 20s
# before the slow path is even attempted.
_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 5
_DOWNLOAD_SOCK_CONNECT_TIMEOUT_SECONDS = 5
_DOWNLOAD_SOCK_READ_TIMEOUT_SECONDS = 15
_DOWNLOAD_TOTAL_TIMEOUT_SECONDS = 20
_DOWNLOAD_MAX_ATTEMPTS = 6   # = 1 initial + 5 retries


async def _http_get(url: str) -> bytes:
    """Download one media URL with phase-split timeouts + retries.

    Why phase-split (not a single ``total``): the dominant failure mode
    we see is the connection-setup phase stalling, not slow body transfer.
    A single flat ``total=30s`` lets a hung handshake eat the whole budget
    before any retry happens. Splitting the budget makes a stuck
    connection fail fast at 5s and immediately retry against a fresh node,
    which is the path that already reliably succeeds on the second attempt
    in the field.

    Budget per attempt:
      - connect / sock_connect = 5s   (TCP+TLS handshake — the stuck phase)
      - sock_read              = 15s  (body transfer — generous)
      - total                  = 20s  (hard per-attempt ceiling)

    Retries: up to 5 immediate retries (6 attempts total), each on a fresh
    ``ClientSession`` so connection-pool reuse can't pin us to a bad node.
    All attempts combined (120s worst-case) stay well inside the ~5min URL lifetime.
    """
    timeout = aiohttp.ClientTimeout(
        total=_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
        connect=_DOWNLOAD_CONNECT_TIMEOUT_SECONDS,
        sock_connect=_DOWNLOAD_SOCK_CONNECT_TIMEOUT_SECONDS,
        sock_read=_DOWNLOAD_SOCK_READ_TIMEOUT_SECONDS,
    )
    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            # Fresh session per attempt: a bad keep-alive connection in the
            # pool must NOT carry over into the retry, otherwise the retry
            # re-hits the same stuck node and the timeout split is wasted.
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.read()
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            last_exc = exc
            log.warning(
                "media download attempt %d/%d failed for %s: %r",
                attempt, _DOWNLOAD_MAX_ATTEMPTS, url[:80], exc,
            )
            # fall through to next retry, or re-raise after the final attempt
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
