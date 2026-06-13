"""In-process LRU cache: image path → ``data:image/<mime>;base64,...`` URI.

The OpenAI vision payload requires inlining image bytes as base64 data URIs.
We don't store the bytes in agent.history (that would balloon session.json
and re-send identical bytes on every turn); we store relative paths and
materialise the data URI here on demand. The cache key includes file
``mtime_ns`` and ``size`` so a rewritten file invalidates the entry.

The cache also acts as a guard:
* Missing file or read error → returns ``None``.
* Raw size > ``max_inline_bytes`` → depends on ``oversized_image`` strategy:
  - ``"resize"`` (default): auto downscale + re-encode as JPEG, then cache
    the resized result. If the resized bytes still exceed *max_inline_bytes*
    (very unlikely with reasonable settings), fall back to ``None``.
  - ``"reject"``: return ``None`` (callers emit a text placeholder).

Callers (``openai_provider._to_openai_messages``) should fall back to a text
block ``[图:<basename>(已丢失/过大)]`` on ``None``.
"""
from __future__ import annotations

import base64
import logging
import os
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

from ..adapters.wecom_media import sniff_extension

log = logging.getLogger(__name__)

MAX_ENTRIES = 32                # entry-count cap
MAX_TOTAL_BYTES = 32 * 1024 * 1024  # ~32 MB total b64 payload across entries
MAX_INLINE_BYTES = 6 * 1024 * 1024  # raw file size; ~8 MB after b64 < OpenAI cap

# Pillow is optional — the fast path (small images) never touches it.
# Only the resize branch imports it, and we catch ImportError gracefully.
_PIL_UnidentifiedImageError: type | None = None
_PIL_Image: object | None = None


def _lazy_import_pillow() -> tuple[object, type] | None:
    """Import Pillow on demand; return (Image, UnidentifiedImageError) or None."""
    global _PIL_Image, _PIL_UnidentifiedImageError
    if _PIL_Image is not None:
        return _PIL_Image, _PIL_UnidentifiedImageError  # type: ignore[return-value]
    try:
        from PIL import Image, UnidentifiedImageError  # type: ignore[import-untyped]
        _PIL_Image = Image
        _PIL_UnidentifiedImageError = UnidentifiedImageError
        return Image, UnidentifiedImageError
    except ImportError:
        log.warning(
            "Pillow is not installed; oversized images cannot be auto-resized "
            "and will be replaced with text placeholders. "
            "Install with: pip install Pillow"
        )
        return None


_EXT_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _mime_for(data: bytes, ext: str) -> str:
    sniffed = sniff_extension(data, "image")
    if sniffed in _EXT_TO_MIME:
        return _EXT_TO_MIME[sniffed]
    if ext.lower() in _EXT_TO_MIME:
        return _EXT_TO_MIME[ext.lower()]
    return "image/jpeg"


def _resize_image(
    data: bytes,
    *,
    long_side: int,
    quality: int,
) -> bytes | None:
    """Resize *data* (raw image bytes) so the longest dimension is at most
    *long_side* pixels, then re-encode as JPEG at the given *quality*.

    Returns the JPEG bytes, or ``None`` if Pillow is unavailable or the image
    cannot be decoded.
    """
    pillow = _lazy_import_pillow()
    if pillow is None:
        return None
    Image, UnidentifiedImageError = pillow  # type: ignore[misc]

    try:
        img = Image.open(BytesIO(data))
        img.load()  # force full decode so we catch errors early
    except (UnidentifiedImageError, OSError, ValueError, Exception):  # noqa: BLE001
        log.warning("image_cache: Pillow failed to decode image for resize", exc_info=True)
        return None

    # Convert RGBA / palette / other modes to RGB (JPEG has no alpha).
    if img.mode not in ("RGB", "L"):
        # Composite onto white background for alpha channels.
        if img.mode in ("RGBA", "LA", "PA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])  # use alpha as mask
            img = bg
        else:
            img = img.convert("RGB")

    # Proportional downscale via thumbnail (in-place, preserves aspect ratio).
    w, h = img.size
    if w > long_side or h > long_side:
        img.thumbnail((long_side, long_side))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class ImageDataURICache:
    """LRU keyed by ``(abs_path, mtime_ns, size, long_side, quality)``. Both an
    entry-count and a total-bytes ceiling are enforced; oldest entries evict
    first.

    When *oversized_image* is ``"resize"`` (default), files that exceed
    *max_inline_bytes* are automatically resized before encoding; the cache key
    includes the resize parameters so different configs don't collide.
    """

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
        max_inline_bytes: int = MAX_INLINE_BYTES,
        oversized_image: str = "resize",
        resize_long_side: int = 2048,
        resize_quality: int = 85,
    ):
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._max_inline_bytes = max_inline_bytes
        self._oversized_image = oversized_image
        self._resize_long_side = resize_long_side
        self._resize_quality = resize_quality
        self._store: OrderedDict[tuple[str, int, int, int, int], str] = OrderedDict()
        self._total_bytes = 0

    def get(self, path: str | Path) -> str | None:
        """Return a ``data:image/...;base64,...`` URI for the file, or
        ``None`` if the file is missing, unreadable, or too large to send.
        """
        try:
            abs_path = os.path.abspath(str(path))
            stat = os.stat(abs_path)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            log.warning("image_cache: cannot stat %s", path)
            return None

        size = stat.st_size
        if size <= 0:
            log.warning("image_cache: empty file %s", abs_path)
            return None

        # Fast path: file fits within limit — read and cache directly.
        if size <= self._max_inline_bytes:
            return self._read_and_encode(abs_path, stat)

        # Slow path: file exceeds limit.
        if self._oversized_image != "resize":
            log.warning(
                "image_cache: skipping %s (size=%d, max=%d, strategy=%s)",
                abs_path, size, self._max_inline_bytes, self._oversized_image,
            )
            return None

        # Attempt auto-resize.
        return self._resize_and_encode(abs_path, stat)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, abs_path: str, stat: os.stat_result) -> tuple[str, int, int, int, int]:
        return (abs_path, stat.st_mtime_ns, stat.st_size, self._resize_long_side, self._resize_quality)

    def _read_and_encode(self, abs_path: str, stat: os.stat_result) -> str | None:
        """Read a small (within-limit) file, base64-encode, and cache."""
        key = self._cache_key(abs_path, stat)
        cached = self._store.get(key)
        if cached is not None:
            self._store.move_to_end(key)
            return cached

        try:
            with open(abs_path, "rb") as fp:
                data = fp.read()
        except OSError:
            log.exception("image_cache: read failed for %s", abs_path)
            return None

        # Double-check size after read (file could have changed between stat and read).
        if len(data) > self._max_inline_bytes:
            log.warning(
                "image_cache: file grew past limit between stat and read: %s (%d > %d)",
                abs_path, len(data), self._max_inline_bytes,
            )
            return None

        ext = os.path.splitext(abs_path)[1].lstrip(".")
        mime = _mime_for(data, ext)
        b64 = base64.b64encode(data).decode("ascii")
        uri = f"data:{mime};base64,{b64}"

        self._store[key] = uri
        self._total_bytes += len(uri)
        self._evict_if_needed()
        return uri

    def _resize_and_encode(self, abs_path: str, stat: os.stat_result) -> str | None:
        """Read an oversized file, resize + re-encode as JPEG, and cache."""
        key = self._cache_key(abs_path, stat)
        cached = self._store.get(key)
        if cached is not None:
            self._store.move_to_end(key)
            return cached

        try:
            with open(abs_path, "rb") as fp:
                data = fp.read()
        except OSError:
            log.exception("image_cache: read failed for resize of %s", abs_path)
            return None

        resized = _resize_image(
            data,
            long_side=self._resize_long_side,
            quality=self._resize_quality,
        )
        if resized is None:
            log.warning(
                "image_cache: resize failed for %s (size=%d); falling back to placeholder",
                abs_path, len(data),
            )
            return None

        if len(resized) > self._max_inline_bytes:
            log.warning(
                "image_cache: resized %s still too large (%d > %d); giving up",
                abs_path, len(resized), self._max_inline_bytes,
            )
            return None

        b64 = base64.b64encode(resized).decode("ascii")
        uri = f"data:image/jpeg;base64,{b64}"

        self._store[key] = uri
        self._total_bytes += len(uri)
        self._evict_if_needed()
        log.info(
            "image_cache: resized %s (%d → %d bytes, long_side=%d, quality=%d)",
            abs_path, len(data), len(resized), self._resize_long_side, self._resize_quality,
        )
        return uri

    def _evict_if_needed(self) -> None:
        while (
            len(self._store) > self._max_entries
            or self._total_bytes > self._max_total_bytes
        ) and self._store:
            _, victim = self._store.popitem(last=False)
            self._total_bytes -= len(victim)
            if self._total_bytes < 0:
                self._total_bytes = 0


# Module-level default cache shared by the provider; tests can replace it.
_DEFAULT_CACHE: ImageDataURICache | None = None


def default_cache() -> ImageDataURICache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = ImageDataURICache()
    return _DEFAULT_CACHE


def configure_default_cache(
    *,
    max_inline_bytes: int = MAX_INLINE_BYTES,
    oversized_image: str = "resize",
    resize_long_side: int = 2048,
    resize_quality: int = 85,
) -> None:
    """(Re)create the module-level default cache with the given parameters.

    Called once at startup from ``app.py`` after ``load_settings()`` so that
    config.yaml values take effect.
    """
    global _DEFAULT_CACHE
    _DEFAULT_CACHE = ImageDataURICache(
        max_inline_bytes=max_inline_bytes,
        oversized_image=oversized_image,
        resize_long_side=resize_long_side,
        resize_quality=resize_quality,
    )
    log.info(
        "image_cache: configured (max_inline_bytes=%d, oversized=%s, "
        "resize_long_side=%d, resize_quality=%d)",
        max_inline_bytes, oversized_image, resize_long_side, resize_quality,
    )