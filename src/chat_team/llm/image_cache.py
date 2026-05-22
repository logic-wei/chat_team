"""In-process LRU cache: image path → ``data:image/<mime>;base64,...`` URI.

The OpenAI vision payload requires inlining image bytes as base64 data URIs.
We don't store the bytes in agent.history (that would balloon session.json
and re-send identical bytes on every turn); we store relative paths and
materialise the data URI here on demand. The cache key includes file
``mtime_ns`` and ``size`` so a rewritten file invalidates the entry.

The cache also acts as a guard:
* Missing file or read error → returns ``None``.
* Raw size > ``MAX_INLINE_BYTES`` → returns ``None`` (the OpenAI request
  body cap is ~10 MB; base64 expansion is ~4/3, so 6 MB raw is the
  conservative ceiling).

Callers (``openai_provider._to_openai_messages``) should fall back to a
text block ``[图:<basename>(已丢失/过大)]`` on ``None``.
"""
from __future__ import annotations

import base64
import logging
import os
from collections import OrderedDict
from pathlib import Path

from ..adapters.wecom_media import sniff_extension

log = logging.getLogger(__name__)

MAX_ENTRIES = 32                # entry-count cap
MAX_TOTAL_BYTES = 32 * 1024 * 1024  # ~32 MB total b64 payload across entries
MAX_INLINE_BYTES = 6 * 1024 * 1024  # raw file size; ~8 MB after b64 < OpenAI cap


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


class ImageDataURICache:
    """LRU keyed by ``(abs_path, mtime_ns, size)``. Both an entry-count and
    a total-bytes ceiling are enforced; oldest entries evict first."""

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
        max_inline_bytes: int = MAX_INLINE_BYTES,
    ):
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._max_inline_bytes = max_inline_bytes
        self._store: OrderedDict[tuple[str, int, int], str] = OrderedDict()
        self._total_bytes = 0

    def get(self, path: str | Path) -> str | None:
        """Return a ``data:image/...;base64,...`` URI for the file, or
        ``None`` if the file is missing, unreadable, or too large.
        """
        try:
            abs_path = os.path.abspath(str(path))
            stat = os.stat(abs_path)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            log.warning("image_cache: cannot stat %s", path)
            return None

        size = stat.st_size
        if size <= 0 or size > self._max_inline_bytes:
            log.warning(
                "image_cache: skipping %s (size=%d, max=%d)",
                abs_path, size, self._max_inline_bytes,
            )
            return None

        key = (abs_path, stat.st_mtime_ns, size)
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

        ext = os.path.splitext(abs_path)[1].lstrip(".")
        mime = _mime_for(data, ext)
        b64 = base64.b64encode(data).decode("ascii")
        uri = f"data:{mime};base64,{b64}"

        self._store[key] = uri
        self._total_bytes += len(uri)
        self._evict_if_needed()
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
