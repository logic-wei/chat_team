"""LRU cache: (image path, prompt, detail, model) → vision description text.

Sibling to :mod:`image_cache`, but stores the **textual description** that
falls out of a vision LLM call rather than the raw base64 data URI. The
eager vision shim and the ``describe_image`` tool both consult this cache
so a given image+prompt+detail+model is OCR'd at most once per process.

The cache key includes ``mtime_ns`` and ``size`` so a rewritten file
invalidates its entry, plus ``model`` so swapping the vision backend
(e.g. ``gpt-4o`` → ``gpt-4o-mini``) doesn't read stale results.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger(__name__)

MAX_ENTRIES = 128
MAX_TOTAL_BYTES = 1 * 1024 * 1024  # ~1 MB total — descriptions are plain text, far smaller than base64


CacheKey = tuple[str, int, int, str, str, str]
# (abs_path, mtime_ns, size, detail, model, prompt)


class ImageDescriptionCache:
    """Process-level LRU keyed by (path, mtime, size, detail, model, prompt)."""

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ):
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._store: OrderedDict[CacheKey, str] = OrderedDict()
        self._total_bytes = 0

    def _key_for(
        self, path: str | Path, *, detail: str, model: str, prompt: str
    ) -> CacheKey | None:
        try:
            abs_path = os.path.abspath(str(path))
            stat = os.stat(abs_path)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return None
        return (abs_path, stat.st_mtime_ns, stat.st_size, detail, model, prompt)

    def get(
        self, path: str | Path, *, detail: str, model: str, prompt: str
    ) -> str | None:
        key = self._key_for(path, detail=detail, model=model, prompt=prompt)
        if key is None:
            return None
        cached = self._store.get(key)
        if cached is not None:
            self._store.move_to_end(key)
        return cached

    def put(
        self,
        path: str | Path,
        description: str,
        *,
        detail: str,
        model: str,
        prompt: str,
    ) -> None:
        key = self._key_for(path, detail=detail, model=model, prompt=prompt)
        if key is None:
            return
        if key in self._store:
            old = self._store.pop(key)
            self._total_bytes -= len(old)
        self._store[key] = description
        self._total_bytes += len(description)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while (
            len(self._store) > self._max_entries
            or self._total_bytes > self._max_total_bytes
        ) and self._store:
            _, victim = self._store.popitem(last=False)
            self._total_bytes -= len(victim)
            if self._total_bytes < 0:
                self._total_bytes = 0


_DEFAULT_CACHE: ImageDescriptionCache | None = None


def default_cache() -> ImageDescriptionCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = ImageDescriptionCache()
    return _DEFAULT_CACHE
