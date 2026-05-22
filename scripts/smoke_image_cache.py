"""Unit smoke for ImageDataURICache:
* path → data URI
* cache hit (no re-read)
* mtime invalidation
* missing file → None
* oversize file → None
* entry-count eviction
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat_team.llm.image_cache import ImageDataURICache


# Tiny valid PNG (1x1, 67 bytes).
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6364f8ffff3f0000050001017a96cb6c0000000049454e"
    "44ae426082"
)
_JPG_BYTES = bytes.fromhex("ffd8ffe000104a464946000100" + "00" * 32) + b"\xff\xd9"


def write_file(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def test_returns_data_uri():
    print("== test 1: path → data URI ==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.png"
        write_file(p, _PNG_BYTES)
        cache = ImageDataURICache()
        uri = cache.get(p)
        assert uri is not None
        assert uri.startswith("data:image/png;base64,"), uri[:40]
        assert len(uri) > len("data:image/png;base64,")
        print("  ✓", uri[:48], "...")


def test_cache_hit():
    print("== test 2: second get hits cache (no re-read on touched mtime) ==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.jpg"
        write_file(p, _JPG_BYTES)
        cache = ImageDataURICache()
        u1 = cache.get(p)
        u2 = cache.get(p)
        assert u1 is u2, "expected same cached object"
        assert u1.startswith("data:image/jpeg;base64,")
        print("  ✓ second call returned cached URI")


def test_mtime_invalidates():
    print("== test 3: rewriting the file invalidates the entry ==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.png"
        write_file(p, _PNG_BYTES)
        cache = ImageDataURICache()
        u1 = cache.get(p)
        # wait a hair to cross mtime nanosecond resolution on coarse FS
        time.sleep(0.01)
        write_file(p, _JPG_BYTES)
        # Force mtime bump on FS that round to seconds:
        new_time = time.time() + 1
        os.utime(p, (new_time, new_time))
        u2 = cache.get(p)
        assert u2 is not None
        assert u2 != u1, "rewritten file must produce a different cached URI"
        # and the new MIME should reflect JPEG bytes
        assert u2.startswith("data:image/jpeg;base64,")
        print("  ✓ mtime/size key invalidates on rewrite")


def test_missing_file():
    print("== test 4: missing file → None ==")
    cache = ImageDataURICache()
    assert cache.get("/tmp/definitely-not-here-9871234.png") is None
    print("  ✓")


def test_oversize():
    print("== test 5: oversize file → None ==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "huge.png"
        # 7 MB > 6 MB cap. Use sparse-friendly bytes (zeroes).
        p.write_bytes(b"\x00" * (7 * 1024 * 1024))
        cache = ImageDataURICache()
        assert cache.get(p) is None
        print("  ✓ file > MAX_INLINE_BYTES rejected")


def test_entry_eviction():
    print("== test 6: entry-count eviction kicks in ==")
    with tempfile.TemporaryDirectory() as d:
        cache = ImageDataURICache(max_entries=2, max_total_bytes=10 * 1024 * 1024)
        paths = []
        for i in range(3):
            p = Path(d) / f"img-{i}.png"
            # vary by content so each has a unique key
            write_file(p, _PNG_BYTES + bytes([i]) * 4)
            paths.append(p)
            cache.get(p)
        assert len(cache._store) == 2
        # oldest (paths[0]) should have been evicted; recent two remain
        keys = list(cache._store.keys())
        assert all(str(paths[0]) not in k[0] for k in keys), keys
        print("  ✓ oldest entry evicted at cap")


async def main():
    test_returns_data_uri()
    test_cache_hit()
    test_mtime_invalidates()
    test_missing_file()
    test_oversize()
    test_entry_eviction()
    print("\nALL IMAGE-CACHE SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
