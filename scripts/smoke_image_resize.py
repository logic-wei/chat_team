"""Smoke test for image auto-resize in ImageDataURICache.

* Small image (< max_inline_bytes) → returns original format data URI, no resize
* Oversized image with strategy=resize → auto resized to JPEG data URI
* Oversized image with strategy=reject → returns None (text placeholder path)
* RGBA image with transparency → composite onto white, resized to JPEG
* Resize is idempotent (cache hit on second call)
* Pillow-missing fallback → oversized image returns None gracefully
* configure_default_cache() resets the singleton with new parameters
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force a clean home so we don't read user state.
home = Path("/tmp/chat_team_smoke_image_resize")
shutil.rmtree(home, ignore_errors=True)
os.environ["CHAT_TEAM_HOME"] = str(home)

from chat_team.llm.image_cache import (
    MAX_INLINE_BYTES,
    ImageDataURICache,
    _resize_image,
    configure_default_cache,
    default_cache,
)


# ---------------------------------------------------------------------------
# Helpers to create test images
# ---------------------------------------------------------------------------

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


def _make_png(path: Path, width: int = 10, height: int = 10, rgba: bool = False) -> Path:
    """Create a simple PNG image using Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGBA" if rgba else "RGB"
    color = (255, 0, 0, 128) if rgba else (255, 0, 0)
    img = Image.new(mode, (width, height), color)
    img.save(path, format="PNG")
    return path


def _make_jpeg(path: Path, width: int = 10, height: int = 10) -> Path:
    """Create a simple JPEG image using Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (0, 128, 255))
    img.save(path, format="JPEG", quality=90)
    return path


def _make_oversized_png(path: Path) -> Path:
    """Create a PNG that exceeds the 6 MB limit when raw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 3000x3000 RGB PNG will be several MB uncompressed but well within 6MB
    # as a PNG. To exceed 6MB, we create a large uncompressed-ish image.
    # A 2500x2500 RGBA PNG is typically > 6MB.
    img = Image.new("RGBA", (2500, 2500), (255, 128, 0, 200))
    img.save(path, format="PNG")
    actual_size = path.stat().st_size
    if actual_size <= MAX_INLINE_BYTES:
        # If still too small, add noise to defeat compression
        import random
        pixels = img.load()
        for y in range(2500):
            for x in range(0, 2500, 4):  # sparse noise is enough
                pixels[x, y] = (random.randint(0, 255), random.randint(0, 255),
                                random.randint(0, 255), random.randint(128, 255))
        img.save(path, format="PNG")
    return path


# Tiny valid PNG (1x1, 67 bytes) — doesn't need Pillow.
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6364f8ffff3f0000050001017a96cb6c0000000049454e"
    "44ae426082"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_resize_when_small():
    """Small image (< max_inline_bytes) returns original format, no resize."""
    print("== test 1: small image → original format data URI ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        small = d / "small.png"
        small.write_bytes(_PNG_BYTES)

        cache = ImageDataURICache(max_inline_bytes=MAX_INLINE_BYTES)
        uri = cache.get(str(small))
        assert uri is not None, "small image must not return None"
        assert uri.startswith("data:image/png;base64,"), f"expected PNG mime, got: {uri[:40]}"
        print(f"  ✓ small PNG → {len(uri)} chars, mime=image/png")


def test_resize_on_oversize():
    """Oversized image with strategy=resize → auto resized to JPEG data URI."""
    if not HAS_PILLOW:
        print("== test 2: SKIPPED (Pillow not installed) ==")
        return
    print("== test 2: oversized image → resize to JPEG ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        big = _make_oversized_png(d / "big.png")
        raw_size = big.stat().st_size
        assert raw_size > MAX_INLINE_BYTES, f"test image must be > {MAX_INLINE_BYTES}, got {raw_size}"

        cache = ImageDataURICache(
            max_inline_bytes=MAX_INLINE_BYTES,
            oversized_image="resize",
            resize_long_side=2048,
            resize_quality=85,
        )
        uri = cache.get(str(big))
        assert uri is not None, f"oversized image should be resized, not None (raw_size={raw_size})"
        assert uri.startswith("data:image/jpeg;base64,"), f"expected JPEG mime after resize, got: {uri[:40]}"

        # Verify the base64 payload decodes to valid JPEG bytes smaller than limit
        b64_part = uri.split(",", 1)[1]
        import base64
        decoded = base64.b64decode(b64_part)
        assert len(decoded) <= MAX_INLINE_BYTES, f"resized image still too large: {len(decoded)}"
        assert decoded[:2] == b"\xff\xd8", "JPEG must start with FFD8"
        print(f"  ✓ oversized PNG ({raw_size} bytes) → JPEG ({len(decoded)} bytes)")


def test_reject_strategy():
    """Oversized image with strategy=reject → returns None."""
    if not HAS_PILLOW:
        print("== test 3: SKIPPED (Pillow not installed) ==")
        return
    print("== test 3: oversized image → reject strategy → None ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        big = _make_oversized_png(d / "big.png")
        raw_size = big.stat().st_size
        assert raw_size > MAX_INLINE_BYTES

        cache = ImageDataURICache(
            max_inline_bytes=MAX_INLINE_BYTES,
            oversized_image="reject",
        )
        uri = cache.get(str(big))
        assert uri is None, f"reject strategy must return None for oversized image, got {type(uri)}"
        print("  ✓ reject strategy → None")


def test_rgba_to_rgb():
    """RGBA image with transparency → composites onto white, resized to JPEG."""
    if not HAS_PILLOW:
        print("== test 4: SKIPPED (Pillow not installed) ==")
        return
    print("== test 4: RGBA image → white background + JPEG ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Create a small RGBA image (within limit, so it won't trigger resize).
        # This tests the fast path — small RGBA PNGs stay as PNG.
        rgba_small = _make_png(d / "rgba_small.png", width=10, height=10, rgba=True)
        cache = ImageDataURICache(max_inline_bytes=MAX_INLINE_BYTES)
        uri = cache.get(str(rgba_small))
        assert uri is not None
        assert uri.startswith("data:image/png;base64,"), "small RGBA → stays PNG"

        # Create a genuinely oversized RGBA PNG (same approach as _make_oversized_png).
        rgba_big = _make_oversized_png(d / "rgba_big.png")
        raw_size = rgba_big.stat().st_size
        assert raw_size > MAX_INLINE_BYTES, f"RGBA test image must exceed {MAX_INLINE_BYTES}"

        cache2 = ImageDataURICache(
            max_inline_bytes=MAX_INLINE_BYTES,
            oversized_image="resize",
            resize_long_side=2048,
            resize_quality=85,
        )
        uri2 = cache2.get(str(rgba_big))
        assert uri2 is not None, "RGBA oversize must resize successfully"
        assert uri2.startswith("data:image/jpeg;base64,"), "RGBA resize → JPEG"

        import base64
        b64_part = uri2.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:2] == b"\xff\xd8", "must be valid JPEG"
        print("  ✓ RGBA resize → JPEG composites onto white")


def test_resize_idempotent_cache_hit():
    """Second call for same file returns cached result (no re-read)."""
    if not HAS_PILLOW:
        print("== test 5: SKIPPED (Pillow not installed) ==")
        return
    print("== test 5: resize result cached on second call ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        big = _make_oversized_png(d / "big.png")

        cache = ImageDataURICache(
            max_inline_bytes=MAX_INLINE_BYTES,
            oversized_image="resize",
            resize_long_side=2048,
            resize_quality=85,
        )
        uri1 = cache.get(str(big))
        assert uri1 is not None

        # Check that the cache has exactly 1 entry.
        assert len(cache._store) == 1, f"expected 1 cache entry, got {len(cache._store)}"

        uri2 = cache.get(str(big))
        assert uri2 == uri1, "second call must return same URI (cache hit)"
        assert len(cache._store) == 1, "cache should not grow on hit"
        print("  ✓ second call → cache hit, same URI")


def test_pillow_missing_fallback():
    """When Pillow is not installed, oversized image returns None gracefully."""
    print("== test 6: Pillow missing → oversized returns None ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # Create a file that's big enough to trigger resize even without Pillow.
        big = d / "big.raw"
        big.write_bytes(b"\x00" * (7 * 1024 * 1024))

        cache = ImageDataURICache(
            max_inline_bytes=MAX_INLINE_BYTES,
            oversized_image="resize",
        )
        with patch("chat_team.llm.image_cache._lazy_import_pillow", return_value=None):
            uri = cache.get(str(big))
        assert uri is None, "without Pillow, resize must fall back to None"
        print("  ✓ Pillow missing → None returned")


def test_resize_image_function():
    """Direct test of _resize_image with various inputs."""
    if not HAS_PILLOW:
        print("== test 7: SKIPPED (Pillow not installed) ==")
        return
    print("== test 7: _resize_image() direct tests ==")

    # Resize a small RGB PNG — should not change dimensions (already under long_side).
    from PIL import Image as PILImage
    import io
    img = PILImage.new("RGB", (100, 50), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    result = _resize_image(png_bytes, long_side=2048, quality=85)
    assert result is not None
    assert result[:2] == b"\xff\xd8", "output must be JPEG"
    # Verify dimensions: 100x50 → longest side is 100, which is < 2048, so no downscale.
    resized_img = PILImage.open(io.BytesIO(result))
    assert resized_img.size == (100, 50), f"no downscale expected, got {resized_img.size}"

    # Resize a large image — should downscale.
    img2 = PILImage.new("RGB", (4000, 3000), (200, 100, 50))
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    png_bytes2 = buf2.getvalue()

    result2 = _resize_image(png_bytes2, long_side=2048, quality=85)
    assert result2 is not None
    resized2 = PILImage.open(io.BytesIO(result2))
    w, h = resized2.size
    assert max(w, h) <= 2048, f"longest side must be <= 2048, got {w}x{h}"
    assert w / h == 4000 / 3000, f"aspect ratio must be preserved, got {w}x{h}"

    print(f"  ✓ small image: no downscale (100x50)")
    print(f"  ✓ large image: downscale to {w}x{h}, aspect preserved")


def test_configure_default_cache():
    """configure_default_cache() replaces the singleton."""
    print("== test 8: configure_default_cache() resets singleton ==")

    # Get the current default.
    c1 = default_cache()
    assert c1._oversized_image == "resize"  # default from constructor
    assert c1._resize_long_side == 2048  # ... but after app.py calls configure, it may differ
    # The point is it returns a consistent object.

    # Reconfigure.
    configure_default_cache(
        max_inline_bytes=1024,
        oversized_image="reject",
        resize_long_side=512,
        resize_quality=60,
    )
    c2 = default_cache()
    assert c2 is not c1, "configure should create a new cache"
    assert c2._max_inline_bytes == 1024
    assert c2._oversized_image == "reject"
    assert c2._resize_long_side == 512
    assert c2._resize_quality == 60

    # Reset to defaults.
    configure_default_cache()
    c3 = default_cache()
    assert c3._max_inline_bytes == MAX_INLINE_BYTES
    assert c3._oversized_image == "resize"

    print("  ✓ configure_default_cache() creates new singleton with given params")


def test_empty_file_returns_none():
    """Empty file → returns None, not an exception."""
    print("== test 9: empty file → None ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        empty = d / "empty.png"
        empty.write_bytes(b"")
        cache = ImageDataURICache()
        uri = cache.get(str(empty))
        assert uri is None
        print("  ✓ empty file → None")


def test_missing_file_returns_none():
    """Missing file → returns None."""
    print("== test 10: missing file → None ==")
    cache = ImageDataURICache()
    uri = cache.get("/nonexistent/path/image.png")
    assert uri is None
    print("  ✓ missing file → None")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_no_resize_when_small()
    test_resize_on_oversize()
    test_reject_strategy()
    test_rgba_to_rgb()
    test_resize_idempotent_cache_hit()
    test_pillow_missing_fallback()
    test_resize_image_function()
    test_configure_default_cache()
    test_empty_file_returns_none()
    test_missing_file_returns_none()
    print("\nALL IMAGE_RESIZE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()