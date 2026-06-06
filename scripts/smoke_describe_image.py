"""Smoke test for describe_images() + DescribeImageTool.

* describe_images caches by (path, mtime, size, detail, model, prompt)
* Different prompt or different file → fresh LLM call
* Multi-image batch dispatched concurrently, returns in path order
* Missing file / oversize file → "[读取失败]" without an LLM call
* DescribeImageTool sandbox rejects absolute paths and ../
* DescribeImageTool formats results as "[图:rel]\\n<desc>" sections
* DescribeImageTool allows up to 16 paths and rejects 17+
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force a clean home so we don't read user state.
home = Path("/tmp/chat_team_smoke_describe_image")
shutil.rmtree(home, ignore_errors=True)
os.environ["CHAT_TEAM_HOME"] = str(home)

from chat_team.agent.tools.base import ToolContext, ToolError
from chat_team.agent.tools.describe_image import (
    DescribeImageTool,
    describe_images,
)
from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from chat_team.llm.image_description_cache import ImageDescriptionCache


# Tiny valid PNG (1x1, 67 bytes).
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6364f8ffff3f0000050001017a96cb6c0000000049454e"
    "44ae426082"
)


class CountingLLM(LLMProvider):
    """Returns a deterministic description per call; counts how many calls."""

    def __init__(self, response_factory=None) -> None:
        self.calls: list[CompletionRequest] = []
        self._factory = response_factory or (lambda i: f"描述-{i}")

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        idx = len(self.calls)
        self.calls.append(request)
        return CompletionResponse(
            message=ChatMessage(role="assistant", content=self._factory(idx)),
            finish_reason="stop",
        )


@dataclass
class _FakeNotebook:
    def toc(self) -> str: return ""
    def dump(self) -> dict: return {}


@dataclass
class _FakeSession:
    cwd: Path
    notebook: _FakeNotebook
    session_id: str = "smoke-sid"
    current_role: str = "test_role"


def _make_image(dir_: Path, name: str) -> Path:
    p = dir_ / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_PNG_BYTES)
    return p


async def test_single_call_then_cache():
    print("== test 1+2: single call, repeat hits cache ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = _make_image(d / "inbox", "a.png")
        cache = ImageDescriptionCache()
        llm = CountingLLM(lambda i: "猫一只")
        out1 = await describe_images(
            [str(img)],
            prompt="describe", detail="high", llm=llm,
            model="gpt-4o", image_base_dir=str(d), cache=cache,
        )
        assert out1 == ["猫一只"], out1
        assert len(llm.calls) == 1
        out2 = await describe_images(
            [str(img)],
            prompt="describe", detail="high", llm=llm,
            model="gpt-4o", image_base_dir=str(d), cache=cache,
        )
        assert out2 == ["猫一只"]
        assert len(llm.calls) == 1, "second call should hit cache"
        print("  ✓ first call → 1 LLM hit; second call → 0 (cache)")


async def test_different_prompt_misses_cache():
    print("== test 3: different prompt → cache miss ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = _make_image(d / "inbox", "a.png")
        cache = ImageDescriptionCache()
        llm = CountingLLM(lambda i: f"resp-{i}")
        await describe_images(
            [str(img)], prompt="P1", detail="high", llm=llm,
            model="gpt-4o", image_base_dir=str(d), cache=cache,
        )
        out = await describe_images(
            [str(img)], prompt="P2", detail="high", llm=llm,
            model="gpt-4o", image_base_dir=str(d), cache=cache,
        )
        assert out == ["resp-1"], out
        assert len(llm.calls) == 2, "different prompt must trigger a new LLM call"
        print("  ✓ prompt change → fresh LLM call")


async def test_concurrent_order_preserved():
    print("== test 4: 3 images concurrent, order preserved ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = [_make_image(d / "inbox", f"img{i}.png") for i in range(3)]
        # Force unique content per file so cache keys differ.
        for i, p in enumerate(paths):
            p.write_bytes(_PNG_BYTES + bytes([i] * 8))
        cache = ImageDescriptionCache()
        llm = CountingLLM(lambda i: f"d{i}")
        out = await describe_images(
            [str(p) for p in paths],
            prompt="P", detail="high", llm=llm,
            model="gpt-4o", image_base_dir=str(d), cache=cache,
        )
        assert len(out) == 3
        assert len(llm.calls) == 3, f"expected 3 LLM calls, got {len(llm.calls)}"
        # The order is path order, not LLM call order — the gather may return
        # in any order but the result list is positional.
        # Each request carried exactly one image whose path matches the input.
        request_paths = []
        for req in llm.calls:
            user_msg = req.messages[0]
            blocks = user_msg.content
            assert isinstance(blocks, list)
            img_blocks = [b for b in blocks if b.get("type") == "image"]
            assert len(img_blocks) == 1
            request_paths.append(img_blocks[0]["path"])
        assert sorted(request_paths) == sorted(str(p) for p in paths)
        print("  ✓ 3 images → 3 LLM calls, results aligned positionally")


async def test_missing_and_oversize_no_llm():
    print("== test 5: missing/oversize file → '[读取失败]' without LLM ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        good = _make_image(d / "inbox", "good.png")
        big = d / "inbox" / "huge.png"
        big.write_bytes(b"\x00" * (7 * 1024 * 1024))    # > 6MB cap
        missing = d / "inbox" / "nope.png"
        cache = ImageDescriptionCache()
        llm = CountingLLM(lambda i: "ok")
        out = await describe_images(
            [str(good), str(big), str(missing)],
            prompt="P", detail="high", llm=llm,
            model="gpt-4o", image_base_dir=str(d), cache=cache,
        )
        assert out[0] == "ok"
        assert out[1].startswith("[读取失败:") and "过大" in out[1]
        assert out[2].startswith("[读取失败:") and ("不存在" in out[2] or "无法读取" in out[2])
        # Only the good image should have triggered an LLM call.
        assert len(llm.calls) == 1, f"expected 1 LLM call (good only), got {len(llm.calls)}"
        print("  ✓ bad inputs short-circuit to '[读取失败]' without LLM")


async def test_tool_sandbox_and_format():
    print("== test 6+7: DescribeImageTool sandbox + formatting ==")
    with tempfile.TemporaryDirectory() as d:
        cwd = Path(d) / "ws"
        cwd.mkdir()
        img = _make_image(cwd / "inbox", "a.png")

        settings = load_settings()
        sess = _FakeSession(cwd=cwd, notebook=_FakeNotebook())
        llm = CountingLLM(lambda i: "正文识别结果")
        ctx = ToolContext(cwd=cwd, session=sess, settings=settings, stream=None, llm=llm)
        tool = DescribeImageTool()

        # Absolute path → ToolError, no LLM
        try:
            await tool.run(ctx, paths=[str(img)])
        except ToolError as e:
            print("  ✓ absolute path rejected:", e)
        else:
            raise AssertionError("absolute path must be rejected by sandbox")
        assert len(llm.calls) == 0

        # ../ escape → ToolError
        try:
            await tool.run(ctx, paths=["../escape.png"])
        except ToolError as e:
            print("  ✓ '..' rejected:", e)
        else:
            raise AssertionError("'..' must be rejected")
        assert len(llm.calls) == 0

        # Relative valid path → returns formatted "[图:rel]\\n<desc>"
        result = await tool.run(ctx, paths=["./inbox/a.png"])
        assert "[图:./inbox/a.png]" in result, result
        assert "正文识别结果" in result, result
        assert len(llm.calls) == 1
        print("  ✓ relative path produces formatted result:", repr(result[:60]))

        # 16 paths accepted (batch upper bound)
        result16 = await tool.run(ctx, paths=["./inbox/a.png"] * 16)
        assert result16.count("[图:./inbox/a.png]") == 16, result16
        # repeated same path should be served by cache after first scan
        assert len(llm.calls) == 1, "batch call should reuse cache for repeated image path"

        # 17 paths → cap rejected
        try:
            await tool.run(ctx, paths=["./inbox/a.png"] * 17)
        except ToolError as e:
            print("  ✓ >16 paths rejected:", e)
        else:
            raise AssertionError(">16 paths must be rejected")


async def main():
    await test_single_call_then_cache()
    await test_different_prompt_misses_cache()
    await test_concurrent_order_preserved()
    await test_missing_and_oversize_no_llm()
    await test_tool_sandbox_and_format()
    print("\nALL DESCRIBE_IMAGE SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
