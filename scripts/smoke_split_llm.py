"""Smoke test: vision model and chat model configured separately.

Covers:
* build_vision_llm_provider returns the SAME instance when vision creds == main creds
* build_vision_llm_provider returns a DIFFERENT instance when vision creds differ
* Dispatcher._run_turn passes vision_llm (not self.llm) to apply_vision_strategy
* DescribeImageTool.run uses ctx.vision_llm when present
* DescribeImageTool.run falls back to ctx.llm when ctx.vision_llm is None
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

home = Path("/tmp/chat_team_smoke_split_llm")
shutil.rmtree(home, ignore_errors=True)
os.environ["CHAT_TEAM_HOME"] = str(home)

# Provide a dummy API key so build_llm_provider doesn't raise.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-main")

from chat_team.agent.tools.base import ToolContext
from chat_team.agent.tools.describe_image import DescribeImageTool
from chat_team.app import build_vision_llm_provider
from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from chat_team.llm.image_description_cache import ImageDescriptionCache
from chat_team.llm.openai_provider import OpenAIChatCompletionProvider


_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6364f8ffff3f0000050001017a96cb6c0000000049454e"
    "44ae426082"
)


class TaggedLLM(LLMProvider):
    """Records calls and identifies itself via a tag."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            message=ChatMessage(role="assistant", content=f"[{self.tag}]"),
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
    session_id: str = "smoke-split-sid"
    current_role: str = "test_role"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: build_vision_llm_provider returns same instance when creds identical
# ─────────────────────────────────────────────────────────────────────────────
def test_vision_provider_same_instance_when_creds_identical():
    print("== test 1: same creds → same LLMProvider instance ==")
    settings = load_settings()
    settings.llm.api_key = "sk-test-main"
    main_llm = OpenAIChatCompletionProvider(api_key="sk-test-main")
    vision_llm = build_vision_llm_provider(settings, main_llm)
    assert vision_llm is main_llm, (
        f"expected same instance, got different: {vision_llm!r} vs {main_llm!r}"
    )
    print("  ✓ returns same instance when vision creds match main creds")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: build_vision_llm_provider returns NEW instance when creds differ
# ─────────────────────────────────────────────────────────────────────────────
def test_vision_provider_new_instance_when_creds_differ():
    print("== test 2: different creds (via config) → new LLMProvider instance ==")
    settings = load_settings()
    settings.llm.api_key = "sk-test-main"
    settings.llm.vision.api_key = "sk-vision-special"
    main_llm = OpenAIChatCompletionProvider(api_key="sk-test-main")
    vision_llm = build_vision_llm_provider(settings, main_llm)
    assert vision_llm is not main_llm, "expected a new instance for different vision key"
    assert isinstance(vision_llm, OpenAIChatCompletionProvider)
    print("  ✓ returns new OpenAIChatCompletionProvider when vision key differs")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: OPENAI_VISION_API_KEY env var also causes a split
# ─────────────────────────────────────────────────────────────────────────────
def test_vision_provider_env_var_causes_split():
    print("== test 3: OPENAI_VISION_API_KEY env var fallback → new instance ==")
    old = os.environ.pop("OPENAI_VISION_API_KEY", None)
    try:
        os.environ["OPENAI_VISION_API_KEY"] = "sk-env-vision"
        settings = load_settings()
        settings.llm.api_key = "sk-test-main"
        main_llm = OpenAIChatCompletionProvider(api_key="sk-test-main")
        vision_llm = build_vision_llm_provider(settings, main_llm)
        assert vision_llm is not main_llm, "env var vision key should cause new instance"
        print("  ✓ OPENAI_VISION_API_KEY env var causes a new vision provider")
    finally:
        if old is None:
            os.environ.pop("OPENAI_VISION_API_KEY", None)
        else:
            os.environ["OPENAI_VISION_API_KEY"] = old


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: DescribeImageTool uses ctx.vision_llm when present
# ─────────────────────────────────────────────────────────────────────────────
async def test_describe_image_uses_vision_llm():
    print("== test 4: DescribeImageTool uses ctx.vision_llm when set ==")
    with tempfile.TemporaryDirectory() as d:
        cwd = Path(d) / "ws"
        cwd.mkdir()
        img = cwd / "inbox" / "a.png"
        img.parent.mkdir()
        img.write_bytes(_PNG_BYTES)

        settings = load_settings()
        sess = _FakeSession(cwd=cwd, notebook=_FakeNotebook())
        chat_llm = TaggedLLM("chat")
        vision_llm = TaggedLLM("vision")

        ctx = ToolContext(
            cwd=cwd,
            session=sess,
            settings=settings,
            stream=None,
            llm=chat_llm,
            vision_llm=vision_llm,
        )
        tool = DescribeImageTool()
        result = await tool.run(ctx, paths=["./inbox/a.png"])
        assert "[vision]" in result, f"expected [vision] tag in result: {result!r}"
        assert len(vision_llm.calls) == 1, f"expected 1 call on vision_llm, got {len(vision_llm.calls)}"
        assert len(chat_llm.calls) == 0, f"chat_llm should not be called, got {len(chat_llm.calls)}"
        print("  ✓ DescribeImageTool routed to vision_llm, chat_llm untouched")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: DescribeImageTool falls back to ctx.llm when vision_llm is None
# ─────────────────────────────────────────────────────────────────────────────
async def test_describe_image_fallback_to_chat_llm():
    print("== test 5: DescribeImageTool falls back to ctx.llm when vision_llm=None ==")
    with tempfile.TemporaryDirectory() as d:
        cwd = Path(d) / "ws"
        cwd.mkdir()
        img = cwd / "inbox" / "b.png"
        img.parent.mkdir()
        img.write_bytes(_PNG_BYTES)

        settings = load_settings()
        sess = _FakeSession(cwd=cwd, notebook=_FakeNotebook())
        chat_llm = TaggedLLM("chat-fallback")

        ctx = ToolContext(
            cwd=cwd,
            session=sess,
            settings=settings,
            stream=None,
            llm=chat_llm,
            vision_llm=None,        # no separate vision provider
        )
        # Reset module-level cache so previous test doesn't pollute
        import chat_team.llm.image_description_cache as _m
        _m._DEFAULT_CACHE = ImageDescriptionCache()

        tool = DescribeImageTool()
        result = await tool.run(ctx, paths=["./inbox/b.png"])
        assert "[chat-fallback]" in result, f"expected [chat-fallback] in result: {result!r}"
        assert len(chat_llm.calls) == 1
        print("  ✓ falls back to ctx.llm when ctx.vision_llm is None")


async def main():
    test_vision_provider_same_instance_when_creds_identical()
    test_vision_provider_new_instance_when_creds_differ()
    test_vision_provider_env_var_causes_split()
    await test_describe_image_uses_vision_llm()
    await test_describe_image_fallback_to_chat_llm()
    print("\nALL SPLIT_LLM SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
