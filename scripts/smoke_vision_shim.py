"""Smoke test for the vision shim (apply_vision_strategy).

* tool mode + [text, image] → flat string with "[图:rel]" placeholder, no LLM call
* tool mode + plain string → unchanged, no LLM call
* tool mode + text-only list → flattened to string, no LLM call
* direct mode + [text, image] → list returned unchanged, no LLM call
* role.llm.vision_strategy overrides settings.llm.default_vision_strategy
* repeated tool-mode calls never invoke LLM
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

home = Path("/tmp/chat_team_smoke_vision_shim")
shutil.rmtree(home, ignore_errors=True)
os.environ["CHAT_TEAM_HOME"] = str(home)

from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from chat_team.roles.config import Role, RoleLLMConfig
from chat_team.vision_shim import apply_vision_strategy, resolve_vision_strategy


_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6364f8ffff3f0000050001017a96cb6c0000000049454e"
    "44ae426082"
)


class CountingLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        idx = len(self.calls)
        self.calls.append(request)
        # Echo the path so we can spot which image was OCR'd
        user_msg = request.messages[0]
        blocks = user_msg.content
        if isinstance(blocks, list):
            for b in blocks:
                if b.get("type") == "image":
                    return CompletionResponse(
                        message=ChatMessage(
                            role="assistant",
                            content=f"OCR-{os.path.basename(b['path'])}-{idx}",
                        ),
                        finish_reason="stop",
                    )
        return CompletionResponse(
            message=ChatMessage(role="assistant", content=f"resp-{idx}"),
            finish_reason="stop",
        )


def _make_image(d: Path, name: str, payload_extra: bytes = b"") -> Path:
    p = d / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_PNG_BYTES + payload_extra)
    return p


def _role(name: str, vs: str | None = None) -> Role:
    return Role(
        name=name,
        display_name=name,
        description="",
        system_prompt="",
        tools=[],
        llm=RoleLLMConfig(vision_strategy=vs),
    )


async def test_tool_mode_image_block_replaced():
    print("== test 1: tool mode replaces image block with [图:rel] ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _make_image(d / "inbox", "a.png")
        settings = load_settings()
        role = _role("admin")  # no per-role override → settings default = "tool"
        llm = CountingLLM()
        out = await apply_vision_strategy(
            [
                {"type": "text", "text": "看这个"},
                {"type": "image", "path": "./inbox/a.png"},
            ],
            role=role, settings=settings, llm=llm, cwd=d,
        )
        assert isinstance(out, str), f"expected str, got {type(out)}"
        assert "看这个" in out
        assert "[图:./inbox/a.png]" in out
        assert "OCR-a.png-0" not in out
        assert len(llm.calls) == 0
        print("  ✓ result:", repr(out))


async def test_tool_mode_plain_string_passthrough():
    print("== test 2: tool mode + plain string passthrough ==")
    settings = load_settings()
    role = _role("admin")
    llm = CountingLLM()
    out = await apply_vision_strategy(
        "你好", role=role, settings=settings, llm=llm, cwd=Path("/tmp"),
    )
    assert out == "你好"
    assert len(llm.calls) == 0
    print("  ✓ string returned unchanged, 0 LLM calls")


async def test_tool_mode_text_only_list_flattens():
    print("== test 3: tool mode + text-only list flattens to string ==")
    settings = load_settings()
    role = _role("admin")
    llm = CountingLLM()
    out = await apply_vision_strategy(
        [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ],
        role=role, settings=settings, llm=llm, cwd=Path("/tmp"),
    )
    assert isinstance(out, str)
    assert "hello" in out and "world" in out
    assert len(llm.calls) == 0, "no images → no LLM call expected"
    print("  ✓ text-only list →", repr(out))


async def test_direct_mode_passes_through_unchanged():
    print("== test 4: direct mode returns list as-is, no LLM ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _make_image(d / "inbox", "a.png")
        settings = load_settings()
        role = _role("vision_role", vs="direct")
        llm = CountingLLM()
        original = [
            {"type": "text", "text": "看图"},
            {"type": "image", "path": "./inbox/a.png"},
        ]
        out = await apply_vision_strategy(
            original, role=role, settings=settings, llm=llm, cwd=d,
        )
        assert out is original or out == original
        assert isinstance(out, list)
        assert len(llm.calls) == 0
        print("  ✓ direct mode → list passed through, 0 LLM calls")


async def test_role_overrides_settings_default():
    print("== test 5: role.llm.vision_strategy overrides settings default ==")
    settings = load_settings()
    # Settings default is "tool"; role explicitly says "direct"
    assert settings.llm.default_vision_strategy == "tool"
    role = _role("forced_direct", vs="direct")
    assert resolve_vision_strategy(role, settings) == "direct"
    role2 = _role("default_role", vs=None)
    assert resolve_vision_strategy(role2, settings) == "tool"
    role3 = _role("garbage", vs="bogus_value")
    # invalid → falls back to settings default
    assert resolve_vision_strategy(role3, settings) == "tool"
    print("  ✓ role override + invalid fallback both work")


async def test_repeated_tool_mode_never_calls_llm():
    print("== test 6: repeated tool-mode calls never invoke LLM ==")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _make_image(d / "inbox", "a.png", payload_extra=b"\xaa")
        _make_image(d / "inbox", "b.png", payload_extra=b"\xbb")
        _make_image(d / "inbox", "c.png", payload_extra=b"\xcc")
        settings = load_settings()
        role = _role("admin")
        llm = CountingLLM()

        out1 = await apply_vision_strategy(
            [
                {"type": "text", "text": "first"},
                {"type": "image", "path": "./inbox/a.png"},
                {"type": "image", "path": "./inbox/b.png"},
            ],
            role=role, settings=settings, llm=llm, cwd=d,
        )
        assert "[图:./inbox/a.png]" in out1 and "[图:./inbox/b.png]" in out1
        assert len(llm.calls) == 0

        out2 = await apply_vision_strategy(
            [
                {"type": "text", "text": "second"},
                {"type": "image", "path": "./inbox/a.png"},
                {"type": "image", "path": "./inbox/c.png"},
            ],
            role=role, settings=settings, llm=llm, cwd=d,
        )
        assert "[图:./inbox/a.png]" in out2 and "[图:./inbox/c.png]" in out2
        assert len(llm.calls) == 0
        print("  ✓ no pre-OCR call happened in either turn")


async def main():
    await test_tool_mode_image_block_replaced()
    await test_tool_mode_plain_string_passthrough()
    await test_tool_mode_text_only_list_flattens()
    await test_direct_mode_passes_through_unchanged()
    await test_role_overrides_settings_default()
    await test_repeated_tool_mode_never_calls_llm()
    print("\nALL VISION_SHIM SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
