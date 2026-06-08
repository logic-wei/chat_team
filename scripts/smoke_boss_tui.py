"""Smoke test for boss TUI module — import, construction, protocol conformance.

No actual Textual rendering; just verifies the module loads and key classes work.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HOME = Path("/tmp/chat_team_boss_tui_smoke")
shutil.rmtree(HOME, ignore_errors=True)
os.environ["CHAT_TEAM_HOME"] = str(HOME)
os.environ.setdefault("OPENAI_API_KEY", "sk-fake")


from chat_team.boss import build_boss_agent
from chat_team.boss_tui import BossApp, TuiStream
from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)


class FakeLLM(LLMProvider):
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            message=ChatMessage(role="assistant", content="fake reply"),
            usage=None,
        )


async def main() -> None:
    settings = load_settings()
    llm = FakeLLM()
    agent = build_boss_agent(settings, llm)

    # BossApp can be instantiated
    app = BossApp(agent=agent, llm=llm, settings=settings)
    assert app.agent is agent
    assert app.llm is llm
    assert app._pending_bot_widget is None
    assert app._busy is False
    print("OK: BossApp construction")

    # TuiStream conforms to StreamHandle protocol
    stream = TuiStream(app)
    assert hasattr(stream, "push")
    assert hasattr(stream, "status")
    assert hasattr(stream, "finish")
    assert hasattr(stream, "send_image")
    assert hasattr(stream, "send_file")
    print("OK: TuiStream protocol")

    # Slash commands are defined
    from chat_team.boss_tui import SLASH_COMMANDS

    assert len(SLASH_COMMANDS) >= 4
    cmds = [c for c, _ in SLASH_COMMANDS]
    assert "/quit" in cmds
    assert "/clear" in cmds
    assert "/roles" in cmds
    assert "/help" in cmds
    print("OK: slash commands defined")

    print("\nAll boss TUI smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
