"""Smoke test for solo mode: per-bot dispatchers, shared notebook, isolated persistence.

Run:
    CHAT_TEAM_HOME=/tmp/chat_team_smoke_solo python3 scripts/smoke_solo.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_smoke_solo"
os.environ.setdefault("OPENAI_API_KEY", "sk-fake")

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.agent.tools.base import ToolRegistry
from chat_team.agent.tools.notebook_tools import (
    NotebookReadTool,
    NotebookWriteTool,
)
from chat_team.config import BotConfig, Settings, load_settings
from chat_team.dispatcher import Dispatcher
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.roles.config import Role
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager
from chat_team.session.persistence import PersistenceManager, load_state


class CapturingStream:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.final: str | None = None

    async def push(self, chunk: str, *, append: bool = True) -> None:
        pass

    async def status(self, note: str) -> None:
        self.statuses.append(note)

    async def finish(self, final_text: str) -> None:
        self.final = final_text


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        return self._responses.pop(0)


def _text_response(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def _tool_call_response(name: str, args: dict) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=args)],
        ),
        finish_reason="tool_calls",
    )


async def main() -> None:
    home = Path("/tmp/chat_team_smoke_solo")
    if home.exists():
        shutil.rmtree(home)

    settings = load_settings()

    # Create two roles
    role_a = Role(
        name="engineer",
        display_name="工程师",
        description="Engineering role",
        system_prompt="You are an engineer.",
        tools=["notebook_read", "notebook_write"],
    )
    role_b = Role(
        name="designer",
        display_name="设计师",
        description="Design role",
        system_prompt="You are a designer.",
        tools=["notebook_read", "notebook_write"],
    )
    roles = RoleRegistry({"engineer": role_a, "designer": role_b})

    # Build tool registries WITHOUT transfer tool
    def build_tools():
        reg = ToolRegistry()
        reg.register(NotebookReadTool())
        reg.register(NotebookWriteTool())
        return reg

    persistence = PersistenceManager(settings)

    # -- Test 1: Solo mode config parsing --
    settings.mode = "solo"
    settings.bots = [
        BotConfig(name="engineer", bot_id="bot1", secret="s1"),
        BotConfig(name="designer", bot_id="bot2", secret="s2"),
    ]
    print("[1] Config: mode=%s, bots=%d" % (settings.mode, len(settings.bots)))
    assert settings.mode == "solo"
    assert len(settings.bots) == 2
    print("    PASS")

    # -- Test 2: No transfer tool registered --
    from chat_team.app import build_tool_registry
    tools = build_tool_registry(roles, include_transfer=False)
    assert "transfer_to_employee" not in tools.names()
    print("[2] transfer_to_employee NOT in tool registry")
    print("    PASS")

    # -- Test 3: Per-bot dispatchers with fixed_role --
    llm_a = ScriptedLLM([_text_response("工程师回答")])
    llm_b = ScriptedLLM([_text_response("设计师回答")])

    sessions_a = SessionManager(settings, persistence=persistence, solo_role="engineer")
    sessions_b = SessionManager(settings, persistence=persistence, solo_role="designer")

    dispatcher_a = Dispatcher(
        settings, sessions_a, roles, build_tools(), llm_a,
        persistence=persistence, fixed_role="engineer",
    )
    dispatcher_b = Dispatcher(
        settings, sessions_b, roles, build_tools(), llm_b,
        persistence=persistence, fixed_role="designer",
    )

    # Same group chat session_id for both
    session_id = "wecom-group-test123"
    msg = IncomingMessage(
        session_id=session_id,
        user_id="user1",
        text="你好",
        chat_type=ChatType.GROUP,
        msg_id="msg001",
        bot_id="bot1",
    )

    stream_a = CapturingStream()
    await dispatcher_a.handle(msg, stream_a)
    assert stream_a.final == "工程师回答", f"got: {stream_a.final}"
    print("[3] Dispatcher A (engineer) responded correctly")

    stream_b = CapturingStream()
    await dispatcher_b.handle(msg, stream_b)
    assert stream_b.final == "设计师回答", f"got: {stream_b.final}"
    print("    Dispatcher B (designer) responded correctly")
    print("    PASS")

    # -- Test 4: Shared notebook across bots --
    # Write to notebook directly (simulating what the tool does) via engineer's session
    session_a_obj = await sessions_a.get_or_create(session_id)
    session_a_obj.notebook.write("shared_fact", "来自工程师的笔记")

    # Designer's session for same group chat should see the notebook
    session_b_obj2 = await sessions_b.get_or_create(session_id)
    toc = session_b_obj2.notebook.toc()
    assert "shared_fact" in toc, f"notebook toc: {toc}"
    print("[4] Shared notebook: designer sees engineer's write")
    print("    PASS")

    # -- Test 5: Isolated session persistence --
    session_a = await sessions_a.get_or_create(session_id)
    persistence.flush_now(session_a)
    session_b = await sessions_b.get_or_create(session_id)
    persistence.flush_now(session_b)

    cwd = sessions_a.workspace_for(session_id)
    meta = cwd / ".chat_team"
    state_a = load_state(cwd, "session-engineer.json")
    state_b = load_state(cwd, "session-designer.json")
    assert state_a is not None, "session-engineer.json missing"
    assert state_b is not None, "session-designer.json missing"
    assert state_a["current_role"] == "engineer"
    assert state_b["current_role"] == "designer"
    print("[5] Isolated persistence: session-engineer.json and session-designer.json")
    print("    PASS")

    # -- Test 6: Solo mode isolation rule in system prompt --
    session_obj = await sessions_a.get_or_create("wecom-single-test")
    from chat_team.agent.agent import Agent
    from chat_team.skills.registry import SkillRegistry
    agent = Agent(
        role=role_a,
        session=session_obj,
        settings=settings,
        llm=ScriptedLLM([]),
        tools=build_tools(),
        skills=SkillRegistry({}),
    )
    sys_msgs = agent._build_system_messages()
    sys_text = sys_msgs[0].content
    assert "其他机器人各自维护自己的对话" in sys_text, f"isolation rule not found in: {sys_text[:200]}"
    assert "切换员工后" not in sys_text
    print("[6] Solo mode isolation rule in system prompt")
    print("    PASS")

    print("\n=== All solo mode smoke tests passed ===")


if __name__ == "__main__":
    asyncio.run(main())
