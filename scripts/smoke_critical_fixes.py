"""Smoke for the four severe-bug fixes:

1. ``PersistenceManager.schedule`` snapshots SYNCHRONOUSLY under the caller's
   session lock — mutations after schedule() must NOT leak into the eventual
   on-disk file.
2. ``Agent.handle`` rolls back the entire turn (user msg + any partial
   assistant/tool messages) when the LLM raises, so history stays well-formed.
3. Same rollback when the LLM fails mid-tool-loop (after a successful tool call).
4. ``Dispatcher.handle`` keeps running ``_post_turn`` (compact + persist) and
   pushes a fallback ``stream.finish`` even when ``_run_turn`` raises.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_critical_fixes_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.agent.agent import Agent
from chat_team.agent.tools.base import Tool, ToolContext, ToolRegistry
from chat_team.config import load_settings
from chat_team.dispatcher import Dispatcher
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager
from chat_team.session.persistence import PersistenceManager, load_state


class CapturingStream:
    def __init__(self):
        self.statuses, self.final = [], None
    async def push(self, chunk, *, append=True): pass
    async def status(self, note): self.statuses.append(note)
    async def finish(self, text): self.final = text


class FailingLLM(LLMProvider):
    """Raises on the Nth call; replies normally otherwise."""

    def __init__(self, fail_on: list[int], replies: list[CompletionResponse] | None = None):
        self.fail_on = set(fail_on)
        self.replies = list(replies or [])
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError(f"simulated LLM failure on call #{self.calls}")
        if self.replies:
            return self.replies.pop(0)
        return CompletionResponse(
            message=ChatMessage(role="assistant", content="ok"),
            finish_reason="stop",
        )


class TouchTool(Tool):
    name = "touch"
    description = "no-op"
    parameters = {"type": "object", "properties": {}}
    async def run(self, ctx: ToolContext, **kwargs):
        return "touched"


def call_tool(name: str, call_id: str = "tc-1") -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments={})],
        ),
        finish_reason="tool_calls",
    )


def reply(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def imsg(sid: str, text: str) -> IncomingMessage:
    return IncomingMessage(
        session_id=sid, chat_type=ChatType.SINGLE, user_id="u",
        text=text, msg_id=f"m-{text[:8]}", bot_id="bot",
    )


# --------------------------------------------------------------------------
# 1. Persistence snapshot is frozen at schedule() time
# --------------------------------------------------------------------------

async def test_persistence_snapshot_under_lock():
    print("== test 1: schedule() snapshots synchronously, not at flush time ==")
    settings = load_settings()
    settings.session.persistence_debounce_seconds = 0.2

    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-snap")

    # Materialise an agent and pre-populate history with a known marker.
    role = roles.get("team_admin")
    llm = FailingLLM(fail_on=[])
    agent = Agent(role=role, session=sess, settings=settings, llm=llm, tools=tools)
    agent.history.append(ChatMessage(role="user", content="MARKER_AT_SCHEDULE"))
    sess.agents_by_role["team_admin"] = agent
    sess.current_role = "team_admin"

    pm = PersistenceManager(settings)

    # Schedule under "lock held" semantics (we call it directly, mimicking
    # what Dispatcher._post_turn does inside session.lock).
    pm.schedule(sess)

    # Now mutate the live history — these mutations must NOT appear on disk
    # because the snapshot was supposed to be taken synchronously above.
    agent.history.append(ChatMessage(role="assistant", content="MUTATED_AFTER_SCHEDULE"))
    agent.history.append(ChatMessage(role="user", content="ALSO_AFTER"))

    # Also stress-test by adding a brand new role/agent post-schedule —
    # would have raised RuntimeError("dictionary changed size during iteration")
    # under the old code if the snapshot iterated agents_by_role at flush time.
    sess.agents_by_role["new_role"] = Agent(
        role=role, session=sess, settings=settings, llm=llm, tools=tools,
    )

    await asyncio.sleep(0.4)  # debounce + slack

    state = load_state(sess.cwd)
    assert state is not None, "session.json should exist after debounce"
    admin = state["histories"].get("team_admin", [])
    print(f"  on-disk admin history: {[m['content'] for m in admin]}")
    assert len(admin) == 1, f"expected 1 message in snapshot, got {len(admin)}"
    assert admin[0]["content"] == "MARKER_AT_SCHEDULE"
    assert "new_role" not in state["histories"], (
        "agent added after schedule must not appear in the snapshot"
    )
    print("  ✓ snapshot frozen at schedule() time; later mutations didn't leak")


# --------------------------------------------------------------------------
# 2. Agent rolls back on first-call LLM failure
# --------------------------------------------------------------------------

async def test_agent_rollback_first_call():
    print("== test 2: agent rolls back when first LLM call fails ==")
    settings = load_settings()
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-rollback-1")
    role = roles.get("team_admin")

    llm = FailingLLM(fail_on=[1])
    agent = Agent(role=role, session=sess, settings=settings, llm=llm, tools=tools)
    pre_len = len(agent.history)

    raised = False
    try:
        await agent.handle("hi", CapturingStream())
    except RuntimeError as e:
        assert "simulated" in str(e)
        raised = True
    assert raised, "agent.handle should re-raise the LLM failure"
    assert len(agent.history) == pre_len, (
        f"history should be rolled back to {pre_len}, got {len(agent.history)}"
    )
    print(f"  ✓ history clean after failed first call (len={len(agent.history)})")


# --------------------------------------------------------------------------
# 3. Agent rolls back when LLM fails mid-tool-loop
# --------------------------------------------------------------------------

async def test_agent_rollback_mid_loop():
    print("== test 3: agent rolls back full turn when 2nd LLM call fails ==")
    settings = load_settings()
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    tools.register(TouchTool())
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-rollback-2")
    role = roles.get("team_admin")
    role.tools = list(set(role.tools + ["touch"]))

    # Call 1: returns a tool call to `touch`
    # Tool runs successfully
    # Call 2: raises
    llm = FailingLLM(fail_on=[2], replies=[call_tool("touch")])
    agent = Agent(role=role, session=sess, settings=settings, llm=llm, tools=tools)

    raised = False
    try:
        await agent.handle("do it", CapturingStream())
    except RuntimeError:
        raised = True
    assert raised
    assert len(agent.history) == 0, (
        f"history should be empty after rollback, got {len(agent.history)}: "
        f"{[m.role for m in agent.history]}"
    )
    print("  ✓ user + assistant(tool_calls) + tool all rolled back")


# --------------------------------------------------------------------------
# 4. Dispatcher persists + finishes stream even when _run_turn raises
# --------------------------------------------------------------------------

async def test_dispatcher_persists_on_error():
    print("== test 4: dispatcher post_turn + stream.finish run on _run_turn error ==")
    settings = load_settings()
    settings.session.persistence_debounce_seconds = 0.05

    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    pm = PersistenceManager(settings)
    sessions = SessionManager(settings)
    llm = FailingLLM(fail_on=[1])
    disp = Dispatcher(settings, sessions, roles, tools, llm, persistence=pm)

    s = CapturingStream()
    await disp.handle(imsg("sess-disp-err", "你好"), s)

    # stream.finish must have been called with the fallback message — the
    # adapter relies on this to close the WS reply.
    assert s.final is not None, "stream.finish was not called on error path"
    assert "出错" in s.final, f"expected fallback final, got: {s.final!r}"
    print(f"  ✓ stream.finish('{s.final}') called despite _run_turn raising")

    sess = await sessions.get_or_create("sess-disp-err")
    state_path = sess.cwd / ".chat_team" / "session.json"
    await asyncio.sleep(0.2)
    assert state_path.exists(), "persistence.schedule should have fired"

    state = load_state(sess.cwd)
    # Agent rolled back its history before the persist ran, so the failed
    # turn's user message must NOT be on disk — otherwise it would replay.
    admin = state["histories"].get("team_admin", []) if state else []
    assert all(
        "你好" not in (m.get("content") or "") for m in admin
    ), f"failed turn leaked into snapshot: {admin}"
    print("  ✓ session.json written; failed user msg NOT persisted")


async def main():
    await test_persistence_snapshot_under_lock()
    await test_agent_rollback_first_call()
    await test_agent_rollback_mid_loop()
    await test_dispatcher_persists_on_error()
    print("\nALL CRITICAL-FIX SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
