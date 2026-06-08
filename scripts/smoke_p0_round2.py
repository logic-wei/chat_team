"""Smoke for the 2nd batch of P0 fixes:

1. ``WeComBotAdapter._spawn_bg`` keeps a strong ref to in-flight callbacks
   so asyncio's weak-ref-only Task table can't GC them mid-await.
2. ``SessionManager.get_or_create`` is async-safe: N concurrent calls for
   the same fresh session_id all return the *same* Session object (and
   therefore the same ``session.lock``).
3. ``SessionManager._evict_if_needed`` does the pre-eviction flush in a
   worker thread, so a slow ``flush_now`` cannot freeze the event loop.
4. ``Dispatcher.handle`` calls ``stream.finish`` BEFORE ``_post_turn`` —
   the user must see the reply before compaction's LLM round-trip and the
   persistence snapshot run.
"""
from __future__ import annotations

import asyncio
import gc
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_p0_round2"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.adapters.wecom import WeComBotAdapter
from chat_team.agent.agent import Agent
from chat_team.agent.tools.base import ToolRegistry
from chat_team.config import load_settings
from chat_team.dispatcher import Dispatcher
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager
from chat_team.session.persistence import PersistenceManager


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class CapturingStream:
    def __init__(self, on_finish=None):
        self.statuses, self.final = [], None
        self._on_finish = on_finish
    async def push(self, chunk, *, append=True): pass
    async def status(self, note): self.statuses.append(note)
    async def finish(self, text):
        self.final = text
        if self._on_finish is not None:
            self._on_finish()


class ScriptedLLM(LLMProvider):
    def __init__(self, replies, on_call=None):
        self.replies = list(replies)
        self.calls = 0
        self._on_call = on_call

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if self._on_call is not None:
            self._on_call(request)
        if not self.replies:
            raise RuntimeError("ScriptedLLM exhausted")
        return self.replies.pop(0)


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
# 1. _spawn_bg holds a strong ref
# --------------------------------------------------------------------------

async def test_spawn_bg_holds_strong_ref():
    print("== test 1: _spawn_bg keeps in-flight callbacks alive across GC ==")
    settings = load_settings()
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="S")

    completed = {"n": 0}

    async def slow_callback(idx: int):
        # Long enough that GC could plausibly run before completion.
        await asyncio.sleep(0.05)
        completed["n"] += 1

    # Spawn several callbacks but throw away local references to the Task
    # returned — only adapter._bg_tasks should be holding them alive.
    for i in range(5):
        adapter._spawn_bg(slow_callback(i), name=f"cb-{i}")

    # Force several GC cycles immediately. Pre-fix, this could collect the
    # Task wrappers because asyncio only weakly tracks tasks.
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)

    assert len(adapter._bg_tasks) == 5, (
        f"_bg_tasks shrank under GC: {len(adapter._bg_tasks)}"
    )
    print(f"  in-flight refs held: {len(adapter._bg_tasks)}")

    # Drain — all five must complete.
    await asyncio.sleep(0.1)
    assert completed["n"] == 5, f"only {completed['n']}/5 callbacks completed"
    # done_callback should have released the refs.
    assert len(adapter._bg_tasks) == 0, (
        f"_bg_tasks not cleared after completion: {len(adapter._bg_tasks)}"
    )
    print("  ✓ all 5 callbacks completed; _bg_tasks cleared via done_callback")


# --------------------------------------------------------------------------
# 2. Concurrent get_or_create returns one Session
# --------------------------------------------------------------------------

async def test_concurrent_get_or_create_same_session():
    print("== test 2: concurrent get_or_create for same fresh sid → one Session ==")
    settings = load_settings()
    mgr = SessionManager(settings)

    sid = "concurrent-fresh-sid"
    # Fire 20 concurrent creates of the same brand-new session.
    results = await asyncio.gather(*[mgr.get_or_create(sid) for _ in range(20)])

    first = results[0]
    for i, r in enumerate(results[1:], start=1):
        assert r is first, f"result #{i} is a different Session object"
        assert r.lock is first.lock, f"result #{i} has a different lock"
    assert len(mgr.known_sessions()) == 1, mgr.known_sessions()
    print(f"  ✓ 20 concurrent creates returned 1 Session (id={id(first)})")

    # Same-lock invariant matters because two competing Sessions with two
    # different locks would let two turns for the same conversation run in
    # parallel — exactly the race the fix prevents. Validate by acquiring
    # the lock on `first` and confirming a wait on `results[-1].lock` blocks.
    blocked = {"hit": False}

    async def try_lock():
        async with results[-1].lock:
            blocked["hit"] = True

    await first.lock.acquire()
    try:
        t = asyncio.create_task(try_lock())
        await asyncio.sleep(0.05)
        assert not blocked["hit"], "results[-1].lock did not serialise against first.lock"
    finally:
        first.lock.release()
    await t
    assert blocked["hit"]
    print("  ✓ all returned objects share the SAME asyncio.Lock instance")


# --------------------------------------------------------------------------
# 3. Eviction does not block the event loop
# --------------------------------------------------------------------------

async def test_eviction_does_not_block_event_loop():
    print("== test 3: _evict_if_needed flushes in worker thread, loop stays live ==")
    settings = load_settings()
    settings.session.max_in_memory_sessions = 1

    # A persistence stub whose flush_now blocks for 0.3s of WALL time. The
    # fix wraps this in asyncio.to_thread, so a parallel sleep(0) ticker
    # should keep firing while flush_now is running.
    class SlowPersistence:
        def __init__(self):
            self.calls = 0

        def flush_now(self, session) -> None:
            self.calls += 1
            time.sleep(0.3)                   # blocking sleep; pre-fix this froze the loop

    sp = SlowPersistence()
    mgr = SessionManager(settings, persistence=sp)
    await mgr.get_or_create("seed-session")

    ticks = {"n": 0}
    stop = asyncio.Event()

    async def ticker():
        # If the event loop is frozen during the blocking flush, this won't
        # tick. We expect many ticks across the ~0.3s flush window.
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0.01)

    tk = asyncio.create_task(ticker())
    t0 = time.monotonic()
    # Trigger eviction of "seed-session" by adding a second one over cap.
    await mgr.get_or_create("evictor-session")
    elapsed = time.monotonic() - t0
    stop.set()
    await tk

    assert sp.calls == 1, f"flush_now should run once, ran {sp.calls}"
    # The 0.3s flush is in a worker thread, so get_or_create itself awaits
    # it but the loop kept ticking. We expect ≥10 ticks across the window.
    assert ticks["n"] >= 10, (
        f"event loop appears frozen: only {ticks['n']} ticks in {elapsed:.2f}s"
    )
    print(f"  ✓ flush ran ({elapsed:.2f}s wall), loop ticked {ticks['n']} times during it")


# --------------------------------------------------------------------------
# 4. stream.finish runs BEFORE _post_turn (compaction + persistence)
# --------------------------------------------------------------------------

async def test_finish_before_post_turn():
    print("== test 4: stream.finish completes before _post_turn work ==")
    settings = load_settings()
    settings.session.persistence_debounce_seconds = 0.01

    roles = RoleRegistry.load(settings.paths.user_roles_dir)

    timeline: list[tuple[float, str]] = []
    def mark(label: str) -> None:
        timeline.append((time.monotonic(), label))

    def on_llm_call(req):
        mark("llm:agent")

    llm = ScriptedLLM(replies=[reply("您好,我已收到。")], on_call=on_llm_call)

    # Hook into the dispatcher's compactor entrypoint. The point of P0 #4 is
    # that compaction (which can do an LLM round-trip per agent) must NOT
    # block stream.finish. Make our fake compaction take real time so any
    # ordering bug would be obvious.
    from chat_team import dispatcher as dispatcher_mod
    original_compact = dispatcher_mod.maybe_compact

    async def hooked_compact(agent, llm_):
        mark("compact:start")
        await asyncio.sleep(0.05)
        mark("compact:end")
        return False

    dispatcher_mod.maybe_compact = hooked_compact

    class TimedPersistence(PersistenceManager):
        def schedule(self, session):
            mark("persist:schedule")
            super().schedule(session)

    try:
        pm = TimedPersistence(settings)
        tools = ToolRegistry()
        sessions = SessionManager(settings, persistence=pm)
        disp = Dispatcher(settings, sessions, roles, tools, llm, persistence=pm)
        stream = CapturingStream(on_finish=lambda: mark("stream:finish"))
        await disp.handle(imsg("sess-order", "hi"), stream)
    finally:
        dispatcher_mod.maybe_compact = original_compact

    labels = [lbl for _, lbl in timeline]
    print("  timeline:", labels)

    # main-turn LLM call must come first (before finish).
    assert labels.index("llm:agent") < labels.index("stream:finish")
    # finish must precede compaction (which lives inside _post_turn) — the
    # whole point of P0 #4 is that the user sees the reply before we spend
    # an LLM round-trip on summarisation.
    assert labels.index("stream:finish") < labels.index("compact:start"), (
        f"stream.finish was NOT called before compaction started: {labels}"
    )
    # finish must also precede persistence.schedule (also inside _post_turn).
    assert labels.index("stream:finish") < labels.index("persist:schedule"), (
        f"stream.finish was NOT called before persistence.schedule: {labels}"
    )
    print("  ✓ ordering: llm:agent → stream:finish → compact:* → persist:schedule")


# --------------------------------------------------------------------------

async def main():
    await test_spawn_bg_holds_strong_ref()
    await test_concurrent_get_or_create_same_session()
    await test_eviction_does_not_block_event_loop()
    await test_finish_before_post_turn()
    print("\nALL P0 ROUND-2 SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
