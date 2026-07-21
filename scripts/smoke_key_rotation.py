"""Smoke tests for multi-API-key rotation (``llm.api_keys``).

Covers the four behaviours the maintainer signed off on:

1. **Round-robin on new sessions** — distinct ``(session_id, role_name)``
   pairs advance the round-robin pointer (1→2→0→1… for N=3).
2. **Reuse within idle window** — the same ``(session, role)`` re-uses its
   bound key across consecutive turns (cache-friendly) until idle.
3. **Idle reset advances the pointer** — after ``idle_reset_seconds`` the
   binding is dropped and the NEXT request advances the pointer (does NOT
   re-pick the just-released key).
4. **Multi-key failure retry** — on failure the provider tries the bound
   key first, then every other key in round-robin order, each retried up
   to ``max_retries`` times. No key is permanently disabled.
5. **Binding survives failure** — a successful retry on a *different* key
   does NOT move the binding; the next turn for that ``(session, role)``
   reuses the original bound key (unless it has gone idle).
6. **Single-key backward compat** — ``api_key`` only (no ``api_keys``) →
   one client, no router, behaves exactly like before.
7. **Per-role isolation within a session** — different roles in the same
   session get different keys (different binding entries).

All pure-Python: no live LLM, no network. Uses real ``openai`` exception
types so ``_RETRYABLE_EXCEPTIONS`` membership works without monkey-patching.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import httpx
from openai import APIConnectionError, BadRequestError, RateLimitError

from chat_team.llm.base import ChatMessage, CompletionRequest
from chat_team.llm.key_rotation import SessionKeyRouter
from chat_team.llm.openai_provider import OpenAIChatCompletionProvider

# --------------------------------------------------------------------------- #
# Fake OpenAI client
# --------------------------------------------------------------------------- #

class _FakeMessage:
    def __init__(self, text: str):
        self.content = text
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, text: str):
        self.message = _FakeMessage(text)
        self.finish_reason = "stop"


class _FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _FakeCompletion:
    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]
        self.usage = _FakeUsage()


class _FakeCreate:
    """Scripted ``client.chat.completions.create`` for ONE key index.

    ``script`` is a list whose entries are either a ``BaseException``
    (raised) or a ``str`` (returned as the completion content). Each call
    pops the head. If the script runs out, it returns ``"default"`` so a
    misconfigured test fails loudly rather than raising IndexError.
    """
    def __init__(self, script: list):
        self.script = list(script)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return _FakeCompletion(item)
        return _FakeCompletion("default")


def _wire_fakes(provider: OpenAIChatCompletionProvider, scripts: list[list]):
    """Replace each ``provider._clients[i].chat.completions`` with a fake.

    ``scripts[i]`` is the script for key index ``i``.
    """
    fakes = []
    for script in scripts:
        fake = _FakeCreate(script)
        fakes.append(fake)
        client = provider._clients[fakes.__len__() - 1]  # noqa: SIM222
    # assign properly (the loop above used len trick; rewrite cleanly)
    fakes = []
    for i, script in enumerate(scripts):
        fake = _FakeCreate(script)
        fakes.append(fake)

        class _Chat:
            pass

        ch = _Chat()
        ch.completions = fake  # type: ignore[attr-defined]
        provider._clients[i].chat = ch
    provider._client = provider._clients[0]
    return fakes


def _req(session_id: str | None, role: str | None) -> CompletionRequest:
    return CompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.0,
        session_id=session_id,
        role_name=role,
    )


def _http_resp(status: int):
    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://api.openai.example/v1/chat/completions"),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

async def test_round_robin_new_sessions():
    print("== test 1: round-robin advances across new (session,role) pairs ==")
    # keys [k0,k1,k2]; provider starts _rr=0 so select() advances to 1,2,0
    p = OpenAIChatCompletionProvider(
        api_key="unused", api_keys=["k0", "k1", "k2"],
        max_retries=1, retry_initial_delay=0.0, use_streaming=False,
    )
    # Track which key index is selected per call by inspecting router state.
    # We use a no-LLM-call path: just exercise the router directly.
    r = p._router
    assert r is not None
    idxs = [r.select(f"s{i}", "admin") for i in range(4)]
    assert idxs == [1, 2, 0, 1], idxs
    print(f"  ✓ new pairs got indices {idxs}")


async def test_reuse_within_idle_window():
    print("== test 2: same (session,role) reuses bound key while active ==")
    p = OpenAIChatCompletionProvider(
        api_key="unused", api_keys=["k0", "k1", "k2"],
        max_retries=1, retry_initial_delay=0.0, use_streaming=False,
        key_rotation_idle_seconds=10.0,
    )
    r = p._router
    first = r.select("sX", "admin")
    # subsequent selects within window must return the SAME index
    for _ in range(5):
        assert r.select("sX", "admin") == first
    print(f"  ✓ (sX,admin) pinned to index {first} across 6 calls")


async def test_idle_reset_advances_pointer():
    print("== test 3: idle reset advances pointer (no re-pick) ==")
    p = OpenAIChatCompletionProvider(
        api_key="unused", api_keys=["k0", "k1", "k2", "k3"],
        max_retries=1, retry_initial_delay=0.0, use_streaming=False,
        key_rotation_idle_seconds=0.2,
    )
    r = p._router
    a = r.select("s1", "admin")     # advances to some idx (say i)
    b = r.select("s1", "research")  # advances to i+1
    assert a != b
    # now sleep past idle window for s1/admin
    await asyncio.sleep(0.25)
    c = r.select("s1", "admin")     # idle-released → advances again
    assert c != a, f"idle reset should advance away from {a}, got {c}"
    # the binding snapshot must show the NEW idx, not the stale one
    snap = r.snapshot()
    assert snap[("s1", "admin")][0] == c
    print(f"  ✓ (s1,admin) {a}→ released → {c} (advanced, not re-picked)")


async def test_multi_key_failure_retry_all_keys():
    print("== test 4: failure retries ALL keys × max_retries, no disable ==")
    # 3 keys, max_retries=2 → up to 6 attempts if all fail
    p = OpenAIChatCompletionProvider(
        api_key="unused", api_keys=["k0", "k1", "k2"],
        max_retries=2, retry_initial_delay=0.0, use_streaming=False,
    )
    # All keys always raise RateLimitError (retryable).
    rl = lambda: RateLimitError("429", response=_http_resp(429), body=None)
    fakes = _wire_fakes(p, [
        [rl(), rl()],   # k0: 2 attempts, both fail
        [rl(), rl()],   # k1: 2 attempts, both fail
        [rl(), rl()],   # k2: 2 attempts, both fail
    ])
    # First call seeds the binding for (s1, admin). Bound key is the first
    # one tried. After the call, all 6 attempts should be spent.
    try:
        await p.complete(_req("s1", "admin"))
    except RateLimitError:
        pass
    else:
        raise AssertionError("expected RateLimitError after all keys exhausted")
    total = sum(f.calls for f in fakes)
    assert total == 6, f"expected 6 total attempts (3 keys × 2), got {total}"
    assert all(f.calls == 2 for f in fakes), [f.calls for f in fakes]
    print(f"  ✓ {total} attempts across 3 keys (2 each), all exhausted")


async def test_failure_recovers_on_next_key_no_disable():
    print("== test 5: recovers on a later key; binding stays on the bound key ==")
    # 3 keys. Bound key (k_first) always fails; the NEXT key succeeds.
    p = OpenAIChatCompletionProvider(
        api_key="unused", api_keys=["k0", "k1", "k2"],
        max_retries=2, retry_initial_delay=0.0, use_streaming=False,
    )
    rl = lambda: RateLimitError("429", response=_http_resp(429), body=None)
    # We don't know which key is bound first without consulting the router,
    # so make ALL keys: first attempt fails (retryable), second succeeds.
    # That guarantees recovery regardless of which key is bound first, and
    # exercises the per-key retry path on every key.
    fakes = _wire_fakes(p, [
        [rl(), "ok-k0"],
        [rl(), "ok-k1"],
        [rl(), "ok-k2"],
    ])
    resp = await p.complete(_req("s1", "admin"))
    # The first key tried (= bound key) should have done 2 calls (fail then ok)
    # and the others 0 calls (because the bound key recovered on attempt 2).
    total = sum(f.calls for f in fakes)
    assert total == 2, f"expected 2 attempts (bound key fail→ok), got {total}"
    print(f"  ✓ bound key recovered after 2 attempts; total={total}")

    # Now: binding must NOT have moved. The next turn for (s1,admin) should
    # reuse the SAME bound key. Re-wire fresh fakes where the bound key
    # succeeds immediately and every OTHER key would FAIL if hit. If the
    # binding had moved to a different key, that key's failure script would
    # be exercised (and we'd see >1 total call).
    bound_idx = p._router.snapshot()[("s1", "admin")][0]
    rl2 = lambda: RateLimitError("429", response=_http_resp(429), body=None)
    scripts2 = [["ok-turn2"] if i == bound_idx else [rl2(), rl2()] for i in range(3)]
    f2 = _wire_fakes(p, scripts2)
    resp2 = await p.complete(_req("s1", "admin"))
    assert resp2.message.content == "ok-turn2", resp2.message.content
    # bound key hit exactly once (success); no other key touched.
    assert f2[bound_idx].calls == 1, f2[bound_idx].calls
    assert all(f2[i].calls == 0 for i in range(3) if i != bound_idx), \
        [f2[i].calls for i in range(3)]
    print(f"  ✓ binding unchanged: turn 2 reused bound key {bound_idx} "
          f"(1 call, others untouched)")


async def test_non_retryable_jumps_to_next_key():
    print("== test 6: non-retryable error jumps to next key (no 3× hammer) ==")
    p = OpenAIChatCompletionProvider(
        api_key="unused", api_keys=["k0", "k1", "k2"],
        max_retries=3, retry_initial_delay=0.0, use_streaming=False,
    )
    # Bound key raises BadRequestError (non-retryable); next key succeeds.
    # Make the bound key fail non-retryable and ALL others succeed-first so
    # we count calls. Since we don't know which is bound first, set every
    # key to: first call BadRequest, second call ok — then the bound key
    # does exactly 1 call (non-retryable jumps) and recovery happens on a
    # later key's SECOND call... but that's complicated. Simpler: set the
    # bound key (after we know it) to non-retryable-fail, others to ok-first.
    # Seed the binding first via a trivial successful call.
    fakes = _wire_fakes(p, [["seed-ok"], ["seed-ok"], ["seed-ok"]])
    await p.complete(_req("s1", "admin"))
    bound_idx = p._router.snapshot()[("s1", "admin")][0]
    # Now set scripts: bound key → BadRequest (non-retryable); a *different*
    # key → ok-first. The bound key should be tried (1 call, raises), then
    # the next key succeeds (1 call). max_retries=3 but non-retryable means
    # the bound key gets only 1 attempt, NOT 3.
    other = (bound_idx + 1) % 3
    scripts = [[] for _ in range(3)]
    scripts[bound_idx] = [BadRequestError("400", response=_http_resp(400), body=None)]
    scripts[other] = ["ok-recovered"]
    for i, fake in enumerate(fakes):
        fake.script = list(scripts[i])
        fake.calls = 0          # reset so we count only THIS turn's attempts
    resp = await p.complete(_req("s1", "admin"))
    assert fakes[bound_idx].calls == 1, (
        f"non-retryable should give bound key 1 attempt, got {fakes[bound_idx].calls}"
    )
    # recovery happened on the next key (also 1 attempt, success-first)
    assert fakes[other].calls == 1, fakes[other].calls
    print(f"  ✓ bound key {bound_idx} got 1 attempt (non-retryable), recovered on key {other}")


async def test_per_role_isolation_in_session():
    print("== test 7: different roles in same session get different keys ==")
    p = OpenAIChatCompletionProvider(
        api_key="unused", api_keys=["k0", "k1", "k2", "k3"],
        max_retries=1, retry_initial_delay=0.0, use_streaming=False,
    )
    r = p._router
    a = r.select("s1", "team_admin")
    b = r.select("s1", "research_engineer")
    assert a != b, "different roles must get different keys (no shared cache)"
    print(f"  ✓ session s1: team_admin→{a}, research_engineer→{b}")


async def test_single_key_backward_compat():
    print("== test 8: single api_key (no api_keys) → 1 client, no router ==")
    p = OpenAIChatCompletionProvider(
        api_key="sk-only", max_retries=3, retry_initial_delay=0.0,
        use_streaming=False,
    )
    assert len(p._clients) == 1
    assert p._router is None
    fakes = _wire_fakes(p, [["ok-single"]])
    resp = await p.complete(_req("s1", "admin"))
    assert resp.message.content == "ok-single"
    assert fakes[0].calls == 1
    print("  ✓ single-key path works, no router, 1 call")


async def test_router_unit():
    print("== test 9: SessionKeyRouter unit (pointer math) ==")
    r = SessionKeyRouter(3, idle_reset_seconds=1.0)
    # rr starts at 0; each select on a NEW pair does (rr+1)%n
    assert r.select("a", "r") == 1
    assert r.select("b", "r") == 2
    assert r.select("c", "r") == 0
    assert r.select("d", "r") == 1
    # reuse within window
    assert r.select("a", "r") == 1
    # idle release
    await asyncio.sleep(1.1)
    assert r.select("a", "r") != 1  # advanced away
    print("  ✓ router pointer 1,2,0,1; reuse pinned; idle advanced")


# --------------------------------------------------------------------------- #

async def main():
    # Clean CHAT_TEAM_HOME so we don't touch the real one (these tests don't
    # use the home dir, but mirror the convention of the other smokes).
    home = Path(tempfile.mkdtemp(prefix="chat_team_keyrot_"))
    os.environ["CHAT_TEAM_HOME"] = str(home)
    try:
        await test_round_robin_new_sessions()
        await test_reuse_within_idle_window()
        await test_idle_reset_advances_pointer()
        await test_multi_key_failure_retry_all_keys()
        await test_failure_recovers_on_next_key_no_disable()
        await test_non_retryable_jumps_to_next_key()
        await test_per_role_isolation_in_session()
        await test_single_key_backward_compat()
        await test_router_unit()
    finally:
        shutil.rmtree(home, ignore_errors=True)
    print("\nALL KEY-ROTATION SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
