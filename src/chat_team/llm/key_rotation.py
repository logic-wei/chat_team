"""Per-session, per-role API-key rotation for the OpenAI provider.

The provider is constructed with an ordered list of API keys
(``llm.api_keys`` / ``llm.vision.api_keys``). Each
``(session_id, role_name)`` pair is bound to a single key on first use
and reuses that key for the life of the binding so the upstream prefix
cache stays warm across the role's multi-turn conversation within a
session (different roles in the same session have separate histories →
no shared cache → they get different keys, which is fine).

Bindings are in-memory only — never persisted, never hot-reloaded. A
binding is released after ``idle_reset_seconds`` (default 600s) of no
activity; the next request for that pair advances the round-robin
pointer (it does *not* re-pick the just-released key — the pointer
keeps moving forward).

Failure handling (see ``OpenAIChatCompletionProvider.complete``): the
router is *not* consulted on failure. When the bound key's call fails
the provider iterates the *other* keys (each retried ``max_retries``
times) before giving up — but the binding itself stays put: the bound
key is tried again on the *next* turn for that ``(session, role)``
unless it has since gone idle. This matches the maintainer's rule:
"once bound, always this key; only idle reset advances it; failures
never permanently disable a key."
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Binding:
    """One ``(session_id, role_name)`` → key-index binding.

    ``last_seen`` is monotonic-seconds of the most recent successful
    ``select()`` for this pair. A pair whose ``now - last_seen``
    exceeds ``idle_reset_seconds`` is dropped on the next ``select()``
    and re-bound to a freshly-advanced round-robin pointer.
    """
    idx: int
    last_seen: float


class SessionKeyRouter:
    """Thread-safe round-robin key binder keyed by ``(session_id, role_name)``.

    Not async-aware by design: the provider's ``complete()`` is the only
    caller, all under the asyncio event loop (single-threaded execution
    of the critical section). A lock is still used so a future non-async
    caller (or reload-thread inspection) can't observe a torn update.
    """

    def __init__(self, n_keys: int, *, idle_reset_seconds: float = 600.0) -> None:
        if n_keys < 1:
            raise ValueError(f"n_keys must be >= 1, got {n_keys}")
        self._n = n_keys
        self._idle = max(0.0, float(idle_reset_seconds))
        # session_id+role_name → binding. role_name may be None (boss CLI);
        # we still key on the pair so a None-role stream doesn't collide
        # with role-bearing streams of the same session.
        self._bindings: dict[tuple[str | None, str | None], _Binding] = {}
        self._rr = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @property
    def n_keys(self) -> int:
        return self._n

    @property
    def idle_reset_seconds(self) -> float:
        return self._idle

    # ------------------------------------------------------------------ #
    def _next_index(self) -> int:
        """Advance the round-robin pointer and return the new index.

        Always moves forward by one (mod n) — never re-picks the previous
        value, which is what gives "continue the rotation" semantics on
        idle release instead of "re-pick the just-released key".
        """
        self._rr = (self._rr + 1) % self._n
        return self._rr

    def select(
        self,
        session_id: str | None,
        role_name: str | None,
    ) -> int:
        """Return the key index for ``(session_id, role_name)``.

        * Existing & not idle → reuse (cache-friendly).
        * Existing & idle → drop, advance pointer, re-bind.
        * New pair → advance pointer, bind.

        Updates ``last_seen`` on every call.
        """
        now = time.monotonic()
        key = (session_id, role_name)
        with self._lock:
            b = self._bindings.get(key)
            if b is not None and (now - b.last_seen) < self._idle:
                b.last_seen = now
                return b.idx
            # Either no binding, or it went idle — advance and (re)bind.
            if b is not None:
                # Idle release: drop the stale binding so its old idx
                # doesn't leak into the next select on a different pair.
                del self._bindings[key]
            idx = self._next_index()
            self._bindings[key] = _Binding(idx=idx, last_seen=now)
            return idx

    # ------------------------------------------------------------------ #
    def reset_binding(
        self,
        session_id: str | None,
        role_name: str | None,
    ) -> None:
        """Forget the binding for a pair (no-op if absent).

        Currently unused by the provider (failures don't reset bindings,
        per the maintainer's rule) but exposed for tests and future
        "explicit logout" semantics.
        """
        with self._lock:
            self._bindings.pop((session_id, role_name), None)

    def snapshot(self) -> dict[tuple[str | None, str | None], tuple[int, float]]:
        """Read-only copy of bindings — for tests/diagnostics only."""
        with self._lock:
            return {k: (v.idx, v.last_seen) for k, v in self._bindings.items()}
