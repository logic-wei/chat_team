"""SessionManager: lookup-or-create per-conversation Session objects.

Two ops layered on top of the basic cache:

* **LRU eviction** — ``max_in_memory_sessions`` caps the in-memory dict.
  When exceeded, the least-recently-used Session is flushed to its own
  ``session.json`` via the injected :class:`PersistenceManager` and then
  dropped. On the user's next message, ``load_state`` + ``restored_histories``
  rebuild it transparently.

* **Lazy file sweep** — first ``get_or_create`` per session (and once every
  ``sweep_interval_hours`` thereafter) walks ``inbox/``, ``.chat_team/runs/``,
  and ``.chat_team/llm/`` and unlinks files older than ``max_age_days``.
  Without this the workspace grows forever.
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings
from ..paths import sanitize_session_id
from .notebook import Notebook
from .persistence import load_state, restored_histories
from .session import Session

if TYPE_CHECKING:
    from .persistence import PersistenceManager

log = logging.getLogger(__name__)


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        persistence: "PersistenceManager | None" = None,
    ):
        self.settings = settings
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        # Optional — only used to force a synchronous flush before evicting
        # an LRU entry. SessionManager works without it (the eviction just
        # drops the in-memory copy and the next access reloads from disk;
        # any unflushed delta is lost, so always wire it in production).
        self._persistence = persistence
        self._last_sweep: dict[str, float] = {}

    def workspace_for(self, session_id: str) -> Path:
        return self.settings.workspace_root / sanitize_session_id(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            session = self._sessions[session_id]
            self._maybe_sweep(session)
            return session

        cwd = self.workspace_for(session_id)
        cwd.mkdir(parents=True, exist_ok=True)
        meta = cwd / ".chat_team"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "runs").mkdir(parents=True, exist_ok=True)

        notebook = Notebook(meta / "notebook.md", max_bytes=self.settings.notebook.max_bytes)

        # Restore prior current_role + histories from session.json (if any).
        prior = load_state(cwd) or {}
        current_role = prior.get("current_role") or self.settings.default_role
        prior_histories = restored_histories(cwd)

        session = Session(
            session_id=session_id,
            cwd=cwd,
            current_role=current_role,
            notebook=notebook,
            restored_histories=prior_histories,
        )
        self._sessions[session_id] = session
        self._evict_if_needed()
        self._maybe_sweep(session)
        return session

    def known_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    # ---- internals --------------------------------------------------------

    def _evict_if_needed(self) -> None:
        cap = self.settings.session.max_in_memory_sessions
        if cap <= 0:
            return
        while len(self._sessions) > cap:
            sid, victim = self._sessions.popitem(last=False)
            self._last_sweep.pop(sid, None)
            if self._persistence is not None:
                try:
                    self._persistence.flush_now(victim)
                except Exception:                                # noqa: BLE001
                    log.exception("flush-on-evict failed for %s", sid)
            log.info(
                "evicted session %s (in-memory cap %d reached)", sid, cap,
            )

    def _maybe_sweep(self, session: Session) -> None:
        cfg = self.settings.cleanup
        if cfg.max_age_days <= 0:
            return
        now = time.monotonic()
        last = self._last_sweep.get(session.session_id)
        if last is not None and (now - last) < cfg.sweep_interval_hours * 3600.0:
            return
        self._last_sweep[session.session_id] = now
        cutoff = time.time() - cfg.max_age_days * 86400.0
        for rel in cfg.sweep_subdirs:
            target = session.cwd / rel
            try:
                _unlink_older_than(target, cutoff)
            except Exception:                                    # noqa: BLE001
                log.warning(
                    "sweep failed for %s in session %s",
                    target, session.session_id,
                    exc_info=True,
                )


def _unlink_older_than(directory: Path, cutoff_ts: float) -> None:
    """Best-effort: unlink files in ``directory`` whose mtime is older than
    ``cutoff_ts`` (epoch seconds). Subdirectories are not recursed into —
    the sweep targets are flat directories by design. Missing directory is
    silently ignored. Per-file errors are swallowed so one bad inode doesn't
    abort the rest of the sweep."""
    if not directory.exists():
        return
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    if entry.stat(follow_symlinks=False).st_mtime < cutoff_ts:
                        os.unlink(entry.path)
                except OSError:
                    continue
    except FileNotFoundError:
        return
