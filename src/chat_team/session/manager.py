"""SessionManager: lookup-or-create per-conversation Session objects."""
from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..paths import sanitize_session_id
from .notebook import Notebook
from .persistence import load_state, restored_histories
from .session import Session


class SessionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions: dict[str, Session] = {}

    def workspace_for(self, session_id: str) -> Path:
        return self.settings.workspace_root / sanitize_session_id(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if session_id in self._sessions:
            return self._sessions[session_id]

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
        return session

    def known_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())
