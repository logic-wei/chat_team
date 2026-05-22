"""Resolve and initialise the global runtime directory ``~/.chat_team``.

Default config + .env templates live as files under ``chat_team/templates/``
so they're visible in the source tree and editable without touching code.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


def _load_template(name: str) -> str:
    return resources.files("chat_team.templates").joinpath(name).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Paths:
    home: Path
    config_yaml: Path
    dotenv: Path
    user_roles_dir: Path
    workspaces_dir: Path
    logs_dir: Path
    state_dir: Path

    def session_workspace(self, session_id: str) -> Path:
        safe = sanitize_session_id(session_id)
        return self.workspaces_dir / safe


_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_session_id(session_id: str) -> str:
    """Map an arbitrary session id to a filesystem-safe directory name."""
    cleaned = _SAFE_RE.sub("_", session_id).strip("._")
    return cleaned or "default"


def resolve_home() -> Path:
    override = os.environ.get("CHAT_TEAM_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".chat_team"


def init_home(home: Path | None = None) -> Paths:
    """Create ~/.chat_team and seed default config/.env on first run."""
    root = (home or resolve_home())
    root.mkdir(parents=True, exist_ok=True)

    paths = Paths(
        home=root,
        config_yaml=root / "config.yaml",
        dotenv=root / ".env",
        user_roles_dir=root / "roles",
        workspaces_dir=root / "workspaces",
        logs_dir=root / "logs",
        state_dir=root / "state",
    )

    for d in (paths.user_roles_dir, paths.workspaces_dir, paths.logs_dir, paths.state_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not paths.config_yaml.exists():
        paths.config_yaml.write_text(_load_template("config.yaml"), encoding="utf-8")
    if not paths.dotenv.exists():
        paths.dotenv.write_text(_load_template("env.template"), encoding="utf-8")
        try:
            os.chmod(paths.dotenv, 0o600)
        except OSError:
            pass

    return paths
