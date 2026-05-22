"""Load configuration from ``~/.chat_team/config.yaml`` and ``~/.chat_team/.env``."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .paths import Paths, init_home


@dataclass
class SessionConfig:
    msgid_lru_size: int = 500
    persistence_debounce_seconds: float = 10.0
    per_turn_transfer_cap: int = 3


@dataclass
class NotebookConfig:
    max_bytes: int = 4096


@dataclass
class ToolsConfig:
    file_read_max_bytes: int = 1_048_576
    file_write_max_bytes: int = 1_048_576
    shell_timeout_seconds: int = 30
    shell_output_max_bytes: int = 8192


@dataclass
class LLMConfig:
    provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    default_temperature: float = 0.3
    default_history_token_budget: int = 12000


@dataclass
class LoggingConfig:
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass
class Settings:
    paths: Paths
    workspace_root: Path
    default_role: str = "team_admin"
    log_level: str = "INFO"
    session: SessionConfig = field(default_factory=SessionConfig)
    notebook: NotebookConfig = field(default_factory=NotebookConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    env: dict[str, str] = field(default_factory=dict)

    def get_env(self, key: str, default: str | None = None) -> str | None:
        return self.env.get(key) or os.environ.get(key) or default


def _coerce(target: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if hasattr(target, key):
            setattr(target, key, value)


def load_settings(paths: Paths | None = None) -> Settings:
    paths = paths or init_home()
    load_dotenv(paths.dotenv, override=False)

    raw: dict[str, Any] = {}
    if paths.config_yaml.exists():
        loaded = yaml.safe_load(paths.config_yaml.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            raw = loaded

    workspace_root_raw = raw.get("workspace_root", "workspaces")
    workspace_root = Path(workspace_root_raw)
    if not workspace_root.is_absolute():
        workspace_root = paths.home / workspace_root

    settings = Settings(
        paths=paths,
        workspace_root=workspace_root,
        default_role=raw.get("default_role", "team_admin"),
        log_level=raw.get("log_level", "INFO"),
    )
    if isinstance(raw.get("session"), dict):
        _coerce(settings.session, raw["session"])
    if isinstance(raw.get("notebook"), dict):
        _coerce(settings.notebook, raw["notebook"])
    if isinstance(raw.get("tools"), dict):
        _coerce(settings.tools, raw["tools"])
    if isinstance(raw.get("llm"), dict):
        _coerce(settings.llm, raw["llm"])
    if isinstance(raw.get("logging"), dict):
        _coerce(settings.logging, raw["logging"])

    settings.env = {
        k: v for k, v in os.environ.items()
        if k.startswith(("WECOM_", "OPENAI_", "CHAT_TEAM_"))
    }
    workspace_root.mkdir(parents=True, exist_ok=True)
    return settings
