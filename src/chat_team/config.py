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
    # Hard cap on Sessions held in memory. When exceeded, the LRU entry is
    # flushed to session.json and evicted; the user transparently reloads
    # from disk on next message. Without this, every distinct user × bot
    # leaks a Session forever.
    max_in_memory_sessions: int = 1000


@dataclass
class NotebookConfig:
    max_bytes: int = 4096


@dataclass
class ToolsConfig:
    file_read_max_bytes: int = 1_048_576
    file_write_max_bytes: int = 1_048_576
    shell_timeout_seconds: int = 30
    shell_output_max_bytes: int = 8192
    # Extra env-var names dropped from the run_command subprocess on top of
    # the built-in deny-list (OPENAI_*, WECOM_*, ANTHROPIC_*, CHAT_TEAM_*,
    # plus anything containing KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL).
    shell_env_extra_drop: list[str] = field(default_factory=list)


@dataclass
class CleanupConfig:
    # Files older than this in inbox/, .chat_team/runs/, .chat_team/llm/
    # are unlinked on the lazy per-session sweep.
    max_age_days: int = 14
    # Minimum gap between two sweeps of the same session. Sweep is triggered
    # by get_or_create; this throttles thrash on chatty sessions.
    sweep_interval_hours: float = 6.0
    sweep_subdirs: list[str] = field(
        default_factory=lambda: ["inbox", ".chat_team/runs", ".chat_team/llm"]
    )


@dataclass
class LLMConfig:
    provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    default_temperature: float = 0.3
    default_history_token_budget: int = 12000
    default_image_detail: str = "high"   # "low" | "high" | "auto"; consulted when role doesn't set image_detail
    # Vision handling strategy: "tool" runs an eager OCR/describe shim on
    # inbound images and feeds the agent text only; "direct" hands the raw
    # image blocks to the provider every turn (legacy behaviour).
    default_vision_strategy: str = "tool"
    # Optional override for the model used by the eager shim and the
    # describe_image tool. Empty → reuse default_model.
    default_vision_model: str = ""
    # Detail level used by the eager shim. OCR needs "high" to read small
    # printed text reliably; agents can still call describe_image with a
    # different detail to opt in to a cheaper or finer pass.
    default_eager_detail: str = "high"
    # Prompt used by the eager shim. Empty → fall back to the OCR-with-
    # fallback prompt defined in runtime.vision_shim.DEFAULT_OCR_PROMPT.
    default_eager_prompt: str = ""
    # When true, every OpenAI provider call writes a JSON file to
    # <workspace>/.chat_team/llm/ recording the request payload (with
    # base64 image data URIs redacted), the response, token usage, and
    # latency. Off by default — turn on per-install for debugging; do
    # NOT enable in production (one file per call adds up fast and
    # transcripts can contain sensitive user content).
    debug_log_enabled: bool = False
    # Hard ceiling on a single OpenAI request. Without this the AsyncOpenAI
    # client waits forever, and since the dispatcher holds session.lock for
    # the duration of a turn a hung request deadlocks the whole session.
    request_timeout_seconds: float = 60.0
    # Total attempts (including the first) for a single LLM call. Retry is
    # triggered only by transient errors: RateLimitError, APITimeoutError,
    # APIConnectionError, InternalServerError. 4xx other than 429 still raises
    # immediately.
    max_retries: int = 3
    # Base delay before the second attempt; doubled each retry with up to
    # 0.5s of random jitter on top to avoid thundering-herd reconnects.
    retry_initial_delay: float = 1.0


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
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    team_profile: str = ""
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
    if isinstance(raw.get("cleanup"), dict):
        _coerce(settings.cleanup, raw["cleanup"])

    settings.env = {
        k: v for k, v in os.environ.items()
        if k.startswith(("WECOM_", "OPENAI_", "CHAT_TEAM_"))
    }
    if paths.team_md.exists():
        settings.team_profile = paths.team_md.read_text(encoding="utf-8").strip()
    workspace_root.mkdir(parents=True, exist_ok=True)
    return settings
