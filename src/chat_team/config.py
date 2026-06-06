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
    # UX: while the agent is still working, periodically push a transient
    # status line so WeCom users don't stare at a silent chat window.
    progress_status_enabled: bool = True
    progress_status_delay_seconds: float = 1.5
    progress_status_interval_seconds: float = 2.5
    progress_status_text: str = "正在处理,请稍候..."
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
    chat: "LLMChatConfig" = field(default_factory=lambda: LLMChatConfig())
    vision: "LLMVisionConfig" = field(default_factory=lambda: LLMVisionConfig())
    # When true, every OpenAI provider call writes a JSON file to
    # <workspace>/.chat_team/llm/ recording the request payload (with
    # base64 image data URIs redacted), the response, token usage, and
    # latency. Off by default — turn on per-install for debugging; do
    # NOT enable in production (one file per call adds up fast and
    # transcripts can contain sensitive user content).
    debug_log_enabled: bool = False
    # When true, record every outbound HTTP request for LLM calls (headers
    # + full body) to per-session JSON files under <workspace>/.chat_team/
    # llm_http/. Includes sensitive fields (for example Authorization).
    # Keep OFF in production; use only for short local debugging windows.
    http_debug_log_enabled: bool = False
    # When true, provider uses OpenAI streaming under the hood and assembles
    # the final assistant message from chunks. This keeps long calls alive as
    # long as bytes keep arriving, reducing read-timeout failures near the
    # first token boundary.
    use_streaming: bool = True
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
    # Max LLM↔tool round-trips inside a single turn before the agent gives
    # up and returns a fuse message. Skill-driven flows (load SKILL.md →
    # write script → `uv run` → re-read failure → fix deps → re-run → …)
    # routinely need more than the previous fixed cap of 8.
    max_tool_loops_per_turn: int = 16

    # ---- backward-compatible aliases (for existing call sites/tests) ----
    @property
    def default_model(self) -> str:
        return self.chat.model

    @property
    def default_temperature(self) -> float:
        return self.chat.temperature

    @property
    def default_history_token_budget(self) -> int:
        return self.chat.history_token_budget

    @property
    def default_image_detail(self) -> str:
        return self.vision.image_detail

    @property
    def default_vision_strategy(self) -> str:
        return self.vision.strategy

    @property
    def default_vision_model(self) -> str:
        return self.vision.model


@dataclass
class LLMChatConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    history_token_budget: int = 12000
    # Optional reasoning depth for chat turns. Keep empty to let provider/model
    # defaults decide.
    reasoning_effort: str = ""


@dataclass
class LLMVisionConfig:
    # Empty means "reuse chat model".
    model: str = ""
    # Vision handling strategy: "tool" converts images to placeholders;
    # "direct" passes image blocks to the provider.
    strategy: str = "tool"
    image_detail: str = "high"            # "low" | "high" | "auto"
    # Optional reasoning depth for vision calls (describe_image, OCR-ish flows).
    reasoning_effort: str = ""


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
        llm_raw = raw["llm"]
        assert isinstance(llm_raw, dict)
        # Shared llm runtime knobs (provider/retry/timeout/debug/etc).
        _coerce(settings.llm, {
            k: v for k, v in llm_raw.items()
            if k not in {
                "chat", "vision",
                # legacy flat aliases handled below
                "default_model", "default_temperature", "default_history_token_budget",
                "default_vision_model", "default_image_detail", "default_vision_strategy",
                "default_chat_reasoning_effort", "default_vision_reasoning_effort",
            }
        })
        # Legacy flat keys (backward compatibility).
        if "default_model" in llm_raw:
            settings.llm.chat.model = llm_raw["default_model"]
        if "default_temperature" in llm_raw:
            settings.llm.chat.temperature = llm_raw["default_temperature"]
        if "default_history_token_budget" in llm_raw:
            settings.llm.chat.history_token_budget = llm_raw["default_history_token_budget"]
        if "default_vision_model" in llm_raw:
            settings.llm.vision.model = llm_raw["default_vision_model"]
        if "default_image_detail" in llm_raw:
            settings.llm.vision.image_detail = llm_raw["default_image_detail"]
        if "default_vision_strategy" in llm_raw:
            settings.llm.vision.strategy = llm_raw["default_vision_strategy"]
        if "default_chat_reasoning_effort" in llm_raw:
            settings.llm.chat.reasoning_effort = llm_raw["default_chat_reasoning_effort"]
        if "default_vision_reasoning_effort" in llm_raw:
            settings.llm.vision.reasoning_effort = llm_raw["default_vision_reasoning_effort"]

        # New nested layout: llm.chat / llm.vision.
        if isinstance(llm_raw.get("chat"), dict):
            _coerce(settings.llm.chat, llm_raw["chat"])
        if isinstance(llm_raw.get("vision"), dict):
            _coerce(settings.llm.vision, llm_raw["vision"])
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
