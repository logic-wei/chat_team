"""Load configuration from ``~/.chat_team/config.yaml``."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

import yaml
from dotenv import load_dotenv

from .mcp.config import McpServerConfig
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
    api_key: str = ""
    base_url: str = ""
    chat: "LLMChatConfig" = field(default_factory=lambda: LLMChatConfig())
    vision: "LLMVisionConfig" = field(default_factory=lambda: LLMVisionConfig())
    debug_log_enabled: bool = False
    http_debug_log_enabled: bool = False
    use_streaming: bool = True
    request_timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_initial_delay: float = 1.0
    max_tool_loops_per_turn: int = 16


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
    model: str = ""
    strategy: str = "tool"
    image_detail: str = "high"
    reasoning_effort: str = ""
    api_key: str = ""
    base_url: str = ""
    # Image size limits and resize behaviour for vision payloads.
    # When a raw image exceeds max_inline_bytes:
    #   "resize" → auto downscale + re-encode as JPEG (default)
    #   "reject" → replace with [图:xxx(过大,已省略)] placeholder
    max_inline_bytes: int = 6 * 1024 * 1024       # 6 MB raw file size ceiling
    oversized_image: str = "resize"                # resize | reject
    resize_long_side: int = 2048                   # max pixel dimension after resize
    resize_quality: int = 85                       # JPEG quality 1-95


@dataclass
class LoggingConfig:
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass
class McpConfig:
    servers: list[McpServerConfig] = field(default_factory=list)


@dataclass
class PrivateChatConfig:
    """Policy for whether the bot replies to single (private) chats.

    Group chats always pass through; only single chats are gated.

    Modes:
      * ``open``      — reply to everyone
      * ``closed``    — reply to no one
      * ``blacklist`` — reply to everyone except ``blacklist``
      * ``whitelist`` — reply only to ``whitelist`` (DEFAULT — see below)

    Default mode is ``whitelist`` with an empty list, i.e. default-deny:
    a brand-new install or an upgrade that doesn't add a ``private_chat``
    block will NOT reply to any private chat until the maintainer explicitly
    opts in (either by switching ``mode`` or by populating ``whitelist``).
    Group chats are always served regardless.

    IMPORTANT for upgrades: pre-feature deployments replied to every private
    chat. After this change, the bot will go silent on private chats until
    you add ``private_chat: {mode: open}`` (or a populated whitelist). The
    ``team_admin`` role can still be reached via group chats and the
    ``chat-team-boss`` CLI (which bypasses this gate entirely).

    ``blocked_reply`` is sent (as a single text message) to a user whose
    private chat was blocked. Empty string = silent drop. Note: this only
    gates one-to-one single chats; group chats are unaffected.
    """
    mode: str = "whitelist"
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    # Default is NON-empty: a default-deny policy that silently drops would
    # trap users in a dead-end (they can't learn their own userid to ask the
    # maintainer to whitelist them). The {userid} placeholder is substituted
    # by the adapter with the blocked sender's WeCom userid, so a brand-new
    # install self-documents the onboarding path without any config.
    blocked_reply: str = "私聊未开放。你的企业微信账号是 {userid},请联系管理员将其加入白名单。"

    def allows(self, user_id: str) -> bool:
        """True if this user_id should receive a reply in a single chat."""
        if self.mode == "open":
            return True
        if self.mode == "closed":
            return False
        if self.mode == "blacklist":
            return user_id not in self.blacklist
        # "whitelist" or any unrecognised value → fail closed (default-deny).
        # A typo in config.yaml can't accidentally expose the bot to everyone;
        # the maintainer simply adds their own userid to whitelist once.
        return user_id in self.whitelist


@dataclass
class BotConfig:
    """One WeCom bot entry. Solo mode requires ``name`` (= role); team mode doesn't."""
    bot_id: str
    secret: str
    name: str = ""


@dataclass
class Settings:
    paths: Paths
    workspace_root: Path
    default_role: str = "team_admin"
    mode: str = "team"  # "team" (multi-role transfer) | "solo" (one bot = one role)
    bots: list[BotConfig] = field(default_factory=list)
    log_level: str = "INFO"
    session: SessionConfig = field(default_factory=SessionConfig)
    notebook: NotebookConfig = field(default_factory=NotebookConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    private_chat: PrivateChatConfig = field(default_factory=PrivateChatConfig)
    team_profile: str = ""


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

    mode = raw.get("mode", "team")
    bots: list[BotConfig] = []
    if isinstance(raw.get("bots"), list):
        for b in raw["bots"]:
            if isinstance(b, dict) and b.get("bot_id") and b.get("secret"):
                bots.append(BotConfig(
                    bot_id=str(b["bot_id"]),
                    secret=str(b["secret"]),
                    name=str(b.get("name", "")),
                ))

    settings = Settings(
        paths=paths,
        workspace_root=workspace_root,
        default_role=raw.get("default_role", "team_admin"),
        mode=mode,
        bots=bots,
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
        _coerce(settings.llm, {
            k: v for k, v in llm_raw.items()
            if k not in {"chat", "vision"}
        })
        if isinstance(llm_raw.get("chat"), dict):
            _coerce(settings.llm.chat, llm_raw["chat"])
        if isinstance(llm_raw.get("vision"), dict):
            _coerce(settings.llm.vision, llm_raw["vision"])
    if isinstance(raw.get("logging"), dict):
        _coerce(settings.logging, raw["logging"])
    if isinstance(raw.get("cleanup"), dict):
        _coerce(settings.cleanup, raw["cleanup"])

    if isinstance(raw.get("private_chat"), dict):
        pc_raw = raw["private_chat"]
        if not isinstance(pc_raw, dict):
            _log.warning("private_chat must be a mapping; ignoring")
        else:
            valid_modes = {"open", "closed", "blacklist", "whitelist"}
            # Default to whitelist (default-deny) when the block exists but
            # omits ``mode`` — matches PrivateChatConfig's dataclass default.
            mode = str(pc_raw.get("mode", "whitelist")).strip().lower()
            if mode not in valid_modes:
                _log.warning(
                    "private_chat.mode=%r is not one of %s; defaulting to "
                    "'whitelist' (default-deny) — populate private_chat.whitelist "
                    "or set mode: open to enable private replies",
                    pc_raw.get("mode"), sorted(valid_modes),
                )
                mode = "whitelist"
            settings.private_chat = PrivateChatConfig(
                mode=mode,
                whitelist=[str(u).strip() for u in (pc_raw.get("whitelist") or []) if str(u).strip()],
                blacklist=[str(u).strip() for u in (pc_raw.get("blacklist") or []) if str(u).strip()],
                blocked_reply=str(pc_raw.get("blocked_reply", "") or ""),
            )

    if isinstance(raw.get("mcp"), dict):
        servers_raw = raw["mcp"].get("servers")
        if isinstance(servers_raw, dict):
            mcp_servers: list[McpServerConfig] = []
            for srv_name, srv_cfg in servers_raw.items():
                if not isinstance(srv_cfg, dict):
                    continue
                cfg = McpServerConfig(
                    name=str(srv_name),
                    command=srv_cfg.get("command", ""),
                    args=list(srv_cfg.get("args") or []),
                    env=dict(srv_cfg.get("env") or {}),
                    url=srv_cfg.get("url", ""),
                )
                try:
                    cfg.validate()
                    mcp_servers.append(cfg)
                except ValueError:
                    _log.warning("skipping invalid mcp server %r", srv_name, exc_info=True)
            settings.mcp = McpConfig(servers=mcp_servers)

    # Backward compat: if no bots configured in YAML, auto-construct one
    # from WECOM_BOT_ID / WECOM_SECRET env vars (loaded from .env above).
    if not settings.bots:
        env_bot_id = os.environ.get("WECOM_BOT_ID", "")
        env_secret = os.environ.get("WECOM_SECRET", "")
        if env_bot_id and env_secret:
            settings.bots = [BotConfig(bot_id=env_bot_id, secret=env_secret)]

    if paths.team_md.exists():
        settings.team_profile = paths.team_md.read_text(encoding="utf-8").strip()
    workspace_root.mkdir(parents=True, exist_ok=True)
    return settings
