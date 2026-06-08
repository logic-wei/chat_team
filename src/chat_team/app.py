"""Application bootstrap: wires settings + tools + roles + dispatcher + adapter."""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import shutil

from .adapters.base import BotAdapter
from .agent.tools.base import Tool, ToolRegistry
from .agent.tools.describe_image import DescribeImageTool
from .agent.tools.file_tools import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from .agent.tools.media_tools import SendFileTool, SendImageTool
from .agent.tools.notebook_tools import (
    NotebookDeleteTool,
    NotebookReadTool,
    NotebookWriteTool,
)
from .agent.tools.shell_tool import RunCommandTool
from .agent.tools.skill_tools import SkillReadFileTool, SkillTool
from .agent.tools.transfer_tool import TransferToEmployeeTool
from .config import Settings, load_settings
from .dispatcher import Dispatcher
from .llm.base import LLMProvider
from .llm.openai_provider import OpenAIChatCompletionProvider
from .roles.registry import RoleRegistry
from .session.manager import SessionManager
from .session.persistence import PersistenceManager
from .skills.registry import SkillRegistry

log = logging.getLogger(__name__)


def build_tool_registry(
    roles: RoleRegistry,
    skills: SkillRegistry | None = None,
    extra_tools: list[Tool] | None = None,
    *,
    include_transfer: bool = True,
) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ReadFileTool())
    reg.register(WriteFileTool())
    reg.register(EditFileTool())
    reg.register(ListDirTool())
    reg.register(GlobTool())
    reg.register(GrepTool())
    reg.register(RunCommandTool())
    reg.register(NotebookReadTool())
    reg.register(NotebookWriteTool())
    reg.register(NotebookDeleteTool())
    reg.register(SendImageTool())
    reg.register(SendFileTool())
    reg.register(DescribeImageTool())
    if include_transfer:
        reg.register(TransferToEmployeeTool(available_employees=roles.names()))
    if skills is not None and skills.names():
        reg.register(SkillTool(skills=skills, roles=roles))
        reg.register(SkillReadFileTool(skills=skills, roles=roles))
    for tool in extra_tools or []:
        reg.register(tool)
    return reg


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm.provider != "openai":
        raise NotImplementedError(f"llm provider not supported yet: {settings.llm.provider}")
    api_key = settings.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "LLM API key missing — set llm.api_key in ~/.chat_team/config.yaml "
            "or the OPENAI_API_KEY environment variable"
        )
    base_url = settings.llm.base_url or os.environ.get("OPENAI_BASE_URL") or None
    return OpenAIChatCompletionProvider(
        api_key=api_key,
        base_url=base_url,
        debug_log_enabled=settings.llm.debug_log_enabled,
        http_debug_log_enabled=settings.llm.http_debug_log_enabled,
        use_streaming=settings.llm.use_streaming,
        request_timeout_seconds=settings.llm.request_timeout_seconds,
        max_retries=settings.llm.max_retries,
        retry_initial_delay=settings.llm.retry_initial_delay,
    )


def build_vision_llm_provider(settings: Settings, main_llm: LLMProvider) -> LLMProvider:
    """Return an LLM provider for vision/OCR calls.

    Credentials are read exclusively from environment variables, consistent
    with how the main provider reads OPENAI_API_KEY / OPENAI_BASE_URL from .env:

      OPENAI_VISION_API_KEY  — vision API key; falls back to OPENAI_API_KEY
      OPENAI_VISION_BASE_URL — vision base URL; falls back to OPENAI_BASE_URL

    When the resolved vision credentials are identical to the main provider's,
    ``main_llm`` is returned as-is to avoid duplicate connections.
    """
    main_api_key = settings.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
    main_base_url = settings.llm.base_url or os.environ.get("OPENAI_BASE_URL") or None

    vision_api_key = (
        settings.llm.vision.api_key
        or os.environ.get("OPENAI_VISION_API_KEY", "")
        or main_api_key
    )
    vision_base_url = (
        settings.llm.vision.base_url
        or os.environ.get("OPENAI_VISION_BASE_URL", "")
        or main_base_url
    ) or None

    # Reuse the main provider when credentials are identical — no extra connections.
    if vision_api_key == main_api_key and vision_base_url == main_base_url:
        return main_llm

    log.info(
        "vision LLM uses separate credentials (base_url=%s)",
        vision_base_url or "(default)",
    )
    return OpenAIChatCompletionProvider(
        api_key=vision_api_key,
        base_url=vision_base_url,
        debug_log_enabled=settings.llm.debug_log_enabled,
        http_debug_log_enabled=settings.llm.http_debug_log_enabled,
        use_streaming=settings.llm.use_streaming,
        request_timeout_seconds=settings.llm.request_timeout_seconds,
        max_retries=settings.llm.max_retries,
        retry_initial_delay=settings.llm.retry_initial_delay,
    )


def warn_if_uv_missing(roles: RoleRegistry) -> None:
    """WARN when a Python-capable role is loaded but `uv` isn't on PATH.

    Triggered by the same `{skill, run_command}` combination that injects the
    PEP 723 convention into the agent's system prompt (see ``agent.py``).
    Non-fatal: roles that don't run Python skills keep working.
    """
    if shutil.which("uv"):
        return
    if not any({"skill", "run_command"}.issubset(set(r.tools)) for r in roles.all()):
        return
    log.warning(
        "skill+run_command roles loaded but `uv` not on PATH; "
        "Python skills will fail. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


def build_dispatcher(
    settings: Settings,
    extra_tools: list[Tool] | None = None,
) -> Dispatcher:
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    warn_if_uv_missing(roles)
    tools = build_tool_registry(roles, skills, extra_tools=extra_tools)
    persistence = PersistenceManager(settings)
    sessions = SessionManager(settings, persistence=persistence)
    llm = build_llm_provider(settings)
    vision_llm = build_vision_llm_provider(settings, llm)
    return Dispatcher(
        settings, sessions, roles, tools, llm,
        skills=skills, persistence=persistence,
        vision_llm=vision_llm,
    )


def configure_logging(settings: Settings) -> None:
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.paths.logs_dir / "chat_team.log"
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    rotating = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=settings.logging.max_bytes,
        backupCount=settings.logging.backup_count,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[rotating, logging.StreamHandler()],
    )


async def _run_solo(
    settings: Settings,
    mcp_tools: list[Tool],
    mcp_manager,
    adapter_factory,
) -> None:
    """Solo mode: one process, N bots, each pinned to one role."""
    if not settings.bots:
        raise RuntimeError("mode=solo but no bots configured in config.yaml")

    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    warn_if_uv_missing(roles)
    llm = build_llm_provider(settings)
    vision_llm = build_vision_llm_provider(settings, llm)
    persistence = PersistenceManager(settings)

    adapters: list[tuple[BotAdapter, Dispatcher]] = []
    for bot_cfg in settings.bots:
        if not roles.has(bot_cfg.name):
            raise RuntimeError(
                f"solo bot '{bot_cfg.name}' references unknown role; "
                f"available: {roles.names()}"
            )
        tools = build_tool_registry(
            roles, skills, extra_tools=mcp_tools, include_transfer=False,
        )
        sessions = SessionManager(
            settings, persistence=persistence, solo_role=bot_cfg.name,
        )
        dispatcher = Dispatcher(
            settings, sessions, roles, tools, llm,
            skills=skills, persistence=persistence,
            vision_llm=vision_llm, fixed_role=bot_cfg.name,
        )
        adapter: BotAdapter = adapter_factory(
            settings, sessions.workspace_for,
            bot_id=bot_cfg.bot_id, secret=bot_cfg.secret,
            role_name=bot_cfg.name,
        )
        adapter.set_handler(dispatcher.handle)
        adapters.append((adapter, dispatcher))
        log.info("solo bot '%s' configured (bot_id=%s)", bot_cfg.name, bot_cfg.bot_id[:8] + "...")

    try:
        await asyncio.gather(*[a.run_forever() for a, _ in adapters])
    finally:
        for a, _ in adapters:
            await a.close()
        all_sessions: list = []
        for _, d in adapters:
            all_sessions.extend(d.sessions.all_sessions())
        await persistence.flush_all(all_sessions)
        await mcp_manager.close_all()


async def _async_main(adapter_factory) -> None:
    from .mcp.client import McpClientManager

    settings = load_settings()
    configure_logging(settings)
    log.info("chat_team starting; home=%s mode=%s", settings.paths.home, settings.mode)

    mcp_manager = McpClientManager()
    mcp_tools: list[Tool] = []
    if settings.mcp.servers:
        mcp_tools = await mcp_manager.connect_all(settings.mcp.servers)
        log.info(
            "MCP: %d tool(s) from %d server(s)",
            len(mcp_tools),
            len(settings.mcp.servers),
        )

    if settings.mode == "solo":
        await _run_solo(settings, mcp_tools, mcp_manager, adapter_factory)
        return

    dispatcher = build_dispatcher(settings, extra_tools=mcp_tools)
    bot = settings.bots[0] if settings.bots else None
    adapter: BotAdapter = adapter_factory(
        settings, dispatcher.sessions.workspace_for,
        bot_id=bot.bot_id if bot else None,
        secret=bot.secret if bot else None,
    )
    adapter.set_handler(dispatcher.handle)
    try:
        # Prefer run_forever (handles transient WS loss); fall back to the
        # one-shot connect+run for adapters that don't implement it.
        runner = getattr(adapter, "run_forever", None)
        if runner is not None:
            await runner()
        else:
            await adapter.connect()
            await adapter.run()
    finally:
        await adapter.close()
        if dispatcher.persistence is not None:
            await dispatcher.persistence.flush_all(dispatcher.sessions.all_sessions())
        await mcp_manager.close_all()


def run() -> None:
    """Default entry point — uses WeCom adapter (stage 3 implements it)."""
    from .adapters.wecom import WeComBotAdapter   # imported lazily for stage 2 testability

    asyncio.run(_async_main(WeComBotAdapter))
