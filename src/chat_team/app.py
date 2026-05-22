"""Application bootstrap: wires settings + tools + roles + dispatcher + adapter."""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os

from .adapters.base import BotAdapter
from .agent.tools.base import ToolRegistry
from .agent.tools.file_tools import ListDirTool, ReadFileTool, WriteFileTool
from .agent.tools.media_tools import SendFileTool, SendImageTool
from .agent.tools.notebook_tools import (
    NotebookDeleteTool,
    NotebookReadTool,
    NotebookWriteTool,
)
from .agent.tools.shell_tool import RunCommandTool
from .agent.tools.transfer_tool import TransferToEmployeeTool
from .config import Settings, load_settings
from .dispatcher import Dispatcher
from .llm.base import LLMProvider
from .llm.openai_provider import OpenAIChatCompletionProvider
from .roles.registry import RoleRegistry
from .session.manager import SessionManager
from .session.persistence import PersistenceManager

log = logging.getLogger(__name__)


def build_tool_registry(roles: RoleRegistry) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ReadFileTool())
    reg.register(WriteFileTool())
    reg.register(ListDirTool())
    reg.register(RunCommandTool())
    reg.register(NotebookReadTool())
    reg.register(NotebookWriteTool())
    reg.register(NotebookDeleteTool())
    reg.register(SendImageTool())
    reg.register(SendFileTool())
    reg.register(TransferToEmployeeTool(available_employees=roles.names()))
    return reg


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm.provider != "openai":
        raise NotImplementedError(f"llm provider not supported yet: {settings.llm.provider}")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing — set it in ~/.chat_team/.env")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    return OpenAIChatCompletionProvider(api_key=api_key, base_url=base_url)


def build_dispatcher(settings: Settings) -> Dispatcher:
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = build_tool_registry(roles)
    sessions = SessionManager(settings)
    llm = build_llm_provider(settings)
    persistence = PersistenceManager(settings)
    return Dispatcher(settings, sessions, roles, tools, llm, persistence=persistence)


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


async def _async_main(adapter_factory) -> None:
    settings = load_settings()
    configure_logging(settings)
    log.info("chat_team starting; home=%s", settings.paths.home)
    dispatcher = build_dispatcher(settings)
    adapter: BotAdapter = adapter_factory(settings, dispatcher.sessions.workspace_for)
    adapter.set_handler(dispatcher.handle)
    await adapter.connect()
    try:
        await adapter.run()
    finally:
        await adapter.close()
        if dispatcher.persistence is not None:
            await dispatcher.persistence.flush_all(dispatcher.sessions.all_sessions())


def run() -> None:
    """Default entry point — uses WeCom adapter (stage 3 implements it)."""
    from .adapters.wecom import WeComBotAdapter   # imported lazily for stage 2 testability

    asyncio.run(_async_main(WeComBotAdapter))
