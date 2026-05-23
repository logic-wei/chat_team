"""Application bootstrap: wires settings + tools + roles + dispatcher + adapter."""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import shutil

from .adapters.base import BotAdapter
from .agent.tools.base import ToolRegistry
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
    reg.register(TransferToEmployeeTool(available_employees=roles.names()))
    if skills is not None and skills.names():
        reg.register(SkillTool(skills=skills, roles=roles))
        reg.register(SkillReadFileTool(skills=skills, roles=roles))
    return reg


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm.provider != "openai":
        raise NotImplementedError(f"llm provider not supported yet: {settings.llm.provider}")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing — set it in ~/.chat_team/.env")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    return OpenAIChatCompletionProvider(
        api_key=api_key,
        base_url=base_url,
        debug_log_enabled=settings.llm.debug_log_enabled,
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


def build_dispatcher(settings: Settings) -> Dispatcher:
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    warn_if_uv_missing(roles)
    tools = build_tool_registry(roles, skills)
    sessions = SessionManager(settings)
    llm = build_llm_provider(settings)
    persistence = PersistenceManager(settings)
    return Dispatcher(
        settings, sessions, roles, tools, llm,
        skills=skills, persistence=persistence,
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
