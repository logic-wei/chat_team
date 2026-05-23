"""Dispatcher: glues IncomingMessage → Session → Agent → Response.

Owns transfer handling (per-turn cap, handoff_note injection, re-invoking
the new agent on the original user message).
"""
from __future__ import annotations

import logging

from .adapters.base import IncomingMessage, StreamHandle
from .agent.agent import Agent
from .agent.compactor import maybe_compact
from .agent.tools.base import ToolRegistry, TransferRequested
from .config import Settings
from .llm.base import LLMProvider
from .roles.registry import RoleRegistry
from .session.manager import SessionManager
from .session.persistence import PersistenceManager
from .session.session import PendingHandoff, Session
from .skills.registry import SkillRegistry
from .vision_shim import apply_vision_strategy

log = logging.getLogger(__name__)


class Dispatcher:
    def __init__(
        self,
        settings: Settings,
        sessions: SessionManager,
        roles: RoleRegistry,
        tools: ToolRegistry,
        llm: LLMProvider,
        skills: SkillRegistry | None = None,
        persistence: PersistenceManager | None = None,
    ):
        self.settings = settings
        self.sessions = sessions
        self.roles = roles
        self.tools = tools
        self.llm = llm
        self.skills = skills if skills is not None else SkillRegistry({})
        self.persistence = persistence

    async def handle(self, msg: IncomingMessage, stream: StreamHandle) -> None:
        session = self.sessions.get_or_create(msg.session_id)
        # Prefer rich content_blocks (multi-modal); fall back to flat text for
        # adapters that haven't been upgraded.
        user_content = msg.content_blocks if msg.content_blocks else msg.text
        async with session.lock:
            try:
                final = await self._run_turn(session, user_content, stream)
            finally:
                session.reset_turn_counters()
            await stream.finish(final)
            await self._post_turn(session)

    async def _post_turn(self, session: Session) -> None:
        for agent in list(session.agents_by_role.values()):
            try:
                await maybe_compact(agent, self.llm)
            except Exception:                                  # noqa: BLE001
                log.exception("compaction failed for role=%s", agent.role.name)
        if self.persistence is not None:
            self.persistence.schedule(session)

    async def _run_turn(
        self,
        session: Session,
        user_content,
        stream: StreamHandle,
    ) -> str:
        cap = self.settings.session.per_turn_transfer_cap
        original_content = user_content
        while True:
            agent = self._agent_for(session, session.current_role)
            self._apply_pending_handoff(agent, session)
            # Apply the *current* role's vision strategy to the *original*
            # user content. On transfer, the new role re-evaluates the same
            # raw image blocks under its own strategy (and shares the eager
            # description cache, so re-OCR is essentially free).
            transformed = await apply_vision_strategy(
                original_content,
                role=agent.role,
                settings=self.settings,
                llm=self.llm,
                cwd=session.cwd,
                session_id=session.session_id,
            )
            try:
                return await agent.handle(transformed, stream)
            except TransferRequested as t:
                session.transfer_count_this_turn += 1
                if session.transfer_count_this_turn >= cap:
                    log.warning(
                        "session %s hit transfer cap %d; forcing %s to answer",
                        session.session_id, cap, session.current_role,
                    )
                    agent.queue_system_note(
                        f"[强制回答] 你已经在本轮触发过 {cap} 次员工交接,"
                        f"现在必须直接给用户答复,不再调用 transfer_to_employee。"
                    )
                    return await agent.handle(
                        f"(系统提示: 交接次数达上限,请你直接回答用户的最近一次提问。)",
                        stream,
                    )
                if not self.roles.has(t.target):
                    log.warning("transfer target %s unknown; staying with %s", t.target, session.current_role)
                    agent.queue_system_note(
                        f"[交接失败] 没有名为 {t.target} 的同事,请直接回答用户。"
                    )
                    return await agent.handle(
                        "(系统提示: 上一次交接失败,请直接回答用户的最近一次提问。)",
                        stream,
                    )
                await stream.status(f"交接给 {self.roles.get(t.target).display_name}")
                session.pending_handoff = PendingHandoff(
                    from_role=session.current_role,
                    to_role=t.target,
                    reason=t.reason,
                    note=t.handoff_note,
                )
                session.current_role = t.target
                # loop again with the new role and the same user_text

    def _apply_pending_handoff(self, agent: Agent, session: Session) -> None:
        if not session.pending_handoff:
            return
        h = session.pending_handoff
        if h.to_role != agent.role.name:
            return
        agent.queue_system_note(
            f"[交接备忘] 来自同事 {h.from_role}。原因: {h.reason}\n备忘: {h.note}\n"
            "请基于此备忘和团队记事本继续服务用户;若你也认为应该再次转给别人,请慎重。"
        )
        session.pending_handoff = None

    def _agent_for(self, session: Session, role_name: str) -> Agent:
        if role_name in session.agents_by_role:
            return session.agents_by_role[role_name]
        role = self.roles.get(role_name)
        agent = Agent(
            role=role,
            session=session,
            settings=self.settings,
            llm=self.llm,
            tools=self.tools,
            skills=self.skills,
        )
        # First time we materialise this role for the session — adopt any
        # history loaded from disk and consume it (one-shot).
        prior = session.restored_histories.pop(role_name, None)
        if prior:
            agent.history = list(prior)
        session.agents_by_role[role_name] = agent
        return agent
