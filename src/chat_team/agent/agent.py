"""Agent: a (Role × Session) instance running the chat + tool loop.

One Agent owns one role's message history within one session. It does NOT
know about adapters or platforms — it returns the final assistant text and
optionally pushes status notes via the supplied StreamHandle.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..adapters.base import StreamHandle
from ..config import Settings
from ..llm.base import (
    ChatMessage,
    CompletionRequest,
    LLMProvider,
    ToolCall,
)
from ..roles.config import Role
from .tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    TransferRequested,
    stringify_result,
)

if TYPE_CHECKING:
    from ..session.session import Session

log = logging.getLogger(__name__)

MAX_TOOL_LOOPS = 8                         # bound the chat+tool loop per turn


@dataclass
class Agent:
    role: Role
    session: "Session"
    settings: Settings
    llm: LLMProvider
    tools: ToolRegistry
    history: list[ChatMessage] = field(default_factory=list)
    pending_system_inject: list[str] = field(default_factory=list)

    def reset_turn(self) -> None:
        # Clear per-turn buffers; called when caller hands a new user message.
        pass

    def queue_system_note(self, note: str) -> None:
        """Inject a one-shot system message at the start of the next chat call."""
        self.pending_system_inject.append(note)

    # ---- prompt assembly ---------------------------------------------------

    def _build_system_messages(self) -> list[ChatMessage]:
        toc = self.session.notebook.toc()
        blocks: list[str] = [self.role.system_prompt]
        if self.settings.team_profile:
            blocks.append("[团队信息]\n" + self.settings.team_profile)
        blocks.append("\n".join([
            f"[当前角色] {self.role.name} ({self.role.display_name})",
            f"[团队记事本目录] {toc}",
            "[隔离规则] 你只看得到自己的对话历史;切换员工后,新同事看不到你的轨迹。",
        ]))
        full = "\n\n".join(b for b in blocks if b).strip()
        msgs = [ChatMessage(role="system", content=full)]
        for note in self.pending_system_inject:
            msgs.append(ChatMessage(role="system", content=note))
        self.pending_system_inject.clear()
        return msgs

    def _all_employee_roster_keys(self) -> list[str]:
        # placeholder for future employee roster; kept to avoid future refactor.
        return []

    # ---- main loop ---------------------------------------------------------

    async def handle(self, user_text: str, stream: StreamHandle) -> str:
        self.history.append(ChatMessage(role="user", content=user_text))

        for loop_idx in range(MAX_TOOL_LOOPS):
            sys_msgs = self._build_system_messages()
            request = CompletionRequest(
                messages=sys_msgs + self.history,
                tools=self.tools.specs_for(self.role.tools),
                model=self._model(),
                temperature=self._temperature(),
            )
            response = await self.llm.complete(request)
            assistant = response.message
            self.history.append(assistant)

            if not assistant.tool_calls:
                return assistant.content or ""

            # Run each tool call serially. Surface errors back to the LLM.
            for call in assistant.tool_calls:
                await stream.status(f"调用工具: {call.name}")
                try:
                    result = await self._invoke_tool(call)
                except TransferRequested as transfer:
                    # Close the dangling tool_call in our own history so this
                    # role's transcript stays well-formed if it's revisited.
                    self.history.append(ChatMessage(
                        role="tool",
                        content=f"[transferred] target={transfer.target}",
                        tool_call_id=call.id,
                        name=call.name,
                    ))
                    raise                                  # propagate to dispatcher
                except ToolError as err:
                    result = f"[tool_error] {err}"
                except Exception as err:                   # noqa: BLE001
                    log.exception("tool %s raised", call.name)
                    result = f"[tool_error] {type(err).__name__}: {err}"
                self.history.append(ChatMessage(
                    role="tool",
                    content=stringify_result(result),
                    tool_call_id=call.id,
                    name=call.name,
                ))

        # safety fuse — too many loops without a final answer
        return "(已达到工具循环上限,本轮未给出最终答复)"

    async def _invoke_tool(self, call: ToolCall) -> Any:
        if not self.tools.has(call.name):
            raise ToolError(f"unknown tool: {call.name}")
        tool: Tool = self.tools.get(call.name)
        ctx = ToolContext(cwd=self.session.cwd, session=self.session, settings=self.settings)
        return await tool.run(ctx, **(call.arguments or {}))

    # ---- model resolution --------------------------------------------------

    def _model(self) -> str:
        return self.role.llm.model or self.settings.llm.default_model

    def _temperature(self) -> float:
        if self.role.llm.temperature is None:
            return self.settings.llm.default_temperature
        return self.role.llm.temperature
