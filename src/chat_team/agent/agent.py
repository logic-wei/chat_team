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

from ..adapters.base import ContentBlock, StreamHandle
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
    from ..skills.registry import SkillRegistry

log = logging.getLogger(__name__)

# Injected into a role's system prompt only when the role exposes both `skill`
# and `run_command`. The combination is a strong signal the agent will be asked
# to execute Python emitted by skill bodies — community skills can't be modified
# to declare deps, so we teach the agent a uniform `uv run` + PEP 723 pattern
# that resolves third-party imports without polluting the host environment.
PYTHON_UV_CONVENTION = """[Python 执行约定]
当你需要执行 Python 脚本且引入第三方库时,请使用 PEP 723 inline metadata + `uv run`,不要直接 `pip install` 也不要假设库已安装:

    # /// script
    # dependencies = ["pkg-a", "pkg-b"]
    # ///
    import pkg_a
    ...

执行: `uv run script.py`。uv 会自动下载并隔离依赖。首次新依赖可能需要 30-60s 下载。"""


@dataclass
class Agent:
    role: Role
    session: "Session"
    settings: Settings
    llm: LLMProvider
    tools: ToolRegistry
    skills: "SkillRegistry | None" = None
    vision_llm: LLMProvider | None = None
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
        skills_block = self._render_skills_block()
        if skills_block:
            blocks.append(skills_block)
        if {"skill", "run_command"}.issubset(set(self.role.tools)):
            blocks.append(PYTHON_UV_CONVENTION)
        blocks.append("\n".join([
            f"[当前角色] {self.role.name} ({self.role.display_name})",
            f"[当前工作目录] {self.session.cwd}",
            f"[团队记事本目录] {toc}",
            "[隔离规则] 你只看得到自己的对话历史;切换员工后,新同事看不到你的轨迹。",
            "[路径规则] 业务输入/输出文件必须位于当前工作目录及其子目录;"
            "调用 skill 脚本时可进入 skill 目录执行,但 --input/--output 仍必须指向当前工作目录内的文件。",
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

    def _render_skills_block(self) -> str:
        has_skill_tool = "skill" in set(self.role.tools)
        if not self.role.skills or self.skills is None:
            return ""
        if not has_skill_tool:
            # Role is misconfigured: it whitelists skills but cannot call the
            # skill tool. Make the limitation explicit to prevent filesystem
            # scavenging attempts (e.g. run_command + find /...).
            return (
                "[skills 配置异常]\n"
                "当前角色声明了 skills,但 tools 未包含 skill,因此你无法读取任何 skill 正文。\n"
                "不要通过 run_command/read_file 在文件系统中搜索 SKILL.md 或 ~/.chat_team/skills。"
            )
        toc = self.skills.render_toc(self.role.skills)
        if not toc:
            return ""
        return (
            "[可用 skills]\n"
            "只通过 skill(name=...) 按名字加载正文;需要辅助文件时用 "
            "skill_read_file(skill=..., path=...)。\n"
            "若 skill 含脚本/资源,以 skill() 返回中的 [本 skill 目录] 为准。\n"
            "禁止使用 run_command/read_file 在系统目录中查找 skill 文件。\n"
            + toc
        )

    # ---- main loop ---------------------------------------------------------

    async def handle(
        self,
        user_content: str | list[ContentBlock],
        stream: StreamHandle,
    ) -> str:
        # Accept either a flat string (legacy / synthetic system-injected
        # turns from the dispatcher) or a list of ContentBlocks (multi-modal
        # user message from the adapter).
        # Snapshot the history length BEFORE appending the user message so
        # we can roll back the entire turn if the LLM call (or an unexpected
        # exception) fails partway through. Without rollback a failed first
        # call leaves a dangling user message; a failed mid-tool-loop call
        # leaves an assistant(tool_calls) without all of its tool replies.
        pre_turn_len = len(self.history)
        self.history.append(ChatMessage(role="user", content=user_content))

        try:
            for loop_idx in range(self.settings.llm.max_tool_loops_per_turn):
                sys_msgs = self._build_system_messages()
                request = CompletionRequest(
                    messages=sys_msgs + self.history,
                    tools=self.tools.specs_for(self.role.tools),
                    model=self._model(),
                    temperature=self._temperature(),
                    image_detail=self._image_detail(),
                    image_base_dir=self.session.cwd,
                    session_id=self.session.session_id,
                    role_name=self.role.name,
                    call_kind="agent",
                    debug_log_dir=self.session.cwd / ".chat_team" / "llm",
                    # Replace stream preview with cumulative text so users see
                    # live progress while the provider is still generating.
                    stream_text_callback=lambda text: stream.push(text, append=False),
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
                        result = await self._invoke_tool(call, stream)
                    except TransferRequested as transfer:
                        # Close the dangling tool_call in our own history so
                        # this role's transcript stays well-formed if it's
                        # revisited. Don't roll back — the closed sequence is
                        # valid OpenAI history that the dispatcher relies on.
                        self.history.append(ChatMessage(
                            role="tool",
                            content=f"[transferred] target={transfer.target}",
                            tool_call_id=call.id,
                            name=call.name,
                        ))
                        raise                              # propagate to dispatcher
                    except ToolError as err:
                        result = f"[tool_error] {err}"
                    except Exception as err:               # noqa: BLE001
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
        except TransferRequested:
            raise
        except Exception:
            # LLM call timed out / 5xx'd / network died. Drop everything we
            # appended this turn so the next turn (or the next time this
            # role is reopened) doesn't see a malformed transcript.
            del self.history[pre_turn_len:]
            raise

    async def _invoke_tool(self, call: ToolCall, stream: StreamHandle) -> Any:
        if not self.tools.has(call.name):
            raise ToolError(f"unknown tool: {call.name}")
        tool: Tool = self.tools.get(call.name)
        ctx = ToolContext(
            cwd=self.session.cwd,
            session=self.session,
            settings=self.settings,
            stream=stream,
            llm=self.llm,
            vision_llm=self.vision_llm,
        )
        return await tool.run(ctx, **(call.arguments or {}))

    # ---- model resolution --------------------------------------------------

    def _model(self) -> str:
        return self.role.llm.model or self.settings.llm.default_model

    def _temperature(self) -> float:
        if self.role.llm.temperature is None:
            return self.settings.llm.default_temperature
        return self.role.llm.temperature

    def _image_detail(self) -> str:
        return (
            self.role.llm.image_detail
            or self.settings.llm.default_image_detail
        )
