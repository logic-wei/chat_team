"""End-to-end smoke test for the dispatcher with a deterministic fake LLM.

Run:
    CHAT_TEAM_HOME=/tmp/chat_team_smoke python3 scripts/smoke_dispatch.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.agent.tools.base import ToolRegistry
from chat_team.agent.tools.file_tools import ListDirTool, ReadFileTool
from chat_team.agent.tools.notebook_tools import (
    NotebookDeleteTool,
    NotebookReadTool,
    NotebookWriteTool,
)
from chat_team.agent.tools.transfer_tool import TransferToEmployeeTool
from chat_team.config import load_settings
from chat_team.dispatcher import Dispatcher
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager


class CapturingStream:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.final: str | None = None

    async def push(self, chunk: str, *, append: bool = True) -> None:
        pass

    async def status(self, note: str) -> None:
        self.statuses.append(note)

    async def finish(self, final_text: str) -> None:
        self.final = final_text


class ScriptedLLM(LLMProvider):
    """Returns a queued sequence of responses, one per call."""

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self._responses.pop(0)


def reply(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def call(name: str, args: dict, call_id: str = "tc-1") -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        ),
        finish_reason="tool_calls",
    )


def build(settings, llm: LLMProvider) -> Dispatcher:
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(ListDirTool())
    tools.register(NotebookReadTool())
    tools.register(NotebookWriteTool())
    tools.register(NotebookDeleteTool())
    tools.register(TransferToEmployeeTool(available_employees=roles.names()))
    return Dispatcher(settings, SessionManager(settings), roles, tools, llm)


def msg(session_id: str, text: str) -> IncomingMessage:
    return IncomingMessage(
        session_id=session_id,
        chat_type=ChatType.SINGLE,
        user_id="u1",
        text=text,
        msg_id="m-" + text[:6],
        bot_id="bot1",
    )


async def main() -> None:
    home = Path("/tmp/chat_team_smoke")
    shutil.rmtree(home, ignore_errors=True)
    import os
    os.environ["CHAT_TEAM_HOME"] = str(home)

    # Seed a test-only role so the smoke doesn't depend on the builtin set,
    # which intentionally ships with only `team_admin`.
    roles_dir = home / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    (roles_dir / "engineer.yaml").write_text(
        "name: engineer\n"
        "display_name: 测试工程师\n"
        "system_prompt: |\n"
        "  你是测试用的研发同事。\n"
        "tools:\n"
        "  - read_file\n"
        "  - write_file\n"
        "  - list_dir\n"
        "  - run_command\n"
        "  - notebook_read\n"
        "  - notebook_write\n"
        "  - transfer_to_employee\n"
        "llm:\n"
        "  temperature: 0.2\n"
        "  history_token_budget: 16000\n",
        encoding="utf-8",
    )

    settings = load_settings()

    # ---- Test 1: admin replies directly without tool use --------------------
    print("== test 1: admin direct reply ==")
    llm1 = ScriptedLLM([reply("你好,我是小管,有什么可以帮你的?")])
    disp = build(settings, llm1)
    s = CapturingStream()
    await disp.handle(msg("wecom-single-bot1-u1", "你好"), s)
    assert s.final and "小管" in s.final, f"final={s.final!r}"
    print("  final:", s.final)

    # ---- Test 2: admin transfers to the seeded test engineer role -----------
    print("== test 2: admin → engineer transfer ==")
    llm2 = ScriptedLLM([
        call("transfer_to_employee", {
            "employee": "engineer",
            "reason": "用户需要看代码",
            "handoff_note": "用户想了解 src/main.py,请帮看",
        }, call_id="tc-transfer"),
        reply("你好,我是小研,我来看下代码。"),
    ])
    disp2 = build(settings, llm2)
    s2 = CapturingStream()
    await disp2.handle(msg("wecom-single-bot1-u2", "我想找研发帮我看代码"), s2)
    print("  statuses:", s2.statuses)
    print("  final:", s2.final)
    assert s2.final and "小研" in s2.final
    sess = await disp2.sessions.get_or_create("wecom-single-bot1-u2")
    assert sess.current_role == "engineer", sess.current_role
    print("  current_role after turn:", sess.current_role)

    # ---- Test 3: notebook write + read --------------------------------------
    print("== test 3: notebook write/read by tool ==")
    llm3 = ScriptedLLM([
        call("notebook_write", {"key": "user_name", "value": "张三"}, call_id="tc-w"),
        call("notebook_read", {"key": "user_name"}, call_id="tc-r"),
        reply("我已记下:用户名是张三。"),
    ])
    disp3 = build(settings, llm3)
    s3 = CapturingStream()
    await disp3.handle(msg("wecom-single-bot1-u3", "我叫张三"), s3)
    print("  final:", s3.final)
    sess3 = await disp3.sessions.get_or_create("wecom-single-bot1-u3")
    assert sess3.notebook.read("user_name") == "张三"
    print("  notebook[user_name]:", sess3.notebook.read("user_name"))
    print("  notebook toc:", sess3.notebook.toc())

    # ---- Test 4: file sandbox (read_file blocks ../) ------------------------
    print("== test 4: file sandbox ==")
    sess4 = await disp3.sessions.get_or_create("wecom-single-bot1-u4")
    (sess4.cwd / "hello.txt").write_text("hi from workspace", encoding="utf-8")
    llm4 = ScriptedLLM([
        call("read_file", {"path": "hello.txt"}, call_id="tc-rf1"),
        call("read_file", {"path": "../escape.txt"}, call_id="tc-rf2"),
        reply("读取完成。"),
    ])
    disp4 = build(settings, llm4)
    # reuse session's cwd by routing the same session_id
    s4 = CapturingStream()
    await disp4.handle(msg("wecom-single-bot1-u4", "读 hello.txt 和 ../escape.txt"), s4)
    print("  final:", s4.final)
    # check tool messages in agent history
    sess4b = await disp4.sessions.get_or_create("wecom-single-bot1-u4")
    agent = sess4b.agents_by_role[sess4b.current_role]
    tool_results = [m.content for m in agent.history if m.role == "tool"]
    print("  tool results:", tool_results)
    assert any("hi from workspace" in r for r in tool_results)
    assert any("not allowed" in r for r in tool_results)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
