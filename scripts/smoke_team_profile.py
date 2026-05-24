"""End-to-end smoke for the global team profile (~/.chat_team/team.md).

Covers:
* Missing/empty team.md  → no `[团队信息]` block in the system prompt.
* Populated team.md      → block appears verbatim in the system prompt.
* Compactor untouched    → its summarize call's system prompt remains the
  compactor's own ("你是会话历史压缩器…"), no team profile leakage.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fresh_home(tag: str) -> Path:
    home = Path(f"/tmp/chat_team_team_profile_{tag}")
    shutil.rmtree(home, ignore_errors=True)
    return home


from chat_team.agent.agent import Agent
from chat_team.agent.compactor import maybe_compact
from chat_team.agent.tools.base import ToolRegistry
from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager


class CapturingLLM(LLMProvider):
    """Records every CompletionRequest seen; returns canned replies."""

    def __init__(self, replies: list[CompletionResponse]) -> None:
        self.requests: list[CompletionRequest] = []
        self._replies = list(replies)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._replies:
            raise RuntimeError("CapturingLLM exhausted")
        return self._replies.pop(0)


async def _build_agent(home: Path):
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-team-profile")
    role = roles.get("team_admin")
    return settings, Agent(
        role=role,
        session=sess,
        settings=settings,
        llm=CapturingLLM([]),
        tools=ToolRegistry(),
    )


async def test_missing_team_md_no_injection() -> None:
    print("== test 1: empty team.md → no [团队信息] block ==")
    home = _fresh_home("missing")
    home.mkdir(parents=True, exist_ok=True)
    # Pre-create an empty team.md so init_home doesn't seed the template.
    (home / "team.md").write_text("", encoding="utf-8")
    settings, agent = await _build_agent(home)
    assert settings.team_profile == "", f"team_profile should be empty, got {settings.team_profile!r}"
    msgs = agent._build_system_messages()
    body = msgs[0].content or ""
    assert "[团队信息]" not in body, f"unexpected team block in:\n{body}"
    assert "[当前角色]" in body
    print("  ✓ no [团队信息] block when team.md is empty")


async def test_populated_team_md_injects_block() -> None:
    print("== test 2: populated team.md → block injected verbatim ==")
    home = _fresh_home("populated")
    home.mkdir(parents=True, exist_ok=True)
    profile = "## 我们是谁\n上海某某科技 · 客户成功部\n\n## 我们做什么\n对接 SaaS 售后,负责续约。"
    (home / "team.md").write_text(profile, encoding="utf-8")
    settings, agent = await _build_agent(home)
    assert settings.team_profile == profile, f"team_profile mismatch: {settings.team_profile!r}"
    msgs = agent._build_system_messages()
    body = msgs[0].content or ""
    assert "[团队信息]" in body, f"missing team block in:\n{body}"
    assert "上海某某科技" in body
    assert "对接 SaaS 售后" in body
    assert "[当前角色]" in body, "meta lines must still be present"
    print("  ✓ [团队信息] block present with verbatim content")


async def test_compactor_untouched() -> None:
    """maybe_compact uses its own system prompt; the team profile must not
    leak into the summarize call."""
    print("== test 3: compactor summarize call unaffected ==")
    home = _fresh_home("compactor")
    home.mkdir(parents=True, exist_ok=True)
    profile = "## 我们是谁\n敏感公司画像不应进入压缩器"
    (home / "team.md").write_text(profile, encoding="utf-8")
    settings, agent = await _build_agent(home)
    # Tighten budget and feed enough history to trigger compaction.
    agent.role.llm.history_token_budget = 50
    for i in range(10):
        agent.history.append(ChatMessage(role="user", content=f"用户 {i}: " + "x" * 50))
        agent.history.append(ChatMessage(role="assistant", content=f"回答 {i}: " + "y" * 50))

    canned = CompletionResponse(
        message=ChatMessage(role="assistant", content="(压缩摘要)"),
        finish_reason="stop",
    )
    llm = CapturingLLM([canned])
    did = await maybe_compact(agent, llm)
    assert did, "compaction should have run"
    assert len(llm.requests) == 1, "expected exactly one summarize LLM call"
    sys_msgs = [m for m in llm.requests[0].messages if m.role == "system"]
    assert len(sys_msgs) == 1, f"compactor call should have exactly one system msg, got {len(sys_msgs)}"
    sys_text = sys_msgs[0].content or ""
    assert sys_text.startswith("你是会话历史压缩器"), \
        f"compactor's own system prompt was overwritten:\n{sys_text}"
    assert "团队信息" not in sys_text, "team profile leaked into compactor"
    assert "敏感公司画像" not in sys_text, "team profile leaked into compactor"
    print("  ✓ compactor.summarize unaffected by team_profile")


async def main() -> None:
    await test_missing_team_md_no_injection()
    await test_populated_team_md_injects_block()
    await test_compactor_untouched()
    print("\nALL TEAM-PROFILE SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
