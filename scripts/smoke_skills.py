"""End-to-end smoke for the skills feature.

Covers:
* SkillRegistry.load discovers user skills, skips malformed ones (warning).
* Agent._build_system_messages injects ``[可用 skills]`` block, filtered by
  ``role.skills ∩ registry.names()``; first-line-only description rendering.
* SkillTool returns the SKILL.md body for a whitelisted skill, lists aux
  files, and refuses skills not in the current role's whitelist.
* SkillReadFileTool reads aux files under the skill dir and rejects ``..``.
* Compactor's summarize call does NOT see the skills block (sterile prompt
  remains intact).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fresh_home(tag: str) -> Path:
    home = Path(f"/tmp/chat_team_skills_{tag}")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    return home


from chat_team.agent.agent import Agent
from chat_team.agent.compactor import maybe_compact
from chat_team.agent.tools.base import ToolContext, ToolError, ToolRegistry
from chat_team.agent.tools.skill_tools import SkillReadFileTool, SkillTool
from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from chat_team.roles.config import Role, RoleLLMConfig
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager
from chat_team.skills.registry import SkillRegistry


class CapturingLLM(LLMProvider):
    def __init__(self, replies: list[CompletionResponse]) -> None:
        self.requests: list[CompletionRequest] = []
        self._replies = list(replies)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._replies:
            raise RuntimeError("CapturingLLM exhausted")
        return self._replies.pop(0)


def _write_skill(home: Path, name: str, description: str, body: str, aux: dict[str, str] | None = None) -> Path:
    import json
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    # JSON-encode the description so multi-line values stay valid YAML
    # (YAML accepts JSON-style double-quoted scalars with \n escapes).
    desc_yaml = json.dumps(description, ensure_ascii=False)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc_yaml}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    for rel, content in (aux or {}).items():
        target = skill_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return skill_dir


def _make_role(name: str, skills: list[str], tools: list[str] | None = None) -> Role:
    return Role(
        name=name,
        display_name=name,
        description="",
        system_prompt="测试角色",
        tools=list(tools) if tools is not None else ["skill", "skill_read_file"],
        skills=list(skills),
        llm=RoleLLMConfig(),
    )


def _setup(tag: str, *, role_skills: list[str]):
    home = _fresh_home(tag)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    return home, settings


async def test_load_filters_malformed() -> None:
    print("== test 1: load skips malformed skill, picks up valid ones ==")
    home, settings = _setup("load", role_skills=[])

    _write_skill(home, "pr_review", "给 PR 写 checklist 式 review", "正文 A\n## checklist\n- 项目 1\n")
    _write_skill(home, "translate_zh",
                 "中英互译规约\n（多行 description 仅取首行进 TOC）",
                 "正文 B")
    # malformed: missing frontmatter
    bad1 = home / "skills" / "no_frontmatter"
    bad1.mkdir(parents=True, exist_ok=True)
    (bad1 / "SKILL.md").write_text("just a body without frontmatter\n", encoding="utf-8")
    # malformed: name mismatch (dir name != frontmatter name)
    bad2 = home / "skills" / "wrong_dir_name"
    bad2.mkdir(parents=True, exist_ok=True)
    (bad2 / "SKILL.md").write_text(
        "---\nname: actually_other_name\ndescription: x\n---\nbody\n",
        encoding="utf-8",
    )
    # malformed: missing description
    bad3 = home / "skills" / "no_desc"
    bad3.mkdir(parents=True, exist_ok=True)
    (bad3 / "SKILL.md").write_text(
        "---\nname: no_desc\n---\nbody\n",
        encoding="utf-8",
    )
    # malformed: missing SKILL.md altogether
    (home / "skills" / "empty_dir").mkdir(parents=True, exist_ok=True)

    reg = SkillRegistry.load(settings.paths.user_skills_dir)
    names = reg.names()
    assert names == ["pr_review", "translate_zh"], f"got {names!r}"
    assert reg.get("translate_zh").description.startswith("中英互译规约"), "description preserved"
    print("  ✓ valid skills loaded; malformed dirs skipped")


async def test_system_prompt_injection() -> None:
    print("== test 2: Agent system prompt includes [可用 skills] filtered by role ==")
    home, settings = _setup("prompt", role_skills=["pr_review"])
    _write_skill(home, "pr_review", "给 PR 写 checklist 式 review\n（多行说明）", "正文")
    _write_skill(home, "translate_zh", "中英互译规约", "翻译指引")
    skills = SkillRegistry.load(settings.paths.user_skills_dir)

    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-prompt")
    role = _make_role("test_role", skills=["pr_review", "nonexistent"])
    sess.current_role = role.name

    agent = Agent(
        role=role, session=sess, settings=settings,
        llm=CapturingLLM([]), tools=ToolRegistry(), skills=skills,
    )
    body = agent._build_system_messages()[0].content or ""
    assert "[可用 skills]" in body, f"missing skills block:\n{body}"
    assert "只通过 skill(name=...)" in body
    assert "禁止使用 run_command/read_file" in body
    # Description should be first-line-only.
    assert "给 PR 写 checklist 式 review" in body
    assert "（多行说明）" not in body, "multi-line description was not truncated"
    # Filtered: translate_zh is in registry but not in role.skills.
    assert "translate_zh" not in body, "non-whitelisted skill leaked into TOC"
    # Filtered: 'nonexistent' is in role.skills but not in registry; silently dropped.
    assert "nonexistent" not in body
    print("  ✓ block present, filtered by role.skills ∩ registry.names()")

    # Empty whitelist → no block.
    role_no_skills = _make_role("test_role_2", skills=[])
    sess2 = await sessions.get_or_create("sess-prompt-no-skills")
    agent2 = Agent(
        role=role_no_skills, session=sess2, settings=settings,
        llm=CapturingLLM([]), tools=ToolRegistry(), skills=skills,
    )
    body2 = agent2._build_system_messages()[0].content or ""
    assert "[可用 skills]" not in body2, "block leaked when role has no skills"
    print("  ✓ empty role.skills → no block")


async def test_system_prompt_misconfigured_role_without_skill_tool() -> None:
    print("== test 2b: role has skills but no skill tool => explicit misconfig warning ==")
    home, settings = _setup("prompt-misconfig", role_skills=["pr_review"])
    _write_skill(home, "pr_review", "给 PR 写 checklist 式 review", "正文")
    skills = SkillRegistry.load(settings.paths.user_skills_dir)

    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-prompt-misconfig")
    role = _make_role("test_role_misconfig", skills=["pr_review"], tools=["read_file", "run_command"])
    sess.current_role = role.name

    agent = Agent(
        role=role, session=sess, settings=settings,
        llm=CapturingLLM([]), tools=ToolRegistry(), skills=skills,
    )
    body = agent._build_system_messages()[0].content or ""
    assert "[可用 skills]" not in body, "skills TOC should not be shown without skill tool"
    assert "[skills 配置异常]" in body, f"missing misconfig warning:\n{body}"
    assert "不要通过 run_command/read_file" in body
    print("  ✓ misconfigured role gets explicit no-search warning")


async def test_skill_tool_run() -> None:
    print("== test 3: SkillTool.run returns body, lists aux, gates on whitelist ==")
    home, settings = _setup("tool", role_skills=[])
    _write_skill(home, "pr_review", "给 PR 写 checklist", "## 流程\n1. 通览 diff\n2. 跑测试",
                 aux={"checklist.md": "- [ ] 边界值"})
    _write_skill(home, "translate_zh", "中英互译规约", "翻译规约正文")
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    roles = RoleRegistry({_make_role("test_role", ["pr_review"]).name: _make_role("test_role", ["pr_review"])})

    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-tool")
    sess.current_role = "test_role"

    tool = SkillTool(skills=skills, roles=roles)
    ctx = ToolContext(cwd=sess.cwd, session=sess, settings=settings)
    out = await tool.run(ctx, name="pr_review")
    assert "## 流程" in out and "通览 diff" in out, f"body missing:\n{out}"
    assert "[本 skill 目录]" in out, "skill directory hint missing"
    assert str((home / "skills" / "pr_review").resolve()) in out
    assert "[本 skill 附带辅助文件]" in out, "aux files listing missing"
    assert "checklist.md" in out

    # tolerant matching: quoted/space-padded and TOC-like value
    out2 = await tool.run(ctx, name='  "pr_review"  ')
    assert "## 流程" in out2
    out3 = await tool.run(ctx, name="pr_review: 给 PR 写 checklist")
    assert "## 流程" in out3

    # not in whitelist
    try:
        await tool.run(ctx, name="translate_zh")
    except ToolError as e:
        assert "白名单" in str(e), f"unexpected error: {e}"
    else:
        raise AssertionError("expected ToolError for non-whitelisted skill")

    # unknown skill
    try:
        await tool.run(ctx, name="does_not_exist")
    except ToolError as e:
        assert "unknown skill" in str(e)
    else:
        raise AssertionError("expected ToolError for unknown skill")

    print("  ✓ run() body+aux ok; whitelist + unknown name guarded")


async def test_skill_read_file() -> None:
    print("== test 4: SkillReadFileTool reads aux files; rejects escape ==")
    home, settings = _setup("readfile", role_skills=[])
    _write_skill(home, "pr_review", "给 PR 写 checklist", "正文",
                 aux={"checklist.md": "## checklist\n- 边界值\n",
                      "examples/good_pr.md": "good example"})
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    role = _make_role("test_role", ["pr_review"])
    roles = RoleRegistry({role.name: role})

    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-readfile")
    sess.current_role = "test_role"

    tool = SkillReadFileTool(skills=skills, roles=roles)
    ctx = ToolContext(cwd=sess.cwd, session=sess, settings=settings)

    out = await tool.run(ctx, skill="pr_review", path="checklist.md")
    assert "边界值" in out, f"unexpected: {out}"
    out_q = await tool.run(ctx, skill="'pr_review'", path="checklist.md")
    assert "边界值" in out_q
    out2 = await tool.run(ctx, skill="pr_review", path="examples/good_pr.md")
    assert out2.strip() == "good example"

    # ../ escape
    try:
        await tool.run(ctx, skill="pr_review", path="../../etc/passwd")
    except ToolError as e:
        assert "absolute" in str(e) or "escape" in str(e) or ".." in str(e)
    else:
        raise AssertionError("expected ToolError for ../ escape")

    # absolute path
    try:
        await tool.run(ctx, skill="pr_review", path="/etc/passwd")
    except ToolError:
        pass
    else:
        raise AssertionError("expected ToolError for absolute path")

    # SKILL.md re-read refused
    try:
        await tool.run(ctx, skill="pr_review", path="SKILL.md")
    except ToolError as e:
        assert "SKILL.md" in str(e)
    else:
        raise AssertionError("expected ToolError for SKILL.md re-read")

    print("  ✓ aux read ok; escape + absolute + SKILL.md re-read all blocked")


async def test_compactor_unaffected() -> None:
    print("== test 5: compactor's summarize call does not see [可用 skills] ==")
    home, settings = _setup("compactor", role_skills=[])
    _write_skill(home, "secret_skill", "敏感 skill 描述不应进入压缩器", "secret body")
    skills = SkillRegistry.load(settings.paths.user_skills_dir)

    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-compactor")
    role = _make_role("test_role", ["secret_skill"])
    sess.current_role = role.name
    agent = Agent(
        role=role, session=sess, settings=settings,
        llm=CapturingLLM([]), tools=ToolRegistry(), skills=skills,
    )
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
    assert len(llm.requests) == 1
    sys_msgs = [m for m in llm.requests[0].messages if m.role == "system"]
    assert len(sys_msgs) == 1
    sys_text = sys_msgs[0].content or ""
    assert "[可用 skills]" not in sys_text, "skills block leaked into compactor prompt"
    assert "secret_skill" not in sys_text
    assert "敏感 skill" not in sys_text
    print("  ✓ compactor sterile; no skills leakage")


async def test_python_uv_convention() -> None:
    print("== test 6: [Python 执行约定] gated on role having both skill + run_command ==")
    home, settings = _setup("uv_convention", role_skills=[])
    _write_skill(home, "pr_review", "给 PR 写 checklist", "正文")
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    sessions = SessionManager(settings)

    async def _prompt_for(role: Role, sid: str) -> str:
        sess = await sessions.get_or_create(sid)
        sess.current_role = role.name
        agent = Agent(
            role=role, session=sess, settings=settings,
            llm=CapturingLLM([]), tools=ToolRegistry(), skills=skills,
        )
        return agent._build_system_messages()[0].content or ""

    role_a = _make_role("role_a", ["pr_review"], tools=["skill", "run_command"])
    body_a = await _prompt_for(role_a, "sess-uv-a")
    assert "[Python 执行约定]" in body_a, "block missing for skill+run_command role"
    assert "uv run" in body_a and "# /// script" in body_a
    print("  ✓ skill + run_command → block present")

    role_b = _make_role("role_b", ["pr_review"], tools=["skill", "skill_read_file"])
    body_b = await _prompt_for(role_b, "sess-uv-b")
    assert "[Python 执行约定]" not in body_b, "block leaked into skill-only role"
    print("  ✓ skill without run_command → block absent")

    role_c = _make_role("role_c", [], tools=["run_command", "read_file"])
    body_c = await _prompt_for(role_c, "sess-uv-c")
    assert "[Python 执行约定]" not in body_c, "block leaked into run_command-only role"
    print("  ✓ run_command without skill → block absent")


async def test_uv_missing_warn() -> None:
    print("== test 7: warn_if_uv_missing fires only when uv absent AND a role needs it ==")
    import logging
    from unittest.mock import patch

    from chat_team.app import warn_if_uv_missing

    role_needs = _make_role("needs_python", [], tools=["skill", "run_command"])
    role_safe = _make_role("text_only", [], tools=["skill", "read_file"])

    handler = logging.Handler()
    captured: list[logging.LogRecord] = []
    handler.emit = captured.append   # type: ignore[assignment]
    logger = logging.getLogger("chat_team.app")
    logger.addHandler(handler)
    try:
        # uv missing + a role that needs it → WARN
        with patch("chat_team.app.shutil.which", return_value=None):
            captured.clear()
            warn_if_uv_missing(RoleRegistry({role_needs.name: role_needs}))
            warns = [r for r in captured if r.levelno == logging.WARNING]
            assert warns and "uv" in warns[0].getMessage(), f"missing WARN, got {captured!r}"
        print("  ✓ uv absent + skill+run_command role → WARN logged")

        # uv missing but no role needs it → silent
        with patch("chat_team.app.shutil.which", return_value=None):
            captured.clear()
            warn_if_uv_missing(RoleRegistry({role_safe.name: role_safe}))
            warns = [r for r in captured if r.levelno == logging.WARNING]
            assert not warns, f"unexpected WARN: {[r.getMessage() for r in warns]}"
        print("  ✓ uv absent but no Python-capable role → silent")

        # uv present → silent regardless
        with patch("chat_team.app.shutil.which", return_value="/usr/local/bin/uv"):
            captured.clear()
            warn_if_uv_missing(RoleRegistry({role_needs.name: role_needs}))
            warns = [r for r in captured if r.levelno == logging.WARNING]
            assert not warns, f"unexpected WARN when uv present: {[r.getMessage() for r in warns]}"
        print("  ✓ uv present → silent")
    finally:
        logger.removeHandler(handler)


async def main() -> None:
    await test_load_filters_malformed()
    await test_system_prompt_injection()
    await test_system_prompt_misconfigured_role_without_skill_tool()
    await test_skill_tool_run()
    await test_skill_read_file()
    await test_compactor_unaffected()
    await test_python_uv_convention()
    await test_uv_missing_warn()
    print("\nALL SKILLS SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
