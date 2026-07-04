"""Smoke for the SkillTool trigger-keyword gate.

Covers:
* SKILL.md frontmatter `trigger_keywords` parses into a tuple.
* SkillTool.run refuses to load a protected skill when the most recent
  user message contains none of the keywords — this is the code-level
  backstop for the "LLM ignores 红线规则 and self-invokes" bug.
* SkillTool.run loads the same skill fine when a keyword IS present.
* Gate is a no-op when the skill has no trigger_keywords.

Run: ``python3 scripts/smoke_skill_trigger.py`` — no network, no LLM.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_skill_trigger_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.agent.tools.base import ToolContext  # noqa: E402
from chat_team.agent.tools.shell_tool import RunCommandTool  # noqa: E402
from chat_team.agent.tools.skill_tools import SkillTool, ToolError  # noqa: E402
from chat_team.roles.registry import RoleRegistry  # noqa: E402
from chat_team.skills.registry import SkillRegistry  # noqa: E402


def _make_skill(home: Path, name: str, *, body: str, trigger_keywords=None) -> Path:
    sd = home / "skills" / name
    sd.mkdir(parents=True, exist_ok=True)
    fm = ["---", f"name: {name}", f"description: {name} skill for tests"]
    if trigger_keywords:
        fm.append("trigger_keywords:")
        for kw in trigger_keywords:
            fm.append(f"  - {kw}")
    fm += ["---", ""]
    (sd / "SKILL.md").write_text("\n".join(fm) + body, encoding="utf-8")
    return sd


def _make_role(home: Path, name: str, skills: list[str]) -> Role:
    rd = home / "roles"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"{name}.yaml").write_text(
        f"name: {name}\n"
        f"display_name: {name}\n"
        f"system_prompt: you are {name}\n"
        f"tools: [skill, skill_read_file]\n"
        f"skills: {skills}\n",
        encoding="utf-8",
    )
    return Role.from_dict({
        "name": name, "system_prompt": f"you are {name}",
        "tools": ["skill", "skill_read_file"], "skills": skills,
    })


def _ctx_with_history(session, history: list[dict]):
    """Build a ToolContext whose session has agents_by_role pre-populated."""
    # Minimal stub agent exposing .history.
    session.current_role = "tester"
    session.agents_by_role = {"tester": SimpleNamespace(history=history)}
    return ToolContext(cwd=Path("/tmp"), session=session, settings=None, llm=None)


async def _run_skill(session, skill_name: str, history: list[dict]) -> str:
    tool = session._tool
    ctx = _ctx_with_history(session, history)
    return await tool.run(ctx, name=skill_name)


async def main():
    home = Path("/tmp/chat_team_skill_trigger_smoke")
    # protected skill + unprotected skill
    _make_skill(home, "report-gen",
                body="# report\nthe body", trigger_keywords=["报告", "出报告", "PDF"])
    _make_skill(home, "free-skill", body="# free\nthe body")

    skills = SkillRegistry.load(home / "skills")
    assert skills.get("report-gen").trigger_keywords == ("报告", "出报告", "PDF")
    assert skills.get("free-skill").trigger_keywords == ()

    # Build role registry in-memory so we control skills whitelist exactly.
    from chat_team.roles.config import Role as _Role
    tester_role = _Role(
        name="tester", display_name="tester", description="test",
        system_prompt="you are tester",
        tools=["skill", "skill_read_file"],
        skills=["report-gen", "free-skill"],
    )
    roles = RoleRegistry({"tester": tester_role})

    tool = SkillTool(skills=skills, roles=roles)
    session = SimpleNamespace(_tool=tool, current_role="tester",
                              agents_by_role={})

    # --- 1. protected skill WITHOUT keyword in user message → ToolError ---
    history_no_kw = [
        {"role": "user", "content": "看看这几张图\n[图:a.jpg]"},
        {"role": "assistant", "content": "已识别"},
        {"role": "user", "content": "@bot"},          # most recent — no keyword
    ]
    try:
        await _run_skill(session, "report-gen", history_no_kw)
        assert False, "expected ToolError (no trigger keyword)"
    except ToolError as exc:
        msg = str(exc)
        assert "受触发词保护" in msg, msg
        assert "报告" in msg, msg
        print("✓ protected skill blocked when keyword absent")

    # --- 2. protected skill WITH keyword → loads body ---
    history_kw = [
        {"role": "user", "content": "出报告"},       # most recent — has keyword
    ]
    body = await _run_skill(session, "report-gen", history_kw)
    assert "the body" in body, body
    print("✓ protected skill loads when keyword present")

    # --- 3. case-insensitive match (PDF vs pdf) ---
    history_pdf = [{"role": "user", "content": "我要 pdf 版本"}]
    body = await _run_skill(session, "report-gen", history_pdf)
    assert "the body" in body
    print("✓ case-insensitive keyword match works")

    # --- 4. unprotected skill ignores gate entirely ---
    body = await _run_skill(session, "free-skill", history_no_kw)
    assert "the body" in body
    print("✓ unprotected skill not gated")

    # --- 5. vision-block content renders via blocks_to_text for scan ---
    history_vision = [
        {"role": "user", "content": [
            {"type": "image", "path": "./inbox/x.jpg"},
            {"type": "text", "text": "出报告"},       # keyword in vision turn
        ]},
    ]
    body = await _run_skill(session, "report-gen", history_vision)
    assert "the body" in body
    print("✓ vision-block user message scanned correctly")

    # ---------------- run_command bypass prevention ----------------
    # The LLM has been observed bypassing skill() by directly shelling out
    # to skill scripts. The RunCommandTool must enforce the same gate.

    # Build a RunCommandTool instance with the same skill registry.
    shell_tool = RunCommandTool(skills=skills, roles=roles)

    # --- 6. run_command referencing protected skill dir WITHOUT keyword → block ---
    # Mock the actual exec so we only test the gate (no real subprocess).
    async def _no_op_run(self, *a, **kw):
        return "(mocked)"

    shell_tool_run_orig = shell_tool.run
    # Patch _execute so we never shell out; just the gate matters.
    # We can't easily mock subprocess, so just call the gate helper directly
    # AND confirm a real .run() with the bypass command raises before exec.
    try:
        # Use a known-good skills directory path. The SkillRegistry resolves
        # to absolute paths under our test home.
        report_skill = skills.get("report-gen")
        bypass_cmd = f"cd {report_skill.directory} && python gen.py"
        ctx_blocked = _ctx_with_history(
            SimpleNamespace(current_role="tester", agents_by_role={}),
            history_no_kw,  # most recent user msg = "@bot" (no keyword)
        )
        # We need a session that exposes the tool registry; simplest is to
        # set ctx.settings so timeout parsing doesn't fail. Use a stub.
        ctx_blocked.settings = SimpleNamespace(tools=SimpleNamespace(
            shell_timeout_seconds=10, shell_env_extra_drop=[],
            shell_output_max_bytes=1024))
        await shell_tool.run(ctx_blocked, command=bypass_cmd)
        assert False, "expected ToolError from run_command bypass"
    except ToolError as exc:
        msg = str(exc)
        assert "受触发词保护" in msg, msg
        assert "report-gen" in msg, msg
        print("✓ run_command bypass blocked (no trigger keyword)")

    # --- 7. same bypass command WITH trigger keyword → gate passes ---
    # (We can't easily mock subprocess here; instead, verify the gate
    # itself does NOT raise by calling the helper directly.)
    from chat_team.agent.tools.skill_tools import scan_command_for_protected_skills
    ctx_kw = _ctx_with_history(
        SimpleNamespace(current_role="tester", agents_by_role={}),
        history_kw,  # most recent user msg = "出报告"
    )
    # No raise = gate passed.
    scan_command_for_protected_skills(bypass_cmd, skills, ctx_kw)
    print("✓ run_command bypass allowed when trigger keyword present")

    # --- 8. unrelated command (no skill reference) is never gated ---
    ctx_clean = _ctx_with_history(
        SimpleNamespace(current_role="tester", agents_by_role={}),
        history_no_kw,
    )
    scan_command_for_protected_skills("ls -la /tmp", skills, ctx_clean)
    scan_command_for_protected_skills("echo hello", skills, ctx_clean)
    print("✓ unrelated commands not gated")

    print("ALL SKILL TRIGGER-GATE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
