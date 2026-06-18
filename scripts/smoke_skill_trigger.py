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

    print("ALL SKILL TRIGGER-GATE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
