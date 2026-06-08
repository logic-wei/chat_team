"""Smoke test for ReadDeployConfigTool (boss-side, read-only).

Covers:
* team mode: returns mode + default_role + bot count, no secrets visible.
* solo mode: returns per-bot name, no bot_id / secret visible.
* Empty / missing config: returns sensible defaults.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat_team.agent.tools.base import ToolContext
from chat_team.agent.tools.team_tools import ReadDeployConfigTool
from chat_team.config import load_settings
from chat_team.session.notebook import Notebook
from chat_team.session.session import Session


def _fresh_home(tag: str) -> Path:
    home = Path(f"/tmp/chat_team_deploy_cfg_{tag}")
    shutil.rmtree(home, ignore_errors=True)
    return home


def _make_ctx(home: Path) -> ToolContext:
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    notebook = Notebook(home / ".boss_notebook.md")
    session = Session(
        session_id="__test__", cwd=home, current_role="boss", notebook=notebook,
    )
    return ToolContext(cwd=home, session=session, settings=settings)


async def test_team_mode() -> None:
    print("== test 1: read_deploy_config — team mode ==")
    home = _fresh_home("team")
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.yaml"
    cfg.write_text(
        "bots:\n"
        "  - bot_id: wk_abc123\n"
        "    secret: sk-super-secret-key-do-not-leak\n"
        "default_role: team_admin\n",
        encoding="utf-8",
    )
    ctx = _make_ctx(home)
    tool = ReadDeployConfigTool()
    out = await tool.run(ctx)
    assert "mode: team" in out, f"expected 'mode: team' in:\n{out}"
    assert "default_role: team_admin" in out, f"expected default_role in:\n{out}"
    assert "wk_abc123" not in out, f"bot_id leaked:\n{out}"
    assert "sk-super-secret" not in out, f"secret leaked:\n{out}"
    assert "secret" not in out.lower(), f"word 'secret' appeared:\n{out}"
    print(f"  output:\n{out}")
    print("  ✓ team mode ok, no credentials visible")


async def test_solo_mode() -> None:
    print("== test 2: read_deploy_config — solo mode ==")
    home = _fresh_home("solo")
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.yaml"
    cfg.write_text(
        "mode: solo\n"
        "default_role: research_engineer\n"
        "bots:\n"
        "  - bot_id: bot_111\n"
        "    secret: sec_111\n"
        "    name: research_engineer\n"
        "  - bot_id: bot_222\n"
        "    secret: sec_222\n"
        "    name: customer_service\n",
        encoding="utf-8",
    )
    ctx = _make_ctx(home)
    tool = ReadDeployConfigTool()
    out = await tool.run(ctx)
    assert "mode: solo" in out, f"expected 'mode: solo' in:\n{out}"
    assert "research_engineer" in out, f"expected bot name in:\n{out}"
    assert "customer_service" in out, f"expected bot name in:\n{out}"
    assert "bot_111" not in out, f"bot_id leaked:\n{out}"
    assert "bot_222" not in out, f"bot_id leaked:\n{out}"
    assert "sec_111" not in out, f"secret leaked:\n{out}"
    assert "sec_222" not in out, f"secret leaked:\n{out}"
    assert "2个" in out, f"expected bot count in:\n{out}"
    print(f"  output:\n{out}")
    print("  ✓ solo mode ok, names visible, no credentials")


async def test_empty_config() -> None:
    print("== test 3: read_deploy_config — empty config ==")
    home = _fresh_home("empty")
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.yaml"
    cfg.write_text("", encoding="utf-8")
    ctx = _make_ctx(home)
    tool = ReadDeployConfigTool()
    out = await tool.run(ctx)
    assert "mode: team" in out, f"expected default mode in:\n{out}"
    assert "default_role: team_admin" in out, f"expected default role in:\n{out}"
    assert "未配置" in out, f"expected empty bots hint in:\n{out}"
    print(f"  output:\n{out}")
    print("  ✓ empty config returns sensible defaults")


async def test_solo_missing_name() -> None:
    print("== test 4: read_deploy_config — solo bot without name ==")
    home = _fresh_home("solo_noname")
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.yaml"
    cfg.write_text(
        "mode: solo\n"
        "bots:\n"
        "  - bot_id: bot_x\n"
        "    secret: sec_x\n",
        encoding="utf-8",
    )
    ctx = _make_ctx(home)
    tool = ReadDeployConfigTool()
    out = await tool.run(ctx)
    assert "未指定" in out, f"expected missing-name hint in:\n{out}"
    assert "bot_x" not in out, f"bot_id leaked:\n{out}"
    print(f"  output:\n{out}")
    print("  ✓ solo bot without name shows hint, no credentials")


async def main() -> None:
    await test_team_mode()
    await test_solo_mode()
    await test_empty_config()
    await test_solo_missing_name()
    print("\n=== ALL deploy-config smoke tests passed ===")


if __name__ == "__main__":
    asyncio.run(main())
