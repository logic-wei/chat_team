"""Smoke test for stage-4 tools: write_file, run_command, output truncation,
sandbox enforcement (./../). No LLM, no WS — pure Python."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_tools_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.agent.tools.base import ToolContext, ToolError
from chat_team.agent.tools.file_tools import (
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from chat_team.agent.tools.shell_tool import RunCommandTool
from chat_team.config import load_settings
from chat_team.session.manager import SessionManager


async def main():
    settings = load_settings()
    sessions = SessionManager(settings)
    sess = sessions.get_or_create("test-session-tools")
    ctx = ToolContext(cwd=sess.cwd, session=sess, settings=settings)

    # write_file: ok
    out = await WriteFileTool().run(ctx, path="hello.md", content="hi 你好")
    print("write_file:", out)
    assert (sess.cwd / "hello.md").read_text(encoding="utf-8") == "hi 你好"

    # write_file: rejects ../
    try:
        await WriteFileTool().run(ctx, path="../escape.md", content="x")
    except ToolError as e:
        print("write_file ../ rejected:", e)
    else:
        raise AssertionError("write_file did not reject ../")

    # write_file: nested path auto-mkdir
    await WriteFileTool().run(ctx, path="sub/dir/note.txt", content="nested")
    assert (sess.cwd / "sub/dir/note.txt").read_text(encoding="utf-8") == "nested"

    # read_file: ok
    txt = await ReadFileTool().run(ctx, path="hello.md")
    print("read_file:", txt)
    assert "hi" in txt

    # list_dir: shows entries, hides .chat_team
    listing = await ListDirTool().run(ctx, path=".")
    print("list_dir:\n", listing)
    assert "hello.md" in listing
    assert ".chat_team" not in listing

    # run_command: small output
    out = await RunCommandTool().run(ctx, command="echo hello && pwd")
    print("run_command small:\n", out)
    assert "hello" in out
    assert str(sess.cwd) in out
    assert "exit_code=0" in out

    # run_command: cwd is enforced (touch a file)
    out = await RunCommandTool().run(ctx, command="touch from_shell.txt && ls from_shell.txt")
    assert (sess.cwd / "from_shell.txt").exists()
    print("run_command touch ok")

    # run_command: output truncation (generate >8KB)
    out = await RunCommandTool().run(ctx, command="python3 -c 'print(\"x\"*20000)'")
    print("run_command truncation header:", out.split('\n', 1)[0])
    assert "truncated" in out
    log_rel = out.split("full_log=")[1].split()[0]
    full = (sess.cwd / log_rel).read_bytes()
    assert len(full) >= 20000

    # run_command: timeout
    out = await RunCommandTool().run(ctx, command="sleep 5", timeout=1)
    print("run_command timeout:", out.split('\n', 1)[0])
    assert "TIMEOUT" in out

    print("\nALL TOOL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
