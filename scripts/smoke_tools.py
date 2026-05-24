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
    EditFileTool,
    GlobTool,
    GrepTool,
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
    sess = await sessions.get_or_create("test-session-tools")
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

    # ---- edit_file ----
    await WriteFileTool().run(ctx, path="edit_target.txt", content="alpha\nbeta\ngamma\nbeta\n")
    out = await EditFileTool().run(ctx, path="edit_target.txt", old="alpha", new="ALPHA")
    print("edit_file single:", out)
    assert (sess.cwd / "edit_target.txt").read_text() == "ALPHA\nbeta\ngamma\nbeta\n"

    # multi-match without replace_all → error
    try:
        await EditFileTool().run(ctx, path="edit_target.txt", old="beta", new="B")
    except ToolError as e:
        print("edit_file multi-match rejected:", e)
    else:
        raise AssertionError("edit_file should reject multi-match without replace_all")

    # multi-match with replace_all → ok
    out = await EditFileTool().run(ctx, path="edit_target.txt", old="beta", new="B", replace_all=True)
    print("edit_file replace_all:", out)
    assert (sess.cwd / "edit_target.txt").read_text() == "ALPHA\nB\ngamma\nB\n"

    # not found → error
    try:
        await EditFileTool().run(ctx, path="edit_target.txt", old="nope", new="x")
    except ToolError as e:
        print("edit_file not-found rejected:", e)
    else:
        raise AssertionError("edit_file should reject missing old")

    # empty old → error
    try:
        await EditFileTool().run(ctx, path="edit_target.txt", old="", new="x")
    except ToolError as e:
        print("edit_file empty-old rejected:", e)
    else:
        raise AssertionError("edit_file should reject empty old")

    # ---- glob ----
    await WriteFileTool().run(ctx, path="docs/a.md", content="# A")
    await WriteFileTool().run(ctx, path="docs/b.md", content="# B")
    await WriteFileTool().run(ctx, path="docs/inner/c.md", content="# C")
    out = await GlobTool().run(ctx, pattern="**/*.md")
    print("glob **/*.md:\n", out)
    assert "docs/a.md" in out and "docs/b.md" in out and "docs/inner/c.md" in out
    assert ".chat_team" not in out

    out = await GlobTool().run(ctx, pattern="docs/*.md")
    assert "docs/a.md" in out and "docs/inner/c.md" not in out

    out = await GlobTool().run(ctx, pattern="*.nope")
    assert out == "(no matches)"

    # truncation
    out = await GlobTool().run(ctx, pattern="**/*.md", max_results=1)
    assert "[truncated" in out

    # ---- grep ----
    await WriteFileTool().run(ctx, path="src/x.py", content="def hello():\n    return 'world'\n")
    await WriteFileTool().run(ctx, path="src/y.py", content="def Hello():\n    return 'WORLD'\n")
    out = await GrepTool().run(ctx, pattern=r"def hello")
    print("grep def hello:\n", out)
    assert "src/x.py:1:" in out and "src/y.py" not in out

    out = await GrepTool().run(ctx, pattern=r"def hello", ignore_case=True)
    assert "src/x.py:1:" in out and "src/y.py:1:" in out

    # non-UTF-8 file is skipped silently
    (sess.cwd / "blob.bin").write_bytes(b"\xff\xfehello\xff")
    out = await GrepTool().run(ctx, pattern=r"hello")
    assert "blob.bin" not in out
    assert "src/x.py:1:" in out

    # max_results truncation
    await WriteFileTool().run(ctx, path="many.txt", content="\n".join(["match"] * 10) + "\n")
    out = await GrepTool().run(ctx, pattern=r"match", max_results=3)
    assert "[truncated at 3]" in out

    # ---- read_file offset/limit ----
    body = "\n".join(f"line{i}" for i in range(100)) + "\n"
    await WriteFileTool().run(ctx, path="big.txt", content=body)
    out = await ReadFileTool().run(ctx, path="big.txt", offset=10, limit=5)
    print("read_file offset/limit:\n", out)
    assert "line10\n" in out and "line14\n" in out
    assert "line9\n" not in out and "line15\n" not in out
    assert "[shown lines 10..15 of 100]" in out

    # offset only
    out = await ReadFileTool().run(ctx, path="big.txt", offset=98)
    assert "line98\n" in out and "line99\n" in out and "line97\n" not in out

    # offset=0,limit unset returns full text (no annotation)
    out = await ReadFileTool().run(ctx, path="big.txt")
    assert "[shown lines" not in out

    # ---- .chat_team isolation enforced by _resolve_under ----
    try:
        await ReadFileTool().run(ctx, path=".chat_team/session.json")
    except ToolError as e:
        print("read .chat_team rejected:", e)
    else:
        raise AssertionError("read_file should reject .chat_team paths")

    try:
        await WriteFileTool().run(ctx, path=".chat_team/oops", content="x")
    except ToolError as e:
        print("write .chat_team rejected:", e)
    else:
        raise AssertionError("write_file should reject .chat_team paths")

    try:
        await EditFileTool().run(ctx, path=".chat_team/session.json", old="a", new="b")
    except ToolError as e:
        print("edit .chat_team rejected:", e)
    else:
        raise AssertionError("edit_file should reject .chat_team paths")

    # glob/grep filter .chat_team (run_command's runs/*.log lives under there)
    out = await GlobTool().run(ctx, pattern="**/*")
    assert ".chat_team" not in out
    out = await GrepTool().run(ctx, pattern=r".+")
    assert ".chat_team" not in out

    print("\nALL TOOL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
