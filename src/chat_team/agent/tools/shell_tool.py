"""Shell command execution tool, sandboxed to the session's cwd.

* Runs via ``asyncio.create_subprocess_shell`` so we don't block the loop.
* Hard timeout from settings (default 30s).
* Stdout+stderr captured; returned to LLM truncated; full transcript saved
  under ``<cwd>/.chat_team/runs/<ts>.log`` so the agent can re-read it via
  ``read_file`` if needed.
* Subprocess env is scrubbed via :func:`_scrub_env` — secrets like
  ``OPENAI_API_KEY`` / ``WECOM_SECRET`` never reach the child, so a
  prompt-injection that talks the LLM into running ``printenv | curl …``
  has nothing to leak.
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Any, Iterable

from .base import Tool, ToolContext, ToolError


_DENY_PREFIXES = ("OPENAI_", "WECOM_", "ANTHROPIC_", "CHAT_TEAM_")
_DENY_SUBSTRINGS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


def _scrub_env(
    source: dict[str, str] | os._Environ[str],
    extra_drop: Iterable[str] = (),
) -> dict[str, str]:
    """Filter out env vars likely to carry secrets before exec'ing a child.

    Drops anything matching a built-in prefix (``OPENAI_*``, ``WECOM_*``,
    ``ANTHROPIC_*``, ``CHAT_TEAM_*``), anything whose name contains a
    sensitive substring (``KEY`` / ``SECRET`` / ``TOKEN`` / ``PASSWORD``
    / ``CREDENTIAL``, case-insensitive), and any name in ``extra_drop``.
    Everything else (PATH, HOME, LANG, PROXY, …) is passed through so
    user scripts keep working.
    """
    drop_set = set(extra_drop)
    out: dict[str, str] = {}
    for k, v in source.items():
        if k in drop_set:
            continue
        if any(k.startswith(p) for p in _DENY_PREFIXES):
            continue
        upper = k.upper()
        if any(s in upper for s in _DENY_SUBSTRINGS):
            continue
        out[k] = v
    return out


def _truncate(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> bytes:
    """SIGTERM the whole process group, drain stdout, escalate to SIGKILL.

    With start_new_session=True the child bash is its own session/pgrp leader,
    so its pid IS the pgid and signalling that pgid hits every descendant.
    """
    pid = proc.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        return stdout_bytes or b""
    except asyncio.TimeoutError:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout_bytes, _ = await proc.communicate()
            return stdout_bytes or b""
        except Exception:                                    # noqa: BLE001
            return b""


class RunCommandTool(Tool):
    name = "run_command"

    def __init__(self, skills=None, roles=None):
        # Optional reference to the SkillRegistry / RoleRegistry so that
        # run() can enforce trigger-keyword gates when the LLM tries to
        # bypass the protected ``skill()`` tool by directly executing a
        # skill's scripts via shell. Default None = no gate (used by smokes
        # that don't care about skills).
        self._skills = skills
        self._roles = roles
    description = (
        "在当前会话的工作目录中执行一条 shell 命令(bash -c)。受超时与输出大小上限保护。"
        "输出会截断回显;完整 stdout/stderr 落到 .chat_team/runs/<ts>.log,可通过 read_file 取回。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的完整命令(将以 bash -c 运行)"},
            "timeout": {
                "type": "integer",
                "description": "可选:覆盖默认超时秒数(默认见全局配置)",
            },
        },
        "required": ["command"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command must be a non-empty string")

        # Trigger-keyword gate: refuse to run any command that references a
        # protected skill's directory unless the user said a trigger keyword.
        # Closes the bypass where the LLM avoids skill(name=...) and shells
        # out to the skill scripts directly.
        if self._skills is not None:
            # Local import avoids a hard dep cycle (skill_tools imports base).
            from .skill_tools import scan_command_for_protected_skills
            scan_command_for_protected_skills(command, self._skills, ctx)

        timeout = kwargs.get("timeout") or ctx.settings.tools.shell_timeout_seconds
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            raise ToolError("timeout must be an integer (seconds)")
        if timeout <= 0 or timeout > 600:
            raise ToolError("timeout must be in (0, 600] seconds")

        if ctx.stream is not None:
            await ctx.stream.status("run_command 运行中...")

        runs_dir = ctx.cwd / ".chat_team" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = runs_dir / f"{ts}-{os.urandom(2).hex()}.log"

        # start_new_session=True puts bash into its own process group so that
        # `bash -c "sleep 1000 &"` style children share the pgid and can be
        # killed together on timeout. Without this, proc.kill() only kills
        # bash and orphans the children to the init reaper.
        scrubbed_env = _scrub_env(
            os.environ, ctx.settings.tools.shell_env_extra_drop,
        )
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(ctx.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            env=scrubbed_env,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True
            stdout_bytes = await _terminate_process_group(proc)

        return_code = proc.returncode if proc.returncode is not None else -1
        if ctx.stream is not None:
            if timed_out:
                await ctx.stream.status("run_command 超时,已终止")
            else:
                await ctx.stream.status(f"run_command 完成(exit={return_code})")
        try:
            log_path.write_bytes(stdout_bytes or b"")
        except OSError:
            pass

        max_bytes = ctx.settings.tools.shell_output_max_bytes
        body, truncated = _truncate(
            (stdout_bytes or b"").decode("utf-8", errors="replace"), max_bytes
        )
        rel_log = str(log_path.relative_to(ctx.cwd))
        header = (
            f"exit_code={return_code}"
            + (" (TIMEOUT)" if timed_out else "")
            + f"; full_log={rel_log}"
            + (f" (truncated, total {len(stdout_bytes)} bytes)" if truncated else "")
        )
        return f"{header}\n---\n{body}"
