"""Shell command execution tool, sandboxed to the session's cwd.

* Runs via ``asyncio.create_subprocess_shell`` so we don't block the loop.
* Hard timeout from settings (default 30s).
* Stdout+stderr captured; returned to LLM truncated; full transcript saved
  under ``<cwd>/.chat_team/runs/<ts>.log`` so the agent can re-read it via
  ``read_file`` if needed.
* No environment scrubbing for now — the operator is trusted; tighten in a
  later stage if the bot ever runs untrusted prompts.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolError


def _truncate(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


class RunCommandTool(Tool):
    name = "run_command"
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
        timeout = kwargs.get("timeout") or ctx.settings.tools.shell_timeout_seconds
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            raise ToolError("timeout must be an integer (seconds)")
        if timeout <= 0 or timeout > 600:
            raise ToolError("timeout must be in (0, 600] seconds")

        runs_dir = ctx.cwd / ".chat_team" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = runs_dir / f"{ts}-{os.urandom(2).hex()}.log"

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(ctx.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            try:
                stdout_bytes, _ = await proc.communicate()
            except Exception:                                # noqa: BLE001
                stdout_bytes = b""
            timed_out = True

        return_code = proc.returncode if proc.returncode is not None else -1
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
