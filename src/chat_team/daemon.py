"""Run the bot as a background daemon that survives SSH disconnect.

Default invocation backgrounds the process (double-fork detach); pass
``--foreground`` to run in the current terminal — the mode to use under
systemd/supervisor. ``--stop`` SIGTERMs a running daemon.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable

log_file_for_errors = None  # set by callers if they want errors echoed somewhere


def _process_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists (``kill 0`` succeeds)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Not ours, but it exists — treat as alive.
        return True
    return True


def _clear_pid(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def daemonize_and_run(out_path: Path, pid_path: Path, run: Callable[[], None]) -> None:
    """Detach from the controlling terminal and run ``run`` in the background.

    Classic double-fork: the first child becomes a session leader via
    ``setsid`` (dropping the controlling TTY), then forks again so the final
    daemon can never re-acquire a TTY. stdin → /dev/null, stdout/stderr →
    ``out_path`` (so prints and uncaught tracebacks survive instead of dying
    with the SSH session). The daemon PID is recorded in ``pid_path`` and
    removed on clean exit.

    The original process blocks just long enough to confirm the daemon came up
    (or crashed on a bad config), prints a status line, then returns.
    """
    # Refuse to double-start.
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text(encoding="utf-8").strip())
            if _process_alive(old_pid):
                print(
                    f"chat_team already running (pid={old_pid}); "
                    f"use --stop first.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except (ValueError, OSError):
            pass
        _clear_pid(pid_path)  # stale

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # First fork — the parent reports status and exits.
    first_pid = os.fork()
    if first_pid > 0:
        os.waitpid(first_pid, 0)
        _report_startup(out_path, pid_path)
        return

    # First child: new session, no controlling TTY.
    os.setsid()
    os.umask(0o022)

    # Second fork — the daemon can never reacquire a TTY.
    second_pid = os.fork()
    if second_pid > 0:
        os._exit(0)

    # ---- Grandchild: the actual daemon ----
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(devnull_fd, 0)   # stdin  → /dev/null
    os.dup2(out_fd, 1)       # stdout → out file
    os.dup2(out_fd, 2)       # stderr → out file
    if devnull_fd > 2:
        os.close(devnull_fd)
    if out_fd > 2:
        os.close(out_fd)

    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(_clear_pid, pid_path)

    run()


_STARTUP_WAIT = 1.0   # max seconds to wait for the daemon to write its PID
_STARTUP_GRACE = 1.5  # seconds to confirm the daemon stays up after PID write


def _report_startup(out_path: Path, pid_path: Path) -> None:
    """Wait for the PID file, then watch the daemon through a grace period.

    The PID is written *before* ``asyncio.run`` starts, so "PID file exists" only
    proves the process forked — not that it initialised. A bad config (missing
    key, unparseable YAML, ...) crashes the daemon within tens of milliseconds,
    well inside ``_STARTUP_GRACE``; if it dies there we surface the error from
    ``out_path`` instead of falsely reporting success.
    """
    deadline = time.monotonic() + _STARTUP_WAIT
    pid: int | None = None
    while time.monotonic() < deadline:
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pid = None
                break
            if _process_alive(pid):
                break
            pid = None
            break
        time.sleep(0.05)

    if pid is None or not _process_alive(pid):
        _fail(out_path)

    grace_deadline = time.monotonic() + _STARTUP_GRACE
    while time.monotonic() < grace_deadline:
        if not _process_alive(pid):
            _fail(out_path)
        time.sleep(0.1)

    print(f"chat_team started in background (pid={pid}).")
    print(f"  stdout/stderr → {out_path}")
    print(f"  log           → {out_path.parent / "chat_team.log"}")
    print(f"  pid file      → {pid_path}")
    print("  stop with:    chat-team --stop")


def _fail(out_path: Path) -> None:
    print("chat_team failed to start. Recent output:", file=sys.stderr)
    print(f"  {out_path}", file=sys.stderr)
    _print_tail(out_path)
    sys.exit(1)


def _print_tail(out_path: Path) -> None:
    try:
        with open(out_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 2000))
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        sys.stderr.write(text)
        if text and not text.endswith("\n"):
            sys.stderr.write("\n")
    except OSError:
        pass


def stop_daemon(pid_path: Path, timeout: float = 10.0) -> int:
    """SIGTERM the daemon named by ``pid_path``; SIGKILL on timeout.

    Returns a process exit code (0 = stopped, 1 = nothing to stop).
    """
    if not pid_path.exists():
        print(f"chat_team is not running (no pid file at {pid_path}).", file=sys.stderr)
        return 1
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as e:
        print(f"invalid pid file {pid_path}: {e}", file=sys.stderr)
        _clear_pid(pid_path)
        return 1

    if not _process_alive(pid):
        print(f"chat_team is not running (stale pid file, pid={pid}).", file=sys.stderr)
        _clear_pid(pid_path)
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_pid(pid_path)
        print(f"chat_team stopped (pid={pid}).")
        return 0

    print(f"sending SIGTERM to chat_team (pid={pid})...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            _clear_pid(pid_path)
            print(f"chat_team stopped (pid={pid}).")
            return 0
        time.sleep(0.1)

    print(f"no exit within {timeout:.0f}s; sending SIGKILL...", file=sys.stderr)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _clear_pid(pid_path)
    print(f"chat_team force-killed (pid={pid}).", file=sys.stderr)
    return 0


def reload_daemon(pid_path: Path) -> int:
    """SIGHUP the daemon named by ``pid_path`` to trigger a hot reload.

    The running process's SIGHUP handler (installed in
    ``app._run_with_shutdown``) re-reads config.yaml / team.md / roles / skills
    and applies the changes in place — without dropping the WebSocket or
    interrupting in-flight turns. Returns a process exit code.

    Unlike ``stop_daemon``, this doesn't wait: SIGHUP is asynchronous by
    nature (the reload touches disk + maybe an LLM call for compaction is in
    flight), and the daemon logs the reload result to
    ``~/.chat_team/logs/chat_team.log``.
    """
    if not hasattr(signal, "SIGHUP"):
        print("SIGHUP not available on this platform; hot reload unsupported.",
              file=sys.stderr)
        return 1
    if not pid_path.exists():
        print(f"chat_team is not running (no pid file at {pid_path}).", file=sys.stderr)
        return 1
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as e:
        print(f"invalid pid file {pid_path}: {e}", file=sys.stderr)
        _clear_pid(pid_path)
        return 1
    if not _process_alive(pid):
        print(f"chat_team is not running (stale pid file, pid={pid}).", file=sys.stderr)
        _clear_pid(pid_path)
        return 1
    try:
        os.kill(pid, signal.SIGHUP)
    except ProcessLookupError:
        _clear_pid(pid_path)
        print(f"chat_team is not running (pid={pid} vanished).", file=sys.stderr)
        return 1
    print(
        f"SIGHUP sent to chat_team (pid={pid}); hot reload triggered. "
        f"Check ~/.chat_team/logs/chat_team.log for the result."
    )
    return 0
