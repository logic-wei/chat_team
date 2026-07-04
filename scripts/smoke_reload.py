"""Smoke tests for the hot-reload feature (SIGHUP / ``chat-team --reload``).

Covers, end to end:
  1. ``reload_settings`` mutates the live ``Settings`` in place and reports
     applied vs. requires-restart correctly (incl. the int/float false-positive
     guard — a no-change reload must report zero changes).
  2. ``RoleRegistry.reload_in_place`` atomically swaps the internal dict so
     existing references see new roles (add / remove / change deltas).
  3. ``SkillRegistry.reload_in_place`` mirrors the same for skills.
  4. ``TransferToEmployeeTool.update_employees`` refreshes the JSON-schema enum.
  5. ``OpenAIChatCompletionProvider.apply_runtime_overrides`` updates the four
     runtime knobs without rebuilding the SDK client.
  6. ``Reloader.reload`` orchestrates the whole stack: settings + registries +
     transfer enum + live agent role-swap (history preserved) + image cache +
     LLM overrides, and reports orphans when a role disappears mid-session.
  7. SIGHUP actually fires ``on_sighup`` inside ``_run_with_shutdown`` without
     cancelling the main task (the WebSocket stays up).
  8. ``reload_daemon`` CLI helper returns the right exit code when no pid file
     exists (so ``chat-team --reload`` on a stopped bot fails cleanly).

Pure-Python: no live WS, no live LLM, no network.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_reload_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.app import _run_with_shutdown, configure_logging
from chat_team.agent.agent import Agent
from chat_team.agent.tools.base import ToolRegistry
from chat_team.agent.tools.transfer_tool import TransferToEmployeeTool
from chat_team.config import load_settings, reload_settings
from chat_team.daemon import reload_daemon
from chat_team.llm.base import ChatMessage, LLMProvider
from chat_team.llm.openai_provider import OpenAIChatCompletionProvider
from chat_team.reload import Reloader
from chat_team.roles.registry import RoleRegistry
from chat_team.session.notebook import Notebook
from chat_team.session.session import Session
from chat_team.skills.registry import SkillRegistry


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _FakeLLM(LLMProvider):
    """LLMProvider stub that supports apply_runtime_overrides (or not)."""

    def __init__(self, *, supports_overrides: bool = True):
        self._supports = supports_overrides
        if supports_overrides:
            # mimic the real provider's mutable knobs
            self._debug_log_enabled = False
            self._use_streaming = True
            self._max_retries = 3
            self._retry_initial_delay = 1.0

    def apply_runtime_overrides(self, *, debug_log_enabled, use_streaming,
                                max_retries, retry_initial_delay):
        if not self._supports:
            return []
        changed = []
        if self._debug_log_enabled != debug_log_enabled:
            self._debug_log_enabled = debug_log_enabled
            changed.append("debug_log_enabled")
        if self._use_streaming != use_streaming:
            self._use_streaming = use_streaming
            changed.append("use_streaming")
        new_r = max(1, int(max_retries))
        if self._max_retries != new_r:
            self._max_retries = new_r
            changed.append("max_retries")
        new_d = max(0.0, float(retry_initial_delay))
        if self._retry_initial_delay != new_d:
            self._retry_initial_delay = new_d
            changed.append("retry_initial_delay")
        return changed

    async def complete(self, req):  # type: ignore[override]
        return None


class _FakeSessions:
    def __init__(self):
        self._s: list[Session] = []

    def all_sessions(self):
        return self._s


class _FakeDispatcher:
    """Minimal stand-in for Dispatcher with the attrs Reloader reads."""
    def __init__(self, settings, llm):
        self.settings = settings
        self.roles = RoleRegistry.load(settings.paths.user_roles_dir)
        self.skills = SkillRegistry.load(settings.paths.user_skills_dir)
        self.tools = ToolRegistry()
        self.tools.register(TransferToEmployeeTool(self.roles.names()))
        self.llm = llm
        self._vision_llm = None
        self._fixed_role = None
        self.sessions = _FakeSessions()


def _write_role(home: Path, name: str, prompt: str) -> None:
    (home / "roles" / f"{name}.yaml").write_text(
        f"name: {name}\nsystem_prompt: {prompt}\ntools: []\n", encoding="utf-8",
    )


def _reseed_clean_config(settings):
    """Restore a known-good minimal config.yaml so one test's mutations don't
    bleed into the next test's ``load_settings()``."""
    settings.paths.config_yaml.write_text(
        "default_role: team_admin\nmode: team\n",
        encoding="utf-8",
    )




# --------------------------------------------------------------------------
# 1. reload_settings
# --------------------------------------------------------------------------

def test_reload_settings_applies_and_reports():
    print("== test 1: reload_settings applies + reports ==")
    settings = load_settings()
    _reseed_clean_config(settings)
    # Re-load to pick up the clean config so the no-change baseline holds.
    settings = load_settings()
    # No-change reload: the seeded template uses int values that must NOT trip
    # a false "changed" against the float dataclass defaults.
    r0 = reload_settings(settings)
    assert not r0.applied and not r0.requires_restart, (
        f"no-change reload must be empty, got {r0.summary()}"
    )
    # Real change.
    settings.paths.config_yaml.write_text(
        "session:\n  per_turn_transfer_cap: 9\nprivate_chat:\n  mode: open\n",
        encoding="utf-8",
    )
    r1 = reload_settings(settings)
    assert "session.per_turn_transfer_cap" in r1.applied
    assert "private_chat" in r1.applied
    assert settings.session.per_turn_transfer_cap == 9
    assert settings.private_chat.mode == "open"
    # Structural change → requires_restart, NOT applied.
    settings.paths.config_yaml.write_text("mode: solo\n", encoding="utf-8")
    r2 = reload_settings(settings)
    assert "mode" in r2.requires_restart
    assert settings.mode == "team", "mode must NOT be mutated live"
    # Bad YAML → error, nothing applied.
    try:
        settings.paths.config_yaml.write_text(":\n  : bad", encoding="utf-8")
        r3 = reload_settings(settings)
        assert r3.errors, "bad YAML must populate errors"
        assert settings.mode == "team", "settings must be untouched on parse error"
    finally:
        _reseed_clean_config(settings)
    print("   OK")


# --------------------------------------------------------------------------
# 2. RoleRegistry.reload_in_place
# --------------------------------------------------------------------------

def test_role_registry_reload_in_place():
    print("== test 2: RoleRegistry.reload_in_place atomic swap ==")
    settings = load_settings()
    _reseed_clean_config(settings)
    settings = load_settings()
    reg = RoleRegistry.load(settings.paths.user_roles_dir)
    base_names = set(reg.names())
    held_ref = reg  # the registry OBJECT held by e.g. RunCommandTool
    # Add a role.
    _write_role(settings.paths.home, "vip", "vip prompt")
    added, removed, changed = reg.reload_in_place(settings.paths.user_roles_dir)
    assert added == ["vip"], added
    assert "vip" in reg.names()
    assert held_ref is reg, "registry object identity must be stable"
    assert reg.get("vip").system_prompt == "vip prompt"
    # Change vip's prompt.
    _write_role(settings.paths.home, "vip", "vip prompt v2")
    a2, r2, c2 = reg.reload_in_place(settings.paths.user_roles_dir)
    assert c2 == ["vip"], c2
    assert reg.get("vip").system_prompt == "vip prompt v2"
    # Remove vip.
    (settings.paths.home / "roles" / "vip.yaml").unlink()
    a3, r3, c3 = reg.reload_in_place(settings.paths.user_roles_dir)
    assert r3 == ["vip"], r3
    assert "vip" not in reg.names()
    assert set(reg.names()) == base_names
    print("   OK")


# --------------------------------------------------------------------------
# 3. SkillRegistry.reload_in_place
# --------------------------------------------------------------------------

def test_skill_registry_reload_in_place():
    print("== test 3: SkillRegistry.reload_in_place atomic swap ==")
    settings = load_settings()
    _reseed_clean_config(settings)
    settings = load_settings()
    reg = SkillRegistry.load(settings.paths.user_skills_dir)
    held_ref = reg
    # Add a skill dir.
    sk = settings.paths.user_skills_dir / "demo"
    sk.mkdir(parents=True, exist_ok=True)
    (sk / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo skill\n---\nbody v1\n",
        encoding="utf-8",
    )
    added, removed, changed = reg.reload_in_place(settings.paths.user_skills_dir)
    assert added == ["demo"], added
    assert held_ref is reg
    assert "demo" in reg.names()
    # Change body.
    (sk / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo skill v2\n---\nbody v2\n",
        encoding="utf-8",
    )
    a2, r2, c2 = reg.reload_in_place(settings.paths.user_skills_dir)
    assert c2 == ["demo"], c2
    assert reg.get("demo").description == "a demo skill v2"
    # Remove.
    shutil.rmtree(sk)
    a3, r3, c3 = reg.reload_in_place(settings.paths.user_skills_dir)
    assert r3 == ["demo"], r3
    print("   OK")


# --------------------------------------------------------------------------
# 4. TransferToEmployeeTool.update_employees
# --------------------------------------------------------------------------

def test_transfer_tool_update_employees():
    print("== test 4: TransferToEmployeeTool.update_employees ==")
    t = TransferToEmployeeTool(["a", "b"])
    assert t.parameters["properties"]["employee"]["enum"] == ["a", "b"]
    assert t.update_employees(["a", "b", "c"]) is True
    assert t.parameters["properties"]["employee"]["enum"] == ["a", "b", "c"]
    assert t.update_employees(["a", "b", "c"]) is False, "no-op must return False"
    assert t.update_employees(["a"]) is True
    assert t.parameters["properties"]["employee"]["enum"] == ["a"]
    print("   OK")


# --------------------------------------------------------------------------
# 5. OpenAIChatCompletionProvider.apply_runtime_overrides
# --------------------------------------------------------------------------

def test_provider_apply_runtime_overrides():
    print("== test 5: provider.apply_runtime_overrides ==")
    prov = OpenAIChatCompletionProvider(
        api_key="k", debug_log_enabled=False, use_streaming=True,
        max_retries=3, retry_initial_delay=1.0,
    )
    changed = prov.apply_runtime_overrides(
        debug_log_enabled=True, use_streaming=False,
        max_retries=5, retry_initial_delay=2.0,
    )
    assert changed == ["debug_log_enabled", "use_streaming", "max_retries", "retry_initial_delay"]
    assert prov._debug_log_enabled is True
    assert prov._use_streaming is False
    assert prov._max_retries == 5
    assert prov._retry_initial_delay == 2.0
    no = prov.apply_runtime_overrides(
        debug_log_enabled=True, use_streaming=False,
        max_retries=5, retry_initial_delay=2.0,
    )
    assert no == [], "no-op must return []"
    print("   OK")


# --------------------------------------------------------------------------
# 6. Reloader.reload end-to-end
# --------------------------------------------------------------------------

def test_reloader_end_to_end():
    print("== test 6: Reloader.reload end-to-end ==")
    settings = load_settings()
    _reseed_clean_config(settings)
    settings = load_settings()
    disp = _FakeDispatcher(settings, _FakeLLM())
    # Materialise a live agent for team_admin so we can verify role-swap +
    # history preservation.
    sess = Session(
        "s1", Path("/tmp"), "team_admin", Notebook(Path("/tmp/nb.md"), 4096),
    )
    agent = Agent(disp.roles.get("team_admin"), sess, settings, disp.llm, disp.tools)
    agent.history = [ChatMessage(role="user", content="hi")]
    sess.agents_by_role["team_admin"] = agent
    disp.sessions._s.append(sess)
    orig_role_obj = agent.role

    reloader = Reloader(settings, [disp], reconfigure_logging=configure_logging)

    # No-change reload: 0 swaps, 0 errors.
    r0 = reloader.reload()
    assert r0.ok and not r0.agents_role_swapped, r0.summary()

    # Change a settings field, change team_admin prompt, add vip role.
    settings.paths.config_yaml.write_text(
        "session:\n  per_turn_transfer_cap: 5\nllm:\n  debug_log_enabled: true\n",
        encoding="utf-8",
    )
    _write_role(settings.paths.home, "team_admin", "CHANGED PROMPT")
    _write_role(settings.paths.home, "vip", "vip prompt")
    r1 = reloader.reload()
    assert r1.ok, r1.summary()
    assert "session.per_turn_transfer_cap" in r1.settings.applied
    assert "vip" in r1.roles_deltas[0][1], r1.roles_deltas
    assert "team_admin" in r1.roles_deltas[0][3], r1.roles_deltas  # changed
    assert r1.transfer_enum_updated
    assert "vip" in disp.tools.get("transfer_to_employee").available
    assert agent.role is not orig_role_obj, "agent.role should be a new object"
    assert agent.role.system_prompt == "CHANGED PROMPT"
    assert agent.history == [ChatMessage(role="user", content="hi")], "history preserved"
    assert settings.session.per_turn_transfer_cap == 5
    assert "debug_log_enabled" in r1.llm_overrides_applied
    assert disp.llm._debug_log_enabled is True

    # Remove vip → enum shrinks.
    (settings.paths.home / "roles" / "vip.yaml").unlink()
    r2 = reloader.reload()
    assert "vip" not in disp.tools.get("transfer_to_employee").available

    # Orphan: agent holds a role name that no longer exists on disk.
    ghost = type("R", (), {"name": "ghost_role", "system_prompt": "x"})()
    sess.agents_by_role.clear()
    sess.agents_by_role["ghost_role"] = agent
    agent.role = ghost
    r3 = reloader.reload()
    assert r3.agents_role_orphaned >= 1, r3.summary()
    assert agent.role is ghost, "orphaned agent keeps its frozen role"
    print("   OK")


# --------------------------------------------------------------------------
# 7. SIGHUP actually triggers on_sighup without cancelling the main task
# --------------------------------------------------------------------------

def test_sighup_triggers_reload_without_cancellation():
    print("== test 7: SIGHUP → on_sighup, main task keeps running ==")
    if not hasattr(signal, "SIGHUP"):
        print("   SKIP (no SIGHUP on this platform)")
        return
    settings = load_settings()
    _reseed_clean_config(settings)
    settings = load_settings()
    disp = _FakeDispatcher(settings, _FakeLLM())
    reloader = Reloader(settings, [disp], reconfigure_logging=configure_logging)
    fired = []
    orig = reloader.reload

    def recording_reload():
        fired.append(True)
        return orig()

    reloader.reload = recording_reload

    async def long_running():
        # Block forever (until cancelled by the test teardown).
        await asyncio.Event().wait()

    async def runner():
        await _run_with_shutdown(long_running(), on_sighup=reloader.reload)

    async def driver():
        task = asyncio.create_task(runner())
        await asyncio.sleep(0.15)  # let signal handlers install
        os.kill(os.getpid(), signal.SIGHUP)
        await asyncio.sleep(0.3)   # let the handler fire
        assert fired, "SIGHUP should have triggered on_sighup"
        # Main task must STILL be running (not cancelled by SIGHUP).
        assert not task.done(), "SIGHUP must not cancel the main task"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())
    print("   OK")


# --------------------------------------------------------------------------
# 8. reload_daemon returns exit 1 when no pid file
# --------------------------------------------------------------------------

def test_reload_daemon_no_pid():
    print("== test 8: reload_daemon returns 1 when no pid file ==")
    pid_path = Path(os.environ["CHAT_TEAM_HOME"]) / "chat_team.pid"
    if pid_path.exists():
        pid_path.unlink()
    code = reload_daemon(pid_path)
    assert code == 1, f"expected exit 1, got {code}"
    print("   OK")


# --------------------------------------------------------------------------

def main() -> None:
    test_reload_settings_applies_and_reports()
    test_role_registry_reload_in_place()
    test_skill_registry_reload_in_place()
    test_transfer_tool_update_employees()
    test_provider_apply_runtime_overrides()
    test_reloader_end_to_end()
    test_sighup_triggers_reload_without_cancellation()
    test_reload_daemon_no_pid()
    print("\nALL smoke_reload tests passed.")


if __name__ == "__main__":
    main()
