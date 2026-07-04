"""Hot-reload orchestrator: re-read config + roles + skills + team.md and
apply the changes to a *running* process without dropping the WebSocket.

Triggered by SIGHUP (see ``app._install_sighup_handler``) or by the
``chat-team --reload`` CLI subcommand (which sends SIGHUP to the daemon PID).

Design
------
The reloader mutates shared objects **in place** wherever possible so the
existing wiring (Dispatcher → SessionManager / Agent / ToolRegistry /
LLMProvider) keeps its references and just sees new values on the next read:

* ``Settings`` — ``reload_settings`` overwrites fields on the live instance.
* ``RoleRegistry`` / ``SkillRegistry`` — ``reload_in_place`` atomically swaps
  the internal dict, so every holder of the registry instance sees new
  contents. ``RunCommandTool.roles`` / ``SkillTool.skills`` are unaffected
  because they hold the *registry*, not its dict.
* ``TransferToEmployeeTool`` — ``update_employees`` rebuilds the enum.
* Live ``Agent`` instances — their ``role`` attribute is swapped to the
  reloaded ``Role`` object (history is untouched; the next turn's system
  prompt rebuild picks up the new prompt/tools).
* ``OpenAIChatCompletionProvider`` — ``apply_runtime_overrides`` updates the
  four knobs not baked into the SDK client.
* ``ImageDataURICache`` singleton — ``configure_default_cache`` recreates it
  with the new vision resize knobs.

What stays restart-only (reported, not applied):
  ``mode``, ``bots``, ``workspace_root``, ``mcp``,
  ``llm.{api_key, base_url, request_timeout_seconds, http_debug_log_enabled}``,
  ``llm.vision.{api_key, base_url}``.

Concurrency: reload is **best-effort atomic at the attribute level**. Python
attribute assignment is atomic under the GIL, so individual field swaps are
safe. A turn in flight may see pre- or post-reload config for a single read,
but never corrupted state. We deliberately do NOT try to acquire every
session lock (there can be hundreds); the cost of one turn using slightly
mixed config is nil compared to the cost of stalling all sessions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .config import ReloadReport, reload_settings
from .llm.image_cache import configure_default_cache

if TYPE_CHECKING:
    from .dispatcher import Dispatcher
    from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class CombinedReloadReport:
    """Aggregate result of one reload pass across all dispatchers."""
    settings: ReloadReport = field(default_factory=ReloadReport)
    # Per-dispatcher registry deltas. In team mode there's one entry; in solo
    # mode there's one per bot. Index aligns with ``Reloader.dispatchers``.
    roles_deltas: list[tuple[str, list[str], list[str], list[str]]] = field(default_factory=list)
    skills_deltas: list[tuple[str, list[str], list[str], list[str]]] = field(default_factory=list)
    transfer_enum_updated: bool = False
    agents_role_swapped: int = 0
    agents_role_orphaned: int = 0   # role disappeared after reload; agent left as-is
    llm_overrides_applied: list[str] = field(default_factory=list)
    image_cache_reconfigured: bool = False
    logging_reconfigured: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.settings.ok

    def summary(self) -> str:
        if self.errors:
            return "reload FAILED: " + "; ".join(self.errors)
        parts: list[str] = []
        if self.settings.applied:
            parts.append("settings.applied=[" + ", ".join(self.settings.applied) + "]")
        if self.settings.requires_restart:
            parts.append("settings.requires_restart=[" + ", ".join(self.settings.requires_restart) + "]")
        if self.roles_deltas:
            rd = self.roles_deltas[0]
            parts.append(f"roles(+{len(rd[1])} -{len(rd[2])} ~{len(rd[3])})")
        if self.skills_deltas:
            sd = self.skills_deltas[0]
            parts.append(f"skills(+{len(sd[1])} -{len(sd[2])} ~{len(sd[3])})")
        if self.transfer_enum_updated:
            parts.append("transfer_enum_refreshed")
        if self.agents_role_swapped:
            parts.append(f"agents_role_swapped={self.agents_role_swapped}")
        if self.agents_role_orphaned:
            parts.append(f"agents_role_orphaned={self.agents_role_orphaned}")
        if self.llm_overrides_applied:
            parts.append("llm_overrides=[" + ", ".join(self.llm_overrides_applied) + "]")
        if self.image_cache_reconfigured:
            parts.append("image_cache_reconfigured")
        if self.logging_reconfigured:
            parts.append("logging_reconfigured")
        return "reload OK; " + (" ".join(parts) if parts else "no changes")


class Reloader:
    """Owns the live ``Settings`` + one or more ``Dispatcher`` instances and
    applies a hot reload to all of them.

    Constructed once in ``app._async_main`` (team: 1 dispatcher; solo: N) and
    invoked from the SIGHUP handler. ``reconfigure_logging`` is injected as a
    callable (rather than imported) to avoid a circular import with ``app.py``
    and to keep logging policy in one place.
    """

    def __init__(
        self,
        settings: "Settings",
        dispatchers: "list[Dispatcher]",
        *,
        reconfigure_logging: "Callable[[Settings], None] | None" = None,
    ):
        self.settings = settings
        self.dispatchers = list(dispatchers)
        self._reconfigure_logging = reconfigure_logging

    def reload(self) -> CombinedReloadReport:
        report = CombinedReloadReport()

        # 1. Settings (config.yaml + team.md) — mutate in place.
        report.settings = reload_settings(self.settings)
        if not report.settings.ok:
            report.errors.extend(report.settings.errors)
            log.error("reload aborted: %s", report.settings.summary())
            return report

        # 2. Logging — reconfigure level + rotation if the policy changed.
        if self._reconfigure_logging is not None:
            try:
                self._reconfigure_logging(self.settings)
                # Only flag if log_level or logging.* actually moved; cheap to
                # always mark since reconfigure is idempotent and the operator
                # asked for a reload.
                report.logging_reconfigured = (
                    "log_level" in report.settings.applied
                    or any(a.startswith("logging.") for a in report.settings.applied)
                )
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"logging: {exc!r}")
                log.exception("logging reconfigure failed during reload")

        # 3. Per-dispatcher: registries, transfer enum, agent role swap, llm.
        for disp in self.dispatchers:
            label = getattr(disp, "_fixed_role", None) or "team"
            try:
                self._reload_dispatcher(disp, label, report)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"dispatcher[{label}]: {exc!r}")
                log.exception("reload failed for dispatcher %s", label)

        # 4. Image cache singleton — recreate with new vision resize knobs.
        #    The provider reads default_cache() per call, so the new instance
        #    is picked up immediately. Only flag if a vision cache knob moved.
        try:
            v = self.settings.llm.vision
            configure_default_cache(
                max_inline_bytes=v.max_inline_bytes,
                oversized_image=v.oversized_image,
                resize_long_side=v.resize_long_side,
                resize_quality=v.resize_quality,
            )
            report.image_cache_reconfigured = any(
                a.startswith("llm.vision.")
                and a.split(".")[-1] in {
                    "max_inline_bytes", "oversized_image",
                    "resize_long_side", "resize_quality",
                }
                for a in report.settings.applied
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"image_cache: {exc!r}")
            log.exception("image_cache reconfigure failed during reload")

        log.info("%s", report.summary())
        return report

    def _reload_dispatcher(
        self, disp: "Dispatcher", label: str, report: CombinedReloadReport,
    ) -> None:
        # --- roles ---
        added, removed, changed = disp.roles.reload_in_place(
            self.settings.paths.user_roles_dir,
        )
        report.roles_deltas.append((label, added, removed, changed))
        if added or removed or changed:
            log.info(
                "dispatcher[%s] roles reloaded: +%s -%s ~%s",
                label, added, removed, changed,
            )

        # --- skills ---
        s_added, s_removed, s_changed = disp.skills.reload_in_place(
            self.settings.paths.user_skills_dir,
        )
        report.skills_deltas.append((label, s_added, s_removed, s_changed))
        if s_added or s_removed or s_changed:
            log.info(
                "dispatcher[%s] skills reloaded: +%s -%s ~%s",
                label, s_added, s_removed, s_changed,
            )

        # --- transfer_to_employee enum (team mode only; solo skips transfer) ---
        transfer_tool = None
        if disp.tools and disp.tools.has("transfer_to_employee"):
            transfer_tool = disp.tools.get("transfer_to_employee")
        if transfer_tool is not None and hasattr(transfer_tool, "update_employees"):
            updated = transfer_tool.update_employees(disp.roles.names())
            if updated:
                report.transfer_enum_updated = True
                log.info(
                    "dispatcher[%s] transfer_to_employee enum refreshed -> %s",
                    label, transfer_tool.available,
                )

        # --- live agents: swap role references ---
        # Walk every in-memory session and replace each agent's `role` with the
        # reloaded Role object (same name). History is untouched; the next turn
        # rebuilds the system prompt from the new role. Agents whose role was
        # deleted from disk are left running with their frozen role (a running
        # conversation shouldn't be killed mid-flight) and counted as orphaned.
        for session in disp.sessions.all_sessions():
            for agent in list(session.agents_by_role.values()):
                role_name = agent.role.name
                if disp.roles.has(role_name):
                    new_role = disp.roles.get(role_name)
                    # Compare by value (dataclass __eq__) not identity: a reload
                    # re-parses every YAML into fresh objects, so an identity
                    # check would count a no-content-change reload as a swap.
                    # Only real content changes (prompt/tools/llm overrides)
                    # should swap and reset the agent's next-turn prompt.
                    if new_role != agent.role:
                        agent.role = new_role
                        report.agents_role_swapped += 1
                else:
                    report.agents_role_orphaned += 1
                    log.warning(
                        "dispatcher[%s] session=%s role=%s was removed from "
                        "disk; existing agent keeps running with the old role "
                        "until the session ends",
                        label, session.session_id, role_name,
                    )

        # --- LLM runtime overrides (chat + vision providers) ---
        applied_knobs = self._apply_llm_overrides(disp.llm)
        if disp._vision_llm is not None and disp._vision_llm is not disp.llm:
            applied_knobs.extend(self._apply_llm_overrides(disp._vision_llm))
        # De-dup (chat+vision may share the provider instance).
        seen: set[str] = set()
        for k in applied_knobs:
            if k not in seen:
                seen.add(k)
                report.llm_overrides_applied.append(k)

    def _apply_llm_overrides(self, llm: Any) -> list[str]:
        """Push the four safe runtime knobs onto a provider, if it supports it.

        Reads from ``self.settings.llm`` (already reloaded in place by step 1).
        Providers without ``apply_runtime_overrides`` (e.g. a mock in tests)
        are silently skipped — they'll just keep their construction-time knobs.
        """
        if not hasattr(llm, "apply_runtime_overrides"):
            return []
        return llm.apply_runtime_overrides(
            debug_log_enabled=self.settings.llm.debug_log_enabled,
            use_streaming=self.settings.llm.use_streaming,
            max_retries=self.settings.llm.max_retries,
            retry_initial_delay=self.settings.llm.retry_initial_delay,
        )
