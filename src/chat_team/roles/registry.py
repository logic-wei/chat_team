"""Discover roles from builtin yaml directory + user override directory.

Load order: src/chat_team/roles/builtin/*.yaml first, then
~/.chat_team/roles/*.yaml — same ``name`` in user dir overrides the builtin.
"""
from __future__ import annotations

from pathlib import Path

from .config import Role

BUILTIN_DIR = Path(__file__).parent / "builtin"


class RoleRegistry:
    def __init__(self, roles: dict[str, Role]):
        self._roles = roles

    @classmethod
    def load(cls, user_roles_dir: Path | None = None) -> "RoleRegistry":
        roles: dict[str, Role] = {}
        for path in sorted(BUILTIN_DIR.glob("*.yaml")):
            role = Role.from_yaml(path)
            roles[role.name] = role
        if user_roles_dir and user_roles_dir.exists():
            for path in sorted(user_roles_dir.glob("*.yaml")):
                role = Role.from_yaml(path)
                roles[role.name] = role
        return cls(roles)

    def get(self, name: str) -> Role:
        if name not in self._roles:
            raise KeyError(f"unknown role: {name}")
        return self._roles[name]

    def has(self, name: str) -> bool:
        return name in self._roles

    def names(self) -> list[str]:
        return sorted(self._roles.keys())

    def all(self) -> list[Role]:
        return [self._roles[n] for n in self.names()]

    def reload_in_place(self, user_roles_dir: "Path | None" = None) -> tuple[list[str], list[str], list[str]]:
        """Re-scan disk and atomically swap the internal role dict.

        Returns ``(added, removed, changed_names)`` so the caller can decide
        what follow-up work is needed (e.g. refresh the ``transfer_to_employee``
        enum, swap ``agent.role`` references on live agents).

        Atomicity: a brand-new dict is built off to the side, then assigned to
        ``self._roles`` in one statement. Every component that already holds a
        reference to *this* registry instance (``Dispatcher.roles``,
        ``RunCommandTool.roles``, ``SkillTool.roles``) sees the new contents on
        its next ``get``/``has``/``names``/``all`` call — no re-wiring needed.

        Live ``Agent`` instances, however, hold a reference to a *Role object*
        (not the registry), so a reloaded Role won't reach them automatically;
        the ``Reloader`` in ``app.py`` walks live sessions and swaps
        ``agent.role`` for roles that still exist after reload.
        """
        old = self._roles
        fresh = type(self).load(user_roles_dir)
        new = fresh._roles
        old_names = set(old)
        new_names = set(new)
        added = sorted(new_names - old_names)
        removed = sorted(old_names - new_names)
        changed = sorted(
            n for n in (old_names & new_names)
            if old[n] != new[n]
        )
        # Atomic swap — a single attribute assignment under the GIL.
        self._roles = new
        return added, removed, changed

