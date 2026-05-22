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
