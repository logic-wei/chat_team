"""Discover skills from builtin directory + user override directory.

Load order: ``src/chat_team/skills/builtin/<name>/SKILL.md`` first, then
``~/.chat_team/skills/<name>/SKILL.md`` — same name in user dir overrides
the builtin.

A malformed skill (missing SKILL.md / bad frontmatter / name mismatch) is
logged at WARNING and skipped, so one bad directory cannot bring down the
whole load.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Skill

log = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).parent / "builtin"


class SkillRegistry:
    def __init__(self, skills: dict[str, Skill]):
        self._skills = skills

    @classmethod
    def load(cls, user_skills_dir: Path | None = None) -> "SkillRegistry":
        skills: dict[str, Skill] = {}
        cls._scan_into(BUILTIN_DIR, skills, source_label="builtin")
        if user_skills_dir is not None and user_skills_dir.exists():
            cls._scan_into(user_skills_dir, skills, source_label="user")
        return cls(skills)

    @staticmethod
    def _scan_into(root: Path, sink: dict[str, Skill], source_label: str) -> None:
        if not root.exists():
            return
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                continue
            try:
                skill = Skill.from_dir(child)
            except ValueError as exc:
                log.warning("skipping %s skill at %s: %s", source_label, child, exc)
                continue
            sink[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill: {name}")
        return self._skills[name]

    def has(self, name: str) -> bool:
        return name in self._skills

    def names(self) -> list[str]:
        return sorted(self._skills.keys())

    def all(self) -> list[Skill]:
        return [self._skills[n] for n in self.names()]

    def reload_in_place(self, user_skills_dir: "Path | None" = None) -> tuple[list[str], list[str], list[str]]:
        """Re-scan disk and atomically swap the internal skill dict.

        Mirrors ``RoleRegistry.reload_in_place``: build a fresh dict off to
        the side, then assign ``self._skills = new`` in one statement so every
        holder of this registry instance (``Dispatcher.skills``,
        ``SkillTool.skills``, ``SkillReadFileTool.skills``) sees the new
        skills on its next call. Returns ``(added, removed, changed)``.
        """
        old = self._skills
        fresh = type(self).load(user_skills_dir)
        new = fresh._skills
        old_names = set(old)
        new_names = set(new)
        added = sorted(new_names - old_names)
        removed = sorted(old_names - new_names)
        changed = sorted(
            n for n in (old_names & new_names)
            if old[n] != new[n]
        )
        self._skills = new
        return added, removed, changed


    def render_toc(self, allowed: list[str]) -> str:
        """Render a ``- name: description-first-line`` list for the system prompt.

        Filters by an allowlist (typically ``role.skills``); unknown names are
        silently dropped, matching ``ToolRegistry.specs_for``'s behaviour.
        Multi-line descriptions are truncated to their first line so a long
        description can't blow up the system prompt.
        """
        lines: list[str] = []
        for name in allowed:
            skill = self._skills.get(name)
            if skill is None:
                continue
            first = skill.description.split("\n", 1)[0].strip()
            lines.append(f"- {skill.name}: {first}")
        return "\n".join(lines)
