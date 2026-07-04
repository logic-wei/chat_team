"""Skill dataclass + SKILL.md frontmatter parser.

A skill is a directory under ``~/.chat_team/skills/<name>/`` (or the package
builtin dir) containing a ``SKILL.md`` file with YAML frontmatter and
instructional body. The body is what the ``skill`` tool returns to the LLM.
Auxiliary files in the same directory are reachable via ``skill_read_file``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Skill:
    name: str
    description: str
    body: str
    directory: Path
    # Optional trigger-keyword gate. When set, SkillTool.run refuses to
    # load this skill unless the most recent user message contains at
    # least one of these substrings (case-insensitive). Used for skills
    # that MUST be explicitly requested (e.g. report generation), where
    # a prompt-only "禁止自作主张" constraint is unreliable.
    trigger_keywords: tuple[str, ...] = ()

    @classmethod
    def from_dir(cls, directory: Path) -> "Skill":
        skill_md = directory / "SKILL.md"
        if not skill_md.exists() or not skill_md.is_file():
            raise ValueError(f"missing SKILL.md in {directory}")
        text = skill_md.read_text(encoding="utf-8")
        front, body = _parse_frontmatter(text, source=skill_md)
        name = front.get("name")
        description = front.get("description")
        if not name or not isinstance(name, str):
            raise ValueError(f"SKILL.md missing required 'name': {skill_md}")
        if not description or not isinstance(description, str):
            raise ValueError(f"SKILL.md missing required 'description': {skill_md}")
        if name != directory.name:
            raise ValueError(
                f"SKILL.md name mismatch in {skill_md}: "
                f"frontmatter name={name!r} but directory={directory.name!r}; they must match"
            )
        # Optional trigger_keywords: a list of substrings. Validated to be
        # a list of non-empty strings; bad shape → ignored with no fail.
        tk_raw = front.get("trigger_keywords") or []
        trigger_keywords: tuple[str, ...] = ()
        if isinstance(tk_raw, list):
            trigger_keywords = tuple(
                str(s).strip() for s in tk_raw if isinstance(s, str) and s.strip()
            )
        return cls(
            name=name,
            description=description.strip(),
            body=body.strip(),
            directory=directory,
            trigger_keywords=trigger_keywords,
        )


def _parse_frontmatter(text: str, source: Path) -> tuple[dict[str, Any], str]:
    """Split a markdown file with leading ``---`` YAML frontmatter.

    Returns (frontmatter_dict, body). Raises ValueError if the frontmatter
    fence is malformed. A file without a leading ``---`` is rejected so
    skills always carry name/description metadata.
    """
    if not text.startswith("---"):
        raise ValueError(
            f"SKILL.md must start with YAML frontmatter (--- ... ---): {source}"
        )
    rest = text[3:]
    # Tolerate either '\n' or '\r\n' after the opening fence.
    if rest.startswith("\n"):
        rest = rest[1:]
    elif rest.startswith("\r\n"):
        rest = rest[2:]
    end = rest.find("\n---")
    if end < 0:
        raise ValueError(f"SKILL.md frontmatter not terminated by '---': {source}")
    front_text = rest[:end]
    body = rest[end + 4:]
    if body.startswith("\n"):
        body = body[1:]
    elif body.startswith("\r\n"):
        body = body[2:]
    try:
        front = yaml.safe_load(front_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"SKILL.md frontmatter YAML invalid in {source}: {exc}")
    if not isinstance(front, dict):
        raise ValueError(f"SKILL.md frontmatter must be a mapping: {source}")
    return front, body
