"""``skill`` and ``skill_read_file`` — load instructional skill packs by name.

A *skill* is a directory under ``~/.chat_team/skills/<name>/`` (or the
package builtin dir) holding a ``SKILL.md`` (frontmatter + body) and
optional auxiliary files. Skills are the no-code path to extending agent
capabilities: write a markdown doc, list the skill's name in a role's
``skills:`` whitelist, and the agent will see it in its system prompt
TOC and can pull the body via the ``skill`` tool.

Per-role gating happens twice:
1. ``Agent._build_system_messages`` only lists skills in
   ``role.skills ∩ registry.names()`` — what the LLM sees.
2. ``SkillTool.run`` re-checks against the current role's whitelist at
   invocation time — defence in depth if the LLM hallucinates a name.

The registry-level ``enum`` in ``parameters`` is the full set across all
roles; per-role narrowing of the enum would require one tool instance per
role and isn't worth the complexity.
"""
from __future__ import annotations

import difflib
import os
import unicodedata
from pathlib import Path
from typing import Any

from ...roles.registry import RoleRegistry
from ...skills.registry import SkillRegistry
from .base import Tool, ToolContext, ToolError


def _resolve_under_skill(skill_dir: Path, rel: str) -> Path:
    """Reject absolute paths and ``..``; ensure result is inside ``skill_dir``."""
    if not isinstance(rel, str) or not rel:
        raise ToolError("path must be a non-empty string")
    if os.path.isabs(rel) or ".." in Path(rel).parts:
        raise ToolError("absolute paths and '..' are not allowed; use a path relative to the skill directory")
    candidate = (skill_dir / rel).resolve()
    base = skill_dir.resolve()
    try:
        common = os.path.commonpath([str(candidate), str(base)])
    except ValueError:
        raise ToolError("path escapes the skill directory")
    if common != str(base):
        raise ToolError("path escapes the skill directory")
    return candidate


def _allowed_skills_for_current_role(
    ctx: ToolContext, roles: RoleRegistry, skills: SkillRegistry,
) -> list[str]:
    role_name = ctx.session.current_role
    try:
        role = roles.get(role_name)
    except KeyError:
        return []
    return [n for n in role.skills if skills.has(n)]


def _clean_skill_name(raw: str) -> str:
    """Normalize a model-produced skill token into a lookup candidate."""
    s = unicodedata.normalize("NFKC", raw).strip()
    # Common wrappers a model may include when copying from markdown.
    while len(s) >= 2 and (
        (s[0] == "`" and s[-1] == "`")
        or (s[0] == "'" and s[-1] == "'")
        or (s[0] == '"' and s[-1] == '"')
        or (s[0] == "“" and s[-1] == "”")
        or (s[0] == "‘" and s[-1] == "’")
    ):
        s = s[1:-1].strip()
    # TOC copy/paste artifacts such as "- name: ...".
    s = s.lstrip("-* ").strip()
    return s


def _resolve_skill_name_or_raise(raw_name: Any, skills: SkillRegistry) -> str:
    """Resolve a user/model-provided name to a concrete registry key."""
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ToolError("name/skill must be a non-empty string")
    if skills.has(raw_name):
        return raw_name

    cleaned = _clean_skill_name(raw_name)
    if skills.has(cleaned):
        return cleaned

    # Support occasional "name: description" copies from the TOC.
    split_candidates = [cleaned]
    for sep in (":", "："):
        if sep in cleaned:
            split_candidates.append(cleaned.split(sep, 1)[0].strip())
    for cand in split_candidates:
        if skills.has(cand):
            return cand

    names = skills.names()
    lowered = [n for n in names if n.casefold() == cleaned.casefold()]
    if len(lowered) == 1:
        return lowered[0]
    if len(lowered) > 1:
        raise ToolError(
            f"skill 名称不唯一: {raw_name!r}; 请使用精确名称: {', '.join(sorted(lowered))}"
        )

    suggestions = difflib.get_close_matches(cleaned, names, n=3, cutoff=0.6)
    hint = f"; maybe: {', '.join(suggestions)}" if suggestions else ""
    raise ToolError(f"unknown skill: {raw_name}{hint}")


class SkillTool(Tool):
    name = "skill"
    description = (
        "按名加载一个 skill 的正文(SKILL.md body),把它当作本次回答的操作指引/规约。"
        "可用的 skill 名见你的系统提示中的 [可用 skills] 块。skill 一般会指明做事流程、"
        "checklist 或风格要求;调用后请按 body 内容执行。"
    )

    def __init__(self, skills: SkillRegistry, roles: RoleRegistry):
        self._skills = skills
        self._roles = roles
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": skills.names(),
                    "description": "skill 名(见系统提示 [可用 skills] 块);必须严格匹配。",
                },
            },
            "required": ["name"],
        }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        name = _resolve_skill_name_or_raise(kwargs.get("name"), self._skills)
        allowed = _allowed_skills_for_current_role(ctx, self._roles, self._skills)
        if name not in allowed:
            raise ToolError(
                f"skill {name!r} 不在当前角色 ({ctx.session.current_role}) 的 skills 白名单内;"
                f"白名单: {', '.join(allowed) if allowed else '(空)'}"
            )
        skill = self._skills.get(name)
        body = skill.body
        aux = _list_aux_files(skill.directory)
        work_dir = ctx.cwd.resolve()
        body = (
            f"{body}\n\n"
            f"[本 skill 目录] {skill.directory}\n"
            f"[当前会话工作目录] {work_dir}\n"
            "需要使用该 skill 自带脚本/资源文件时,请直接基于 skill 目录路径操作,不要在系统目录中盲目搜索。\n"
            "调用 skill 脚本时可切换到 skill 目录执行命令,但所有业务输入/输出文件路径"
            "(如 --input/--output)必须指向当前会话工作目录及其子目录。"
        )
        if aux:
            files_line = ", ".join(aux)
            body = (
                f"{body}\n\n[本 skill 附带辅助文件] {files_line}\n"
                f"如需查看,调用 skill_read_file(skill={name!r}, path=...)。"
            )
        return body


class SkillReadFileTool(Tool):
    name = "skill_read_file"
    description = (
        "读取某个 skill 目录下的辅助文件(相对路径)。仅在 SKILL.md body 提示你需要时调用。"
        "拒绝绝对路径和 ../;只读,UTF-8 文本。"
    )

    def __init__(self, skills: SkillRegistry, roles: RoleRegistry):
        self._skills = skills
        self._roles = roles
        self.parameters = {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "enum": skills.names(),
                    "description": "skill 名;必须是当前角色已开通的 skill。",
                },
                "path": {
                    "type": "string",
                    "description": "相对该 skill 目录的文件路径,例如 'checklist.md' 或 'examples/good_pr.md'。",
                },
            },
            "required": ["skill", "path"],
        }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        skill_name = _resolve_skill_name_or_raise(kwargs.get("skill"), self._skills)
        rel = kwargs.get("path")
        allowed = _allowed_skills_for_current_role(ctx, self._roles, self._skills)
        if skill_name not in allowed:
            raise ToolError(
                f"skill {skill_name!r} 不在当前角色 ({ctx.session.current_role}) 的 skills 白名单内"
            )
        skill = self._skills.get(skill_name)
        target = _resolve_under_skill(skill.directory, rel)
        if target.name == "SKILL.md" and target.parent == skill.directory.resolve():
            raise ToolError("SKILL.md 已通过 skill 工具返回,无需再读")
        if not target.exists():
            raise ToolError(f"file not found: {rel}")
        if not target.is_file():
            raise ToolError(f"not a regular file: {rel}")
        max_bytes = ctx.settings.tools.file_read_max_bytes
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolError(f"file too large ({size} bytes > {max_bytes}); refusing to read")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError("file is not valid UTF-8 text")


def _list_aux_files(skill_dir: Path) -> list[str]:
    """Relative paths of every regular file in the skill dir except SKILL.md."""
    base = skill_dir.resolve()
    out: list[str] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if rel.parts and rel.parts[0].startswith("."):
            continue
        if str(rel) == "SKILL.md":
            continue
        out.append(str(rel))
    return out
