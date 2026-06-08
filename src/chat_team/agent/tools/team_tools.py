"""Boss-side tools that read/write ~/.chat_team/team.md and ~/.chat_team/roles/*.

These tools deliberately bypass the cwd sandbox: they operate on absolute
paths under ``settings.paths`` (the chat_team home), which is exactly what
``file_tools.py`` refuses to do. They are only registered by
``chat_team.boss`` and never reach the WeCom-side dispatcher.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ...roles.config import Role
from ...roles.registry import BUILTIN_DIR
from ...skills.config import Skill
from ...skills.registry import BUILTIN_DIR as SKILL_BUILTIN_DIR
from .base import Tool, ToolContext, ToolError

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _atomic_write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(target.parent),
        prefix=f".{target.name}.", suffix=".tmp", delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    os.replace(tmp_path, target)


def _builtin_names() -> set[str]:
    return {p.stem for p in BUILTIN_DIR.glob("*.yaml")}


class ListRolesTool(Tool):
    name = "list_roles"
    description = (
        "列出当前所有虚拟员工(builtin + 用户自定义)。返回每个 role 的 name / "
        "display_name / source(builtin|user)/ tools 列表,方便了解现状。"
    )
    parameters = {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        user_dir = ctx.settings.paths.user_roles_dir
        builtin_names = _builtin_names()
        seen: dict[str, dict[str, Any]] = {}
        for path in sorted(BUILTIN_DIR.glob("*.yaml")):
            role = Role.from_yaml(path)
            seen[role.name] = {
                "name": role.name,
                "display_name": role.display_name,
                "source": "builtin",
                "tools": role.tools,
                "description": role.description,
            }
        if user_dir.exists():
            for path in sorted(user_dir.glob("*.yaml")):
                role = Role.from_yaml(path)
                seen[role.name] = {
                    "name": role.name,
                    "display_name": role.display_name,
                    "source": "user_override" if role.name in builtin_names else "user",
                    "tools": role.tools,
                    "description": role.description,
                }
        if not seen:
            return "(no roles)"
        lines = []
        for entry in sorted(seen.values(), key=lambda e: e["name"]):
            tools = ", ".join(entry["tools"]) if entry["tools"] else "-"
            desc = entry["description"] or ""
            lines.append(
                f"- {entry['name']} ({entry['display_name']}) [{entry['source']}] "
                f"tools=[{tools}]"
                + (f"\n  desc: {desc}" if desc else "")
            )
        return "\n".join(lines)


class ReadRoleTool(Tool):
    name = "read_role"
    description = (
        "读出某个 role 的 YAML 原文。先在用户目录(~/.chat_team/roles/)查找,"
        "找不到再回落到 builtin。"
    )
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "role 的 name 字段"}},
        "required": ["name"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        name = kwargs.get("name")
        if not name or not isinstance(name, str):
            raise ToolError("name must be a non-empty string")
        user_path = ctx.settings.paths.user_roles_dir / f"{name}.yaml"
        if user_path.exists():
            return user_path.read_text(encoding="utf-8")
        builtin_path = BUILTIN_DIR / f"{name}.yaml"
        if builtin_path.exists():
            return f"# source: builtin (read-only here; write_role 会写到用户目录覆盖)\n{builtin_path.read_text(encoding='utf-8')}"
        raise ToolError(f"role not found: {name}")


class WriteRoleTool(Tool):
    name = "write_role"
    description = (
        "把一段 YAML 写入 ~/.chat_team/roles/<name>.yaml(已存在则覆盖)。"
        "写盘前会先解析校验:必须是合法 YAML mapping、必须含 name 字段、且 name 必须等于本工具的 name 参数。"
        "注意:写盘会立刻持久化,调用前请先把完整 YAML 给用户看并确认。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "role 名称(英文小写下划线,例如 data_analyst);也作为文件名 <name>.yaml",
            },
            "yaml_content": {
                "type": "string",
                "description": "完整 YAML 文本;顶层必须是 mapping,且其 name 字段等于上面的 name 参数",
            },
        },
        "required": ["name", "yaml_content"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        name = kwargs.get("name")
        yaml_content = kwargs.get("yaml_content")
        if not name or not isinstance(name, str):
            raise ToolError("name must be a non-empty string")
        if not _NAME_RE.match(name):
            raise ToolError(
                f"invalid role name: {name!r}; use lowercase letters, digits, underscore; must start with a letter"
            )
        if not isinstance(yaml_content, str) or not yaml_content.strip():
            raise ToolError("yaml_content must be a non-empty string")
        try:
            raw = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            raise ToolError(f"yaml parse failed: {exc}")
        if not isinstance(raw, dict):
            raise ToolError("yaml_content must be a top-level mapping")
        try:
            role = Role.from_dict(raw)
        except ValueError as exc:
            raise ToolError(f"role schema invalid: {exc}")
        if role.name != name:
            raise ToolError(
                f"name mismatch: tool arg name={name!r} but yaml name={role.name!r}; they must match"
            )
        target = ctx.settings.paths.user_roles_dir / f"{name}.yaml"
        existed = target.exists()
        _atomic_write(target, yaml_content if yaml_content.endswith("\n") else yaml_content + "\n")
        verb = "overwrote" if existed else "created"
        return f"{verb} {target}"


class DeleteRoleTool(Tool):
    name = "delete_role"
    description = (
        "从 ~/.chat_team/roles/ 删除 <name>.yaml。"
        "拒绝删 builtin role(builtin 自带,无法从用户目录删除)。"
        "删盘前请向用户确认。"
    )
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "要删除的 role 名"}},
        "required": ["name"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        name = kwargs.get("name")
        if not name or not isinstance(name, str):
            raise ToolError("name must be a non-empty string")
        if name in _builtin_names():
            raise ToolError(
                f"refusing to delete builtin role: {name}; builtin roles are shipped with the package"
            )
        target = ctx.settings.paths.user_roles_dir / f"{name}.yaml"
        if not target.exists():
            raise ToolError(f"role file not found: {target}")
        target.unlink()
        return f"deleted {target}"


class ReadTeamProfileTool(Tool):
    name = "read_team_profile"
    description = "读 ~/.chat_team/team.md(全局团队画像)的当前内容。空字符串表示未配置。"
    parameters = {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        path = ctx.settings.paths.team_md
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")


class WriteTeamProfileTool(Tool):
    name = "write_team_profile"
    description = (
        "覆盖 ~/.chat_team/team.md(全局团队画像)。"
        "传空字符串等同于清空文件,下次启动 chat-team 时不再注入 [团队信息] 块。"
        "注意:写盘会立刻持久化,调用前请先把完整内容给用户看并确认。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "team.md 的完整新内容;空串=清空",
            },
        },
        "required": ["content"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        content = kwargs.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        target = ctx.settings.paths.team_md
        _atomic_write(target, content)
        size = len(content.encode("utf-8"))
        return f"wrote {size} bytes to {target}"


class ListAvailableToolsTool(Tool):
    name = "list_available_tools"
    description = (
        "列出 chat-team 主进程里所有可在 role YAML 的 tools 字段引用的工具名(及简介)。"
        "请只用这里返回的名字给 role 配 tools。"
    )
    parameters = {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        # Build a fresh main-runtime tool registry to enumerate names. Use an
        # empty RoleRegistry so the transfer enum has no dependency on real roles.
        # Pass the live SkillRegistry so skill / skill_read_file appear when
        # the user has at least one skill defined — otherwise the boss would
        # tell the user `tools: [skill]` isn't valid.
        from ...app import build_tool_registry            # local import: avoid cycle
        from ...roles.registry import RoleRegistry
        from ...skills.registry import SkillRegistry

        skills = SkillRegistry.load(ctx.settings.paths.user_skills_dir)
        reg = build_tool_registry(RoleRegistry({}), skills)
        specs = sorted(reg.specs_for(list(reg._tools.keys())), key=lambda s: s.name)  # noqa: SLF001
        if not specs:
            return "(no tools registered)"
        return "\n".join(f"- {s.name}: {s.description}" for s in specs)


def _builtin_skill_names() -> set[str]:
    if not SKILL_BUILTIN_DIR.exists():
        return set()
    return {p.name for p in SKILL_BUILTIN_DIR.iterdir() if p.is_dir() and (p / "SKILL.md").exists()}


class ListSkillsTool(Tool):
    name = "list_skills"
    description = (
        "列出当前可用的 skill(builtin + 用户自定义),含 name/description/source。"
        "用户在 role YAML 的 skills 字段里引用 skill 时,只能用这里返回的 name。"
        "skill 是『不写代码就能扩展能力』的 markdown 能力包,放在 ~/.chat_team/skills/<name>/SKILL.md。"
    )
    parameters = {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        user_dir = ctx.settings.paths.user_skills_dir
        builtin_names = _builtin_skill_names()
        seen: dict[str, dict[str, Any]] = {}
        if SKILL_BUILTIN_DIR.exists():
            for child in sorted(SKILL_BUILTIN_DIR.iterdir()):
                if not child.is_dir():
                    continue
                try:
                    sk = Skill.from_dir(child)
                except ValueError:
                    continue
                seen[sk.name] = {
                    "name": sk.name,
                    "description": sk.description,
                    "source": "builtin",
                }
        if user_dir.exists():
            for child in sorted(user_dir.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                try:
                    sk = Skill.from_dir(child)
                except ValueError as exc:
                    seen[child.name] = {
                        "name": child.name,
                        "description": f"(加载失败: {exc})",
                        "source": "user_broken",
                    }
                    continue
                seen[sk.name] = {
                    "name": sk.name,
                    "description": sk.description,
                    "source": "user_override" if sk.name in builtin_names else "user",
                }
        if not seen:
            return (
                "(暂无 skill)\n"
                "提示:skill 是 ~/.chat_team/skills/<name>/SKILL.md(YAML frontmatter 必含 name+description)。"
                "本期 boss 不直接编辑 skill 文件;请用户手工创建或后续版本支持。"
            )
        lines = []
        for entry in sorted(seen.values(), key=lambda e: e["name"]):
            first_line = entry["description"].split("\n", 1)[0].strip()
            lines.append(f"- {entry['name']} [{entry['source']}]: {first_line}")
        return "\n".join(lines)


class ReadDeployConfigTool(Tool):
    name = "read_deploy_config"
    description = (
        "读取 config.yaml 中的部署拓扑(mode / default_role / bots 角色绑定)。"
        "不返回任何凭证(bot_id / secret 不可见)。"
    )
    parameters = {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        path = ctx.settings.paths.config_yaml
        raw: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                raw = loaded

        mode = raw.get("mode", "team")
        default_role = raw.get("default_role", "team_admin")
        bots_raw = raw.get("bots") if isinstance(raw.get("bots"), list) else []

        lines = [f"mode: {mode}", f"default_role: {default_role}"]
        if not bots_raw:
            lines.append("bots: (未配置)")
        elif mode == "solo":
            lines.append(f"bots ({len(bots_raw)}个):")
            for i, b in enumerate(bots_raw):
                name = b.get("name", "") if isinstance(b, dict) else ""
                lines.append(f"  {i + 1}. name: {name or '(未指定)'}")
        else:
            lines.append(f"bots: {len(bots_raw)}个(team 模式无需 name)")
        return "\n".join(lines)
