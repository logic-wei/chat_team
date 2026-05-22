"""CLI: print all tools registered in chat-team's main runtime.

Run: ``chat-team-tools``. Useful when authoring role YAMLs by hand — the
``tools:`` field can only reference names that appear here. Mirrors the
output of the boss's ``list_available_tools`` tool but does not require
spinning up an LLM session.
"""
from __future__ import annotations

from .app import build_tool_registry
from .roles.registry import RoleRegistry

# Group label per source module. Display order follows insertion order;
# unknown modules fall back to the module name itself.
_GROUPS: dict[str, str] = {
    "file_tools": "文件系统",
    "shell_tool": "命令执行",
    "notebook_tools": "团队记事本",
    "media_tools": "媒体回复",
    "transfer_tool": "协作路由",
}


def _module_of(tool) -> str:
    return tool.__class__.__module__.rsplit(".", 1)[-1]


def _format_catalog() -> str:
    reg = build_tool_registry(RoleRegistry({}))
    tools = list(reg._tools.values())                       # noqa: SLF001
    if not tools:
        return "(no tools registered)"

    grouped: dict[str, list] = {}
    for tool in tools:
        grouped.setdefault(_module_of(tool), []).append(tool)

    width = max(len(t.name) for t in tools)
    lines = ["chat-team 主进程已注册的工具(可在 role YAML 的 tools 字段引用):"]

    ordered = list(_GROUPS) + [m for m in grouped if m not in _GROUPS]
    for module in ordered:
        if module not in grouped:
            continue
        label = _GROUPS.get(module, module)
        lines.append("")
        lines.append(f"[{label}]")
        for tool in sorted(grouped[module], key=lambda t: t.name):
            desc = (tool.description or "").splitlines()[0]
            lines.append(f"  {tool.name.ljust(width)}  {desc}")

    lines.append("")
    lines.append(f"共 {len(tools)} 个工具。把名字原样填到 role YAML 的 tools: 列表里即可。")
    return "\n".join(lines)


def run() -> None:
    print(_format_catalog())


if __name__ == "__main__":
    run()
