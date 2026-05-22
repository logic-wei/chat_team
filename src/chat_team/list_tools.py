"""CLI: print all tools registered in chat-team's main runtime.

Run: ``chat-team-tools``. Useful when authoring role YAMLs by hand — the
``tools:`` field can only reference names that appear here. Mirrors the
output of the boss's ``list_available_tools`` tool but does not require
spinning up an LLM session.
"""
from __future__ import annotations

from .app import build_tool_registry
from .roles.registry import RoleRegistry


def _format_catalog() -> str:
    reg = build_tool_registry(RoleRegistry({}))
    names = sorted(reg._tools.keys())                       # noqa: SLF001
    if not names:
        return "(no tools registered)"
    width = max(len(n) for n in names)
    lines = ["chat-team 主进程已注册的工具(可在 role YAML 的 tools 字段引用):", ""]
    for name in names:
        desc = (reg.get(name).description or "").splitlines()[0]
        lines.append(f"  {name.ljust(width)}  {desc}")
    lines.append("")
    lines.append(f"共 {len(names)} 个工具。把名字原样填到 role YAML 的 tools: 列表里即可。")
    return "\n".join(lines)


def run() -> None:
    print(_format_catalog())


if __name__ == "__main__":
    run()
