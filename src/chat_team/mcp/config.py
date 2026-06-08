"""MCP server configuration dataclass."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class McpServerConfig:
    """One MCP server entry from config.yaml ``mcp.servers``."""

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""

    def validate(self) -> None:
        if not self.name or not _NAME_RE.match(self.name):
            raise ValueError(
                f"mcp server name must match [a-zA-Z0-9_-]+: {self.name!r}"
            )
        if "__" in self.name:
            raise ValueError(
                f"mcp server name must not contain '__': {self.name!r}"
            )
        has_command = bool(self.command)
        has_url = bool(self.url)
        if has_command == has_url:
            raise ValueError(
                f"mcp server {self.name!r}: set exactly one of 'command' (stdio) or 'url' (sse)"
            )
