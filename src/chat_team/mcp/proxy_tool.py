"""McpProxyTool — wraps a single MCP server tool as a local Tool subclass."""
from __future__ import annotations

from typing import Any

from ..agent.tools.base import Tool, ToolContext, ToolError


class McpProxyTool(Tool):
    """Bridges an MCP tool into the local ToolRegistry.

    Registered as ``mcp__<server>__<tool>`` so role YAMLs (via
    ``mcp_servers:``) or explicit ``tools:`` entries can reference it.
    """

    def __init__(self, server_name: str, mcp_tool: Any, session: Any) -> None:
        self.name = f"mcp__{server_name}__{mcp_tool.name}"
        self.description = mcp_tool.description or ""
        self.parameters = mcp_tool.inputSchema or {"type": "object", "properties": {}}
        self.server_name = server_name
        self._session = session
        self._remote_name: str = mcp_tool.name

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        try:
            result = await self._session.call_tool(self._remote_name, kwargs or None)
        except Exception as exc:
            raise ToolError(f"MCP tool {self.name} failed: {exc}") from exc

        if result.isError:
            parts = [c.text for c in result.content if hasattr(c, "text")]
            raise ToolError("\n".join(parts) or "MCP tool returned an error")

        parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "data") and hasattr(item, "mimeType"):
                parts.append(f"[image: {item.mimeType}]")
            else:
                parts.append(str(item))
        return "\n".join(parts) or "(no output)"
