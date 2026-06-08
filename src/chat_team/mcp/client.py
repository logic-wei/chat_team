"""McpClientManager — manages MCP server connections and exposes proxy tools."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import McpServerConfig
from .proxy_tool import McpProxyTool

log = logging.getLogger(__name__)


@dataclass
class _ServerHandle:
    config: McpServerConfig
    session: Any = None
    transport_cm: Any = None
    session_cm: Any = None


class McpClientManager:
    """Lifecycle manager for MCP server connections.

    Call ``connect_all`` at startup, ``close_all`` in the finally block.
    """

    def __init__(self) -> None:
        self._handles: list[_ServerHandle] = field(default_factory=list)
        self._handles = []

    async def connect_all(
        self, configs: list[McpServerConfig],
    ) -> list[McpProxyTool]:
        proxy_tools: list[McpProxyTool] = []
        for cfg in configs:
            try:
                tools = await self._connect_one(cfg)
                proxy_tools.extend(tools)
            except Exception:
                log.warning(
                    "MCP server %r failed to connect; skipping",
                    cfg.name,
                    exc_info=True,
                )
        return proxy_tools

    async def _connect_one(self, cfg: McpServerConfig) -> list[McpProxyTool]:
        from mcp.client.session import ClientSession

        if cfg.command:
            read_stream, write_stream, transport_cm = await self._open_stdio(cfg)
        else:
            read_stream, write_stream, transport_cm = await self._open_sse(cfg)

        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        handle = _ServerHandle(
            config=cfg,
            session=session,
            transport_cm=transport_cm,
            session_cm=session_cm,
        )
        self._handles.append(handle)

        await session.initialize()
        result = await session.list_tools()
        tools = [
            McpProxyTool(server_name=cfg.name, mcp_tool=t, session=session)
            for t in result.tools
        ]
        log.info(
            "MCP server %r: %d tool(s) — %s",
            cfg.name,
            len(tools),
            ", ".join(t.name for t in tools),
        )
        return tools

    async def _open_stdio(self, cfg: McpServerConfig) -> tuple[Any, Any, Any]:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        import os
        env = {**os.environ, **cfg.env} if cfg.env else None
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=env,
        )
        cm = stdio_client(params)
        streams = await cm.__aenter__()
        return streams[0], streams[1], cm

    async def _open_sse(self, cfg: McpServerConfig) -> tuple[Any, Any, Any]:
        from mcp.client.sse import sse_client

        cm = sse_client(cfg.url)
        streams = await cm.__aenter__()
        return streams[0], streams[1], cm

    async def close_all(self) -> None:
        for handle in reversed(self._handles):
            try:
                if handle.session_cm is not None:
                    await handle.session_cm.__aexit__(None, None, None)
            except Exception:
                log.debug(
                    "error closing MCP session for %r", handle.config.name,
                    exc_info=True,
                )
            try:
                if handle.transport_cm is not None:
                    await handle.transport_cm.__aexit__(None, None, None)
            except Exception:
                log.debug(
                    "error closing MCP transport for %r", handle.config.name,
                    exc_info=True,
                )
        self._handles.clear()
