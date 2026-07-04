"""McpClientManager — manages MCP server connections and exposes proxy tools."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from .config import McpServerConfig
from .proxy_tool import McpProxyTool

log = logging.getLogger(__name__)

# Hard cap on each async-context-manager __aexit__ during shutdown. The
# MCP stdio transport's anyio task group waits on a reader task that can
# hang (or its cancel scope can get into an inconsistent state under
# cancellation), which would otherwise block --stop / SIGTERM forever.
# Abandoning the Python-side close is safe: when this process exits, the
# stdio pipe to the child MCP server closes and the child exits on EOF.
_CLOSE_TIMEOUT_SECONDS = 5.0


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
            await self._close_cm("session", handle.config.name, handle.session_cm)
            await self._close_cm(
                "transport", handle.config.name, handle.transport_cm,
            )
        self._handles.clear()

    async def _close_cm(
        self, kind: str, server_name: str, cm: Any,
    ) -> None:
        """Close one async CM with a hard timeout.

        Uses ``asyncio.wait`` (not ``wait_for``) so the abandoned task is NOT
        cancelled — anyio's cancel scope may already be in an inconsistent
        state, and cancelling it again can re-raise the cross-task
        ``RuntimeError`` we are trying to dodge. A pending task left here is
        harmless: it (or its child process) gets cleaned up when the loop
        closes and the stdio pipe goes away.
        """
        if cm is None:
            return
        task = asyncio.ensure_future(cm.__aexit__(None, None, None))
        done, _pending = await asyncio.wait(
            {task}, timeout=_CLOSE_TIMEOUT_SECONDS,
        )
        if task in done:
            exc = task.exception()
            if exc is not None:
                # anyio may surface BaseExceptionGroup here; debug-level so it
                # doesn't spam on every clean shutdown.
                log.debug(
                    "MCP %s close for %r raised: %r",
                    kind, server_name, exc,
                )
        else:
            log.warning(
                "MCP %s close for %r timed out after %.0fs; abandoning "
                "(child MCP server exits when its stdio pipe closes on "
                "process exit)",
                kind, server_name, _CLOSE_TIMEOUT_SECONDS,
            )
