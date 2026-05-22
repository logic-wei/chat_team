"""Tool ABC + ToolContext + ToolRegistry.

Tools are pure logic; they receive a ``ToolContext`` so they have access
to the session's working dir / notebook / settings without globals.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...llm.base import ToolSpec

if TYPE_CHECKING:                       # avoid circular imports at runtime
    from ...config import Settings
    from ...session.session import Session


class ToolError(Exception):
    """Raised by a tool to signal a recoverable error returned to the LLM."""


class TransferRequested(Exception):
    """Raised by transfer_to_employee — caught by the dispatcher to switch role.

    Carries the requested target plus the structured handoff note.
    """

    def __init__(self, target: str, reason: str, handoff_note: str):
        super().__init__(f"transfer to {target}")
        self.target = target
        self.reason = reason
        self.handoff_note = handoff_note


@dataclass
class ToolContext:
    cwd: Path
    session: "Session"
    settings: "Settings"


class Tool(abc.ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @abc.abstractmethod
    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        """Return a string result fed back to the LLM as a tool message."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"tool missing name: {tool!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def specs_for(self, names: list[str]) -> list[ToolSpec]:
        return [self._tools[n].spec() for n in names if n in self._tools]


def stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
