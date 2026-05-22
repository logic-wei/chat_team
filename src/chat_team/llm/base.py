"""LLM provider abstraction. Concrete providers wrap a vendor SDK
and translate to/from the dataclasses below."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool"
    name: str | None = None


@dataclass
class ToolSpec:
    """JSON-Schema description of a tool, fed to the LLM."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class CompletionRequest:
    messages: list[ChatMessage]
    tools: list[ToolSpec] = field(default_factory=list)
    model: str = ""
    temperature: float = 0.3
    max_tokens: int | None = None


@dataclass
class CompletionResponse:
    message: ChatMessage          # assistant message (may contain tool_calls)
    finish_reason: str            # "stop" | "tool_calls" | "length" | ...
    raw: Any | None = None


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
