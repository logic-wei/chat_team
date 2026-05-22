"""Bot adapter abstraction. Concrete adapters translate platform-specific
events to the internal ``IncomingMessage`` / ``OutgoingMessage`` shapes."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol


class ChatType(str, Enum):
    SINGLE = "single"
    GROUP = "group"


@dataclass
class IncomingMessage:
    """Normalised inbound message handed to the dispatcher."""
    session_id: str
    chat_type: ChatType
    user_id: str           # sender's wxid
    text: str              # already-stripped of @bot prefix for groups
    msg_id: str            # platform message id (used for dedup)
    bot_id: str            # which bot received this
    raw: dict[str, Any] = field(default_factory=dict)
    chat_id: str | None = None      # group chat id, None for single
    user_name: str | None = None    # display name if available
    reply_token: Any | None = None  # adapter-specific (e.g. WeCom req_id)


@dataclass
class OutgoingMessage:
    text: str
    finish: bool = True


class StreamHandle(Protocol):
    """Live handle to push streaming updates back to the user.

    The adapter creates one per inbound message that warrants streaming.
    Implementations are expected to throttle / coalesce internally.
    """

    async def push(self, chunk: str, *, append: bool = True) -> None: ...
    async def status(self, note: str) -> None: ...
    async def finish(self, final_text: str) -> None: ...


MessageHandler = Callable[[IncomingMessage, StreamHandle], Awaitable[None]]


class BotAdapter(abc.ABC):
    """Lifecycle: ``await connect()`` → registers handler → ``await run()`` blocks."""

    @abc.abstractmethod
    def set_handler(self, handler: MessageHandler) -> None: ...

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def run(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...
