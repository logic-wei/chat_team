"""OpenAI Chat Completion provider."""
from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from .base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": m.content,
            })
            continue
        msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.role == "assistant" and m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
        if m.name:
            msg["name"] = m.name
        out.append(msg)
    return out


def _to_openai_tools(specs) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters or {"type": "object", "properties": {}},
            },
        }
        for s in specs
    ]


class OpenAIChatCompletionProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": _to_openai_messages(request.messages),
            "temperature": request.temperature,
        }
        if request.tools:
            kwargs["tools"] = _to_openai_tools(request.tools)
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        chat_msg = ChatMessage(
            role="assistant",
            content=msg.content or "",
            tool_calls=tool_calls,
        )
        return CompletionResponse(
            message=chat_msg,
            finish_reason=choice.finish_reason or "stop",
            raw=completion,
        )
