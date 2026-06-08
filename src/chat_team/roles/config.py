"""Role config dataclass + YAML loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RoleLLMConfig:
    model: str = ""                      # falls back to settings.llm.chat.model
    temperature: float | None = None     # falls back to settings.llm.chat.temperature
    history_token_budget: int | None = None
    image_detail: str | None = None      # "low" | "high" | "auto"; falls back to settings.llm.vision.image_detail
    # "tool" → inbound images become placeholder text blocks (no pre-OCR).
    # "direct" → pass image blocks straight to the provider (high-fidelity
    # multi-turn visual chat).
    # None → fall back to settings.llm.vision.strategy.
    vision_strategy: str | None = None


@dataclass
class Role:
    name: str
    display_name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    llm: RoleLLMConfig = field(default_factory=RoleLLMConfig)
    welcome_message: str | None = None   # used for enter_chat events

    @classmethod
    def from_yaml(cls, path: Path) -> "Role":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"role yaml must be a mapping: {path}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Role":
        llm_raw = raw.get("llm") or {}
        llm = RoleLLMConfig(
            model=llm_raw.get("model", ""),
            temperature=llm_raw.get("temperature"),
            history_token_budget=llm_raw.get("history_token_budget"),
            image_detail=llm_raw.get("image_detail"),
            vision_strategy=llm_raw.get("vision_strategy"),
        )
        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("role yaml missing required 'name'")
        skills_raw = raw.get("skills") or []
        if not isinstance(skills_raw, list) or not all(isinstance(s, str) for s in skills_raw):
            raise ValueError("role yaml 'skills' must be a list of strings")
        mcp_servers_raw = raw.get("mcp_servers") or []
        if not isinstance(mcp_servers_raw, list) or not all(
            isinstance(s, str) for s in mcp_servers_raw
        ):
            raise ValueError("role yaml 'mcp_servers' must be a list of strings")
        return cls(
            name=name,
            display_name=raw.get("display_name", name),
            description=raw.get("description", ""),
            system_prompt=raw.get("system_prompt", "").strip(),
            tools=list(raw.get("tools") or []),
            skills=list(skills_raw),
            mcp_servers=list(mcp_servers_raw),
            llm=llm,
            welcome_message=raw.get("welcome_message"),
        )
