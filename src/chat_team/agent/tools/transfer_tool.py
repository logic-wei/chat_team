"""transfer_to_employee — universal handoff tool injected into every role."""
from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolError, TransferRequested


class TransferToEmployeeTool(Tool):
    name = "transfer_to_employee"
    description = (
        "把当前会话交接给另一位虚拟同事接手。当用户希望换人对接、或当前需求超出你能力范围时调用。"
        "调用前请把 handoff_note 写得有用:用户当前的核心诉求、已经达成的进展或决策、新同事接手时要立刻关注的点。"
    )

    def __init__(self, available_employees: list[str]):
        self.available = sorted(set(available_employees))
        self.parameters = {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "string",
                    "enum": self.available,
                    "description": "目标员工的 name(角色 yaml 中的 name 字段)",
                },
                "reason": {
                    "type": "string",
                    "description": "为什么交接(给用户/调用方看的简短说明)",
                },
                "handoff_note": {
                    "type": "string",
                    "description": "给接手同事的交接备忘:用户诉求/进展/注意事项,200 字以内。",
                },
            },
            "required": ["employee", "reason", "handoff_note"],
        }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        target = kwargs.get("employee")
        reason = kwargs.get("reason", "")
        handoff = kwargs.get("handoff_note", "")
        if not target or target not in self.available:
            raise ToolError(
                f"unknown employee '{target}'. available: {', '.join(self.available)}"
            )
        if target == ctx.session.current_role:
            raise ToolError(f"cannot transfer to yourself ({target})")
        raise TransferRequested(target=target, reason=reason, handoff_note=handoff)
