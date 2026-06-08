"""Textual TUI for the chat-team-boss CLI."""
from __future__ import annotations

import logging
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Header, Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from .agent.agent import Agent
from .agent.compactor import maybe_compact
from .agent.tools.base import TransferRequested
from .app import build_llm_provider, configure_logging
from .boss import BOSS_ROLE, build_boss_agent
from .config import load_settings
from .llm.base import LLMProvider

log = logging.getLogger(__name__)

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/quit", "退出程序"),
    ("/clear", "清空聊天记录"),
    ("/roles", "列出当前角色"),
    ("/help", "显示可用命令"),
]


class TuiStream:
    """StreamHandle that routes updates to the Textual app."""

    def __init__(self, app: BossApp) -> None:
        self._app = app

    async def push(self, chunk: str, *, append: bool = True) -> None:  # noqa: ARG002
        self._app._update_pending_bot(chunk)

    async def status(self, note: str) -> None:
        self._app._set_status(f"  ▸ {note}")

    async def finish(self, final_text: str) -> None:  # noqa: ARG002
        pass

    async def send_image(self, path: Path, *, filename: str | None = None) -> None:  # noqa: ARG002
        pass

    async def send_file(self, path: Path, *, filename: str | None = None) -> None:  # noqa: ARG002
        pass


class SlashInput(Input):
    """Input that intercepts arrow/tab/esc keys for slash command navigation."""

    def _on_key(self, event) -> None:
        app: BossApp = self.app  # type: ignore[assignment]
        if app._popup_visible:
            if event.key == "up":
                app._popup_move(-1)
                event.prevent_default()
                event.stop()
                return
            if event.key == "down":
                app._popup_move(1)
                event.prevent_default()
                event.stop()
                return
            if event.key in ("tab", "right"):
                app._popup_accept()
                event.prevent_default()
                event.stop()
                return
            if event.key == "escape":
                app._popup_hide()
                event.prevent_default()
                event.stop()
                return


class BossApp(App):
    """Textual app for the boss conversational CLI."""

    TITLE = "chat-team boss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", show=False),
        Binding("ctrl+d", "quit", "退出", show=False),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }
    #chat-log {
        height: 1fr;
        padding: 0 1;
    }
    .user-msg {
        color: $text;
        background: $surface;
        margin: 0 0 1 2;
        padding: 0 1;
    }
    .system-msg {
        color: $text-muted;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    .bot-msg {
        margin: 0 0 1 0;
        padding: 0 1;
    }
    #status-bar {
        height: 1;
        dock: bottom;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    #slash-popup {
        dock: bottom;
        height: auto;
        max-height: 8;
        display: none;
        background: $surface;
        border: tall $accent;
        padding: 0;
    }
    #slash-popup.visible {
        display: block;
    }
    #chat-input {
        dock: bottom;
        margin: 0;
    }
    """

    def __init__(self, agent: Agent, llm: LLMProvider, settings) -> None:
        super().__init__()
        self.agent = agent
        self.llm = llm
        self.settings = settings
        self._pending_bot_widget: Markdown | None = None
        self._busy = False
        self._popup_visible = False
        self._popup_commands: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-log")
        yield Static("", id="status-bar")
        yield OptionList(id="slash-popup")
        yield SlashInput(placeholder="输入消息... (/ 查看命令)", id="chat-input")

    def on_mount(self) -> None:
        self._show_intro()
        self.query_one("#chat-input", SlashInput).focus()

    def _show_intro(self) -> None:
        home = self.settings.paths.home
        team_state = "已配置" if self.settings.team_profile else "未配置"
        user_dir = self.settings.paths.user_roles_dir
        user_roles = sorted(p.stem for p in user_dir.glob("*.yaml")) if user_dir.exists() else []
        roles_str = ", ".join(user_roles) if user_roles else "(无)"
        mode = "solo" if self.settings.mode == "solo" else "team"
        intro = (
            f"配置: {home}  |  team.md: {team_state}  |  "
            f"角色: {roles_str}  |  模式: {mode}\n"
            "直接说想做什么 —— 比如: '加一个数据分析师'、'改一下 team.md'。\n"
            "输入 / 查看可用命令。"
        )
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(intro, classes="system-msg"))

    # --- Slash popup logic ---

    def on_input_changed(self, event: Input.Changed) -> None:
        value = event.value
        if value.startswith("/"):
            prefix = value.lower()
            matches = [
                (cmd, desc) for cmd, desc in SLASH_COMMANDS if cmd.startswith(prefix)
            ]
            if matches:
                self._popup_show(matches)
            else:
                self._popup_hide()
        else:
            self._popup_hide()

    def _popup_show(self, matches: list[tuple[str, str]]) -> None:
        popup = self.query_one("#slash-popup", OptionList)
        popup.clear_options()
        self._popup_commands = []
        for cmd, desc in matches:
            popup.add_option(Option(f"{cmd}  — {desc}", id=cmd))
            self._popup_commands.append(cmd)
        popup.highlighted = 0
        popup.add_class("visible")
        self._popup_visible = True

    def _popup_hide(self) -> None:
        if self._popup_visible:
            popup = self.query_one("#slash-popup", OptionList)
            popup.remove_class("visible")
            self._popup_visible = False
            self._popup_commands = []

    def _popup_move(self, delta: int) -> None:
        popup = self.query_one("#slash-popup", OptionList)
        current = popup.highlighted
        if current is None:
            popup.highlighted = 0
        else:
            count = popup.option_count
            popup.highlighted = max(0, min(count - 1, current + delta))

    def _popup_accept(self) -> None:
        popup = self.query_one("#slash-popup", OptionList)
        idx = popup.highlighted
        if idx is not None and idx < len(self._popup_commands):
            cmd = self._popup_commands[idx]
            inp = self.query_one("#chat-input", SlashInput)
            inp.value = cmd
            inp.cursor_position = len(cmd)
        self._popup_hide()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        cmd = event.option.id
        if cmd:
            inp = self.query_one("#chat-input", SlashInput)
            inp.value = cmd
            inp.cursor_position = len(cmd)
        self._popup_hide()
        self.query_one("#chat-input", SlashInput).focus()

    # --- Message handling ---

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one("#chat-input", SlashInput).clear()
        self._popup_hide()
        if not text:
            return
        if text in {"/quit", "/exit"}:
            self.exit()
            return
        if text == "/clear":
            self._do_clear()
            return
        if text == "/roles":
            self._do_roles()
            return
        if text == "/help":
            self._do_help()
            return
        if self._busy:
            return
        self._append_user_message(text)
        self._busy = True
        self.run_worker(self._agent_turn(text), exclusive=True)

    def _do_clear(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.remove_children()
        chat_log.mount(Static("[已清空聊天记录]", classes="system-msg"))

    def _do_roles(self) -> None:
        user_dir = self.settings.paths.user_roles_dir
        user_roles = sorted(p.stem for p in user_dir.glob("*.yaml")) if user_dir.exists() else []
        roles_str = ", ".join(user_roles) if user_roles else "(无)"
        self._append_system_message(f"当前角色: {roles_str}")

    def _do_help(self) -> None:
        lines = ["可用命令:"]
        for cmd, desc in SLASH_COMMANDS:
            lines.append(f"  {cmd}  — {desc}")
        self._append_system_message("\n".join(lines))

    # --- Agent turn ---

    async def _agent_turn(self, text: str) -> None:
        self._create_pending_bot()
        stream = TuiStream(self)
        try:
            reply = await self.agent.handle(text, stream)
        except TransferRequested:
            reply = "[boss 不支持员工切换,请直接告诉我你想做什么]"
        except Exception as exc:  # noqa: BLE001
            log.exception("boss turn failed")
            reply = f"[出错: {type(exc).__name__}: {exc}]"
        self._finalize_bot_message(reply)
        self._set_status("")
        self._busy = False
        try:
            await maybe_compact(self.agent, self.llm)
        except Exception:  # noqa: BLE001
            log.exception("boss compaction failed (non-fatal)")

    # --- Widget helpers ---

    def _append_user_message(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(f"你 > {text}", classes="user-msg"))
        chat_log.scroll_end(animate=False)

    def _append_system_message(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(text, classes="system-msg"))
        chat_log.scroll_end(animate=False)

    def _create_pending_bot(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        widget = Markdown("...", classes="bot-msg")
        chat_log.mount(widget)
        chat_log.scroll_end(animate=False)
        self._pending_bot_widget = widget

    def _update_pending_bot(self, text: str) -> None:
        if self._pending_bot_widget is not None:
            self._pending_bot_widget.update(text)
            chat_log = self.query_one("#chat-log", VerticalScroll)
            chat_log.scroll_end(animate=False)

    def _finalize_bot_message(self, text: str) -> None:
        if self._pending_bot_widget is not None:
            self._pending_bot_widget.update(text)
            self._pending_bot_widget = None
            chat_log = self.query_one("#chat-log", VerticalScroll)
            chat_log.scroll_end(animate=False)

    def _set_status(self, note: str) -> None:
        self.query_one("#status-bar", Static).update(note)


def run_tui() -> None:
    settings = load_settings()
    configure_logging(settings, file_only=True)
    log.info("chat-team-boss TUI starting; home=%s", settings.paths.home)
    llm = build_llm_provider(settings)
    agent = build_boss_agent(settings, llm)
    app = BossApp(agent=agent, llm=llm, settings=settings)
    app.run()
