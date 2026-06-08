"""CLI-only "boss" agent: a conversational helper that writes ~/.chat_team/team.md
and ~/.chat_team/roles/*.yaml on the user's behalf.

Run with the ``chat-team-boss`` console script (registered in pyproject.toml)
or ``python -m chat_team.boss``. Reuses the same Agent/LLM/ToolRegistry plumbing
as the WeCom bot, but the boss role is hardcoded here and is NOT registered in
``RoleRegistry`` — so it never appears in the WeCom employee list or in
``transfer_to_employee``'s enum.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from .agent.agent import Agent
from .agent.compactor import maybe_compact
from .agent.tools.base import ToolRegistry, TransferRequested
from .agent.tools.team_tools import (
    DeleteRoleTool,
    ListAvailableToolsTool,
    ListRolesTool,
    ListSkillsTool,
    ReadDeployConfigTool,
    ReadRoleTool,
    ReadTeamProfileTool,
    WriteRoleTool,
    WriteTeamProfileTool,
)
from .app import build_llm_provider, configure_logging
from .config import Settings, load_settings
from .llm.base import LLMProvider
from .roles.config import Role, RoleLLMConfig
from .session.notebook import Notebook
from .session.session import Session

log = logging.getLogger(__name__)


BOSS_ROLE = Role(
    name="boss",
    display_name="团队搭建助手",
    description="(CLI-only) 通过对话帮你写 ~/.chat_team/team.md 和角色 YAML。",
    system_prompt=(
        "你是 chat_team 项目的『团队搭建助手』,负责帮用户搭建/维护他们的虚拟员工团队。\n"
        "\n"
        "你只在命令行里出现,不会出现在企业微信员工列表里。你的核心职责是把用户的"
        "需求翻译成两类配置文件并落盘:\n"
        "  1) ~/.chat_team/team.md —— 全局团队画像(自由 markdown,会被注入到每个员工每轮的 system prompt)。\n"
        "  2) ~/.chat_team/roles/<name>.yaml —— 单个虚拟员工的角色定义。\n"
        "\n"
        "[工作流]\n"
        "- 开场或收到模糊需求时,先用 list_roles + read_team_profile 摸清现状再问问题。\n"
        "- 做大幅改动前先口头跟用户对齐:他想新增/修改/删除哪个员工?这个员工要做什么?\n"
        "- 写盘前必须先把完整 YAML/markdown 全文贴给用户看,明确询问『是否确认写入?』,"
        "得到肯定回复后才调用 write_role / write_team_profile / delete_role。\n"
        "- 写完落盘后用一两句话告诉用户:文件路径 + 改动摘要 + 提醒『重启 chat-team 后生效』。\n"
        "\n"
        "[role YAML 字段速查]\n"
        "- name (必填): 英文小写下划线,例如 data_analyst;也作为文件名 <name>.yaml。\n"
        "- display_name: 给企微用户看的中文名,例如 『数据分析师』。\n"
        "- description: 一句话职责简介,会出现在 transfer_to_employee 的 enum 里。\n"
        "- system_prompt: 角色的人格 + 行为指令(多行字符串)。\n"
        "- tools: 工具名列表 —— **只能使用 list_available_tools 返回的名字**。\n"
        "- skills: 可选,字符串列表,该角色可调用的 skill 白名单 —— **只能使用 list_skills 返回的 name**。\n"
        "  写了 skills 的角色,**tools 里必须同时含 `skill`**(以及可选的 `skill_read_file`),否则 LLM 调不到。\n"
        "- llm.{model, temperature, history_token_budget, image_detail, vision_strategy}: 角色侧可选。\n"
        "- 全局配置在 config.yaml 的 llm.chat / llm.vision（含 reasoning_effort）。\n"
        "- welcome_message: 可选,企微 enter_chat 时发的欢迎语。\n"
        "\n"
        "[配 tools 的经验]\n"
        "- 一个能转交工作的『前台』角色一般要 transfer_to_employee + notebook_read + notebook_write。\n"
        "- 干活的角色按需挑 read_file / write_file / list_dir / run_command;以及通常需要 notebook_read 看团队备忘。\n"
        "- 想让该角色用某些 skill 时:tools 加 `skill`(必需) + 可选 `skill_read_file`(skill 带辅助文件时再加),"
        "skills 列出具体名字。\n"
        "- 不确定就先 list_available_tools / list_skills 拿一份当前真实可用清单。\n"
        "\n"
        "[skill 是什么]\n"
        "- skill 是一份 markdown 能力包(~/.chat_team/skills/<name>/SKILL.md),含 YAML frontmatter "
        "(name + description) + 正文(指令/checklist/规约)。\n"
        "- 角色在系统提示里会看到 [可用 skills] TOC,需要时调 `skill(name=...)` 拉正文照做;有辅助文件时用 "
        "`skill_read_file(skill=..., path=...)` 读取。\n"
        "- 用户提到『某种规约/checklist/做事流程』时,先 list_skills 看是否已经有现成 skill 可白名单进来,"
        "再考虑写到 system_prompt 里。\n"
        "- 本期 boss 不直接编辑也不调用 skill,只能 list_skills 摸底;skill 文件由用户在 ~/.chat_team/skills/ 下手工维护。\n"
        "\n"
        "[团队画像 (team.md)]\n"
        "- 自由 markdown,内容会被原样注入,建议 ≤ 300 字。\n"
        "- 写空字符串 = 清空 = 关闭注入(向后兼容)。\n"
        "\n"
        "[部署模式]\n"
        "config.yaml 中的 mode 决定运行方式,由用户手动配置,boss 不修改:\n"
        "- team(默认):一个 bot,多角色,通过 transfer_to_employee 转接。\n"
        "- solo:每个 bot 绑一个角色(bots[].name = 角色名),无 transfer_to_employee,"
        "各 bot 独立对话,同群聊共享 notebook。\n"
        "\n"
        "用 read_deploy_config 查看当前模式。根据模式给出对口建议:\n"
        "- solo 模式:角色不要配 transfer_to_employee;提醒用户在 config.yaml 的 bots 里"
        "添加对应条目(name 必须匹配角色名)。\n"
        "- team 模式:前台角色需要 transfer_to_employee;注意 default_role 是新会话入口。\n"
        "模式切换和 bot 凭证配置请引导用户手动编辑 config.yaml。\n"
        "\n"
        "[语气]\n"
        "全程使用中文。简洁、直接、给推荐。能少让用户打字就少让用户打字 —— 你应该主动给出"
        "默认值并请用户确认或修改,而不是反复发问。"
    ),
    tools=[
        "list_roles",
        "read_role",
        "write_role",
        "delete_role",
        "read_team_profile",
        "write_team_profile",
        "list_available_tools",
        "list_skills",
        "read_deploy_config",
    ],
    llm=RoleLLMConfig(temperature=0.4, history_token_budget=12000),
    welcome_message=None,
)


class StdoutStream:
    """StreamHandle implementation that prints status notes to stderr.

    Final assistant text is returned by ``Agent.handle`` directly, so the chat
    loop prints it itself; this stream only surfaces tool-call status updates.
    """

    async def push(self, chunk: str, *, append: bool = True) -> None:  # noqa: ARG002
        return None

    async def status(self, note: str) -> None:
        print(f"  ▸ {note}", file=sys.stderr, flush=True)

    async def finish(self, final_text: str) -> None:  # noqa: ARG002
        return None

    async def send_image(self, path: Path, *, filename: str | None = None) -> None:  # noqa: ARG002
        raise NotImplementedError("boss CLI does not support media replies")

    async def send_file(self, path: Path, *, filename: str | None = None) -> None:  # noqa: ARG002
        raise NotImplementedError("boss CLI does not support media replies")


def build_boss_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ListRolesTool())
    reg.register(ReadRoleTool())
    reg.register(WriteRoleTool())
    reg.register(DeleteRoleTool())
    reg.register(ReadTeamProfileTool())
    reg.register(WriteTeamProfileTool())
    reg.register(ListAvailableToolsTool())
    reg.register(ListSkillsTool())
    reg.register(ReadDeployConfigTool())
    return reg


def build_boss_agent(settings: Settings, llm: LLMProvider) -> Agent:
    notebook = Notebook(settings.paths.state_dir / ".boss_notebook.md")
    session = Session(
        session_id="__boss__",
        cwd=settings.paths.home,
        current_role="boss",
        notebook=notebook,
    )
    tools = build_boss_tool_registry()
    return Agent(
        role=BOSS_ROLE,
        session=session,
        settings=settings,
        llm=llm,
        tools=tools,
    )


def _print_intro(settings: Settings) -> None:
    print("=" * 60)
    print("chat-team 团队搭建助手 (boss)")
    print("=" * 60)
    print(f"配置目录: {settings.paths.home}")
    team_state = "已配置" if settings.team_profile else "未配置"
    print(f"team.md: {team_state}")
    user_dir = settings.paths.user_roles_dir
    user_roles = sorted(p.stem for p in user_dir.glob("*.yaml")) if user_dir.exists() else []
    print(f"用户自定义角色: {', '.join(user_roles) if user_roles else '(无)'}")
    mode_label = "solo（一 bot 一角色）" if settings.mode == "solo" else "team（多角色转接）"
    print(f"部署模式: {mode_label}")
    print()
    print("直接说想做什么 —— 比如:'加一个数据分析师'、'我们公司是 X,改一下 team.md'。")
    print("回车空输入 = 跳过;输入 /quit 或 Ctrl-D 退出。")
    print("=" * 60)
    print()


async def chat_loop(agent: Agent, llm: LLMProvider) -> None:
    while True:
        try:
            text = input("你 > ")
        except (EOFError, KeyboardInterrupt):
            print()  # newline after ^D / ^C
            break
        text = text.strip()
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            break
        stream = StdoutStream()
        try:
            reply = await agent.handle(text, stream)
        except TransferRequested:
            print("[boss 不支持员工切换,请直接告诉我你想做什么]\n")
            continue
        except Exception as exc:                          # noqa: BLE001
            log.exception("boss turn failed")
            print(f"[出错: {type(exc).__name__}: {exc}]\n")
            continue
        print(f"\nboss > {reply}\n")
        try:
            await maybe_compact(agent, llm)
        except Exception:                                  # noqa: BLE001
            log.exception("boss compaction failed (non-fatal)")


async def _async_main() -> None:
    settings = load_settings()
    configure_logging(settings)
    log.info("chat-team-boss starting; home=%s", settings.paths.home)
    llm = build_llm_provider(settings)
    agent = build_boss_agent(settings, llm)
    _print_intro(settings)
    await chat_loop(agent, llm)
    print("再见。")


def run() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="chat-team-boss")
    parser.add_argument(
        "--no-tui", action="store_true",
        help="Use plain stdin/stdout instead of the TUI",
    )
    args = parser.parse_args()

    if args.no_tui or not sys.stdin.isatty():
        asyncio.run(_async_main())
    else:
        from .boss_tui import run_tui

        run_tui()


if __name__ == "__main__":
    run()
