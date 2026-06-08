"""End-to-end smoke for the CLI boss agent (no real LLM, no network).

Covers:
* list_available_tools enumerates the main-runtime tool catalog.
* write_role validates YAML and refuses bad name mismatch / parse error.
* Through Agent.handle: bad YAML → tool_error fed back → retry with good YAML
  → file lands on disk and parses back via Role.from_yaml.
* read/write team_profile round-trip.
* delete_role refuses to remove a builtin (team_admin).
* Boss role does NOT leak into RoleRegistry.names().
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fresh_home(tag: str) -> Path:
    home = Path(f"/tmp/chat_team_boss_{tag}")
    shutil.rmtree(home, ignore_errors=True)
    return home


from chat_team.agent.tools.base import ToolContext, ToolError
from chat_team.agent.tools.team_tools import (
    DeleteRoleTool,
    ListAvailableToolsTool,
    ListRolesTool,
    ReadTeamProfileTool,
    WriteRoleTool,
    WriteTeamProfileTool,
)
from chat_team.boss import BOSS_ROLE, build_boss_agent, build_boss_tool_registry
from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.roles.config import Role
from chat_team.roles.registry import RoleRegistry
from chat_team.session.notebook import Notebook
from chat_team.session.session import Session


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self._responses.pop(0)


def reply(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def call(name: str, args: dict, call_id: str = "tc-1") -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        ),
        finish_reason="tool_calls",
    )


class NullStream:
    async def push(self, chunk: str, *, append: bool = True) -> None: pass
    async def status(self, note: str) -> None: pass
    async def finish(self, final_text: str) -> None: pass


def _ctx(settings) -> ToolContext:
    """Build a minimal ToolContext for direct tool invocation."""
    notebook = Notebook(settings.paths.state_dir / ".boss_notebook.md")
    sess = Session(
        session_id="__boss_test__",
        cwd=settings.paths.home,
        current_role="boss",
        notebook=notebook,
    )
    return ToolContext(cwd=sess.cwd, session=sess, settings=settings)


async def test_list_available_tools() -> None:
    print("== test 1: list_available_tools ==")
    home = _fresh_home("avail")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    out = await ListAvailableToolsTool().run(_ctx(settings))
    for must in ("read_file", "write_file", "list_dir", "run_command",
                 "notebook_read", "notebook_write", "transfer_to_employee"):
        assert must in out, f"expected {must!r} in:\n{out}"
    # boss tools should NOT appear in the main runtime catalog (they are
    # boss-private — registered only by build_boss_tool_registry).
    assert "list_available_tools" not in out, f"boss tool leaked into main catalog:\n{out}"
    print("  ✓ catalog includes main tools, excludes boss-private tools")


async def test_write_role_validation() -> None:
    print("== test 2: write_role validation (direct tool calls) ==")
    home = _fresh_home("write_role_unit")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    ctx = _ctx(settings)
    tool = WriteRoleTool()

    # Bad YAML parse
    try:
        await tool.run(ctx, name="x1", yaml_content="this is: : not: valid: : yaml:")
    except ToolError as exc:
        print(f"  ✓ bad yaml rejected: {exc}")
    else:
        raise AssertionError("expected ToolError for bad yaml")

    # Mapping but missing 'name'
    try:
        await tool.run(ctx, name="x1", yaml_content="display_name: 缺 name 字段\n")
    except ToolError as exc:
        print(f"  ✓ missing-name rejected: {exc}")
    else:
        raise AssertionError("expected ToolError for missing name")

    # name mismatch between arg and yaml
    try:
        await tool.run(ctx, name="alpha", yaml_content="name: beta\nsystem_prompt: hi\n")
    except ToolError as exc:
        print(f"  ✓ name mismatch rejected: {exc}")
    else:
        raise AssertionError("expected ToolError for name mismatch")

    # Bad name shape
    try:
        await tool.run(ctx, name="Bad-Name", yaml_content="name: Bad-Name\n")
    except ToolError as exc:
        print(f"  ✓ bad name shape rejected: {exc}")
    else:
        raise AssertionError("expected ToolError for invalid name shape")

    # Happy path
    yaml_ok = (
        "name: data_analyst\n"
        "display_name: 数据分析师\n"
        "description: 处理 SQL/报表类需求\n"
        "system_prompt: |\n"
        "  你是数据分析师小数,优先用中文回答。\n"
        "tools:\n"
        "  - read_file\n"
        "  - run_command\n"
    )
    out = await tool.run(ctx, name="data_analyst", yaml_content=yaml_ok)
    print(f"  ✓ happy write: {out}")
    target = home / "roles" / "data_analyst.yaml"
    assert target.exists()
    role = Role.from_yaml(target)
    assert role.name == "data_analyst"
    assert role.display_name == "数据分析师"
    assert "read_file" in role.tools
    print("  ✓ role round-tripped via Role.from_yaml")


async def test_agent_loop_retry_on_bad_yaml() -> None:
    """Through the real Agent loop: bad YAML → tool_error → retry → success."""
    print("== test 3: agent loop with bad-then-good yaml ==")
    home = _fresh_home("agent_loop")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()

    bad_yaml = "this is: : not yaml:"
    good_yaml = (
        "name: customer_service\n"
        "display_name: 客服专员\n"
        "system_prompt: |\n"
        "  你是客服小服。\n"
        "tools:\n"
        "  - notebook_read\n"
    )
    llm = ScriptedLLM([
        call("write_role", {"name": "customer_service", "yaml_content": bad_yaml}, call_id="tc-bad"),
        call("write_role", {"name": "customer_service", "yaml_content": good_yaml}, call_id="tc-good"),
        reply("已写入 ~/.chat_team/roles/customer_service.yaml — 重启 chat-team 后生效。"),
    ])
    agent = build_boss_agent(settings, llm)
    final = await agent.handle("加一个客服角色叫小服", NullStream())
    assert "customer_service.yaml" in final, f"unexpected final reply: {final!r}"
    target = home / "roles" / "customer_service.yaml"
    assert target.exists(), "good yaml should have been written"
    Role.from_yaml(target)  # parse must succeed

    # The bad-call result must be a [tool_error] surfaced back to the LLM as
    # a tool message so it can self-correct.
    tool_msgs = [m for m in agent.history if m.role == "tool"]
    assert len(tool_msgs) >= 2, f"expected >=2 tool msgs, got {len(tool_msgs)}"
    assert any("[tool_error]" in (m.content or "") for m in tool_msgs), \
        f"first attempt should be a tool_error; got: {[m.content for m in tool_msgs]}"
    assert any("customer_service" in (m.content or "") and "tool_error" not in (m.content or "")
               for m in tool_msgs), "second attempt should be a success result"
    print(f"  ✓ agent retried after [tool_error] and wrote {target.name}")


async def test_team_profile_roundtrip() -> None:
    print("== test 4: team_profile read/write round-trip ==")
    home = _fresh_home("team_profile")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    ctx = _ctx(settings)
    new_text = "## 我们是谁\n上海某某公司 · 客户成功部\n"
    out = await WriteTeamProfileTool().run(ctx, content=new_text)
    print(f"  ✓ wrote: {out}")
    got = await ReadTeamProfileTool().run(ctx)
    assert got == new_text, f"round-trip mismatch:\nwrote={new_text!r}\nread ={got!r}"
    # And empty string clears.
    await WriteTeamProfileTool().run(ctx, content="")
    cleared = await ReadTeamProfileTool().run(ctx)
    assert cleared == "", f"expected empty after clear, got {cleared!r}"
    print("  ✓ empty content clears file")


async def test_delete_builtin_refused() -> None:
    print("== test 5: delete_role refuses builtin ==")
    home = _fresh_home("delete_builtin")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    ctx = _ctx(settings)
    try:
        await DeleteRoleTool().run(ctx, name="team_admin")
    except ToolError as exc:
        print(f"  ✓ delete builtin refused: {exc}")
    else:
        raise AssertionError("expected ToolError when deleting builtin team_admin")


async def test_list_roles_includes_builtin_and_user() -> None:
    print("== test 6: list_roles surfaces builtin + user roles ==")
    home = _fresh_home("list_roles")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    ctx = _ctx(settings)
    # Drop a user role.
    await WriteRoleTool().run(
        ctx, name="qa", yaml_content="name: qa\ndisplay_name: 测试工程师\n",
    )
    listing = await ListRolesTool().run(ctx)
    assert "team_admin" in listing and "[builtin]" in listing
    assert "qa" in listing and "[user]" in listing
    print("  ✓ listing covers builtin + user")


def test_boss_not_in_registry() -> None:
    print("== test 7: BOSS_ROLE not in RoleRegistry ==")
    home = _fresh_home("registry")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    names = RoleRegistry.load(settings.paths.user_roles_dir).names()
    assert "boss" not in names, f"boss leaked into registry: {names}"
    # Also verify the boss role hardcoded name is what we expect (unchanged).
    assert BOSS_ROLE.name == "boss"
    print(f"  ✓ registry names = {names}; boss not present")


def test_boss_tool_registry_complete() -> None:
    print("== test 8: build_boss_tool_registry has all expected tools ==")
    reg = build_boss_tool_registry()
    expected = {
        "list_roles", "read_role", "write_role", "delete_role",
        "read_team_profile", "write_team_profile",
        "list_available_tools", "list_skills",
        "read_deploy_config",
    }
    got = set(reg._tools.keys())  # noqa: SLF001
    assert got == expected, f"tool mismatch: missing={expected - got}, extra={got - expected}"
    print(f"  ✓ {sorted(got)}")


async def test_list_skills_and_role_with_skills() -> None:
    """list_skills surfaces user skills; write_role accepts a skills: whitelist."""
    print("== test 9: list_skills + write_role with skills whitelist ==")
    home = _fresh_home("skills")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["CHAT_TEAM_HOME"] = str(home)
    settings = load_settings()
    ctx = _ctx(settings)

    from chat_team.agent.tools.team_tools import ListSkillsTool

    # Seed two user skills.
    skill_dir = home / "skills" / "pr_review"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        '---\nname: pr_review\ndescription: "给 PR 写 checklist 式 review"\n---\n\nbody A\n',
        encoding="utf-8",
    )
    skill_dir2 = home / "skills" / "translate_zh"
    skill_dir2.mkdir(parents=True, exist_ok=True)
    (skill_dir2 / "SKILL.md").write_text(
        '---\nname: translate_zh\ndescription: "中英互译规约"\n---\n\nbody B\n',
        encoding="utf-8",
    )

    listing = await ListSkillsTool().run(ctx)
    assert "pr_review" in listing and "[user]" in listing, f"missing pr_review:\n{listing}"
    assert "translate_zh" in listing
    assert "中英互译规约" in listing
    print(f"  ✓ list_skills surfaces both: \n    {listing.replace(chr(10), chr(10)+'    ')}")

    # list_available_tools should now include skill / skill_read_file because
    # at least one skill is defined.
    avail = await ListAvailableToolsTool().run(ctx)
    assert "skill:" in avail or "skill " in avail, f"main 'skill' tool missing in:\n{avail}"
    assert "skill_read_file" in avail, f"skill_read_file missing in:\n{avail}"
    print("  ✓ list_available_tools shows skill / skill_read_file when skills exist")

    # write_role with a skills whitelist should round-trip cleanly.
    yaml_with_skills = (
        "name: reviewer\n"
        "display_name: 评审员\n"
        "system_prompt: |\n"
        "  你是评审员小评。\n"
        "tools:\n"
        "  - read_file\n"
        "  - skill\n"
        "  - skill_read_file\n"
        "skills:\n"
        "  - pr_review\n"
    )
    out = await WriteRoleTool().run(ctx, name="reviewer", yaml_content=yaml_with_skills)
    print(f"  ✓ write_role with skills: {out}")
    role = Role.from_yaml(home / "roles" / "reviewer.yaml")
    assert role.skills == ["pr_review"], f"skills field lost: {role.skills}"
    assert "skill" in role.tools
    print("  ✓ Role.from_yaml preserves skills whitelist")


async def main() -> None:
    await test_list_available_tools()
    await test_write_role_validation()
    await test_agent_loop_retry_on_bad_yaml()
    await test_team_profile_roundtrip()
    await test_delete_builtin_refused()
    await test_list_roles_includes_builtin_and_user()
    test_boss_not_in_registry()
    test_boss_tool_registry_complete()
    await test_list_skills_and_role_with_skills()
    print("\nALL BOSS SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
