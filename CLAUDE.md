# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A WeCom (企业微信) AI Bot that fronts a team of role-differentiated "virtual employees". The only builtin role is `team_admin` (the front-desk/receptionist); every other role is user-defined as a YAML in `~/.chat_team/roles/`. Sample non-builtin roles (`research_engineer`, `customer_service`) live under `docs/examples/roles/` for users to copy. One employee handles a session at a time; the LLM transfers control via a `transfer_to_employee` tool call. Each chat session gets its own working directory; sessions are fully isolated.

## Commands

```bash
# Run the bot (long-connection WeCom WebSocket).
# First run seeds ~/.chat_team/{config.yaml,.env,roles,workspaces,logs,state}.
python main.py

# Smoke tests — all are pure Python, no LLM/network. Run individually.
python scripts/smoke_dispatch.py                  # dispatcher + agent + tools (scripted LLM)
python scripts/smoke_transfer.py                  # multi-hop transfer + cap + unknown target
python scripts/smoke_tools.py                     # write/read/list/run_command + sandbox
python scripts/smoke_wecom_parse.py               # adapter parse + LRU + stream frame shape
python scripts/smoke_compaction_persistence.py    # tiktoken compaction + session.json round-trip
python scripts/smoke_media_events.py              # AES-256-CBC decrypt + enter_chat/disconnected
python scripts/smoke_team_profile.py              # team.md injection + compactor isolation
python scripts/smoke_boss.py                      # CLI boss agent + team_tools

# Conversational team-setup CLI (not the WeCom bot — see "Boss agent" below).
chat-team-boss

# Print all tools registered in the main runtime, for hand-authored role YAMLs.
chat-team-tools          # or: python -m chat_team.list_tools
```

There is no test framework — smokes are async `main()` scripts that print and assert. Add new smokes alongside the existing ones; they all set `CHAT_TEAM_HOME=/tmp/...` and `shutil.rmtree` it at startup so they don't pollute the real `~/.chat_team`.

## Runtime directory

All persistent state lives under `~/.chat_team/` (override with `CHAT_TEAM_HOME`):

```
~/.chat_team/
  config.yaml              # global knobs; defaults written on first run
  .env                     # WECOM_BOT_ID, WECOM_SECRET, OPENAI_API_KEY, OPENAI_BASE_URL
  team.md                  # global team profile; injected into every agent's system prompt
  roles/                   # user-defined role YAMLs override builtins by name
  workspaces/<sid>/        # one per chat session
    inbox/                 # decrypted inbound media lands here
    .chat_team/
      session.json         # current_role + per-role histories (debounced, atomic)
      notebook.md          # shared "team whiteboard", ## key blocks, 4KB cap
      notebook.index.json  # updated_at sidecar
      runs/<ts>.log        # full shell stdout (tool returns truncated)
  logs/                    # rotating chat_team.log
  state/                   # cross-session bits (currently empty, reserved)
```

`paths.sanitize_session_id` is the only function that maps `session_id → directory name`. Both `SessionManager.workspace_for` and `WeComBotAdapter._save_media` go through it (the adapter via the `workspace_resolver` callback wired in `app.py`). Don't recompute paths anywhere else.

## Layered architecture

```
WeComBotAdapter   ── parses WS frames, manages stream replies, decrypts media
        ↓ IncomingMessage
Dispatcher        ── owns session.lock, transfer loop, post-turn compact + persist
        ↓ session_id
SessionManager    ── workspace_for / get_or_create; restores from session.json
        ↓
Session           ── cwd, current_role, agents_by_role, notebook, lock,
                     pending_handoff, transfer_count_this_turn, restored_histories
        ↓
Agent (per role)  ── owns this role's history; runs chat+tool loop
        ↓
Tool              ── ToolContext(cwd, session, settings); sandboxed I/O
```

**One Agent = one role × one session.** Histories are NEVER shared across roles. Cross-role facts go through `notebook.md` (a Markdown file with `## key` blocks) — agents see only the TOC injected into their system prompt, and fetch values via `notebook_read`.

### Boss agent (CLI-only)

`src/chat_team/boss.py` is a separate entry point (`chat-team-boss`) that reuses `Agent` + `LLMProvider` + a fresh `ToolRegistry` of `team_tools`. It runs a stdin/stdout chat loop so the user can shape `~/.chat_team/team.md` and `~/.chat_team/roles/*.yaml` conversationally instead of hand-editing YAML.

Key isolation points:
- **`BOSS_ROLE` is hardcoded in `boss.py`** and is NOT registered in `RoleRegistry`. The WeCom-side `transfer_to_employee` enum and `enter_chat` flow never see it.
- **Boss tools** (`agent/tools/team_tools.py`) deliberately bypass the cwd sandbox and operate on absolute paths under `settings.paths.user_roles_dir` / `settings.paths.team_md`. They are NOT registered in `app.build_tool_registry` and never reach the dispatcher.
- **`list_available_tools`** dynamically calls `app.build_tool_registry(RoleRegistry({}))` to enumerate the *main-runtime* tool catalog so the boss recommends valid `tools:` names — picks up any new tool without editing the boss.
- **`write_role`** validates input (yaml parse → `Role.from_dict` → name match → atomic write) before touching disk; bad YAML returns `ToolError` and the LLM self-corrects on retry.
- **Confirmation is by-prompt, not by-CLI**: the boss's system prompt requires it to paste the full proposed YAML/markdown to the user and ask "是否确认写入?" before invoking any write tool. There is no separate y/N gate.
- **No persistence**: each `chat-team-boss` invocation starts a fresh chat. The durable state lives in the role YAMLs and `team.md` it edits.

## Non-obvious mechanics

**Transfer flow.** `transfer_to_employee` raises `TransferRequested` (a special exception, not a `ToolError`). It bubbles from tool → agent → dispatcher. Before re-raising, the agent appends a synthetic `tool` message (`"[transferred] target=X"`) to its OWN history so the dangling `tool_calls` is closed — otherwise reopening that role later breaks OpenAI history validation. The dispatcher catches it, increments `transfer_count_this_turn`, and either:
- transfers (sets `current_role`, queues a `PendingHandoff`, re-loops with the SAME user_text on the new agent), or
- forces the current agent to answer (cap reached / unknown target) by injecting a synthetic system note.

**Handoff notes are one-shot.** `agent.queue_system_note` puts a string into `pending_system_inject`; `_build_system_messages` emits it as a system message and clears the buffer. It is NOT persisted to `agent.history` — that prevents re-injection every turn. If you want it persistent, you need a different mechanism.

**Post-turn pipeline.** `Dispatcher._post_turn` runs `compactor.maybe_compact(agent)` for every agent in the session, then `persistence.schedule(session)`. Both happen INSIDE the session lock to avoid races with the next inbound message. Compaction may make an LLM call.

**Compaction boundary.** `_find_keep_boundary` always lands on a `user` message — never split an `assistant(tool_calls)` + `tool` pair, otherwise the next OpenAI request 400s. Default keeps the last 6 user turns verbatim and replaces the rest with one `[历史摘要]` system message at the head of `agent.history`. Token budget is `role.llm.history_token_budget` falling back to `settings.llm.default_history_token_budget`.

**History restoration is lazy.** `SessionManager.get_or_create` reads `session.json` on first touch, populates `session.restored_histories` and `session.current_role`. The dispatcher's `_agent_for` consumes (pops) the entry only when that role is materialised — that way restoring a session for which an unused role had a long history doesn't pay any cost until that role is actually needed.

**WS write serialization.** Three cooperating asyncio tasks: reader / heartbeat (30s `ping`) / writer. The writer is the only thing that calls `ws.send` — everything else `await self._enqueue_write(payload)`. This prevents stream frames from interleaving with heartbeats or two replies stomping on each other. Stream pushes are throttled to `STREAM_PUSH_MIN_INTERVAL=1.0s` except `finish=true`, which always sends.

**msgid dedup.** WeCom can replay callbacks; `_LRU(500)` per-adapter dedups both `aibot_msg_callback` and `aibot_event_callback` by `msgid`.

**Group `@bot` stripping.** Done in `_handle_msg_callback` AFTER text resolution (so the strip applies to the joined `mixed` text too), via `_MENTION_RE = r"^@\S+\s+"`.

**Media decryption.** Each `image`/`file`/`video` payload carries its own per-URL `aeskey` (NOT the global EncodingAESKey from registration). Decode base64 → 32 bytes (AES-256). IV = first 16 bytes. AES-CBC + PKCS#7. The download URL is valid for 5 minutes — fetch immediately. Files land in `<cwd>/inbox/<ts>-<msgid>.<ext>`; extension comes from magic-byte sniffing (jpg/png/gif/webp/pdf/zip/mp4) or the msgtype default. The adapter's `workspace_resolver` callback (wired from `SessionManager.workspace_for`) decides where they go.

**Tool sandbox.** `_resolve_under(cwd, rel)` rejects absolute paths and `..`, then double-checks via `os.path.realpath` + `os.path.commonpath`. `list_dir` hides anything starting with `.chat_team` so the LLM doesn't see internal metadata. `run_command` runs through `bash -c` with `cwd=ctx.cwd`, hard timeout from settings, output truncated to `shell_output_max_bytes` (full log to `.chat_team/runs/<ts>-<rand>.log` so the LLM can re-read via `read_file` if needed).

**Team profile injection.** `~/.chat_team/team.md` is read once by `load_settings` into `settings.team_profile` (stripped); when non-empty, `Agent._build_system_messages` splices it as a `[团队信息]` block alongside the role prompt and meta lines. Empty/missing file → no block, behaviour unchanged. The compactor's `_summarize` uses its own sterile system prompt (`compactor.py:100-107`) and is intentionally NOT touched. No hot reload — edits to `team.md` only take effect on the next `chat-team` start.

## Adding a role / tool

**Role** — drop a YAML in `src/chat_team/roles/builtin/` (committed) or `~/.chat_team/roles/` (user override, takes precedence). Required fields: `name`, `system_prompt`, `tools` (a subset of registered tool names). Optional: `display_name`, `welcome_message` (used for `enter_chat`), `llm.{model,temperature,history_token_budget}`. No code changes needed — `RoleRegistry.load` picks it up and `transfer_to_employee`'s enum is rebuilt from `roles.names()`.

**Tool** — subclass `Tool` (`src/chat_team/agent/tools/base.py`), set `name`/`description`/`parameters` (JSON schema), implement `async run(ctx, **kwargs)`. Register in `app.build_tool_registry`. Reference it in role YAMLs that should expose it. Raise `ToolError` for recoverable failures (returned to the LLM as a tool message); raise `TransferRequested` only if you're implementing role-switch semantics.

## Things to not break

- **Don't put system messages into `agent.history`.** System content is rebuilt every turn by `Agent._build_system_messages` (role prompt + notebook TOC + pending one-shot injects). The history list is for `user`/`assistant`/`tool` only. The compactor's summary head is the sole exception and is intentional — but `_find_keep_boundary` must continue to return 0 when the head is already a system summary, otherwise we'd compact a compaction.
- **Don't call `ws.send` directly from anywhere except the writer task.** Use `_enqueue_write`.
- **Don't share notebooks or histories across sessions.** Each `Session` instance owns its own `Notebook` pointed at its own `notebook.md`; SessionManager keys by raw `session_id` (sanitization is path-only).
- **Don't catch `TransferRequested` in tools.** Only `Agent.handle` (which closes the dangling tool_call) and `Dispatcher._run_turn` (which acts on it) should see it.
- **Don't put `BOSS_ROLE` into `roles/builtin/` or any user role dir.** It would be picked up by `RoleRegistry.load` and leak into `transfer_to_employee`'s enum + WeCom's enter_chat flow. Boss must stay hardcoded in `boss.py` and registered nowhere.
- **Don't add boss-side tools (`team_tools.py`) to `app.build_tool_registry`.** They sidestep the cwd sandbox by design and would let any WeCom-side role overwrite arbitrary files under `~/.chat_team/`.
