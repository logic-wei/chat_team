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
python scripts/smoke_describe_image.py            # describe_images() cache + DescribeImageTool sandbox
python scripts/smoke_vision_shim.py               # eager OCR shim (image blocks → text in user content)
python scripts/smoke_llm_debug_log.py             # per-call LLM debug log: redaction + write + seq
python scripts/smoke_skills.py                    # SkillRegistry + skill / skill_read_file tools + system-prompt TOC
python scripts/smoke_critical_fixes.py            # persistence-race snapshot + agent turn rollback + dispatcher post_turn on _run_turn error
python scripts/smoke_p0_fixes.py                  # reconnect + env scrub + LRU + janitor + LLM retry
python scripts/smoke_split_llm.py                 # vision_llm vs chat_llm split: build_vision_llm_provider + DescribeImageTool routing
python scripts/smoke_p0_round2.py                 # _bg_tasks strong-ref + concurrent get_or_create + eviction off event loop + finish-before-post_turn
python scripts/smoke_mcp.py                       # MCP proxy tool + config parsing + agent integration

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
  skills/                  # user-defined skill dirs (<name>/SKILL.md + aux files) override builtins by name
  workspaces/<sid>/        # one per chat session
    inbox/                 # decrypted inbound media lands here
    .chat_team/
      session.json         # current_role + per-role histories (debounced, atomic)
      notebook.md          # shared "team whiteboard", ## key blocks, 4KB cap
      notebook.index.json  # updated_at sidecar
      runs/<ts>.log        # full shell stdout (tool returns truncated)
      llm/<ts>-<seq>-<role>-<kind>.json  # per-LLM-call debug record (when llm.debug_log_enabled)
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

**Group `@bot` stripping.** `_strip_mention_from_first_text` is applied inside `_resolve_inbound_blocks` to the **first text block of the current message ONLY** — never to a quote interior, and never to non-leading text blocks. This is a behavioural change from the pre-vision adapter (which used `_MENTION_RE` on the joined string and would strip any leading-position `@x ` regardless of which fragment it came from). The `[image, text("@bot hi")]` ordering is now stripped where it previously wasn't — intentional improvement. `_MENTION_RE` itself is still exported for the boss + tests.

**Vision content blocks.** The adapter no longer flattens inbound user content to a string. `IncomingMessage.content_blocks: list[ContentBlock]` carries an ordered mix of `{"type":"text","text":...}` and `{"type":"image","path":"./inbox/<file>"}` items; `inbound.text` is its `blocks_to_text` rendering (`[图:<basename>]` for images), kept for logging/dedup/stream previews. `ChatMessage.content` is `str | list[ContentBlock]`; only **user** messages ever carry list content. Persistence stores list content verbatim in `session.json` (legacy string content reads through unchanged). The compactor and OpenAI provider both branch on `isinstance(content, list)` and use `blocks_to_text` for any string-only path (token counting, summary input, tool/assistant/system messages). Out of scope: `file` / `video` / `voice` always degrade to text placeholders even inside vision turns; tool-result strings never become list content. **In the default `tool` vision strategy, the dispatcher pre-processes `content_blocks` through `vision_shim.apply_vision_strategy` before reaching the agent — image blocks are replaced with `[图:rel]\n<desc>` text, so what lands in `agent.history` is a flat string. List content only flows into history for roles that explicitly opt into `vision_strategy: direct`.**

**Vision strategy (eager OCR shim).** Default `settings.llm.vision.strategy = "tool"` runs every inbound image through `describe_images()` *before* `agent.handle` — pre-OCR'd descriptions land in the user message text, so `agent.history` is text-only and the compactor counts real token weight. The agent never sees raw images in tool mode (except via the `describe_image` tool when it wants a different prompt). Why eager-not-lazy: agent reliability ("forgot to call the tool") is eliminated, OCR-heavy multi-turn workloads see ~6× token savings, and the per-turn LLM call upcharge is repaid by cross-turn caching. **`direct` strategy** keeps the original list content and falls back to in-context vision (the pre-shim behaviour) — set `llm.vision_strategy: direct` on a role YAML for high-fidelity visual chat (e.g. art/diagram analysis). Invalid `vision_strategy` values silently fall back to the settings default with a warning. `default_eager_prompt` (in `settings.llm`) is the prompt fed to OCR — defaults to OCR-priority with a fallback short caption; users can override per-deployment in `~/.chat_team/config.yaml`. `default_eager_detail` defaults to `"high"` because `low` can't read small text.

**Split vision / chat providers.** 视觉/OCR 调用可以走独立的 API 端点。凭证统一通过 `.env` 环境变量配置（与主模型一致）：`OPENAI_VISION_API_KEY` 和 `OPENAI_VISION_BASE_URL`，不设则回落至主 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。`config.yaml` 的 `llm` 节点中用 `llm.vision.model` 指定视觉模型名称（留空则复用 `llm.chat.model`）。当凭证与主模型相同时复用同一 `LLMProvider` 实例（不多开连接）。`app.build_vision_llm_provider` 处理此逻辑；`Dispatcher` 持有 `self._vision_llm` 并传给急切 OCR shim（`apply_vision_strategy`）和 `Agent` 构造（`describe_image` 工具经 `ToolContext.vision_llm` 使用）。Compactor 始终用聊天模型。

**Image description cache.** `chat_team.llm.image_description_cache.ImageDescriptionCache` is a process-level LRU keyed by `(abs_path, mtime_ns, size, detail, model, prompt)` → description text. Caps: `MAX_ENTRIES=128`, `MAX_TOTAL_BYTES≈1MB`. Module-level singleton via `default_cache()`; same image with same prompt+detail+model is OCR'd exactly once across roles, sessions, and turns within a process. Different prompt or different file mtime/size invalidates. The cache is shared by both the eager shim AND the `describe_image` tool, so an agent re-querying with a custom prompt only pays for prompts not already cached.

**Image base64 cache.** `chat_team.llm.image_cache.ImageDataURICache` is a module-level LRU keyed by `(abs_path, mtime_ns, size)` → `data:image/<mime>;base64,...`. Caps: `MAX_ENTRIES=32`, `MAX_TOTAL_BYTES≈32MB`, per-image `MAX_INLINE_BYTES=6MB` (raw — base64 ≈ 8MB, leaves headroom under OpenAI's ~10MB request limit). Missing file → `None`; oversize → `None`. The provider degrades both cases to `[图:<name>(已丢失)]` / `(过大,已省略)` text blocks so a single bad image doesn't fail the whole turn. MIME map reuses `wecom_media.sniff_extension` (jpg/png/gif/webp; everything else → image/jpeg).

**Quote (引用) flattening.** WeCom's `quote` field (sibling of `msgtype` on the body) can be text / image / mixed and is handled recursively by the same `_flatten_payload` that handles the current message. The resolver wraps the quote sequence between two text blocks `[引用开始]` / `[引用结束 — 以下为本条新消息]` and prepends them before the current-message blocks; `coalesce_text_blocks` then merges adjacent text spans. The @bot strip is applied to the current-side blocks BEFORE the quote sequence is concatenated, so a quote whose first item is `@bot` is never touched.

**`image_detail` knob.** Defaults to `settings.llm.vision.image_detail = "high"` (~1600 tokens per 1024² image on `gpt-4o`-class models). **Only matters in `vision_strategy: direct` mode** — in the default `tool` mode, the agent never receives raw images, so `image_detail` on the agent's role is moot. The eager shim itself uses `settings.llm.default_eager_detail` (also `"high"` by default). Override per role with `llm.image_detail: low|high|auto`. The provider stamps `detail` on every `image_url` part it builds; `image_base_dir` is plumbed from `session.cwd` so `./inbox/<file>` paths in history resolve correctly. Compactor token-counting drift (placeholder vs. real vision tokens) is no longer an issue in the default mode because history is text — only direct-mode roles still need to budget for it.

**Media decryption.** Each `image`/`file`/`video` payload carries its own per-URL `aeskey` (NOT the global EncodingAESKey from registration). Decode base64 → 32 bytes (AES-256). IV = first 16 bytes. AES-CBC + PKCS#7. The download URL is valid for 5 minutes — fetch immediately. Files land in `<cwd>/inbox/<ts>-<msgid>-<idx>.<ext>` (the `idx` suffix prevents same-second collisions on multi-image bursts inside one `mixed` message); extension comes from magic-byte sniffing (jpg/png/gif/webp/pdf/zip/mp4) or the msgtype default. The adapter's `workspace_resolver` callback (wired from `SessionManager.workspace_for`) decides where they go.

**Tool sandbox.** `_resolve_under(cwd, rel)` rejects absolute paths and `..`, then double-checks via `os.path.realpath` + `os.path.commonpath`. `list_dir` hides anything starting with `.chat_team` so the LLM doesn't see internal metadata. `run_command` runs through `bash -c` with `cwd=ctx.cwd`, hard timeout from settings, output truncated to `shell_output_max_bytes` (full log to `.chat_team/runs/<ts>-<rand>.log` so the LLM can re-read via `read_file` if needed).

**Team profile injection.** `~/.chat_team/team.md` is read once by `load_settings` into `settings.team_profile` (stripped); when non-empty, `Agent._build_system_messages` splices it as a `[团队信息]` block alongside the role prompt and meta lines. Empty/missing file → no block, behaviour unchanged. The compactor's `_summarize` uses its own sterile system prompt (`compactor.py:100-107`) and is intentionally NOT touched. No hot reload — edits to `team.md` only take effect on the next `chat-team` start.

**LLM debug log.** Opt-in: set `llm.debug_log_enabled: true` in `~/.chat_team/config.yaml` (default off — one file per call piles up fast and transcripts can carry sensitive user content, so production must stay off). When on, every call into `OpenAIChatCompletionProvider.complete` writes a JSON file to `<workspace>/.chat_team/llm/<ts>-<seq>-<role>-<kind>.json`. The record carries the full request payload (messages + tools + model + temperature + max_tokens), the response (content + tool_calls + finish_reason + usage from `completion.usage.model_dump()`), and `latency_ms`. Three `call_kind` values: `agent` (main turn), `compactor` (post-turn summary), `vision` (eager OCR shim + `describe_image` tool). Failures write the same file with `error=repr(exc)` and `response=null` before re-raising. **Base64 image data URIs are redacted** to `[redacted: <mime> <bytes> bytes]` via `chat_team.llm.debug_logger.redact_messages` — files stay grep-able. Per-session monotonic `seq` (process-local dict keyed by `session_id`) keeps filenames sortable when the millisecond clock collides. The provider's `_maybe_write_log` reuses the exact `messages_payload` it built for OpenAI (no re-serialisation), so what you see in the log is what the API saw. Writes are best-effort: a write failure is logged at WARNING and the call still returns normally.

## Adding a role / tool

**Role** — drop a YAML in `src/chat_team/roles/builtin/` (committed) or `~/.chat_team/roles/` (user override, takes precedence). Required fields: `name`, `system_prompt`, `tools` (a subset of registered tool names). Optional: `display_name`, `welcome_message` (used for `enter_chat`), `llm.{model,temperature,history_token_budget,image_detail,vision_strategy}`, `mcp_servers` (list of MCP server names from `config.yaml`). `vision_strategy: tool|direct` overrides the global default — set `direct` on a role that needs raw images in context (e.g. art critique). No code changes needed — `RoleRegistry.load` picks it up and `transfer_to_employee`'s enum is rebuilt from `roles.names()`.

**Tool** — subclass `Tool` (`src/chat_team/agent/tools/base.py`), set `name`/`description`/`parameters` (JSON schema), implement `async run(ctx, **kwargs)`. Register in `app.build_tool_registry`. Reference it in role YAMLs that should expose it. Raise `ToolError` for recoverable failures (returned to the LLM as a tool message); raise `TransferRequested` only if you're implementing role-switch semantics. If your tool needs to call the LLM (e.g. vision/embedding), read `ctx.llm` — `ToolContext.llm` is wired to `agent.llm` so the tool reuses the same provider configuration.

**Skill** — a no-code capability pack: drop a directory at `~/.chat_team/skills/<name>/` containing `SKILL.md` (YAML frontmatter + markdown body) plus optional auxiliary files. Frontmatter must have `name` (must equal the directory name) and `description` (single line preferred — multi-line works but only the first line lands in the system-prompt TOC). Body is whatever instructions you want the agent to follow when invoked. To expose a skill to a role, list it under `skills:` in the role YAML AND include `skill` (and optionally `skill_read_file`) in `tools:`. The agent sees a `[可用 skills] - name: description` block in its system prompt, fetches the body via `skill(name=...)`, and reads aux files via `skill_read_file(skill=..., path=...)`. `SkillRegistry.load` mirrors `RoleRegistry.load`: builtin (`src/chat_team/skills/builtin/`) first, user dir overrides by name. Malformed skills (missing/invalid frontmatter, name/dir mismatch, missing SKILL.md) are logged at WARNING and skipped — one bad dir won't break the rest. Per-role gating happens twice: once when rendering the TOC (filtered to `role.skills ∩ registry.names()`) and again at tool invocation in `SkillTool.run`. The `enum` on the JSON-schema parameters is the full registry (one tool instance for all roles), so the runtime check is the real gate. No hot reload — restart `chat-team` after adding/editing skills.

**Python deps for skills (uv + PEP 723).** SKILL.md format is deliberately kept 100% compatible with community skills (frontmatter only carries `name` + `description`), so per-skill dependency declarations are out of scope. Instead: when a role's `tools` contains **both** `skill` and `run_command`, `Agent._build_system_messages` splices in the `PYTHON_UV_CONVENTION` block (`agent/agent.py`) — it tells the agent to write Python scripts with PEP 723 inline metadata (`# /// script\n# dependencies = [...]\n# ///`) and run them via `uv run script.py`. `uv` resolves deps into its global content-addressed env cache (`~/.cache/uv/environments-v2/<hash>/`), shared across sessions/roles/workspaces with zero per-workspace state. `app.warn_if_uv_missing` logs a WARNING on startup if any loaded role would need `uv` but it isn't on PATH; the bot still runs (non-Python skills unaffected). Why not per-workspace venv: 100 sessions × 100MB site-packages is wasteful in a multi-tenant bot. Why not `pip_install` tool: reactive install-on-import-fail costs a full LLM turn per missed dep.

## MCP (Model Context Protocol)

MCP 让角色无需写 Python 即可使用外部工具服务器。用户在 `config.yaml` 声明 MCP 服务器，在角色 YAML 的 `mcp_servers` 字段引用，启动时自动发现并注册工具。

**配置。** `config.yaml` 新增 `mcp:` 节，字典风格（与 Claude Desktop 等 MCP 客户端一致）：

```yaml
mcp:
  servers:
    filesystem:                          # 服务器名 — 角色 YAML 中引用
      command: npx                       # stdio transport: 启动子进程
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    github:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:                               # 传给子进程的额外环境变量
        GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
    remote_api:
      url: http://localhost:8080/sse     # SSE transport: 连接远程服务
```

Transport 自动推断：有 `command` → stdio，有 `url` → SSE，二选一。服务器名须匹配 `[a-zA-Z0-9_-]+` 且不含 `__`（双下划线用于注册名分隔）。

**角色引用。** 在角色 YAML 中添加 `mcp_servers` 字段，列出要使用的服务器名：

```yaml
name: developer
tools: [read_file, write_file, run_command, transfer_to_employee]
mcp_servers: [filesystem, github]        # 该角色可使用这两个 MCP 服务器的所有工具
```

**注册名。** MCP 工具注册为 `mcp__<server>__<tool>`（如 `mcp__filesystem__read_file`），符合 OpenAI function name regex。`Agent._effective_tool_names()` 在每轮调用 LLM 时自动展开 `role.mcp_servers` 为具体工具名，与内置工具合并后传给 `ToolRegistry.specs_for()`。

**生命周期。** `app._async_main` 在 `build_dispatcher` 之前调用 `McpClientManager.connect_all()` 连接所有配置的服务器、发现工具、创建 `McpProxyTool` 实例，然后通过 `build_tool_registry(extra_tools=...)` 注入 `ToolRegistry`。`finally` 块调用 `close_all()` 关闭连接。单个服务器连接失败记 WARNING 并跳过，不阻塞启动。

**工具调用。** `McpProxyTool.run()` 调用 MCP SDK 的 `session.call_tool()`，结果中的 `TextContent` 直接取 `.text`，`ImageContent` 降级为 `[image: mime]` 占位符。MCP 返回 `isError=True` 或底层异常均包装为 `ToolError`，走已有的 agent 错误处理流程。

**关键文件：** `src/chat_team/mcp/config.py`（配置 dataclass）、`src/chat_team/mcp/client.py`（`McpClientManager` 生命周期）、`src/chat_team/mcp/proxy_tool.py`（`McpProxyTool(Tool)` 桥接）。

**限制。** 当前只支持 MCP Tools，不支持 Resources / Prompts。不支持热加载——修改 MCP 配置需重启 bot。`chat-team-tools` CLI 不列出 MCP 工具（它们是运行时动态发现的）。

## Things to not break

- **Don't put system messages into `agent.history`.** System content is rebuilt every turn by `Agent._build_system_messages` (role prompt + notebook TOC + pending one-shot injects). The history list is for `user`/`assistant`/`tool` only. The compactor's summary head is the sole exception and is intentional — but `_find_keep_boundary` must continue to return 0 when the head is already a system summary, otherwise we'd compact a compaction.
- **Don't call `ws.send` directly from anywhere except the writer task.** Use `_enqueue_write`.
- **Don't share notebooks or histories across sessions.** Each `Session` instance owns its own `Notebook` pointed at its own `notebook.md`; SessionManager keys by raw `session_id` (sanitization is path-only).
- **Don't catch `TransferRequested` in tools.** Only `Agent.handle` (which closes the dangling tool_call) and `Dispatcher._run_turn` (which acts on it) should see it.
- **Don't put `BOSS_ROLE` into `roles/builtin/` or any user role dir.** It would be picked up by `RoleRegistry.load` and leak into `transfer_to_employee`'s enum + WeCom's enter_chat flow. Boss must stay hardcoded in `boss.py` and registered nowhere.
- **Don't add boss-side tools (`team_tools.py`) to `app.build_tool_registry`.** They sidestep the cwd sandbox by design and would let any WeCom-side role overwrite arbitrary files under `~/.chat_team/`.
