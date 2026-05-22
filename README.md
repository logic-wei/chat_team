# chat_team

企业微信 (WeCom) 智能机器人 —— 在一个机器人后面隐藏一支可扩展的“虚拟员工团队”。
用户面对单一入口,后端按需把会话交接给最合适的角色:默认 `team_admin` (接待),
可一键转给 `research_engineer` (读写代码、跑命令)、`customer_service` (答疑闲聊),
或你自己写的角色。每个会话都有独立工作目录,会话之间完全隔离。

## 特性

- **长连接 (WebSocket) 模式**:无需公网回调地址,启动即在线,自带应用层心跳。
- **流式回复**:用户先看到“思考中…”占位,过程中持续刷新,最终 `finish=true` 收尾。
- **角色 = 一份 YAML**:加角色不用改派发代码;放进 `~/.chat_team/roles/` 即可覆盖内置。
- **会话级工具沙箱**:文件读写 / shell 都被限制在 `~/.chat_team/workspaces/<sid>/` 内,
  路径穿越、绝对路径、symlink 逃逸全部拒绝;shell 输出超阈值会截断,原文落盘可回查。
- **共享 notebook**:跨员工的“团队白板”—— Markdown 文件,`## key` 分块,
  历史里只看到 TOC,值由 `notebook_read` 工具按需取,避免上下文污染。
- **自动压缩**:每个角色独立 token 预算,超阈值时 LLM 摘要早期消息插回历史头。
- **持久化**:`session.json` debounce 10s 原子刷盘,重启后会话历史和当前在岗员工照旧。
- **媒体落地**:image / file / video 通过每条消息独立的 AES-256-CBC `aeskey` 解密后
  自动入 `<cwd>/inbox/`,Agent 收到一条文本指针。

## 安装

需要 Python ≥ 3.11。建议在虚拟环境中安装:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

只在 `pyproject.toml` 改动 / 增加非 Python 资源 / 重建 venv 时才需要再次 `pip install -e .`;
普通改源码不需要重装(editable 安装把 `src/chat_team` 软链进了 `site-packages`)。

## 配置

首次启动会在 `~/.chat_team/` 下生成默认配置:

```
~/.chat_team/
  config.yaml      # 全局参数 (token 预算、shell 超时、日志轮转……)
  .env             # 凭证 (mode 0600);WECOM_BOT_ID / WECOM_SECRET / OPENAI_*
  roles/           # 用户自定义角色 YAML,同名优先于内置
  workspaces/      # 每个会话一个子目录
  logs/            # chat_team.log,RotatingFileHandler
  state/           # 跨会话状态保留位
```

把企微机器人后台拿到的 BotID / Secret 和 OpenAI 密钥填进 `~/.chat_team/.env`:

```bash
WECOM_BOT_ID=...
WECOM_SECRET=...
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1   # 替换成内部代理 / vLLM / Ollama 网关也行
```

测试时可用 `CHAT_TEAM_HOME=/tmp/chat_team_dev` 把整个根目录搬走,避免污染真实环境。

## 运行

```bash
python main.py
# 等价于:
chat-team
```

`pip install -e .` 之后会注册 `chat-team` 命令(由 `pyproject.toml` 的
`[project.scripts]` 提供),分发时一条命令即可。

正常启动应看到:

```
INFO chat_team.app | chat_team starting; home=/Users/<you>/.chat_team
INFO chat_team.adapters.wecom | connecting to wss://openws.work.weixin.qq.com
INFO chat_team.adapters.wecom | subscribe ok: {...errcode: 0...}
```

之后就可以在企微里 @ 机器人或单聊。日志同时打到 stderr 和 `~/.chat_team/logs/chat_team.log`
(按 10MB × 5 份轮转)。

## 加一个角色

```yaml
# ~/.chat_team/roles/data_analyst.yaml
name: data_analyst
display_name: 数据分析师
description: 处理 SQL、报表、数据探索类需求。
system_prompt: |
  你是数据分析师,名字叫"小数"。回答前先用 read_file 看一下当前目录的样本。
tools:
  - read_file
  - list_dir
  - run_command
  - notebook_read
  - notebook_write
  - transfer_to_employee
welcome_message: 你好,我是数据分析师小数,有数据相关的问题随时问我。
llm:
  model: gpt-4o-mini
  temperature: 0.2
  history_token_budget: 16000
```

不需要重启 —— 不,要重启,角色注册表是进程启动时加载的。但**不需要改任何代码**,
也不需要重新 `pip install`。`transfer_to_employee` 的目标 enum 会在下次启动时自动包含 `data_analyst`。

## 加一个工具

`src/chat_team/agent/tools/` 下子类化 `Tool`,实现 `async run(ctx, **kwargs)`,
然后在 `app.build_tool_registry` 里 `reg.register(...)`,最后在角色 YAML 的 `tools:`
里引用名字即可。可恢复错误抛 `ToolError`(会作为 tool 消息回给 LLM)。

## 烟囱测试 (不依赖 LLM / 网络)

```bash
python scripts/smoke_dispatch.py                  # 派发器 + Agent + 工具
python scripts/smoke_transfer.py                  # 多跳交接 + 上限 + 未知目标
python scripts/smoke_tools.py                     # 文件 / shell + 沙箱
python scripts/smoke_wecom_parse.py               # adapter 解析 + LRU + stream 帧形
python scripts/smoke_compaction_persistence.py    # tiktoken 压缩 + session.json 往返
python scripts/smoke_media_events.py              # AES-256-CBC + enter_chat / disconnected
```

每个 smoke 都把 `CHAT_TEAM_HOME` 指到 `/tmp/...` 并在启动时 `rmtree`,所以反复跑安全。

## 设计要点

详细架构、不显眼的细节、不要踩的地雷见 [`CLAUDE.md`](./CLAUDE.md)。给 Claude Code 看的,
也给人看 —— 它把"为什么这样写"明明白白讲了一遍。

## 已知限制 (v1)

- **单实例运行**。企微"新连接踢旧",同一 BotID 不能多副本启动,会互踢。
  后续考虑外部锁 / leader 选举。
- **只支持 OpenAI Chat Completion**。Anthropic / Gemini 通过 `LLMProvider` 子类后续扩展。
- **文件回传未实现**(`aibot_upload_media_init/chunk/finish`),目前研发员工产出文件只能
  用 markdown 链接告知用户路径。

## 协议

- 协议规约: `docs/wechat_bot_api.md`, `docs/wechat_bot_接收消息.md`
