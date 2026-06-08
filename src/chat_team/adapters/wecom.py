"""WeCom (企业微信) AI Bot adapter — long-connection (WebSocket) mode.

Implements the wire protocol documented in ``docs/wechat_bot_api.md`` /
``docs/wechat_bot_接收消息.md`` directly on top of ``websockets``. Three
co-operating asyncio tasks run while a connection is alive:

* **reader**  — pulls frames from the socket, dispatches by ``cmd``.
* **heartbeat** — sends ``{"cmd": "ping"}`` every 30 seconds.
* **writer**  — single drain task on an ``asyncio.Queue`` that holds JSON
  payloads to send. All writes go through this so frames cannot interleave.

Each ``aibot_msg_callback`` is dispatched as a fire-and-forget task so a
slow user turn does not block the reader.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import random
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import websockets

from ..config import Settings
from . import wecom_media
from .base import (
    BotAdapter,
    ChatType,
    ContentBlock,
    IncomingMessage,
    MessageHandler,
    StreamHandle,
    blocks_to_text,
    coalesce_text_blocks,
)

WorkspaceResolver = Callable[[str], Path]

log = logging.getLogger(__name__)

WECOM_WS_URL = "wss://openws.work.weixin.qq.com"
HEARTBEAT_INTERVAL = 30
STREAM_PUSH_MIN_INTERVAL = 1.0          # seconds between intermediate stream frames
WRITE_QUEUE_MAXSIZE = 1024

# Reconnect backoff: 1s, 2s, 4s, … capped at 5 min; +0.5s jitter per attempt.
RECONNECT_INITIAL_DELAY = 1.0
RECONNECT_MAX_DELAY = 300.0
RECONNECT_JITTER_CAP = 0.5

UPLOAD_CHUNK_SIZE = 256 * 1024          # raw bytes per chunk; ~341KB base64, under 512KB cap
UPLOAD_RESPONSE_TIMEOUT = 30.0          # per-step ack timeout
MEDIA_SIZE_LIMITS = {
    "image": 10 * 1024 * 1024,
    "file": 20 * 1024 * 1024,
}

_MENTION_RE = re.compile(r"^@\S+\s+")


def _strip_mention_from_first_text(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Remove a leading ``@bot `` mention from the first text block.
    Image / non-text blocks before the first text block are preserved as-is.
    Quote markers (added later) live after this strip is applied to the
    current-message side, so quote interiors are never touched."""
    out = list(blocks)
    for i, block in enumerate(out):
        if block.get("type") == "text":
            text = (block.get("text") or "")
            stripped = _MENTION_RE.sub("", text, count=1).strip()
            out[i] = {"type": "text", "text": stripped}
            break
    return out


def _new_req_id() -> str:
    return uuid.uuid4().hex


class _LRU(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def add(self, key: str) -> bool:
        """Return True if key was newly inserted, False if already present."""
        if key in self:
            self.move_to_end(key)
            return False
        self[key] = True
        if len(self) > self.capacity:
            self.popitem(last=False)
        return True


class WeComStreamHandle:
    """Streaming reply backed by aibot_respond_msg / msgtype=stream.

    Throttles intermediate frames to ``STREAM_PUSH_MIN_INTERVAL`` seconds.
    Final ``finish()`` is always sent (un-throttled) with finish=true.
    """

    def __init__(self, adapter: "WeComBotAdapter", req_id: str):
        self._adapter = adapter
        self._req_id = req_id
        self._stream_id = uuid.uuid4().hex
        self._content = ""
        self._last_push = 0.0
        self._closed = False

    async def push(self, chunk: str, *, append: bool = True) -> None:
        if self._closed:
            return
        self._content = (self._content + chunk) if append else chunk
        if time.monotonic() - self._last_push < STREAM_PUSH_MIN_INTERVAL:
            return                                          # throttle silently
        await self._send_frame(self._content, finish=False)

    async def status(self, note: str) -> None:
        if self._closed:
            return
        if time.monotonic() - self._last_push < STREAM_PUSH_MIN_INTERVAL:
            return
        # status messages don't accumulate into the body; they're transient.
        await self._send_frame(f"{self._content}\n\n_{note}_" if self._content else f"_{note}_",
                               finish=False)

    async def finish(self, final_text: str) -> None:
        if self._closed:
            return
        self._closed = True
        text = final_text or "(空回复)"
        await self._send_frame(text, finish=True)

    async def _send_frame(self, content: str, *, finish: bool) -> None:
        payload = {
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": self._req_id},
            "body": {
                "msgtype": "stream",
                "stream": {
                    "id": self._stream_id,
                    "finish": finish,
                    "content": content,
                },
            },
        }
        await self._adapter._enqueue_write(payload)
        self._last_push = time.monotonic()

    async def send_image(self, path: Path, *, filename: str | None = None) -> None:
        await self._send_media(path, kind="image", filename=filename)

    async def send_file(self, path: Path, *, filename: str | None = None) -> None:
        await self._send_media(path, kind="file", filename=filename)

    async def _send_media(self, path: Path, *, kind: str, filename: str | None) -> None:
        data = await asyncio.to_thread(path.read_bytes)
        name = filename or path.name
        media_id = await self._adapter.upload_media(data, kind=kind, filename=name)
        payload = {
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": self._req_id},
            "body": {"msgtype": kind, kind: {"media_id": media_id}},
        }
        await self._adapter._enqueue_write(payload)


class WeComBotAdapter(BotAdapter):
    def __init__(
        self,
        settings: Settings,
        workspace_resolver: WorkspaceResolver | None = None,
        *,
        bot_id: str | None = None,
        secret: str | None = None,
        role_name: str | None = None,
    ):
        self.settings = settings
        self.bot_id = bot_id or settings.get_env("WECOM_BOT_ID") or ""
        self.secret = secret or settings.get_env("WECOM_SECRET") or ""
        self.role_name = role_name
        self._handler: MessageHandler | None = None
        # Per-adapter (lifetime) state ----------------------------------
        # ``_msgid_lru`` survives across reconnects so server-side replays
        # are still deduped.
        self._msgid_lru = _LRU(settings.session.msgid_lru_size)
        self._workspace_resolver = workspace_resolver
        # ``_shutdown`` is set only by close()/SIGINT; signals the outer
        # run_forever loop to exit.
        self._shutdown = asyncio.Event()
        # Strong refs to in-flight callback handlers. asyncio only weakly
        # references tasks (CPython implementation detail) — without this
        # set, a slow callback can be garbage-collected mid-execution and
        # silently disappear under load. Spans reconnects so an in-flight
        # turn survives a brief WS blip.
        self._bg_tasks: set[asyncio.Task] = set()
        # Per-connection state (rebuilt each connect iteration) ---------
        self._ws: Any = None
        self._write_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(WRITE_QUEUE_MAXSIZE)
        self._tasks: list[asyncio.Task] = []
        # ``_connection_dead`` flips when the current connection ends for
        # any reason (server close, disconnected_event, write error, …).
        # Recreated per connection so a fresh wait() resolves correctly.
        self._connection_dead = asyncio.Event()
        self._pending_acks: dict[str, asyncio.Future] = {}

    # ---- BotAdapter interface ---------------------------------------------

    def set_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def connect(self) -> None:
        """Open one WS connection and subscribe.

        Kept for backward-compat with code/tests that drive ``connect()`` +
        ``run()`` directly. In production use ``run_forever()`` instead,
        which handles transient connection loss with exponential backoff.
        """
        if not self.bot_id or not self.secret:
            raise RuntimeError("WECOM_BOT_ID / WECOM_SECRET missing in ~/.chat_team/.env")
        log.info("connecting to %s", WECOM_WS_URL)
        # WeCom uses application-level heartbeat ({"cmd":"ping"} every ~30s);
        # disable the websockets library's protocol-level auto-ping so it does
        # not close the socket when WeCom predictably ignores it.
        self._ws = await websockets.connect(
            WECOM_WS_URL,
            max_size=8 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
        )
        await self._subscribe()

    async def run(self) -> None:
        """Run one connection until it dies. Use ``run_forever`` in prod."""
        if self._ws is None:
            raise RuntimeError("call connect() before run()")
        self._tasks = [
            asyncio.create_task(self._writer_loop(), name="wecom-writer"),
            asyncio.create_task(self._heartbeat_loop(), name="wecom-heartbeat"),
            asyncio.create_task(self._reader_loop(), name="wecom-reader"),
        ]
        try:
            await self._connection_dead.wait()
        finally:
            await self._tear_down_connection()

    async def run_forever(self) -> None:
        """Production entry point: keep reconnecting until ``close()``.

        On any non-shutdown disconnect, sleeps with exponential backoff
        (1s, 2s, 4s, …, capped at 5 min plus jitter) then reconnects.
        Backoff resets to 1s after a connection that survived long enough
        to subscribe successfully.
        """
        backoff = RECONNECT_INITIAL_DELAY
        while not self._shutdown.is_set():
            try:
                await self._open_connection()
                # Subscribe succeeded; future disconnects are "transient".
                backoff = RECONNECT_INITIAL_DELAY
                await self._serve_one_connection()
            except Exception as exc:                             # noqa: BLE001
                log.warning("ws connection ended: %r", exc)
            finally:
                await self._tear_down_connection()
            if self._shutdown.is_set():
                break
            sleep_for = backoff + random.uniform(0, RECONNECT_JITTER_CAP)
            log.info("reconnecting in %.2fs", sleep_for)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=sleep_for)
                break                                            # shutdown won the race
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, RECONNECT_MAX_DELAY)

    async def close(self) -> None:
        self._shutdown.set()
        self._connection_dead.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:                                # noqa: BLE001
                pass
            self._ws = None

    # ---- connection lifecycle --------------------------------------------

    async def _open_connection(self) -> None:
        """Reset per-connection state, open socket, subscribe."""
        if not self.bot_id or not self.secret:
            raise RuntimeError("WECOM_BOT_ID / WECOM_SECRET missing in ~/.chat_team/.env")
        self._connection_dead = asyncio.Event()
        self._write_queue = asyncio.Queue(WRITE_QUEUE_MAXSIZE)
        self._pending_acks = {}
        log.info("connecting to %s", WECOM_WS_URL)
        self._ws = await websockets.connect(
            WECOM_WS_URL,
            max_size=8 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
        )
        await self._subscribe()

    async def _serve_one_connection(self) -> None:
        """Start the 3 cooperating tasks and wait for the connection to die."""
        self._tasks = [
            asyncio.create_task(self._writer_loop(), name="wecom-writer"),
            asyncio.create_task(self._heartbeat_loop(), name="wecom-heartbeat"),
            asyncio.create_task(self._reader_loop(), name="wecom-reader"),
        ]
        await self._connection_dead.wait()

    async def _tear_down_connection(self) -> None:
        """Cancel per-connection tasks, fail any in-flight ack futures, drop
        the websocket. Safe to call multiple times (idempotent)."""
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):      # noqa: BLE001
                pass
        self._tasks = []
        # Any caller blocked on _send_and_await must wake up with an error
        # rather than hang forever once the socket is gone.
        for req_id, fut in list(self._pending_acks.items()):
            if not fut.done():
                fut.set_exception(
                    ConnectionResetError(f"connection lost; req_id={req_id}")
                )
        self._pending_acks = {}
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:                                # noqa: BLE001
                pass
            self._ws = None

    # ---- internals --------------------------------------------------------

    async def _subscribe(self) -> None:
        req_id = _new_req_id()
        payload = {
            "cmd": "aibot_subscribe",
            "headers": {"req_id": req_id},
            "body": {"bot_id": self.bot_id, "secret": self.secret},
        }
        await self._ws.send(json.dumps(payload, ensure_ascii=False))
        # Wait for the subscribe ack before declaring the connection live.
        raw = await asyncio.wait_for(self._ws.recv(), timeout=15)
        msg = json.loads(raw)
        if msg.get("errcode") not in (0, None):
            raise RuntimeError(f"subscribe failed: {msg!r}")
        log.info("subscribe ok: %s", msg)

    async def _writer_loop(self) -> None:
        while True:
            payload = await self._write_queue.get()
            if payload is None:
                return
            try:
                await self._ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception:                                # noqa: BLE001
                log.warning("ws write failed; signalling reconnect", exc_info=True)
                self._connection_dead.set()
                return

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await self._enqueue_write({"cmd": "ping", "headers": {"req_id": _new_req_id()}})

    async def _reader_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("dropping non-json frame: %r", raw[:200])
                    continue
                cmd = msg.get("cmd") or ""
                if cmd == "aibot_msg_callback":
                    self._spawn_bg(
                        self._handle_msg_callback(msg), name="wecom-msg-cb",
                    )
                elif cmd == "aibot_event_callback":
                    self._spawn_bg(
                        self._handle_event_callback(msg), name="wecom-event-cb",
                    )
                elif cmd == "" and msg.get("errmsg") is not None:
                    self._dispatch_ack(msg)                 # upload/heartbeat/subscribe acks
                else:
                    log.debug("frame ignored: cmd=%s", cmd)
        except websockets.ConnectionClosed as err:
            log.warning("ws closed: %s", err)
        finally:
            # Always flag the connection dead so run_forever can reconnect
            # (or run() can return cleanly). _shutdown is unaffected.
            self._connection_dead.set()

    async def _enqueue_write(self, payload: dict[str, Any]) -> None:
        await self._write_queue.put(payload)

    def _spawn_bg(self, coro, *, name: str | None = None) -> asyncio.Task:
        """Create a background task and hold a strong reference to it until
        completion. Without this, asyncio's weak-ref-only Task tracking can
        let an in-flight handler be garbage-collected mid-await — Python's
        docs explicitly warn about this. ``add_done_callback`` releases the
        ref when the task finishes."""
        task = asyncio.create_task(coro, name=name)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ---- message dispatch -------------------------------------------------

    async def _handle_msg_callback(self, frame: dict[str, Any]) -> None:
        body = frame.get("body") or {}
        msg_id = body.get("msgid") or ""
        if msg_id and not self._msgid_lru.add(msg_id):
            log.info("dedup msgid=%s", msg_id)
            return
        try:
            inbound = self._parse_metadata(frame)
        except Exception:                                    # noqa: BLE001
            log.exception("failed to parse callback")
            return
        if inbound is None:
            return                                           # unsupported msgtype, already logged

        msgtype = body.get("msgtype") or "text"
        try:
            blocks = await self._resolve_inbound_blocks(
                body, msgtype, inbound.session_id, inbound.chat_type,
            )
        except Exception:                                    # noqa: BLE001
            log.exception("failed to resolve inbound blocks for msgtype=%s", msgtype)
            blocks = [{"type": "text",
                       "text": f"[用户发来 {msgtype},但下载/解密失败]"}]
        if blocks is None:
            log.info("unsupported msgtype=%s; ignoring", msgtype)
            return
        blocks = coalesce_text_blocks(blocks)
        if not blocks:
            blocks = [{"type": "text", "text": "(空消息)"}]
        inbound.content_blocks = blocks
        inbound.text = blocks_to_text(blocks)

        handler = self._handler
        if handler is None:
            log.error("no handler registered; dropping message")
            return

        stream = WeComStreamHandle(self, req_id=inbound.reply_token)
        # Initial 思考中 frame so the user sees something immediately.
        await stream._send_frame("思考中…", finish=False)

        try:
            await handler(inbound, stream)
        except Exception:                                    # noqa: BLE001
            log.exception("handler raised")
            try:
                await stream.finish("(系统错误,请稍后再试)")
            except Exception:                                # noqa: BLE001
                pass

    def _parse_metadata(self, frame: dict[str, Any]) -> IncomingMessage | None:
        body = frame.get("body") or {}
        headers = frame.get("headers") or {}
        chat_type_raw = body.get("chattype") or "single"
        chat_type = ChatType.GROUP if chat_type_raw == "group" else ChatType.SINGLE
        chat_id = body.get("chatid")
        aibot_id = body.get("aibotid") or self.bot_id
        from_user = (body.get("from") or {}).get("userid") or "anonymous"

        if chat_type == ChatType.GROUP and chat_id:
            session_id = f"wecom-group-{chat_id}"
        else:
            session_id = f"wecom-single-{aibot_id}-{from_user}"

        return IncomingMessage(
            session_id=session_id,
            chat_type=chat_type,
            user_id=from_user,
            text="",                                          # filled in async stage
            msg_id=body.get("msgid") or "",
            bot_id=aibot_id,
            chat_id=chat_id,
            reply_token=headers.get("req_id"),
            raw=body,
        )

    async def _resolve_inbound_blocks(
        self,
        body: dict[str, Any],
        msgtype: str,
        session_id: str,
        chat_type: ChatType,
    ) -> list[ContentBlock] | None:
        """Build the ordered content-block list for an inbound message body.

        Handles WeCom's sibling ``quote`` field by recursively flattening
        the quote payload and prefixing the current-message blocks with
        text-boundary markers so the LLM can tell what was quoted vs. what
        the user just sent. The group ``@bot `` mention strip is applied to
        the first text block of the *current* message side only — never to
        the quote interior or the marker blocks.

        Returns ``None`` for genuinely unsupported msgtypes so the caller
        can drop the message; otherwise always returns a non-empty list.
        """
        msg_id = body.get("msgid") or ""
        # voice msgtype is unique: WeCom delivers it pre-transcribed text,
        # not media to download. Empty transcription → drop entirely.
        if msgtype == "voice":
            transcribed = ((body.get("voice") or {}).get("content") or "").strip()
            current_blocks: list[ContentBlock] = (
                [{"type": "text", "text": transcribed}] if transcribed else []
            )
            if not current_blocks and not body.get("quote"):
                return None
        else:
            current_blocks = await self._flatten_payload(
                body, msgtype, session_id, msg_id, idx=0,
            )

        if chat_type == ChatType.GROUP:
            current_blocks = _strip_mention_from_first_text(current_blocks)

        quote = body.get("quote") or {}
        if quote and quote.get("msgtype"):
            quote_blocks = await self._flatten_payload(
                quote, quote.get("msgtype"), session_id, f"{msg_id}-q", idx=0,
            )
        else:
            quote_blocks = []

        if not quote_blocks:
            return current_blocks or [{"type": "text", "text": "(空消息)"}]

        # Wrap the quote in text-boundary markers so the model can
        # distinguish quoted context from the new message body.
        return (
            [{"type": "text", "text": "[引用开始]"}]
            + quote_blocks
            + [{"type": "text", "text": "[引用结束 — 以下为本条新消息]"}]
            + current_blocks
        )

    async def _flatten_payload(
        self,
        payload: dict[str, Any],
        msgtype: str,
        session_id: str,
        msg_id: str,
        *,
        idx: int = 0,
    ) -> list[ContentBlock]:
        """Recursively flatten a WeCom message body (or a quote sub-payload)
        into an ordered ContentBlock list. Handles ``text``, ``image``,
        ``mixed`` (recurses into ``msg_item``), and falls back to a text
        placeholder for ``voice`` / ``file`` / ``video`` (their bytes are
        saved to inbox but not sent to the LLM as vision)."""
        if msgtype == "text":
            content = ((payload.get("text") or {}).get("content") or "").strip()
            return [{"type": "text", "text": content}] if content else []

        if msgtype == "image":
            image_payload = payload.get("image") or {}
            rel = await self._save_media_bytes(
                image_payload, "image", session_id, f"{msg_id or 'msg'}-{idx}",
            )
            if rel is None:
                return [{"type": "text", "text": "[图片下载失败]"}]
            return [{"type": "image", "path": rel}]

        if msgtype == "mixed":
            items = (payload.get("mixed") or {}).get("msg_item") or []
            # Sequential iteration (rather than asyncio.gather) keeps things
            # simple; ordering is preserved by construction. WeCom rarely
            # delivers more than a handful of items per mixed message.
            blocks: list[ContentBlock] = []
            for i, it in enumerate(items):
                it_type = it.get("msgtype") or ""
                sub = await self._flatten_payload(
                    it, it_type, session_id, msg_id, idx=i,
                )
                if not sub and it_type:
                    sub = [{"type": "text", "text": f"[未支持的 mixed 子项: {it_type}]"}]
                blocks.extend(sub)
            return blocks

        if msgtype in ("file", "video"):
            sub_payload = payload.get(msgtype) or {}
            placeholder = await self._save_media_placeholder(
                sub_payload, msgtype, session_id, f"{msg_id or 'msg'}-{idx}",
            )
            return [{"type": "text", "text": placeholder}]

        # Anything else: emit a text placeholder so the message is still
        # routable rather than silently dropped.
        return [{"type": "text", "text": f"[未支持: {msgtype}]"}]

    def _dispatch_ack(self, msg: dict[str, Any]) -> None:
        req_id = (msg.get("headers") or {}).get("req_id")
        fut = self._pending_acks.pop(req_id, None) if req_id else None
        if fut is not None and not fut.done():
            fut.set_result(msg)
        else:
            log.debug("ack: %s", msg)

    async def _send_and_await(
        self,
        payload: dict[str, Any],
        *,
        timeout: float = UPLOAD_RESPONSE_TIMEOUT,
    ) -> dict[str, Any]:
        req_id = payload["headers"]["req_id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_acks[req_id] = fut
        try:
            await self._enqueue_write(payload)
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_acks.pop(req_id, None)

    async def upload_media(self, data: bytes, *, kind: str, filename: str) -> str:
        """Upload bytes via aibot_upload_media_init/chunk/finish; return media_id."""
        if kind not in ("image", "file"):
            raise RuntimeError(f"unsupported media kind: {kind}")
        size = len(data)
        if size < 5:
            raise RuntimeError("file too small (WeCom requires ≥5 bytes)")
        cap = MEDIA_SIZE_LIMITS[kind]
        if size > cap:
            raise RuntimeError(f"{kind} exceeds {cap} bytes (got {size})")
        total_chunks = max(1, math.ceil(size / UPLOAD_CHUNK_SIZE))
        md5 = hashlib.md5(data).hexdigest()

        init_resp = await self._send_and_await({
            "cmd": "aibot_upload_media_init",
            "headers": {"req_id": _new_req_id()},
            "body": {
                "type": kind,
                "filename": filename,
                "total_size": size,
                "total_chunks": total_chunks,
                "md5": md5,
            },
        })
        if init_resp.get("errcode") not in (0, None):
            raise RuntimeError(f"upload_init failed: {init_resp!r}")
        upload_id = (init_resp.get("body") or {}).get("upload_id") or ""
        if not upload_id:
            raise RuntimeError(f"upload_init missing upload_id: {init_resp!r}")

        for idx in range(total_chunks):
            chunk = data[idx * UPLOAD_CHUNK_SIZE : (idx + 1) * UPLOAD_CHUNK_SIZE]
            chunk_resp = await self._send_and_await({
                "cmd": "aibot_upload_media_chunk",
                "headers": {"req_id": _new_req_id()},
                "body": {
                    "upload_id": upload_id,
                    "chunk_index": idx,
                    "base64_data": base64.b64encode(chunk).decode("ascii"),
                },
            })
            if chunk_resp.get("errcode") not in (0, None):
                raise RuntimeError(f"upload_chunk[{idx}] failed: {chunk_resp!r}")

        finish_resp = await self._send_and_await({
            "cmd": "aibot_upload_media_finish",
            "headers": {"req_id": _new_req_id()},
            "body": {"upload_id": upload_id},
        })
        if finish_resp.get("errcode") not in (0, None):
            raise RuntimeError(f"upload_finish failed: {finish_resp!r}")
        media_id = (finish_resp.get("body") or {}).get("media_id") or ""
        if not media_id:
            raise RuntimeError(f"upload_finish missing media_id: {finish_resp!r}")
        return media_id

    async def _save_media_bytes(
        self,
        payload: dict[str, Any],
        msgtype: str,
        session_id: str,
        media_tag: str,
    ) -> str | None:
        """Download + decrypt + persist media bytes to the session's inbox.

        Returns the workspace-relative path (``./inbox/<file>``) on success
        or ``None`` on any failure. Caller decides how to render failure —
        image flow turns it into a text block, file/video flow into a
        placeholder string. ``media_tag`` typically encodes msgid + index
        so concurrent downloads don't collide on the same-second filename.
        """
        url = payload.get("url") or ""
        aeskey = payload.get("aeskey") or ""
        if not url or not aeskey:
            log.warning("media payload missing url/aeskey for %s", msgtype)
            return None
        if self._workspace_resolver is None:
            log.warning("no workspace_resolver wired; cannot save media")
            return None
        try:
            plain = await wecom_media.download_and_decrypt(url, aeskey)
        except Exception:                                    # noqa: BLE001
            log.exception("download/decrypt failed for %s", msgtype)
            return None
        cwd = self._workspace_resolver(session_id)
        inbox = cwd / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        ext = wecom_media.sniff_extension(plain, msgtype)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", media_tag)[-32:] or "media"
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{ts}-{safe_tag}.{ext}"
        out = inbox / fname
        await asyncio.to_thread(out.write_bytes, plain)
        return f"./inbox/{fname}"

    async def _save_media_placeholder(
        self,
        payload: dict[str, Any],
        msgtype: str,
        session_id: str,
        media_tag: str,
    ) -> str:
        """Wrap _save_media_bytes for non-vision media (file/video). Always
        returns a printable placeholder, never None — failure becomes a
        ``下载失败`` placeholder so the LLM still sees something."""
        rel = await self._save_media_bytes(payload, msgtype, session_id, media_tag)
        if rel is None:
            return f"[用户发来 {msgtype},但下载失败]"
        try:
            cwd = self._workspace_resolver(session_id)
            size = (cwd / rel.lstrip("./")).stat().st_size
        except Exception:                                    # noqa: BLE001
            size = -1
        size_part = f" ({size} bytes)" if size >= 0 else ""
        return f"[用户发来 {msgtype}: {rel}{size_part}]"

    async def _handle_event_callback(self, frame: dict[str, Any]) -> None:
        body = frame.get("body") or {}
        headers = frame.get("headers") or {}
        event = (body.get("event") or {}).get("eventtype") or ""
        msg_id = body.get("msgid") or ""
        if msg_id and not self._msgid_lru.add(msg_id):
            return
        if event == "enter_chat":
            session_id = self._session_id_from_body(body)
            await self._reply_welcome(headers.get("req_id"), session_id)
        elif event == "disconnected_event":
            log.warning("received disconnected_event; will reconnect")
            self._connection_dead.set()
        else:
            log.info("event ignored: %s", event)

    def _session_id_from_body(self, body: dict[str, Any]) -> str | None:
        """Compute the same session_id _parse_metadata would, using only the
        routing fields available on every callback (msg or event). Returns
        None if the body lacks enough info to identify the session."""
        chat_type_raw = body.get("chattype") or "single"
        chat_id = body.get("chatid")
        aibot_id = body.get("aibotid") or self.bot_id
        from_user = (body.get("from") or {}).get("userid") or ""
        if chat_type_raw == "group" and chat_id:
            return f"wecom-group-{chat_id}"
        if from_user:
            return f"wecom-single-{aibot_id}-{from_user}"
        return None

    async def _reply_welcome(
        self,
        req_id: str | None,
        session_id: str | None = None,
    ) -> None:
        from ..roles.registry import RoleRegistry
        from ..session.persistence import load_state
        roles = RoleRegistry.load(self.settings.paths.user_roles_dir)
        # Solo mode: always use the pinned role.
        if self.role_name and roles.has(self.role_name):
            role_name = self.role_name
        else:
            # Pick whichever role the user will *actually* talk to next: the
            # persisted current_role for this session if any, otherwise the
            # global default. Without this, a returning user sees admin's
            # "我是小管" while their messages still route to e.g. engineer.
            role_name = self.settings.default_role
            if session_id and self._workspace_resolver is not None:
                try:
                    cwd = self._workspace_resolver(session_id)
                    state = load_state(cwd)
                except Exception:                                # noqa: BLE001
                    state = None
                prior_role = (state or {}).get("current_role")
                if prior_role and roles.has(prior_role):
                    role_name = prior_role
        welcome = ""
        if roles.has(role_name):
            welcome = (roles.get(role_name).welcome_message or "").strip()
        if not welcome:
            welcome = "你好,我是这个团队的机器人助手。"
        payload = {
            "cmd": "aibot_respond_welcome_msg",
            "headers": {"req_id": req_id or _new_req_id()},
            "body": {"msgtype": "text", "text": {"content": welcome}},
        }
        await self._enqueue_write(payload)
