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
    IncomingMessage,
    MessageHandler,
    StreamHandle,
)

WorkspaceResolver = Callable[[str], Path]

log = logging.getLogger(__name__)

WECOM_WS_URL = "wss://openws.work.weixin.qq.com"
HEARTBEAT_INTERVAL = 30
STREAM_PUSH_MIN_INTERVAL = 1.0          # seconds between intermediate stream frames
WRITE_QUEUE_MAXSIZE = 1024

UPLOAD_CHUNK_SIZE = 256 * 1024          # raw bytes per chunk; ~341KB base64, under 512KB cap
UPLOAD_RESPONSE_TIMEOUT = 30.0          # per-step ack timeout
MEDIA_SIZE_LIMITS = {
    "image": 10 * 1024 * 1024,
    "file": 20 * 1024 * 1024,
}

_MENTION_RE = re.compile(r"^@\S+\s+")


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
        data = path.read_bytes()
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
    ):
        self.settings = settings
        self.bot_id = settings.get_env("WECOM_BOT_ID") or ""
        self.secret = settings.get_env("WECOM_SECRET") or ""
        self._handler: MessageHandler | None = None
        self._ws: Any = None
        self._write_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(WRITE_QUEUE_MAXSIZE)
        self._msgid_lru = _LRU(settings.session.msgid_lru_size)
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._workspace_resolver = workspace_resolver
        self._pending_acks: dict[str, asyncio.Future] = {}

    # ---- BotAdapter interface ---------------------------------------------

    def set_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def connect(self) -> None:
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
        if self._ws is None:
            raise RuntimeError("call connect() before run()")
        self._tasks = [
            asyncio.create_task(self._writer_loop(), name="wecom-writer"),
            asyncio.create_task(self._heartbeat_loop(), name="wecom-heartbeat"),
            asyncio.create_task(self._reader_loop(), name="wecom-reader"),
        ]
        try:
            await self._stop.wait()
        finally:
            for t in self._tasks:
                t.cancel()
            for t in self._tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    async def close(self) -> None:
        self._stop.set()
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
                log.exception("ws write failed")
                self._stop.set()
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
                    asyncio.create_task(self._handle_msg_callback(msg))
                elif cmd == "aibot_event_callback":
                    asyncio.create_task(self._handle_event_callback(msg))
                elif cmd == "" and msg.get("errmsg") is not None:
                    self._dispatch_ack(msg)                 # upload/heartbeat/subscribe acks
                else:
                    log.debug("frame ignored: cmd=%s", cmd)
        except websockets.ConnectionClosed as err:
            log.warning("ws closed: %s", err)
        finally:
            self._stop.set()

    async def _enqueue_write(self, payload: dict[str, Any]) -> None:
        await self._write_queue.put(payload)

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
            text = await self._resolve_inbound_text(body, msgtype, inbound.session_id)
        except Exception:                                    # noqa: BLE001
            log.exception("failed to resolve inbound text for msgtype=%s", msgtype)
            text = f"[用户发来 {msgtype},但下载/解密失败]"
        if text is None:
            log.info("unsupported msgtype=%s; ignoring", msgtype)
            return
        if inbound.chat_type == ChatType.GROUP:
            text = _MENTION_RE.sub("", text, count=1).strip()
        inbound.text = text

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

    async def _resolve_inbound_text(
        self, body: dict[str, Any], msgtype: str, session_id: str
    ) -> str | None:
        if msgtype == "text":
            return ((body.get("text") or {}).get("content") or "").strip()
        if msgtype == "voice":
            return ((body.get("voice") or {}).get("content") or "").strip() or None
        if msgtype == "mixed":
            return await self._resolve_mixed(body, session_id)
        if msgtype in ("image", "file", "video"):
            payload = body.get(msgtype) or {}
            saved = await self._save_media(payload, msgtype, session_id, body.get("msgid") or "")
            return saved or f"[用户发来 {msgtype},但下载失败]"
        return None

    async def _resolve_mixed(self, body: dict[str, Any], session_id: str) -> str:
        items = (body.get("mixed") or {}).get("msg_item") or []
        chunks: list[str] = []
        for idx, it in enumerate(items):
            it_type = it.get("msgtype") or ""
            if it_type == "text":
                content = (it.get("text") or {}).get("content") or ""
                if content.strip():
                    chunks.append(content.strip())
            elif it_type == "image":
                saved = await self._save_media(
                    it.get("image") or {}, "image", session_id,
                    f"{body.get('msgid') or 'mixed'}-{idx}",
                )
                chunks.append(saved or "[图片下载失败]")
            else:
                chunks.append(f"[未支持的 mixed 子项: {it_type}]")
        return "\n".join(chunks).strip() or "(空消息)"

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

    async def _save_media(
        self,
        payload: dict[str, Any],
        msgtype: str,
        session_id: str,
        media_tag: str,
    ) -> str | None:
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
        out.write_bytes(plain)
        rel = f"./inbox/{fname}"
        return f"[用户发来 {msgtype}: {rel} ({len(plain)} bytes)]"

    async def _handle_event_callback(self, frame: dict[str, Any]) -> None:
        body = frame.get("body") or {}
        headers = frame.get("headers") or {}
        event = (body.get("event") or {}).get("eventtype") or ""
        msg_id = body.get("msgid") or ""
        if msg_id and not self._msgid_lru.add(msg_id):
            return
        if event == "enter_chat":
            await self._reply_welcome(headers.get("req_id"))
        elif event == "disconnected_event":
            log.warning("received disconnected_event; closing")
            self._stop.set()
        else:
            log.info("event ignored: %s", event)

    async def _reply_welcome(self, req_id: str | None) -> None:
        from ..roles.registry import RoleRegistry
        roles = RoleRegistry.load(self.settings.paths.user_roles_dir)
        default = self.settings.default_role
        welcome = ""
        if roles.has(default):
            welcome = (roles.get(default).welcome_message or "").strip()
        if not welcome:
            welcome = "你好,我是这个团队的机器人助手。"
        payload = {
            "cmd": "aibot_respond_welcome_msg",
            "headers": {"req_id": req_id or _new_req_id()},
            "body": {"msgtype": "text", "text": {"content": welcome}},
        }
        await self._enqueue_write(payload)
