"""Telegram, the first messaging app the agent is reached by, as Hermes Agent's gateway and
OpenClaw's channels reach theirs.

The box asks Telegram's Bot API for messages, long polling, so no port is opened to the internet.
The user makes a bot with @BotFather and gives the app its token. Only the people the user allows
talk to it: anyone else who writes is told so once, and their request waits in the app, as
OpenClaw's pairing has it; groups are ignored. Each private chat is one conversation, going on, as
Hermes Agent's are; /new begins another, /stop stops a reply. A reply goes back whole once its turn
ends, "typing…" shown till then, then the files the turn made; files sent are the workspace's,
attached. A task set in a chat's conversation is answered there too.
"""

import html
import json
import queue
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from typing import Any

from leat.agent.agent import Agent, Busy, NotFound
from leat.agent.tools import numbered

API = "https://api.telegram.org"
POLL = 30  # seconds a poll waits for messages
TYPING = 4  # seconds between signs of typing, which Telegram shows for 5
LONGEST = 4096  # characters of a message
FILES = 20 << 20  # bytes of a file the Bot API lets a bot download
KEY = "telegram"  # of the settings: its token, its bot, the people allowed and asking, its chats
WELCOME = "Hi! I'm Leat, your assistant. Write to me as to anyone; /new begins a new conversation."
REFUSED = (
    "Hi! I'm a private assistant: my owner must let you in first, in Leat's settings, where your "
    "request now waits."
)


class TelegramError(Exception):
    """Telegram refused a request, or could not be reached."""


class Unreachable(TelegramError):
    """Telegram could not be reached."""


class Telegram:
    """The agent's Telegram bot, once the app gives it a token, at Telegram's `api`."""

    def __init__(self, agent: Agent, api: str = API):
        self.agent, self.api = agent, api
        # the conversations whose turns a chat began: its id, and of a group's, the message asking
        self.started: dict[str, tuple[int, int | None]] = {}
        self._wake = threading.Event()  # set when the token changes
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._poll, name="leat telegram", daemon=True).start()
        threading.Thread(target=self._deliver, name="leat telegram replies", daemon=True).start()

    def state(self) -> dict[str, Any]:
        """What the owner's app shows: the bot, the people allowed, each as a person of the
        household's, and those asking; not the token."""
        s = self._settings()
        people = {k: list(s.get(k, {}).values()) for k in ("allowed", "requests")}
        return {"type": KEY, "bot": s.get("bot")} | people

    def connect(self, token: str) -> dict[str, Any]:
        """Takes a bot's token, which Telegram must know. Raises TelegramError if it does not."""
        try:
            bot = self._call("getMe", token=token.strip())
        except Unreachable:
            raise
        except TelegramError as e:
            raise TelegramError("it does not know that token: copy it again from @BotFather") from e
        self._change(lambda s: s | {"token": token.strip(), "bot": bot["username"]})
        self._call("setMyCommands", commands=[
            {"command": "new", "description": "Begin a new conversation"},
            {"command": "stop", "description": "Stop the reply"},
        ])  # fmt: skip
        self._wake.set()
        return self.state()

    def disconnect(self) -> None:
        self._change(lambda s: {k: v for k, v in s.items() if k not in ("token", "bot")})
        self._wake.set()

    def allow(self, id: int, person: int | None = None) -> None:
        """Lets in someone who asked, as a person of the household, by its id, and tells them."""

        def allowed(s: dict[str, Any]) -> dict[str, Any]:
            asked = s.get("requests", {}).pop(str(id), None)
            if asked is None:
                raise NotFound(f"no one with the id {id} asked")
            return s | {"allowed": s.get("allowed", {}) | {str(id): asked | {"person": person}}}

        self._change(allowed)
        self._send(id, "You're in! Write to me as to anyone.")

    def refuse(self, id: int) -> None:
        """Turns out a person allowed, or turns down one who asked."""

        def refused(s: dict[str, Any]) -> dict[str, Any]:
            for people in ("allowed", "requests"):
                s.get(people, {}).pop(str(id), None)
            return s

        self._change(refused)

    def _poll(self) -> None:
        # reads the messages people send, each handled in turn; waits for a token if there is
        # none, and a while after Telegram failed. Each bot numbers its updates: the next to read
        # is of the token's.
        polled, offset = None, 0
        while True:
            if not (token := self._settings().get("token")):
                self._wake.wait(POLL)
                self._wake.clear()
                continue
            if token != polled:
                polled, offset = token, 0
            try:
                updates = self._call(
                    "getUpdates", token, offset=offset, timeout=POLL, allowed_updates=["message"]
                )
            except TelegramError:
                self._wake.wait(5)
                self._wake.clear()
                continue
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    self._handle(update.get("message") or {})
                except Exception:  # a bug's: said, and the next message read
                    traceback.print_exc()

    def _handle(self, message: dict[str, Any]) -> None:
        # a message of a private chat: from someone allowed, sent the agent; from someone not, a
        # request for the app. One of a group, to the agent, from someone allowed: asked of it.
        if (kind := message.get("chat", {}).get("type")) in ("group", "supergroup"):
            return self._asked(message)
        if kind != "private":
            return
        person, chat = message["from"], message["chat"]["id"]
        settings = self._settings()
        if (allowed := settings.get("allowed", {}).get(str(person["id"]))) is None:
            if str(person["id"]) not in settings.get("requests", {}):
                who = {"id": person["id"], "name": _name(person), "asked": time.time()}
                self._change(
                    lambda s: s | {"requests": s.get("requests", {}) | {str(who["id"]): who}}
                )
                self._send(chat, REFUSED)
            return
        text = (message.get("text") or message.get("caption") or "").strip()
        whose = allowed.get("person")  # of the household's people
        conversation = settings.get("chats", {}).get(str(chat))
        if (c := self.agent.store.conversation(conversation or "")) is None or c["person"] != whose:
            conversation = None  # deleted, or another's since the person was changed
        if (command := (text.split() or [""])[0].split("@")[0]) in ("/start", "/new", "/stop"):
            return self._command(command, chat, conversation, whose)
        sent = message.get("document") or (message.get("photo") or [None])[-1]
        if sent is not None and sent.get("file_size", 0) > FILES:
            return self._send(
                chat, f"That file is too big: Telegram lets me take {FILES >> 20} MB at most."
            )
        attached = [self._download(sent)] if sent and self.agent.workspace else []
        if not text and not attached:
            return self._send(chat, "I read text and files, but not that yet.")
        try:  # the chat known before the turn can end, which delivering it waits for
            with self._lock:
                id = self.agent.send(conversation, text or "(The files attached.)",
                                     attached=attached, via="telegram", person=whose)  # fmt: skip
                self.started[id] = (chat, None)
        except Busy:
            return self._send(chat, "I'm still working on a reply here: /stop stops it.")
        if id != conversation:
            self._change(lambda s: s | {"chats": s.get("chats", {}) | {str(chat): id}})
        self._call("sendChatAction", chat_id=chat, action="typing")

    def _asked(self, message: dict[str, Any]) -> None:
        # a group's message that names the bot, or answers it, from someone allowed: asked of the
        # agent, with the message it answers, as "@leat is this true?" under another; each
        # person's a conversation of their own in the group, knowing none of theirs. Anyone else
        # the bot answers nothing, as a group is no place to ask to join.
        settings, sender, chat = self._settings(), message["from"], message["chat"]["id"]
        allowed, bot = settings.get("allowed", {}).get(str(sender["id"])), settings.get("bot")
        text = (message.get("text") or message.get("caption") or "").strip()
        answered = message.get("reply_to_message") or {}
        theirs = answered.get("from", {}).get("username") == bot
        if allowed is None or not bot or not (f"@{bot}".lower() in text.lower() or theirs):
            return
        question = re.sub(rf"@{re.escape(bot)}\b", "", text, flags=re.I).strip() or "Is it true?"
        if (quoted := answered.get("text") or answered.get("caption")) and not theirs:
            author = _name(answered.get("from") or {}) or "Someone"
            question = f"{author} wrote in the group:\n> {quoted}\n\n{question}"
        whose, key = allowed.get("person"), f"{chat}:{sender['id']}"
        conversation = settings.get("chats", {}).get(key)
        if (c := self.agent.store.conversation(conversation or "")) is None or c["person"] != whose:
            conversation = None
        try:
            with self._lock:
                id = self.agent.send(conversation, question, via="telegram", person=whose,
                                     shared=True)  # fmt: skip
                self.started[id] = (chat, message["message_id"])
        except Busy:
            return
        if id != conversation:
            self._change(lambda s: s | {"chats": s.get("chats", {}) | {key: id}})
        self._call("sendChatAction", chat_id=chat, action="typing")

    def _command(
        self, command: str, chat: int, conversation: str | None, person: int | None
    ) -> None:
        if command == "/new":
            self._change(lambda s: s | {"chats": {c: v for c, v in s.get("chats", {}).items()
                                                  if c != str(chat)}})  # fmt: skip
            self._send(chat, "A new conversation begins.")
        elif command == "/stop" and conversation is not None:
            self.agent.stop(conversation, person)
        elif command == "/start":
            self._send(chat, WELCOME)

    def _download(self, sent: dict[str, Any]) -> str:
        # puts a file sent, a document or a photo's largest size, in the workspace; returns its
        # name there
        workspace = self.agent.workspace
        assert workspace is not None
        path = self._call("getFile", file_id=sent["file_id"])["file_path"]
        token = self._settings()["token"]
        with urllib.request.urlopen(f"{self.api}/file/bot{token}/{path}", timeout=POLL) as response:
            data = response.read(FILES)
        name = workspace.free(
            sent.get("file_name") or f"photo {time.strftime('%Y-%m-%d %H.%M')}.jpg"
        )
        workspace.path(name).write_bytes(data)
        self.agent.files_changed()
        return name

    def _deliver(self) -> None:
        # sends the replies of the turns chats began, and of the tasks of chats' conversations,
        # once each ends; shows them typing till then
        with self.agent.events.watch() as events:
            typed = 0.0
            while True:
                try:
                    event = events.get(timeout=TYPING)
                except queue.Empty:
                    event = {}
                try:
                    typed = self._typing(typed)
                    self._on(event)
                except TelegramError:
                    pass
                except Exception:
                    traceback.print_exc()

    def _typing(self, typed: float) -> float:
        # shows the chats waiting for a reply typing, each TYPING seconds
        if time.monotonic() - typed < TYPING:
            return typed
        with self._lock:
            waiting = {chat for chat, _ in self.started.values()}
        for chat in waiting:
            self._call("sendChatAction", chat_id=chat, action="typing")
        return time.monotonic()

    def _on(self, event: dict[str, Any]) -> None:
        kind, id = event.get("type"), str(event.get("conversation"))
        if kind == "conversation" and not event["conversation"]["running"]:
            with self._lock:
                started = self.started.pop(event["conversation"]["id"], None)
            if started is not None:
                self._reply(event["conversation"]["id"], *started)
        elif kind == "done":  # a task's turn, of a private chat's conversation
            chats = {
                v: int(c) for c, v in self._settings().get("chats", {}).items() if ":" not in c
            }
            if id in chats:
                self._reply(id, chats[id])
        elif kind == "error":
            with self._lock:
                started = self.started.pop(id, None)
            if started is not None:
                self._send(started[0], f"Sorry, that failed: {event['error']}")
        elif kind == "deleted":  # in the app, as its turn ran: nothing to wait for
            with self._lock:
                self.started.pop(event["id"], None)

    def _reply(self, id: str, chat: int, asked: int | None = None) -> None:
        # the last turn's answer, then the files it made; in a group, as an answer to the message
        # that asked
        messages = self.agent.store.messages(id)
        start = max(i for i, m in enumerate(messages) if m["role"] == "user")
        turn = messages[start + 1 :]
        answer = next((m for m in reversed(turn) if m["role"] == "assistant"), None)
        text = (answer or {}).get("content") or ""
        info = (answer or {}).get("info", {})
        if info.get("stopped"):
            text += "\n\n(Stopped.)"
        if info.get("cut"):
            text += "\n\n(Cut off: the conversation is out of room. /new begins another.)"
        sources = {n: url for url, n in numbered(messages).items()}
        for i, part in enumerate(_parts(text.strip() or "(No answer.)")):
            reply = {"reply_parameters": {"message_id": asked}} if asked and not i else {}
            try:
                self._send(chat, to_html(part, sources), parse_mode="HTML", **reply)
            except TelegramError:  # of formatting Telegram cannot read: as it is
                self._send(chat, part, **reply)
        made = [
            name for m in turn if m["role"] == "tool" for name in m.get("info", {}).get("files", [])
        ]
        workspace = self.agent.workspace
        for name in dict.fromkeys(made):
            if workspace is not None and workspace.path(name).is_file():
                self._document(chat, name, workspace.path(name).read_bytes())

    def _send(self, chat: int, text: str, **options: Any) -> None:
        unlinked = {"is_disabled": True}
        self._call("sendMessage", chat_id=chat, text=text, link_preview_options=unlinked, **options)

    def _document(self, chat: int, name: str, data: bytes) -> None:
        # sends a file, by its name without its folder, in a form's encoding, as the Bot API takes
        boundary, quoted = uuid.uuid4().hex, name.rsplit("/", 1)[-1].replace('"', "'")
        field = f"--{boundary}\r\nContent-Disposition: form-data; name="
        body = b"".join([
            f'{field}"chat_id"\r\n\r\n{chat}\r\n'.encode(),
            f'{field}"document"; filename="{quoted}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n", data,
            f"\r\n--{boundary}--\r\n".encode(),
        ])  # fmt: skip
        self._post("sendDocument", body, f"multipart/form-data; boundary={boundary}")

    def _call(self, method: str, token: str | None = None, **parameters: Any) -> Any:
        # a method's result, with its parameters as JSON
        body = json.dumps(parameters).encode()
        return self._post(method, body, "application/json", token)

    def _post(self, method: str, body: bytes, kind: str, token: str | None = None) -> Any:
        token = token or self._settings().get("token")
        if not token:
            raise TelegramError("no bot's token was given")
        request = urllib.request.Request(
            f"{self.api}/bot{token}/{method}", body, {"Content-Type": kind}
        )
        try:
            with urllib.request.urlopen(request, timeout=POLL + 10) as response:
                answer = json.loads(response.read())
        except urllib.error.HTTPError as e:
            try:
                said = json.loads(e.read()).get("description", str(e))
            except ValueError:
                said = str(e)
            raise TelegramError(said) from e
        except (OSError, ValueError) as e:
            raise Unreachable(f"Telegram is not reachable: {e}") from e
        if not answer.get("ok"):
            raise TelegramError(answer.get("description", "refused"))
        return answer["result"]

    def _settings(self) -> dict[str, Any]:
        return self.agent.store.setting(KEY) or {}

    def _change(self, change: Any) -> None:
        # changes the settings, telling the apps what they show of them
        with self._lock:
            self.agent.store.set_setting(KEY, change(self._settings()))
        self.agent.events.publish(self.state() | {"to": "owner"})


def to_html(markdown: str, sources: dict[int, str] | None = None) -> str:
    """Markdown as Telegram's HTML has it: bold, italics, code, links, quotes, and citations, [1],
    linked to their sources; headings bold, lists' marks bullets, tables in a block as they are."""
    lines, out, i = markdown.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(f"<pre>{html.escape(chr(10).join(code), quote=False)}</pre>")
        elif _ROW.match(line):
            rows = []
            while i < len(lines) and _ROW.match(lines[i]):
                if not _RULE.match(lines[i]):
                    rows.append(lines[i].strip())
                i += 1
            out.append(f"<pre>{html.escape(chr(10).join(rows), quote=False)}</pre>")
            continue
        elif heading := re.match(r"#{1,6}\s+(.*)", line):
            out.append(f"<b>{_inline(heading[1], sources or {})}</b>")
        elif item := re.match(r"(\s*)[-*+]\s+(.*)", line):
            out.append(f"{item[1]}• {_inline(item[2], sources or {})}")
        elif quote := re.match(r">\s?(.*)", line):
            out.append(f"<blockquote>{_inline(quote[1], sources or {})}</blockquote>")
        elif re.fullmatch(r"\s*([-*_]\s*){3,}", line):
            out.append("———")
        else:
            out.append(_inline(line, sources or {}))
        i += 1
    return "\n".join(out)


_ROW = re.compile(r"\s*\|.*\|\s*$")  # a table's row
_RULE = re.compile(r"\s*\|[\s:|-]+\|\s*$")  # the rule under a table's header


def _inline(text: str, sources: dict[int, str]) -> str:
    # a line's spans: code, links, bold, italics, and citations
    out = []
    for i, part in enumerate(re.split(r"`([^`]+)`", text)):
        if i % 2:
            out.append(f"<code>{html.escape(part, quote=False)}</code>")
            continue
        s = html.escape(part, quote=False)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)
        s = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: f"<b>{m[1] or m[2]}</b>", s)
        s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", s)
        s = re.sub(r"\[(\d{1,3})\](?!\()", lambda m: _cited(m, sources), s)
        out.append(s)
    return "".join(out)


def _cited(match: re.Match, sources: dict[int, str]) -> str:
    url = sources.get(int(match[1]))
    return f'<a href="{html.escape(url)}">[{match[1]}]</a>' if url else match[0]


def _parts(text: str) -> list[str]:
    # a text in parts that fit a message each, once Telegram's HTML: split at paragraphs, and a
    # paragraph too long at lines
    parts = [""]
    for paragraph in text.split("\n\n"):
        for piece in _pieces(paragraph, LONGEST // 2):
            if parts[-1] and len(parts[-1]) + len(piece) + 2 > LONGEST // 2:
                parts.append("")
            parts[-1] = f"{parts[-1]}\n\n{piece}" if parts[-1] else piece
    return [part for part in parts if part]


def _pieces(paragraph: str, n: int) -> list[str]:
    # a paragraph in pieces of n characters at most, cut at its lines where it can be
    pieces = []
    while len(paragraph) > n:
        cut = paragraph.rfind("\n", 0, n)
        cut = cut if cut > 0 else n
        pieces.append(paragraph[:cut])
        paragraph = paragraph[cut:].lstrip("\n")
    return [*pieces, paragraph]


def _name(person: dict[str, Any]) -> str:
    name = " ".join(filter(None, (person.get("first_name"), person.get("last_name"))))
    return f"{name} (@{person['username']})" if person.get("username") else name
