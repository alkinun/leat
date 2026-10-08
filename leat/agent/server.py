"""The agent's HTTP server: the app at /, its API at /api, and every change as an event, streamed to
each app watching.

It answers this machine and its home network alone. A request must name the box by an address or
a local name, which a site's page turned on the box by DNS rebinding cannot; and a write must come
from the app's own page, or from no browser, not from another site's page.

Every request but for the app's own files, the household's setup and a device's request to join
comes from a device of the household's, by the secret its cookie holds. A person sees and changes
their own conversations, memories and tasks, and the household's memories and files; the owner
alone the household's people and devices, Telegram, and the engine's model.
"""

import contextlib
import datetime
import http.cookies
import ipaddress
import json
import mimetypes
import queue
import re
import socket
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from leat.agent.agent import Agent, Busy, NotFound
from leat.agent.channels.telegram import Telegram, TelegramError
from leat.agent.client import EngineError
from leat.agent.household import Household
from leat.agent.tools import lists
from leat.agent.tools import tasks as scheduling

APP = Path(__file__).parent / "app"
# the app's files, each served at its path in app/, and their types; the app's pages, /c/<id> one
# conversation's, /memory, /files, /tasks, /lists, /characters and /settings, are index.html
_FILES = (
    "index.html",
    "style.css",
    "app.mjs",
    "markdown.mjs",
    "themes.mjs",
    "logo.svg",
    "vendor/temml/temml.mjs",
    "vendor/temml/Temml-Latin-Modern.css",
    "vendor/temml/Temml.woff2",
    "vendor/temml/latinmodernmath.woff2",
)
_PAGES = ("/", "/memory", "/files", "/tasks", "/lists", "/characters", "/settings")
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
}
_KEEP_ALIVE = 15  # seconds between comments on a quiet event stream, which find its client gone
UPLOAD = 100 << 20  # bytes of a file uploaded at most
BODY = 10 << 20  # bytes of a request's JSON at most, which may come before its device is known
COOKIE = "leat"  # the cookie of a device's secret
_YEARS = 10 * 365 * 86400  # seconds a device keeps its cookie: till it is unpaired
# the types of the workspace's files a browser shows in the page; it downloads the others, as a page
# the model wrote might act as the app's own
_SHOWN = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf", "text/plain"}
# /api/conversations/<id>, and what to do there; /api/memories/<id>; a task's, a Telegram person's,
# a request to join's and what to do with it, a device's and a person's; a file's name, of
# /files/<name> to download it and of /api/files/<name> to upload or delete it
_CONVERSATION = re.compile(r"/api/conversations/([0-9a-f]{12})(/messages|/stop)?")
_MEMORY = re.compile(r"/api/memories/([0-9]+)")
_TASK = re.compile(r"/api/tasks/([0-9]+)(/run)?")
_TELEGRAM = re.compile(r"/api/telegram/people/(-?[0-9]+)")
_REQUEST = re.compile(r"/api/pairings/([0-9a-f]{16})(/allow)?")
_DEVICE = re.compile(r"/api/devices/([0-9]+)")
_PERSON = re.compile(r"/api/people/([0-9]+)")
_CHARACTER = re.compile(r"/api/characters/([0-9]+)")
_LIST = re.compile(r"/api/lists/([0-9]+)(/items)?")
_ITEM = re.compile(r"/api/lists/items/([0-9]+)")
_FILE = re.compile(r"/(?:api/)?files/(.+)")


class Server(ThreadingHTTPServer):
    """Serves `agent` at http://host:port, until shut down, to its household's devices, and its
    `telegram` bot's settings."""

    daemon_threads = True  # event streams end with the server

    def __init__(
        self, agent: Agent, host: str = "127.0.0.1", port: int = 8000,
        telegram: Telegram | None = None,
    ):  # fmt: skip
        self.agent, self.telegram, self.household = agent, telegram, Household(agent)
        super().__init__((host, port), _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # a client that hung up before its answer was written, as one that gave up waiting, is
        # no error to print
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


class _Unpaired(Exception):
    """The request comes from no device of the household's."""


class _Refused(Exception):
    """The request asks what only the household's owner may."""


class _Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        if not self._trusted():
            return self._error(403, "this server answers its own network's requests alone")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        if path in _PAGES or path.startswith("/c/"):
            path = "/index.html"
        if path[1:] in _FILES:
            file = APP / path[1:]
            return self._send(200, _TYPES[file.suffix], file.read_bytes())
        with self._answering():
            me = self._device()
            if path == "/api/me":
                self._json(
                    200, {k: me[k] for k in ("person", "name", "owner")} | {"device": me["id"]}
                )
            elif path == "/api/events":
                self._events(me)
            elif path.startswith("/files/") and (match := _FILE.fullmatch(path)):
                self._download(urllib.parse.unquote(match[1]))
            elif (match := _CONVERSATION.fullmatch(path)) and not match[2]:
                if (conversation := agent.conversation(match[1], me["person"])) is None:
                    raise NotFound(f"there is no conversation {match[1]}")
                self._json(200, conversation)
            else:
                self._error(404, f"there is no GET {path}")

    def do_POST(self) -> None:
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        household, telegram = self.server.household, self.server.telegram
        if (match := _REQUEST.fullmatch(path)) and not match[2]:
            # a POST, of this server's page alone: a link another gave could log a browser in
            return self._joined(match[1])
        with self._answering():
            if (size := self._length()) > BODY:
                return self._error(413, f"a request may be {BODY >> 20} MB at most")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            if path == "/api/setup":  # the household's first person, its owner
                secret = household.setup(_text(body, "name"), _device(self.headers))
                return self._json(200, {}, secret)
            if path == "/api/pairings":  # a device's request to join
                return self._json(200, household.ask(_text(body, "name"), _device(self.headers)))
            me = self._device()
            person, match = me["person"], _CONVERSATION.fullmatch(path)
            if path == "/api/conversations":  # with one of the household's characters, if given
                played = int(body["character"]) if body.get("character") else None
                id = agent.send(None, *_message(body), person=person, character=played)
                self._json(200, {"id": id})
            elif match and match[2] == "/messages":
                self._json(200, {"id": agent.send(match[1], *_message(body), person=person)})
            elif match and match[2] == "/stop":
                agent.stop(match[1], person)
                self._json(200, {})
            elif path == "/api/memories":
                category = body.get("category", "about")
                category = category if isinstance(category, str) else ""
                self._json(200, agent.remember(_text(body, "text"), category, person=person))
            elif path == "/api/memories/restore":
                self._json(200, agent.restore(int(body.get("id", 0)), person))
            elif path == "/api/tasks":  # a task the user schedules themselves, as a suggestion
                at = scheduling.when(_text(body, "at"), datetime.datetime.now())
                repeat = body.get("repeat", "once")
                condition = body.get("only_if") if isinstance(body.get("only_if"), str) else None
                self._json(200, agent.schedule(_text(body, "prompt"), at, str(repeat), None,
                                               person, condition))  # fmt: skip
            elif (match := _TASK.fullmatch(path)) and match[2]:  # a task run now, to try it
                self._json(200, {"id": agent.run_now(int(match[1]), person)})
            elif path == "/api/characters":
                self._json(200, agent.add_character(_text(body, "name"), _text(body, "about")))
            elif path == "/api/lists":
                lists.make(agent, _text(body, "name"))
                lists.changed(agent)
                self._json(200, {})
            elif (match := _LIST.fullmatch(path)) and match[2]:
                found = next((li for li in agent.store.lists() if li["id"] == int(match[1])), None)
                if found is None:
                    raise NotFound(f"there is no list {match[1]}")
                self._json(200, lists.add(agent, found["name"], [_text(body, "text")]).info)
            elif match := _PERSON.fullmatch(path):  # whether they are a child
                _owner(me)
                if not isinstance(child := body.get("child"), bool):
                    raise ValueError("child must be a boolean")
                household.set_child(int(match[1]), child)
                self._json(200, {})
            elif (match := _REQUEST.fullmatch(path)) and match[2]:
                _owner(me)
                to = body.get("person")
                name = body.get("name") if isinstance(body.get("name"), str) else None
                self._json(200, household.allow(match[1], int(to) if to else None, name))
            elif path == "/api/telegram" and telegram is not None:
                _owner(me)
                self._json(200, telegram.connect(_text(body, "token")))
            elif path == "/api/telegram/people" and telegram is not None:
                _owner(me)
                whose = int(body.get("person") or person)
                if agent.store.person(whose) is None:
                    raise NotFound(f"there is no person {whose}")
                telegram.allow(int(body.get("id", 0)), whose)
                self._json(200, {})
            elif path == "/api/models/load":
                _owner(me)
                agent.load(_text(body, "model"))
                self._json(200, {})
            else:
                self._error(404, f"there is no POST {path}")

    def do_DELETE(self) -> None:
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        household, telegram = self.server.household, self.server.telegram
        with self._answering():
            me = self._device()
            person = me["person"]
            if (match := _CONVERSATION.fullmatch(path)) and not match[2]:
                agent.delete(match[1], person)
            elif match := _MEMORY.fullmatch(path):
                agent.forget(int(match[1]), person=person)
            elif (match := _TASK.fullmatch(path)) and not match[2]:
                agent.unschedule(int(match[1]), person)
            elif match := _CHARACTER.fullmatch(path):
                agent.remove_character(int(match[1]))
            elif match := _ITEM.fullmatch(path):
                if agent.store.remove_item(int(match[1])) is None:
                    raise NotFound(f"there is no item {match[1]}")
                lists.changed(agent)
            elif (match := _LIST.fullmatch(path)) and not match[2]:
                if not agent.store.remove_list(int(match[1])):
                    raise NotFound(f"there is no list {match[1]}")
                lists.changed(agent)
            elif path.startswith("/api/files/") and agent.workspace is not None:
                agent.workspace.delete(urllib.parse.unquote(path.removeprefix("/api/files/")))
                agent.files_changed()
            elif match := _DEVICE.fullmatch(path):  # the owner's, or a device unpairing itself
                if int(match[1]) != me["id"]:
                    _owner(me)
                household.unpair(int(match[1]))
            elif (match := _REQUEST.fullmatch(path)) and not match[2]:
                _owner(me)
                household.refuse(match[1])
            elif match := _PERSON.fullmatch(path):
                _owner(me)
                household.remove(int(match[1]))
            elif path == "/api/telegram" and telegram is not None:
                _owner(me)
                telegram.disconnect()
            elif (match := _TELEGRAM.fullmatch(path)) and telegram is not None:
                _owner(me)
                telegram.refuse(int(match[1]))
            else:
                return self._error(404, f"there is no DELETE {path}")
            self._json(200, {})

    def do_PUT(self) -> None:
        # a file uploaded to the workspace, by a name of its own: its answer says the one it got
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        with self._answering():
            self._device()
            if not path.startswith("/api/files/") or agent.workspace is None:
                return self._error(404, f"there is no PUT {path}")
            if (size := self._length()) > UPLOAD:
                return self._error(413, f"a file may be {UPLOAD >> 20} MB at most")
            name = agent.workspace.free(urllib.parse.unquote(path.removeprefix("/api/files/")))
            agent.workspace.path(name).write_bytes(self.rfile.read(size))
            agent.files_changed()
            self._json(200, {"name": name})

    @contextlib.contextmanager
    def _answering(self):
        # answers what goes wrong in a request with its status, and what it says
        try:
            yield
        except _Unpaired:
            self._error(401, "this device is not one of the household's: ask to join")
        except _Refused:
            self._error(403, "only the household's owner may do that")
        except (ValueError, RecursionError) as e:  # JSON nested too deep to parse, too
            self._error(400, str(e))
        except (NotFound, LookupError, FileNotFoundError) as e:
            self._error(404, str(e))
        except Busy as e:
            self._error(409, str(e))
        except EngineError as e:
            self._error(502, str(e))
        except TelegramError as e:
            self._error(400, f"Telegram refused it: {e}")

    def _device(self) -> dict[str, Any]:
        # the household's device the request comes from, by its cookie's secret. Raises _Unpaired
        # if it comes from none.
        cookies = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
        secret = cookies[COOKIE].value if COOKIE in cookies else None
        if (device := self.server.household.device(secret)) is None:
            raise _Unpaired
        return device

    def _still(self, person: int | None) -> bool:
        # whether the request's device is still paired, as the person's: unpaired, or its person
        # removed, an open stream of events ends
        try:
            return self._device()["person"] == person
        except _Unpaired:
            return False

    def _joined(self, id: str) -> None:
        # a request to join, asked after: its device's secret once the owner let it in
        try:
            secret = self.server.household.answer(id)
        except LookupError as e:
            return self._error(404, str(e))
        if secret is None:
            return self._json(202, {})
        self._json(200, {}, secret)

    def _download(self, name: str) -> None:
        # a file of the workspace's, shown in the page if it is of a kind that cannot act there,
        # as the content security policy's sandbox has every one
        workspace = self.server.agent.workspace
        try:
            file = workspace.path(name) if workspace else None
        except ValueError:
            file = None
        if file is None or not file.is_file():
            return self._error(404, f"there is no file {name}")
        kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        shown = "inline" if kind in _SHOWN else "attachment"
        data = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        quoted = urllib.parse.quote(file.name)
        self.send_header("Content-Disposition", f"{shown}; filename*=UTF-8''{quoted}")
        self.send_header("Content-Security-Policy", "sandbox")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _events(self, me: dict[str, Any]) -> None:
        # what there is of a device's person's, then every change for them, until the client goes;
        # what is the household's owner's, to theirs alone
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        agent, person = self.server.agent, me["person"]
        with agent.events.watch() as events, contextlib.suppress(OSError):
            self._event({"type": "conversations", "conversations": agent.conversations(person)})
            self._event(agent.memories_event(person))
            self._event(agent.files_event())
            self._event(agent.tasks_event(person))
            self._event(agent.characters_event())
            self._event(lists.event(agent))
            if me["owner"]:
                self._event(self.server.household.state())
                if self.server.telegram is not None:
                    self._event(self.server.telegram.state())
            self._event(agent.models_event())
            checked = time.monotonic()
            while True:
                try:
                    event = events.get(timeout=_KEEP_ALIVE)
                except queue.Empty:
                    event = None
                if time.monotonic() - checked > 1:  # the device still paired, as the person's
                    if not self._still(person):
                        return
                    checked = time.monotonic()
                if event is None:
                    self.wfile.write(b": keep-alive\n\n")
                    continue
                to = event.get("to", person)
                if to == person or to == "owner" and me["owner"]:
                    self._event(event)

    def _trusted(self, write: bool = False) -> bool:
        # a request that names this machine as its own network does, and a write from no browser,
        # which sends no Origin, or from a page of this server
        host = self.headers.get("Host", "")
        if not _local(urllib.parse.urlsplit(f"//{host}").hostname or ""):
            return False
        origin = self.headers.get("Origin")
        return not write or origin is None or urllib.parse.urlsplit(origin).netloc == host

    def _length(self) -> int:
        # the body's length, as its header says. Raises ValueError if it says none there can be.
        if (n := int(self.headers.get("Content-Length") or 0)) < 0:
            raise ValueError("a body's length cannot be negative")
        return n

    def _event(self, event: dict[str, Any]) -> None:
        # an event as the app reads it, without whose it is
        event = {k: v for k, v in event.items() if k != "to"}
        self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())

    def _json(self, status: int, body: dict[str, Any], secret: str | None = None) -> None:
        # an answer of JSON; with a device's secret, the cookie that keeps it, out of the page's
        # reach, and sent to this server's pages alone
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        if secret is not None:
            cookie = f"{COOKIE}={secret}; Path=/; Max-Age={_YEARS}; HttpOnly; SameSite=Strict"
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def _send(self, status: int, kind: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        body: dict[str, Any] = {"error": {"message": message}}
        if status == 401:  # and whether the household is yet to be set up, by its first person
            body["empty"] = self.server.household.empty()
        self._json(status, body)


def _owner(device: dict[str, Any]) -> None:
    # raises _Refused unless the device is the household's owner's
    if not device["owner"]:
        raise _Refused


def _local(host: str) -> bool:
    # an address, or a name of this machine or of its network's: localhost, the machine's own, or
    # an mDNS name, as leat.local. A site's name is none of these.
    with contextlib.suppress(ValueError):
        ipaddress.ip_address(host)
        return True
    return host in ("localhost", socket.gethostname().lower()) or host.endswith(".local")


def _device(headers: Any) -> str:
    # what a device is, in words, of its browser's User-Agent: "Firefox on Android"
    agent = headers.get("User-Agent", "")
    systems = (("Android", "Android"), ("iPhone", "iPhone"), ("iPad", "iPad"), ("Mac OS", "Mac"),
               ("Windows", "Windows"), ("CrOS", "ChromeOS"), ("Linux", "Linux"))  # fmt: skip
    browsers = (("Firefox", "Firefox"), ("Edg/", "Edge"), ("OPR/", "Opera"), ("Chrome", "Chrome"),
                ("Safari", "Safari"))  # fmt: skip
    system = next((name for key, name in systems if key in agent), "")
    browser = next((name for key, name in browsers if key in agent), "")
    return " on ".join(filter(None, (browser, system))) or "A device"


def _text(body: dict[str, Any], key: str) -> str:
    # a body's text of a key, which must be some; raises ValueError if it is not
    if not isinstance(text := body.get(key), str) or not text.strip():
        raise ValueError(f"{key} must be some text")
    return text


def _message(body: dict[str, Any]) -> tuple[str, bool, list[str]]:
    # a message's content, whether the model is to think before it replies, and the files attached
    if not isinstance(content := body.get("content"), str) or not content.strip():
        raise ValueError("a message needs content: some text")
    if not isinstance(think := body.get("think", False), bool):
        raise ValueError("think must be a boolean")
    attached = body.get("files", [])
    if not isinstance(attached, list) or not all(isinstance(name, str) for name in attached):
        raise ValueError("files must be a list of the workspace's files' names")
    return content, think, attached
