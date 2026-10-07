"""The agent's HTTP server: the app at /, its API at /api, and every change as an event, streamed to
each app watching.

It answers this machine and its home network alone. A request must name the box by an address or
a local name, which a site's page turned on the box by DNS rebinding cannot; and a write must come
from the app's own page, or from no browser, not from another site's page.
"""

import contextlib
import ipaddress
import json
import mimetypes
import queue
import re
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from leat.agent.agent import Agent, Busy, NotFound
from leat.agent.channels.telegram import Telegram, TelegramError
from leat.agent.client import EngineError

APP = Path(__file__).parent / "app"
# the app's files, each served at its path in app/, and their types; the app's pages, /c/<id> one
# conversation's, /memory, /files and /tasks, are index.html
_FILES = (
    "index.html",
    "style.css",
    "app.mjs",
    "markdown.mjs",
    "vendor/temml/temml.mjs",
    "vendor/temml/Temml-Latin-Modern.css",
    "vendor/temml/Temml.woff2",
    "vendor/temml/latinmodernmath.woff2",
)
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
}
_KEEP_ALIVE = 15  # seconds between comments on a quiet event stream, which find its client gone
UPLOAD = 100 << 20  # bytes of a file uploaded at most
# the types of the workspace's files a browser shows in the page; it downloads the others, as a page
# the model wrote might act as the app's own
_SHOWN = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf", "text/plain"}
# /api/conversations/<id>, and what to do there; /api/memories/<id>; a file's name, of /files/<name>
# to download it and of /api/files/<name> to upload or delete it
_CONVERSATION = re.compile(r"/api/conversations/([0-9a-f]{12})(/messages|/stop)?")
_MEMORY = re.compile(r"/api/memories/([0-9]+)")
_TASK = re.compile(r"/api/tasks/([0-9]+)")
_PERSON = re.compile(r"/api/telegram/people/(-?[0-9]+)")
_FILE = re.compile(r"/(?:api/)?files/(.+)")


class Server(ThreadingHTTPServer):
    """Serves `agent` at http://host:port, until shut down, and its `telegram` bot's settings."""

    daemon_threads = True  # event streams end with the server

    def __init__(
        self, agent: Agent, host: str = "127.0.0.1", port: int = 8000,
        telegram: Telegram | None = None,
    ):  # fmt: skip
        self.agent, self.telegram = agent, telegram
        super().__init__((host, port), _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        if not self._trusted():
            return self._error(403, "this server answers its own network's requests alone")
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/memory", "/files", "/tasks", "/settings") or path.startswith("/c/"):
            path = "/index.html"
        if path[1:] in _FILES:
            file = APP / path[1:]
            return self._send(200, _TYPES[file.suffix], file.read_bytes())
        if path == "/api/events":
            return self._events()
        if path.startswith("/files/") and (match := _FILE.fullmatch(path)):
            return self._download(urllib.parse.unquote(match[1]))
        if (match := _CONVERSATION.fullmatch(path)) and not match[2]:
            if (conversation := self.server.agent.conversation(match[1])) is None:
                return self._error(404, f"there is no conversation {match[1]}")
            return self._json(200, conversation)
        self._error(404, f"there is no GET {path}")

    def do_POST(self) -> None:
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        try:
            body = json.loads(self.rfile.read(self._length()))
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            match = _CONVERSATION.fullmatch(path)
            if path == "/api/conversations":
                self._json(200, {"id": agent.send(None, *_message(body))})
            elif match and match[2] == "/messages":
                self._json(200, {"id": agent.send(match[1], *_message(body))})
            elif match and match[2] == "/stop":
                agent.stop(match[1])
                self._json(200, {})
            elif path == "/api/memories":
                if not isinstance(text := body.get("text"), str):
                    raise ValueError("a memory needs text")
                category = body.get("category", "about")
                self._json(200, agent.remember(text, category if isinstance(category, str) else ""))
            elif path == "/api/telegram" and self.server.telegram is not None:
                if not isinstance(token := body.get("token"), str) or not token.strip():
                    raise ValueError("a bot needs its token, of @BotFather")
                self._json(200, self.server.telegram.connect(token))
            elif path == "/api/telegram/people" and self.server.telegram is not None:
                self.server.telegram.allow(int(body.get("id", 0)))
                self._json(200, {})
            elif path == "/api/models/load":
                if not isinstance(model := body.get("model"), str):
                    raise ValueError("a load needs a model's id")
                agent.load(model)
                self._json(200, {})
            else:
                self._error(404, f"there is no POST {path}")
        except (ValueError, RecursionError) as e:  # JSON nested too deep to parse, too
            self._error(400, str(e))
        except NotFound as e:
            self._error(404, str(e))
        except Busy as e:
            self._error(409, str(e))
        except EngineError as e:
            self._error(502, str(e))
        except TelegramError as e:
            self._error(400, f"Telegram refused it: {e}")

    def do_DELETE(self) -> None:
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        try:
            if (match := _CONVERSATION.fullmatch(path)) and not match[2]:
                agent.delete(match[1])
            elif match := _MEMORY.fullmatch(path):
                agent.forget(int(match[1]))
            elif match := _TASK.fullmatch(path):
                agent.unschedule(int(match[1]))
            elif path == "/api/telegram" and self.server.telegram is not None:
                self.server.telegram.disconnect()
            elif (match := _PERSON.fullmatch(path)) and self.server.telegram is not None:
                self.server.telegram.refuse(int(match[1]))
            elif path.startswith("/api/files/") and agent.workspace is not None:
                agent.workspace.delete(urllib.parse.unquote(path.removeprefix("/api/files/")))
                agent.files_changed()
            else:
                return self._error(404, f"there is no DELETE {path}")
        except (NotFound, FileNotFoundError, ValueError) as e:
            return self._error(404, str(e))
        self._json(200, {})

    def do_PUT(self) -> None:
        # a file uploaded to the workspace, by a name of its own: its answer says the one it got
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        if not path.startswith("/api/files/") or agent.workspace is None:
            return self._error(404, f"there is no PUT {path}")
        try:
            size = self._length()
        except ValueError as e:
            return self._error(400, str(e))
        if size > UPLOAD:
            return self._error(413, f"a file may be {UPLOAD >> 20} MB at most")
        name = agent.workspace.free(urllib.parse.unquote(path.removeprefix("/api/files/")))
        agent.workspace.path(name).write_bytes(self.rfile.read(size))
        agent.files_changed()
        self._json(200, {"name": name})

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

    def _events(self) -> None:
        # what there is, then every change, until the client goes
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        agent = self.server.agent
        with agent.events.watch() as events, contextlib.suppress(OSError):
            self._event({"type": "conversations", "conversations": agent.conversations()})
            self._event(agent.memories_event())
            self._event(agent.files_event())
            self._event(agent.tasks_event())
            if self.server.telegram is not None:
                self._event(self.server.telegram.state())
            self._event(agent.models_event())
            while True:
                try:
                    self._event(events.get(timeout=_KEEP_ALIVE))
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")

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
        self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())

    def _json(self, status: int, body: dict[str, Any]) -> None:
        self._send(status, "application/json", json.dumps(body, ensure_ascii=False).encode())

    def _send(self, status: int, kind: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message}})


def _local(host: str) -> bool:
    # an address, or a name of this machine or of its network's: localhost, the machine's own, or
    # an mDNS name, as leat.local. A site's name is none of these.
    with contextlib.suppress(ValueError):
        ipaddress.ip_address(host)
        return True
    return host in ("localhost", socket.gethostname().lower()) or host.endswith(".local")


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
