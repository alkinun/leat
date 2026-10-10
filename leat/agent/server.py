"""The agent's HTTP server: the app at /, its API at /api, and every change as an event, streamed to
each app watching.

It answers this machine and its own network alone. A request must name the box by an address or
a local name, which a site's page turned on the box by DNS rebinding cannot; and a write must come
from the app's own page, or from no browser, not from another site's page.

Every request but for the app's own files, the box's setup and a device's request to join comes
from a device paired to one of the box's people, by the secret its cookie holds. A person sees and
changes their own conversations and files, and their own projects and those shared, with their
files; the owner alone the box's people and devices, and the engine's model.

What people do is noted here, as each request asks it, for the owner to look back on: who did what,
where, and of what, never what anyone asked; what Leat does for them, the agent notes.
"""

import contextlib
import http.cookies
import ipaddress
import json
import mimetypes
import queue
import re
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from leat.agent.accounts import Accounts
from leat.agent.agent import Agent, Busy, NotFound, Refused
from leat.agent.client import EngineError
from leat.agent.workspace import Workspace
from leat.chat import EFFORTS

APP = Path(__file__).parent / "app"
# the app's files, each served at its path in app/, and their types; the app's pages, /c/<id> one
# conversation's, /projects/<id> one project's, /projects, /files and /settings, are index.html
_FILES = (
    "index.html",
    "style.css",
    "app.mjs",
    "markdown.mjs",
    "logo.svg",
    "vendor/temml/temml.mjs",
    "vendor/temml/Temml-Latin-Modern.css",
    "vendor/temml/Temml.woff2",
    "vendor/temml/latinmodernmath.woff2",
)
_PAGES = ("/", "/projects", "/files", "/settings", "/activity")
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
PROJECT_NAME = 80  # characters of a project's name at most
INSTRUCTIONS = 20000  # characters of a project's instructions at most
COOKIE = "leat"  # the cookie of a device's secret
_YEARS = 10 * 365 * 86400  # seconds a device keeps its cookie: till it is unpaired
# the types of the workspace's files a browser shows in the page; it downloads the others, as a page
# the model wrote might act as the app's own
_SHOWN = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf", "text/plain"}
# /api/conversations/<id>, and what to do there; a project's, and its page; a request to join's and
# what to do with it, a device's and a person's; a file's project, if one, and name, of
# [/projects/<id>]/files/<name> to download it and of /api[/projects/<id>]/files/<name> to upload
# or delete it
_CONVERSATION = re.compile(r"/api/conversations/([0-9a-f]{12})(/messages|/stop)?")
_PROJECT = re.compile(r"/api/projects/([0-9a-f]{12})")
_WORKFLOW = re.compile(r"/api/workflows/([0-9]+)")
_SYNC = re.compile(r"/api/projects/([0-9a-f]{12})/sync")
_PROJECT_PAGE = re.compile(r"/projects/[0-9a-f]{12}")
_REQUEST = re.compile(r"/api/pairings/([0-9a-f]{16})(/allow)?")
_DEVICE = re.compile(r"/api/devices/([0-9]+)")
_PERSON = re.compile(r"/api/people/([0-9]+)")
_LAW = re.compile(r"/api/library/([^/]+)")  # a law the library keeps, by its abbreviation
_FILE = re.compile(r"/(?:api/)?(?:projects/([0-9a-f]{12})/)?files/(.+)")


class Server(ThreadingHTTPServer):
    """Serves `agent` at http://host:port, until shut down, to the devices of its people."""

    daemon_threads = True  # event streams end with the server

    def __init__(self, agent: Agent, host: str = "127.0.0.1", port: int = 8000):
        self.agent, self.accounts = agent, Accounts(agent)
        super().__init__((host, port), _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # a client that hung up before its answer was written, as one that gave up waiting, is
        # no error to print
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


class _Unpaired(Exception):
    """The request comes from no device paired to the box."""


class _Refused(Exception):
    """The request asks what only the box's owner may."""


class _Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        if not self._trusted():
            return self._error(403, "this server answers its own network's requests alone")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        if path in _PAGES or path.startswith("/c/") or _PROJECT_PAGE.fullmatch(path):
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
            elif path == "/api/overview":  # the box, as its owner looks after it
                _owner(me)
                self._json(200, agent.overview())
            elif path == "/api/activity":  # what was done, the latest first, before ?before
                _owner(me)
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                before = query.get("before", [""])[0]
                if before and not before.isdigit():
                    raise ValueError("before must be the id of a thing done")
                self._json(200, {"activity": agent.store.activity(int(before) if before else None)})
            elif not path.startswith("/api/") and (match := _FILE.fullmatch(path)):
                space = agent.space(match[1], me["person"])
                self._download(space, urllib.parse.unquote(match[2]))
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
        accounts = self.server.accounts
        if (match := _REQUEST.fullmatch(path)) and not match[2]:
            # a POST, of this server's page alone: a link another gave could log a browser in
            return self._joined(match[1])
        with self._answering():
            if (size := self._length()) > BODY:
                return self._error(413, f"a request may be {BODY >> 20} MB at most")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            if path == "/api/setup":  # the box's first person, its owner
                secret = accounts.setup(_text(body, "name"), _device(self.headers))
                agent.note(agent.store.people()[0]["id"], "set Leat up")
                return self._json(200, {}, secret)
            if path == "/api/pairings":  # a device's request to join
                return self._json(200, accounts.ask(_text(body, "name"), _device(self.headers)))
            me = self._device()
            person, match = me["person"], _CONVERSATION.fullmatch(path)
            if path == "/api/conversations":  # in a project, if the body names one
                id = agent.send(None, *_message(body), person=person, project=_held(body))
                agent.note(person, "began a chat", _held(body))
                self._json(200, {"id": id})
            elif path == "/api/projects":
                fields = _project(body, new=True)
                project = agent.add_project(fields.pop("name"), person, **fields)
                agent.note(person, "made the project", project["id"])
                self._json(200, project)
            elif synced := _SYNC.fullmatch(path):  # a project's files synced with a folder, or not
                _owner(me)
                if (source := body.get("source")) is not None and not isinstance(source, str):
                    raise ValueError("source must be a folder's path, or null")
                agent.sync(synced[1], (source or "").strip() or None, person)
                self._json(200, {})
            elif path == "/api/workflows":  # a project's, if the body names one
                workflow = agent.add_workflow(*_workflow(body), person, _held(body))
                agent.note(person, "saved the workflow", workflow["project"], workflow["name"])
                self._json(200, workflow)
            elif match and match[2] == "/messages":
                self._json(200, {"id": agent.send(match[1], *_message(body), person=person)})
            elif match and match[2] == "/stop":
                agent.stop(match[1], person)
                self._json(200, {})
            elif (match := _REQUEST.fullmatch(path)) and match[2]:
                _owner(me)
                to = body.get("person")
                if to is not None and (not isinstance(to, int) or isinstance(to, bool)):
                    raise ValueError("person must be the id of one of this Leat's people")
                name = body.get("name") if isinstance(body.get("name"), str) else None
                allowed = accounts.allow(match[1], to, name)
                agent.note(person, "let in", detail=f"{allowed['name']}'s {allowed['device']}")
                self._json(200, allowed)
            elif path == "/api/models/load":
                _owner(me)
                agent.load(model := _text(body, "model"))
                agent.note(person, "loaded the model", detail=model)
                self._json(200, {})
            elif path == "/api/offline":  # the agent kept from the internet, or let reach it
                _owner(me)
                if not isinstance(offline := body.get("offline"), bool):
                    raise ValueError("offline must be true or false")
                agent.set_offline(offline)
                agent.note(person, f"turned the internet {'off' if offline else 'on'}")
                self._json(200, {})
            elif path == "/api/backups":  # the folder backed up to, or none
                _owner(me)
                if (folder := body.get("folder")) is not None and not isinstance(folder, str):
                    raise ValueError("folder must be a folder's path, or null")
                agent.backups.choose(folder := (folder or "").strip() or None)
                agent.note(person, "chose to back up to" if folder else "stopped backing up",
                           detail=folder)  # fmt: skip
                self._json(200, {})
            elif path == "/api/library":  # a law added, by its abbreviation
                _owner(me)
                agent.library.add(_text(body, "law"), person)
                self._json(200, {})
            elif path == "/api/backups/now":
                _owner(me)
                if not agent.backups.state()["folder"]:
                    raise ValueError("choose a folder to back up to first")
                threading.Thread(target=agent.backups.run, daemon=True).start()
                self._json(200, {})
            else:
                self._error(404, f"there is no POST {path}")

    def do_DELETE(self) -> None:
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        accounts = self.server.accounts
        with self._answering():
            me = self._device()
            person, store = me["person"], agent.store
            if (match := _CONVERSATION.fullmatch(path)) and not match[2]:
                held = (store.conversation(match[1]) or {}).get("project")
                agent.delete(match[1], person)
                agent.note(person, "deleted a chat", held)
            elif match := _PROJECT.fullmatch(path):
                held = (store.project(match[1]) or {}).get("name")
                agent.delete_project(match[1], person)
                store.note(person, "deleted the project", held)
            elif match := _WORKFLOW.fullmatch(path):
                workflow = store.workflow(int(match[1])) or {}
                agent.delete_workflow(int(match[1]), person)
                agent.note(
                    person, "deleted the workflow", workflow.get("project"), workflow.get("name")
                )
            elif path.startswith("/api/") and (match := _FILE.fullmatch(path)):
                agent.space(match[1], person).delete(name := urllib.parse.unquote(match[2]))
                agent.files_changed(match[1], person)
                agent.note(person, "deleted", match[1], name)
            elif match := _DEVICE.fullmatch(path):  # the owner's, or a device unpairing itself
                if int(match[1]) != me["id"]:
                    _owner(me)
                device = next((d for d in store.devices() if d["id"] == int(match[1])), {})
                accounts.unpair(int(match[1]))
                agent.note(person, "unpaired", detail=device.get("name"))
            elif (match := _REQUEST.fullmatch(path)) and not match[2]:
                _owner(me)
                accounts.refuse(match[1])
            elif match := _LAW.fullmatch(path):
                _owner(me)
                agent.library.remove(urllib.parse.unquote(match[1]), person)
            elif match := _PERSON.fullmatch(path):
                _owner(me)
                removed = (store.person(int(match[1])) or {}).get("name")
                accounts.remove(int(match[1]))
                agent.note(person, "removed", detail=removed)
            else:
                return self._error(404, f"there is no DELETE {path}")
            self._json(200, {})

    def do_PATCH(self) -> None:
        # a project changed: its name, instructions, or whether it is shared
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        with self._answering():
            me = self._device()
            if not (match := _PROJECT.fullmatch(path)):
                return self._error(404, f"there is no PATCH {path}")
            if (size := self._length()) > BODY:
                return self._error(413, f"a request may be {BODY >> 20} MB at most")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            changes = _project(body)
            agent.change_project(match[1], me["person"], **changes)
            for key, value in changes.items():  # each change noted
                said = {
                    "name": "renamed the project",
                    "instructions": "changed the instructions of",
                    "shared": "shared the project" if value else "stopped sharing the project",
                }
                agent.note(me["person"], said[key], match[1])  # fmt: skip
            self._json(200, {})

    def do_PUT(self) -> None:
        # a file uploaded to a person's own files, or a project's, by a name of its own, in the
        # folders its path names, a zip unpacked into one of its name if ?unpack: its answer says
        # the name it got, and of a zip how many files it held
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        with self._answering():
            person = self._device()["person"]
            if not path.startswith("/api/") or not (match := _FILE.fullmatch(path)):
                return self._error(404, f"there is no PUT {path}")
            space = agent.space(match[1], person)
            if (size := self._length()) > UPLOAD:
                return self._error(413, f"a file may be {UPLOAD >> 20} MB at most")
            name = space.free(urllib.parse.unquote(match[2]), folders=True)
            space.path(name).parent.mkdir(parents=True, exist_ok=True)
            space.path(name).write_bytes(self.rfile.read(size))
            unpacked = None
            try:  # a zip unpacked into a folder of its name, if asked
                query = urllib.parse.urlsplit(self.path).query
                if "unpack" in urllib.parse.parse_qs(query, keep_blank_values=True):
                    unpacked = space.unpack(name)
            finally:
                agent.files_changed(match[1], person)
            if unpacked is not None:
                agent.note(person, "uploaded", match[1], f"{unpacked[0]}, {unpacked[1]} files")
                return self._json(200, {"name": unpacked[0], "files": unpacked[1]})
            agent.note(person, "uploaded", match[1], name)
            self._json(200, {"name": name})

    @contextlib.contextmanager
    def _answering(self):
        # answers what goes wrong in a request with its status, and what it says
        try:
            yield
        except _Unpaired:
            self._error(401, "this device has not joined this Leat: ask to join")
        except _Refused:
            self._error(403, "only this Leat's owner may do that")
        except Refused as e:
            self._error(403, str(e))
        except (ValueError, RecursionError) as e:  # JSON nested too deep to parse, too
            self._error(400, str(e))
        except (NotFound, LookupError, FileNotFoundError) as e:
            self._error(404, str(e))
        except Busy as e:
            self._error(409, str(e))
        except EngineError as e:
            self._error(502, str(e))

    def _device(self) -> dict[str, Any]:
        # the paired device the request comes from, by its cookie's secret. Raises _Unpaired
        # if it comes from none.
        cookies = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
        secret = cookies[COOKIE].value if COOKIE in cookies else None
        if (device := self.server.accounts.device(secret)) is None:
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
            secret = self.server.accounts.answer(id)
        except LookupError as e:
            return self._error(404, str(e))
        if secret is None:
            return self._json(202, {})
        self._json(200, {}, secret)

    def _download(self, space: Workspace, name: str) -> None:
        # a file of a space's, shown in the page if it is of a kind that cannot act there, as the
        # content security policy's sandbox has every one
        try:
            file = space.path(name)
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
        # what is the owner's, to theirs alone
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        agent, person = self.server.agent, me["person"]
        with agent.events.watch() as events, contextlib.suppress(OSError):
            self._event({"type": "conversations", "conversations": agent.conversations(person)})
            projects = agent.projects(person)
            self._event({"type": "projects", "projects": projects})
            self._event({"type": "workflows", "workflows": agent.workflows(person)})
            for project in [None, *(p["id"] for p in projects)]:
                self._event(agent.files_event(project, person))
            if me["owner"]:
                self._event(self.server.accounts.state())
                self._event(agent.backups.state())
                self._event(agent.library.state())
            self._event(agent.models_event())
            self._event(agent.settings_event())
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
        # the body's length, as its header says. Raises ValueError if it says none there can be,
        # or the body comes in chunks, which this server does not read: an upload would be empty
        if "Transfer-Encoding" in self.headers:
            raise ValueError("a body must come whole, of the length its Content-Length says")
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
        if status == 401:  # and whether the box is yet to be set up, by its first person
            body["empty"] = self.server.accounts.empty()
        self._json(status, body)


def _owner(device: dict[str, Any]) -> None:
    # raises _Refused unless the device is the owner's
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


def _workflow(body: dict[str, Any]) -> tuple[str, str]:
    # a workflow's name and request, of a body; raises ValueError if they are not as they must be
    name, prompt = _text(body, "name").strip(), _text(body, "prompt").strip()
    if len(name) > PROJECT_NAME or len(prompt) > INSTRUCTIONS:
        raise ValueError(f"a workflow's name is {PROJECT_NAME} characters at most, and its "
                         f"request {INSTRUCTIONS}")  # fmt: skip
    return name, prompt


def _held(body: dict[str, Any]) -> str | None:
    # the project a body names, if any; raises ValueError if it names one by other than its id
    if (project := body.get("project")) is not None and not isinstance(project, str):
        raise ValueError("project must be the id of a project")
    return project


def _project(body: dict[str, Any], new: bool = False) -> dict[str, Any]:
    # a project's fields a body gives, its name, instructions and whether it is shared; a `new`
    # one's name must be given. Raises ValueError for one that is not as it must be
    fields: dict[str, Any] = {k: body[k] for k in ("name", "instructions", "shared") if k in body}
    if new or "name" in fields:
        name = _text(body, "name").strip()
        if len(name) > PROJECT_NAME:
            raise ValueError(f"a project's name is {PROJECT_NAME} characters at most")
        fields["name"] = name
    if not isinstance(fields.get("instructions", ""), str):
        raise ValueError("instructions must be text")
    if len(fields.get("instructions", "")) > INSTRUCTIONS:
        raise ValueError(f"a project's instructions are {INSTRUCTIONS} characters at most")
    if not isinstance(fields.get("shared", False), bool):
        raise ValueError("shared must be true or false")
    return fields


def _message(body: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    # a message's content, the effort the model is to reason at, if one, and the files attached
    if not isinstance(content := body.get("content"), str) or not content.strip():
        raise ValueError("a message needs content: some text")
    if (effort := body.get("effort")) is not None and effort not in EFFORTS:
        raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
    attached = body.get("files", [])
    if not isinstance(attached, list) or not all(isinstance(name, str) for name in attached):
        raise ValueError("files must be a list of the workspace's files' names")
    return content, effort, attached
