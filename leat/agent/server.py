"""The agent's HTTP server: the app at /, its API at /api, and every change as an event, streamed to
each app watching.

It answers this machine and its own network alone. A request must name the box by an address or
a local name, which a site's page turned on the box by DNS rebinding cannot; and a write must come
from the app's own page, or from no browser, not from another site's page.

On a box, the server is reached by its name, as k7x2qm.leatbox.net, through the proxy in front of it
that serves the name over HTTPS, with a certificate of its own: it then answers that name alone,
and loopback, as the box's own screen asks; writes come from https://<name>; and a device's cookie
is sent over HTTPS alone, to exactly that name. Every page of the app runs its own scripts, styles
and images alone, framed by none, and a link followed tells no site the box's name.

Every request but for the app's own files, the box's setup and a device's request to join comes
from a device paired to one of the box's people, by the secret its cookie holds. A person sees and
changes their own conversations and files, and their own projects and those shared, with their
files, and lets in their own new devices; an admin looks after the box: its people and devices,
its settings and the engine's model, never another's own conversations or files.

What people do is noted here, as each request asks it, for the admins to look back on: who did what,
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
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from leat.agent.accounts import Accounts
from leat.agent.agent import Agent, Busy, NotFound, Refused
from leat.agent.client import EngineError
from leat.agent.workspace import Workspace
from leat.box import system
from leat.box.watch import Watch
from leat.chat import EFFORTS

APP = Path(__file__).parent / "app"
# the app's files, each served at its path in app/, and their types; the app's pages, /c/<id> one
# conversation's, /projects/<id> one project's, /projects, /files and /settings, are index.html
_FILES = (
    "index.html",
    "mode.js",
    "style.css",
    "app.mjs",
    "markdown.mjs",
    "logo.svg",
)
_PAGES = ("/", "/projects", "/files", "/settings", "/activity")
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
_KEEP_ALIVE = 15  # seconds between comments on a quiet event stream, which find its client gone
UPLOAD = 100 << 20  # bytes of a file uploaded at most
BODY = 10 << 20  # bytes of a request's JSON at most, which may come before its device is known
PROJECT_NAME = 80  # characters of a project's name at most
OFFICE = 60  # characters of an office's name at most, as the box's screen shows it
INSTRUCTIONS = 20000  # characters of a project's instructions at most
COOKIE = "leat"  # the cookie of a device's secret
# of a box served by its name over HTTPS: a cookie browsers send over HTTPS alone, to that name
# alone, as its prefix has them keep it
SECURE_COOKIE = "__Host-leat"
# the app's pages: its own scripts, styles and images alone, and in no other's frame
POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
# of a box served by its name: a year in which a browser reaches it by HTTPS alone
HSTS = "max-age=31536000"
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
# a project's files' earlier versions, or the person's own, to list and to restore
_HISTORY = re.compile(r"/api/(?:projects/([0-9a-f]{12})/)?history")
_PROJECT_PAGE = re.compile(r"/projects/[0-9a-f]{12}")
_REQUEST = re.compile(r"/api/pairings/([0-9a-f]{16})(/allow)?")
_DEVICE = re.compile(r"/api/devices/([0-9]+)")
_WIFI = re.compile(r"/api/wifi/(.+)")
_PERSON = re.compile(r"/api/people/([0-9]+)")
_FILE = re.compile(r"/(?:api/)?(?:projects/([0-9a-f]{12})/)?files/(.+)")


class Server(ThreadingHTTPServer):
    """Serves `agent` at http://host:port, until shut down, to the devices of its people; of a
    `name`, as k7x2qm.leatbox.net, behind the proxy that serves it at https://name."""

    daemon_threads = True  # event streams end with the server

    def __init__(
        self, agent: Agent, host: str = "127.0.0.1", port: int = 8000, name: str | None = None
    ):
        self.name = name.lower() if name else None
        # a box's, set up by the code its screen shows
        self.agent, self.accounts = agent, Accounts(agent, locked=self.name is not None)
        # what of a box needs attention, which its screen and its admins are told
        self.watch = None
        if self.name and agent.workspace:
            self.watch = Watch(self.name, agent.workspace.root,
                               lambda: agent.models_event().get("error"))  # fmt: skip
            self.watch.start()
        self.cookie = SECURE_COOKIE if self.name else COOKIE
        super().__init__((host, port), _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # a client that hung up before its answer was written, as one that gave up waiting, is
        # no error to print
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


class _Unpaired(Exception):
    """The request comes from no device paired to the box."""


class _Refused(Exception):
    """The request asks what only the box's admins may."""


class _Handler(BaseHTTPRequestHandler):
    server: Server

    def version_string(self) -> str:
        return "leat"  # of every answer, which tells no versions

    def log_message(self, format: str, *args: Any) -> None:
        pass  # no access log: which device asked for what, and when, is kept nowhere

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
            if path == "/api/box":  # of the box's own screen, on this machine, never its front door
                if not _loopback(urllib.parse.urlsplit(f"//{self.headers.get('Host', '')}").hostname
                                 or ""):  # fmt: skip
                    return self._error(404, f"there is no GET {path}")
                # and its setup code, while it is to be set up, and a recovery code, while one holds
                accounts, shown = self.server.accounts, agent.box()
                if accounts.locked and accounts.empty():
                    shown["setup"] = accounts.code
                recovery, left = accounts.recovering()
                if recovery:
                    shown |= {"recovery": recovery, "recovery_left": left}
                return self._json(200, shown | {"warnings": self._warnings()})
            me = self._device()
            if path == "/api/me":
                self._json(
                    200, {k: me[k] for k in ("person", "name", "admin")} | {"device": me["id"]}
                )
            elif path == "/api/events":
                self._events(me)
            elif path == "/api/overview":  # the box, as its admins look after it
                _admin(me)
                self._json(200, agent.overview())
            elif path == "/api/wifi":  # the Wi-Fi networks in reach, the admins' to see
                _admin(me)
                self._json(200, _system({"wifi": "scan"}))
            elif path == "/api/update":  # the box's version, and an update ready, if one is
                _admin(me)
                self._json(200, _system({"update": "status"}))
            elif path == "/api/health":  # what of the box needs attention, the admins' to see
                _admin(me)
                self._json(200, {"warnings": self._warnings()})
            elif path == "/api/activity":  # what was done, the latest first, before ?before
                _admin(me)
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                before = query.get("before", [""])[0]
                if before and not before.isdigit():
                    raise ValueError("before must be the id of a thing done")
                self._json(200, {"activity": agent.store.activity(int(before) if before else None)})
            elif not path.startswith("/api/") and (match := _FILE.fullmatch(path)):
                space = agent.space(match[1], me["person"])
                self._download(space, urllib.parse.unquote(match[2]))
            elif match := _HISTORY.fullmatch(path):  # the files' earlier versions
                self._json(200, {"versions": agent.space(match[1], me["person"]).versions()})
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
            if path == "/api/setup":  # the box's first person, its first admin, and its settings
                settings = _settings(body, ("offline", "office", "timezone"))
                zone = settings.pop("timezone", None)
                code = body.get("code") if isinstance(body.get("code"), str) else None
                secret = accounts.setup(_text(body, "name"), _device(self.headers), code)
                agent.change(**settings)
                with contextlib.suppress(ValueError):  # their device's, which a box takes as its
                    # own, if it can; one that is no box keeps its own
                    if zone:
                        agent.change(timezone=zone)
                agent.note(
                    agent.store.people()[0]["id"], "set Leat up", detail=settings.get("office")
                )
                return self._json(200, {}, secret)
            if path == "/api/box/recovery":  # of the box's power button, on this machine alone
                if not _loopback(urllib.parse.urlsplit(f"//{self.headers.get('Host', '')}").hostname
                                 or ""):  # fmt: skip
                    return self._error(404, f"there is no POST {path}")
                if accounts.empty():
                    return self._error(409, "this Leat is not set up: there is no one to recover")
                return self._json(200, {"code": accounts.begin_recovery()})
            if path == "/api/recover":  # a new admin, of the recovery code the box's screen shows
                given = body.get("code") if isinstance(body.get("code"), str) else ""
                person, secret = accounts.recover(
                    str(given), _text(body, "name"), _device(self.headers)
                )
                agent.note(person["id"], "got back in, as an admin, by the box's recovery code")
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
                _admin(me)
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
                to = body.get("person")  # whoever allows it, an admin or not, as Accounts says
                if to is not None and (not isinstance(to, int) or isinstance(to, bool)):
                    raise ValueError("person must be the id of one of this Leat's people")
                name = body.get("name") if isinstance(body.get("name"), str) else None
                allowed = accounts.allow(match[1], me, to, name)
                agent.note(person, "let in", detail=f"{allowed['name']}'s {allowed['device']}")
                self._json(200, allowed)
            elif path == "/api/wifi":  # a Wi-Fi network joined, an admin's to choose
                _admin(me)
                name, passphrase = _text(body, "name"), body.get("passphrase")
                _system({"wifi": "join", "name": name, "passphrase": passphrase})
                agent.note(person, "joined the Wi-Fi network", detail=name)
                self._json(200, {})
            elif path == "/api/update":  # an update looked for now, an admin's to ask
                _admin(me)
                self._json(200, _system({"update": "check"}))
            elif path == "/api/power":  # the box restarted, or switched off, an admin's to do
                _admin(me)
                if (power := body.get("power")) not in ("restart", "off"):
                    raise ValueError('power must be "restart" or "off"')
                _system({"power": power})  # noted once the helper took it, the box going after
                done = "restarted the box" if power == "restart" else "switched the box off"
                agent.note(person, done)
                self._json(200, {})
            elif path == "/api/reset":  # the box's state erased, an admin's to do, once they typed
                _admin(me)  # the office's name, or "reset" of a box of none, as a box is no undo
                word = agent.settings()["office"] or "reset"
                if str(body.get("confirm", "")).strip().casefold() != word.casefold():
                    raise ValueError(f"type {word} to erase everything on this Leat")
                _system({"reset": True})
                self._json(200, {})
            elif path == "/api/models/load":
                _admin(me)
                agent.load(model := _text(body, "model"))
                agent.note(person, "loaded the model", detail=model)
                self._json(200, {})
            elif match := _HISTORY.fullmatch(path):  # a file restored as it was
                name = _text(body, "name")
                agent.space(match[1], person).restore(name, _text(body, "version"))
                agent.files_changed(match[1], person)
                agent.note(person, "restored", match[1], name)
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
            if wifi := _WIFI.fullmatch(path):  # a Wi-Fi network forgotten, an admin's
                _admin(me)
                name = urllib.parse.unquote(wifi[1])
                _system({"wifi": "forget", "name": name})
                agent.note(person, "forgot the Wi-Fi network", detail=name)
                return self._json(200, {})
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
            elif match := _DEVICE.fullmatch(path):  # a person's own, or any, of an admin
                device = next((d for d in store.devices() if d["id"] == int(match[1])), {})
                if device.get("person") != person:
                    _admin(me)
                accounts.unpair(int(match[1]))
                agent.note(person, "unpaired", detail=device.get("name"))
            elif (match := _REQUEST.fullmatch(path)) and not match[2]:
                _admin(me)
                accounts.refuse(match[1])
            elif path == "/api/recovery":  # the recovery code the box's screen shows, voided
                _admin(me)
                if accounts.cancel_recovery():
                    agent.note(person, "cancelled the box's recovery code")
            elif match := _PERSON.fullmatch(path):
                _admin(me)
                removed = (store.person(int(match[1])) or {}).get("name")
                accounts.remove(int(match[1]), person)
                agent.note(person, "removed", detail=removed)
            else:
                return self._error(404, f"there is no DELETE {path}")
            self._json(200, {})

    def do_PATCH(self) -> None:
        # a project changed: its name, instructions, or whether it is shared; a workflow's
        # schedule, set or cleared; a person renamed, of themselves or an admin, or made an admin
        # or not, of an admin; or the box's settings, an admin's
        if not self._trusted(write=True):
            return self._error(403, "requests from other sites' pages are refused")
        path, agent = urllib.parse.urlsplit(self.path).path, self.server.agent
        with self._answering():
            me = self._device()
            flow, match = _WORKFLOW.fullmatch(path), _PROJECT.fullmatch(path)
            changed = _PERSON.fullmatch(path)
            if not flow and not match and not changed and path != "/api/settings":
                return self._error(404, f"there is no PATCH {path}")
            if (size := self._length()) > BODY:
                return self._error(413, f"a request may be {BODY >> 20} MB at most")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            if changed:
                return self._json(200, self._person(int(changed[1]), body, me))
            if path == "/api/settings":
                _admin(me)
                settings = _settings(body, ("offline", "office", "address", "timezone"), only=True)
                agent.change(**settings)
                for key, value in settings.items():  # each change noted
                    said = {
                        "offline": f"turned the internet {'off' if value else 'on'}",
                        "office": "named the office" if value else "took the office's name away",
                        "address": "showed the address on the box's screen" if value
                        else "hid the address from the box's screen",
                        "timezone": "set the time zone",
                    }  # fmt: skip
                    shown = value if key in ("office", "timezone") else None
                    agent.note(me["person"], said[key], detail=shown)
                return self._json(200, {})
            if flow:
                if (when := body.get("schedule")) is not None and not isinstance(when, str):
                    raise ValueError("schedule must be as 'daily 08:00', or null")
                workflow = agent.schedule_workflow(int(flow[1]), when, me["person"])
                done = f"scheduled the workflow, {workflow['schedule']}," if when else (
                    "stopped the schedule of the workflow")  # fmt: skip
                agent.note(me["person"], done, workflow["project"], workflow["name"])
                return self._json(200, workflow)
            assert match is not None
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
            agent.arrived(match[1], person, [unpacked[0] if unpacked else name])
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
            self._error(403, "only this Leat's admins may do that")
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
        name = self.server.cookie
        secret = cookies[name].value if name in cookies else None
        if (device := self.server.accounts.device(secret)) is None:
            raise _Unpaired
        return device

    def _still(self, person: int | None) -> dict[str, Any] | None:
        # the request's device, if it is still paired, as the person's: unpaired, or its person
        # removed, an open stream of events ends
        try:
            device = self._device()
        except _Unpaired:
            return None
        return device if device["person"] == person else None

    def _warnings(self) -> list[dict[str, str]]:
        # what of the box needs attention, as last seen, of a box's; none of a Leat that is no box
        return self.server.watch.warnings if self.server.watch else []

    def _person(self, id: int, body: dict[str, Any], me: dict[str, Any]) -> dict[str, Any]:
        # a person changed: renamed, of themselves or an admin; made an admin or not, of an admin;
        # each noted. Returns the person
        agent, accounts = self.server.agent, self.server.accounts
        if (was := agent.store.person(id)) is None:
            raise LookupError(f"there is no person {id}")
        if unknown := set(body) - {"name", "admin"}:
            raise ValueError(f"a person has no {', '.join(sorted(unknown))} to change")
        if "admin" in body and not isinstance(body["admin"], bool):
            raise ValueError("admin must be true or false")
        if id != me["person"] or "admin" in body:
            _admin(me)
        if "name" in body:
            accounts.rename(id, _text(body, "name"))
            agent.note(me["person"], "renamed", detail=f"{was['name']} as {_text(body, 'name')}")
        if "admin" in body and body["admin"] != bool(was["admin"]):
            accounts.set_admin(id, body["admin"])
            said = "made an admin" if body["admin"] else "made no longer an admin"
            agent.note(me["person"], said, detail=(agent.store.person(id) or was)["name"])
        return agent.store.person(id) or {}

    def _joined(self, id: str) -> None:
        # a request to join, asked after: its device's secret once it was let in
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
        self.end_headers()
        self.wfile.write(data)

    def _events(self, me: dict[str, Any]) -> None:
        # what there is of a device's person's, then every change for them, until the client goes:
        # the admins', to admins alone, as the device's person is one still; the accounts, as the
        # device sees them, each time they change
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
            self._event(self.server.accounts.state(me))
            self._event(agent.models_event())
            self._event(agent.settings_event())
            checked = time.monotonic()
            while True:
                try:
                    event = events.get(timeout=_KEEP_ALIVE)
                except queue.Empty:
                    event = None
                if time.monotonic() - checked > 1 or event and event["type"] == "accounts":
                    if (now := self._still(person)) is None:  # its person's still, as they are
                        return
                    me, checked = now, time.monotonic()
                if event is None:
                    self.wfile.write(b": keep-alive\n\n")
                    continue
                to = event.get("to", person)
                if event["type"] == "accounts":
                    self._event(self.server.accounts.state(me))
                elif to == person or to == "admins" and me["admin"]:
                    self._event(event)

    def _trusted(self, write: bool = False) -> bool:
        # a request that names this machine as its own network does: by the box's name, of one
        # served by its name, which the proxy in front alone reaches, or else by an address or a
        # local name; and by loopback always. A write comes from no browser, which sends no
        # Origin, or from a page of this server's, https://<name> of the box's name
        host, name = self.headers.get("Host", ""), self.server.name
        hostname = urllib.parse.urlsplit(f"//{host}").hostname or ""
        if not (hostname == name or _loopback(hostname) or name is None and _local(hostname)):
            return False
        if not write or (origin := self.headers.get("Origin")) is None:
            return True
        page = urllib.parse.urlsplit(origin)
        return page.netloc == host and (hostname != name or page.scheme == "https")

    def end_headers(self) -> None:
        # of every answer: its type as it says, not sniffed; no address of the box's told a site a
        # link leads to; and of a box served by its name, HTTPS alone from now on
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.server.name:
            self.send_header("Strict-Transport-Security", HSTS)
        super().end_headers()

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
            secure = "; Secure" if self.server.name else ""
            cookie = f"{self.server.cookie}={secret}; Path=/; Max-Age={_YEARS}; HttpOnly{secure}"
            self.send_header("Set-Cookie", f"{cookie}; SameSite=Strict")
        self.end_headers()
        self.wfile.write(data)

    def _send(self, status: int, kind: str, data: bytes) -> None:
        # a file of the app's
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Security-Policy", POLICY)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        body: dict[str, Any] = {"error": {"message": message}}
        if status == 401:  # and whether the box is yet to be set up, by its first person, and
            # whether that takes the code its screen shows
            body["empty"] = self.server.accounts.empty()
            if body["empty"]:
                body["locked"] = self.server.accounts.locked
            else:  # and whether a recovery code holds, the box's power button pressed
                body["recovering"] = self.server.accounts.recovery() is not None
        self._json(status, body)


def _admin(device: dict[str, Any]) -> None:
    # raises _Refused unless the device's person is an admin
    if not device["admin"]:
        raise _Refused


def _local(host: str) -> bool:
    # an address, or a name of this machine or of its network's: localhost, the machine's own, or
    # an mDNS name, as leat.local. A site's name is none of these.
    with contextlib.suppress(ValueError):
        ipaddress.ip_address(host)
        return True
    return host in ("localhost", socket.gethostname().lower()) or host.endswith(".local")


def _loopback(host: str) -> bool:
    # an address or a name of this machine's alone
    with contextlib.suppress(ValueError):
        return ipaddress.ip_address(host).is_loopback
    return host == "localhost"


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


def _system(request: dict[str, Any]) -> dict[str, Any]:
    # the box's system helper's answer. Raises ValueError of its refusal, or of a Leat that is no
    # box, which has none
    try:
        return system.ask(request)
    except OSError as e:
        raise ValueError("this Leat runs on no box of its own, so it cannot do that") from e


def _settings(body: dict[str, Any], keys: tuple[str, ...], only: bool = False) -> dict[str, Any]:
    # the box's settings a body gives, of `keys`, each as it must be, a body of others refused if
    # `only`: whether it is kept from the internet, the office's name, and whether the box's screen
    # shows its address. Raises ValueError
    if only and (others := set(body) - set(keys)):
        raise ValueError(f"there are no settings {', '.join(sorted(others))}")
    settings = {k: body[k] for k in keys if k in body}
    for key in ("offline", "address"):
        if key in settings and not isinstance(settings[key], bool):
            raise ValueError(f"{key} must be true or false")
    if "timezone" in settings and not isinstance(settings["timezone"], str):
        raise ValueError("timezone must be one of the IANA database's, as Europe/Berlin")
    if "office" in settings:
        if not isinstance(settings["office"], str | None):
            raise ValueError("office must be the office's name, or null")
        settings["office"] = " ".join((settings["office"] or "").split()) or None
        if settings["office"] and len(settings["office"]) > OFFICE:
            raise ValueError(f"an office's name is {OFFICE} characters at most")
    return settings


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
