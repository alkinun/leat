"""A person's files and projects as a network drive, of WebDAV, which Windows' Explorer, macOS's
Finder and Word open, so that people drop files into a project and open what Leat made, where they
work, as the app's own files are.

The drive is the box's at /dav, by the box's name over HTTPS, and a computer opens it of a password
of its person's own, which the box keeps the hash of, as a device of theirs, of its kind, "drive",
to unpair: not an app's cookie, nor the other way round. It holds their own files and each project
they see, as the app shows them, and none they do not:

    /dav/My files/…                    their own
    /dav/Projects/<a project's name>/… each project's, as its name is a folder's
    a folder made in Projects          a project of theirs

Whatever is changed or deleted there is kept first, as an earlier version, as the app's files are;
the index reads what changes; a project's workflows that run as files arrive run of those that
arrive here; and the activity notes what was uploaded, changed, moved and deleted, but for the
files programs make for a while beside a document, as Word's ~$ and .tmp. A file's lock, as Word
asks one to edit, is granted and kept for no one: two people editing a document at once both
save, the last's kept, the first's an earlier version.
"""

import email.utils
import mimetypes
import os
import re
import shutil
import time
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

from leat.agent.agent import NotFound, Refused
from leat.agent.workspace import Workspace, hidden

if TYPE_CHECKING:
    from leat.agent.agent import Agent

PREFIX = "/dav"
DAV = "DAV:"
D = "{DAV:}"  # of every tag's name, its namespace
UPLOAD = 100 << 20  # bytes of a file put at most, as the app's uploads
CHUNK = 1 << 16  # bytes read and written at a time
# the drive's two folders at its top, in the box's language
OWN = {"en": "My files", "de": "Meine Dateien"}
PROJECTS = {"en": "Projects", "de": "Projekte"}
METHODS = "OPTIONS, PROPFIND, PROPPATCH, GET, HEAD, PUT, DELETE, MKCOL, MOVE, COPY, LOCK, UNLOCK"
ET.register_namespace("D", DAV)


@dataclass(frozen=True)
class Place:
    """Where a path of the drive's is: its top, "root"; the folder of projects, "projects"; or a
    space, its own files' or a project's, and a name in it, "" of its top."""

    kind: str
    project: str | None = None
    name: str = ""


class Drive:
    """The drive of a person's, as a request to `handler` asks it: their files and projects of
    `agent`'s, its folders named in `language`."""

    def __init__(self, handler: BaseHTTPRequestHandler, agent: "Agent", person: int,
                 language: str | None):  # fmt: skip
        self.handler, self.agent, self.person = handler, agent, person
        self.own, self.projects = OWN.get(language or "", OWN["en"]), PROJECTS.get(
            language or "", PROJECTS["en"])  # fmt: skip

    def answer(self, method: str) -> None:
        """Answers a request of a WebDAV method. Raises NotFound, Refused, ValueError and
        FileNotFoundError of what cannot be done, as the server answers them."""
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.handler.path).path)
        parts = [p for p in path.removeprefix(PREFIX).split("/") if p]
        if method == "MKCOL" and len(parts) == 2 and parts[0] == self.projects:  # a project
            self._body()
            if parts[1] in self._folders():
                return self._send(405, b"")
            self.agent.add_project(parts[1][:80], self.person)
            self.agent.note(self.person, "made the project by the network drive", detail=parts[1])
            return self._send(201, b"")
        place = self.place(path)
        getattr(self, f"_{method.lower()}")(place, path)

    def place(self, path: str) -> Place:
        """The place of a path of the drive's. Raises NotFound of a project the person sees not."""
        parts = [p for p in path.removeprefix(PREFIX).split("/") if p]
        if not parts:
            return Place("root")
        if parts[0] == self.own:
            return Place("space", None, "/".join(parts[1:]))
        if parts[0] != self.projects:
            raise NotFound(f"there is no {parts[0]} in the drive")
        if len(parts) == 1:
            return Place("projects")
        if (project := self._folders().get(parts[1])) is None:
            raise NotFound(f"there is no project {parts[1]}")
        return Place("space", project, "/".join(parts[2:]))

    def _options(self, place: Place, path: str) -> None:
        self._send(200, b"", {"Allow": METHODS})

    def _propfind(self, place: Place, path: str) -> None:
        self._body()  # what is asked of the properties: every one, of each, as most clients ask
        depth = self.handler.headers.get("Depth", "1")
        found = [self._resource(place, path)]
        if depth != "0" and (place.kind != "space" or self._is_folder(place)):
            inside = [f"{path.rstrip('/')}/{n}" for n in self._children(place)]
            found += [self._resource(self.place(p), p) for p in inside]
        status = ET.Element(f"{D}multistatus")
        for href, props in found:
            response = ET.SubElement(status, f"{D}response")
            ET.SubElement(response, f"{D}href").text = _href(href)
            stat = ET.SubElement(response, f"{D}propstat")
            prop = ET.SubElement(stat, f"{D}prop")
            for key, value in props.items():
                element = ET.SubElement(prop, f"{D}{key}")
                if isinstance(value, ET.Element):
                    element.append(value)
                elif value is not None:
                    element.text = value
            ET.SubElement(stat, f"{D}status").text = "HTTP/1.1 200 OK"
        self._xml(207, status)

    def _proppatch(self, place: Place, path: str) -> None:
        # each property set or removed, as Windows sets a file's times, said done, and kept not
        if not self._is_folder(place):
            self._file(place)  # of a file there is, else none
        asked = ET.fromstring(self._body() or b"<x/>")
        status = ET.Element(f"{D}multistatus")
        response = ET.SubElement(status, f"{D}response")
        ET.SubElement(response, f"{D}href").text = _href(path)
        stat = ET.SubElement(response, f"{D}propstat")
        prop = ET.SubElement(stat, f"{D}prop")
        for named in asked.iter(f"{D}prop"):
            for child in named:
                ET.SubElement(prop, child.tag)
        ET.SubElement(stat, f"{D}status").text = "HTTP/1.1 200 OK"
        self._xml(207, status)

    def _get(self, place: Place, path: str, body: bool = True) -> None:
        if place.kind != "space" or self._is_folder(place):
            names = "\n".join(self._children(place)).encode() if body else b""
            return self._send(200, names, {"Content-Type": "text/plain; charset=utf-8"})
        file = self._file(place)
        stat = file.stat()
        headers = {
            "Content-Type": mimetypes.guess_type(file.name)[0] or "application/octet-stream",
            "Content-Length": str(stat.st_size), "Last-Modified": _modified(stat.st_mtime),
            "ETag": _etag(stat),
        }  # fmt: skip
        self.handler.send_response(200)
        for key, value in headers.items():
            self.handler.send_header(key, value)
        self.handler.end_headers()
        if body:
            with file.open("rb") as f:
                shutil.copyfileobj(f, self.handler.wfile, CHUNK)

    def _head(self, place: Place, path: str) -> None:
        self._get(place, path, body=False)

    def _put(self, place: Place, path: str) -> None:
        space, name = self._space(place)
        if not name:
            raise Refused("a folder's place takes no file")
        target = space.path(name)
        if not target.parent.is_dir():
            return self._send(409, b"")  # its folder first, as WebDAV has it
        if target.is_dir():
            raise Refused(f"{name} is a folder")
        partial = target.parent / f".dav-{uuid.uuid4().hex}"
        try:
            with partial.open("wb") as f:
                for chunk in self._chunks():
                    f.write(chunk)
            new = not target.exists()
            if not new:
                space.keep(name)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        action = "uploaded by the network drive" if new else "changed by the network drive"
        self._changed(place.project, action, name, new)
        self._send(201 if new else 204, b"")

    def _delete(self, place: Place, path: str) -> None:
        space, name = self._space(place)
        if not name:
            raise Refused("a project is deleted in Leat's app, never by its drive")
        space.delete(name)
        self._changed(place.project, "deleted by the network drive", name)
        self._send(204, b"")

    def _mkcol(self, place: Place, path: str) -> None:
        self._body()
        if place.kind != "space" or not place.name:  # one there is
            return self._send(405, b"")
        space, name = self._space(place)
        folder = space.path(name)
        if folder.exists():
            return self._send(405, b"")
        if not folder.parent.is_dir():
            return self._send(409, b"")
        folder.mkdir()
        self._send(201, b"")

    def _move(self, place: Place, path: str, copy: bool = False) -> None:
        space, name = self._space(place)
        if not name:
            raise Refused("a project is renamed in Leat's app, never by its drive")
        source = space.path(name)
        if not source.exists():
            raise FileNotFoundError(f"there is no file {name}")
        destination = urllib.parse.unquote(
            urllib.parse.urlsplit(self.handler.headers.get("Destination", "")).path
        )
        there = self.place(destination)
        into, renamed = self._space(there)
        if not renamed:
            raise Refused("a file moves into a folder of a space, not to its place")
        target = into.path(renamed)
        if not target.parent.is_dir():
            return self._send(409, b"")
        existed = target.exists()
        if existed and self.handler.headers.get("Overwrite", "T").upper() == "F":
            return self._send(412, b"")
        if existed:
            into.delete(renamed)  # kept, as any file written over is
        if copy:
            (shutil.copytree if source.is_dir() else shutil.copy2)(source, target)
        else:
            shutil.move(source, target)
            self.agent.files_changed(place.project, self.person)
        if copy:
            action, detail = "copied by the network drive", f"{name} → {renamed}"
        elif _passing(name) and not _passing(renamed):  # as Word saves a document: its temporary
            action, detail = "changed by the network drive", renamed  # file named as it
        else:
            action, detail = "moved by the network drive", f"{name} → {renamed}"
        self._changed(there.project, action, renamed, not existed, detail)
        self._send(204 if existed else 201, b"")

    def _copy(self, place: Place, path: str) -> None:
        self._move(place, path, copy=True)

    def _lock(self, place: Place, path: str) -> None:
        # a lock granted, and kept for no one, as Word asks one before it edits a document; a file
        # there is not, as a lock may make, made empty
        self._body()
        created = False
        if place.kind == "space" and place.name and not self._is_folder(place):
            space, name = self._space(place)
            file = space.path(name)
            if not file.exists() and file.parent.is_dir():
                file.touch()
                created = True
        token = f"opaquelocktoken:{uuid.uuid4()}"
        prop = ET.Element(f"{D}prop")
        active = _sub(prop, "lockdiscovery", "activelock")
        _sub(active, "locktype", "write")
        _sub(active, "lockscope", "exclusive")
        _sub(active, "depth").text = "0"
        _sub(active, "timeout").text = "Second-3600"
        _sub(active, "locktoken", "href").text = token
        _sub(active, "lockroot", "href").text = _href(path)
        self._xml(201 if created else 200, prop, {"Lock-Token": f"<{token}>"})

    def _unlock(self, place: Place, path: str) -> None:
        self._send(204, b"")

    def _resource(self, place: Place, path: str) -> tuple[str, dict[str, Any]]:
        # a path's properties, as PROPFIND answers them
        if place.kind != "space" or self._is_folder(place):
            folder = None if place.kind != "space" else self._space(place)[0].path(place.name)
            when = folder.stat().st_mtime if folder else time.time()
            href = path if path.endswith("/") else f"{path}/"
            return href, {
                "displayname": Path(path.rstrip("/")).name or "Leat",
                "resourcetype": ET.Element(f"{D}collection"),
                "getlastmodified": _modified(when), "creationdate": _created(when),
            }  # fmt: skip
        file = self._file(place)
        stat = file.stat()
        lock = ET.Element(f"{D}lockentry")
        _sub(lock, "lockscope", "exclusive")
        _sub(lock, "locktype", "write")
        return path, {
            "displayname": file.name, "resourcetype": None,
            "getcontentlength": str(stat.st_size), "getlastmodified": _modified(stat.st_mtime),
            "creationdate": _created(stat.st_mtime), "getetag": _etag(stat),
            "getcontenttype": mimetypes.guess_type(file.name)[0] or "application/octet-stream",
            "supportedlock": lock, "lockdiscovery": None,
        }  # fmt: skip

    def _children(self, place: Place) -> list[str]:
        # the names a folder of the drive's holds, but for the hidden, as the app shows none
        if place.kind == "root":
            return [self.own, self.projects]
        if place.kind == "projects":
            return sorted(self._folders())
        folder = self._space(place)[0].path(place.name)
        return sorted(p.name for p in folder.iterdir() if not hidden(p.name))

    def _folders(self) -> dict[str, str]:
        # the projects the person sees, by their folders' names: each its name, made a folder's,
        # and of two of one name, the later's its id's too
        found: dict[str, str] = {}
        for p in sorted(self.agent.projects(self.person), key=lambda p: p["id"]):
            named = re.sub(r'[\\/:*?"<>|]+', "-", p["name"]).strip(" .") or p["id"]
            found[named if named not in found else f"{named} ({p['id'][:6]})"] = p["id"]
        return found

    def _space(self, place: Place) -> tuple[Workspace, str]:
        # the space a place is in, and its name there. Raises Refused of the drive's own folders
        if place.kind != "space":
            raise Refused("the drive's own folders are not changed")
        return self.agent.space(place.project, self.person), place.name

    def _is_folder(self, place: Place) -> bool:
        return place.kind == "space" and self._space(place)[0].path(place.name).is_dir()

    def _file(self, place: Place) -> Path:
        space, name = self._space(place)
        if not (file := space.path(name)).is_file():
            raise FileNotFoundError(f"there is no file {name}")
        return file

    def _changed(self, project: str | None, action: str, name: str, new: bool = False,
                 detail: str | None = None) -> None:  # fmt: skip
        # tells the apps and the index of a space's files changed, and notes what was done, but of
        # a file a program makes for a while; a new file arrived, for the workflows that wait on it
        self.agent.files_changed(project, self.person)
        if _passing(name):
            return
        if new:
            self.agent.arrived(project, self.person, [name])
        self.agent.note(self.person, action, project, detail or name)

    def _chunks(self) -> Any:
        # the request's body, a chunk at a time, of its length, or in HTTP's chunks, as macOS sends
        # a file of unknown length. Raises ValueError of one past UPLOAD
        read, total = self.handler.rfile, 0
        if self.handler.headers.get("Transfer-Encoding", "").lower() == "chunked":
            while size := int(read.readline().split(b";")[0].strip() or b"0", 16):
                if (total := total + size) > UPLOAD:
                    raise ValueError(f"a file may be {UPLOAD >> 20} MB at most")
                yield read.read(size)
                read.readline()
            while read.readline().strip():  # the trailer, to its empty line
                pass
            return
        left = int(self.handler.headers.get("Content-Length") or 0)
        if left > UPLOAD:
            raise ValueError(f"a file may be {UPLOAD >> 20} MB at most")
        while left > 0:
            chunk = read.read(min(CHUNK, left))
            if not chunk:
                break
            left -= len(chunk)
            yield chunk

    def _body(self) -> bytes:
        return b"".join(self._chunks())[: 1 << 20]

    def _xml(self, status: int, root: ET.Element, headers: dict[str, str] | None = None) -> None:
        data = b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root)
        kind = {"Content-Type": 'application/xml; charset="utf-8"'}
        self._send(status, data, kind | (headers or {}))

    def _send(self, status: int, data: bytes, headers: dict[str, str] | None = None) -> None:
        self.handler.send_response(status)
        for key, value in {"DAV": "1, 2", "MS-Author-Via": "DAV", **(headers or {})}.items():
            self.handler.send_header(key, value)
        self.handler.send_header("Content-Length", str(len(data)))
        self.handler.end_headers()
        self.handler.wfile.write(data)


def _sub(parent: ET.Element, *names: str) -> ET.Element:
    # elements of DAV's tags, each in the one before, in `parent`: the last
    for name in names:
        parent = ET.SubElement(parent, f"{D}{name}")
    return parent


def _passing(name: str) -> bool:
    # whether a file is one a program makes for a while beside a document, as Word's ~$ owner file
    # and ~WRL0001.tmp, and macOS's ._ files, which the activity does not note
    base = name.rsplit("/", 1)[-1]
    return base.startswith(("~", ".")) or base.lower().endswith((".tmp", "~"))


def _href(path: str) -> str:
    return urllib.parse.quote(path, safe="/")


def _modified(seconds: float) -> str:
    return email.utils.formatdate(seconds, usegmt=True)


def _created(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def _etag(stat: os.stat_result) -> str:
    return f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
