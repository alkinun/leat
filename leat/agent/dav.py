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

import contextlib
import email.utils
import mimetypes
import os
import re
import shutil
import time
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

from leat.agent.agent import NAME, NotFound, Refused
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
AWAY = 60  # seconds a document moved aside, as Word saves one, is to be named so again
# the documents moved aside, by their spaces' folders and names, and when
_AWAY: dict[tuple[str, str], float] = {}
# the end of the name a Mac's app gives a document it moves aside as it saves it, by macOS's safe
# save: .sb-, the app's hash and a random part
_SAFE_SAVE = re.compile(r"\.sb-[0-9a-f]+-[0-9A-Za-z]+$")
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
        parts = _parts(path)
        if method == "MKCOL" and len(parts) == 2 and parts[0] == self.projects:  # a project
            self._body()
            if parts[1] in self._folders():
                return self._send(405, b"")
            self.agent.add_project(parts[1][:NAME], self.person)
            self.agent.note(self.person, "made the project by the network drive", detail=parts[1])
            return self._send(201, b"")
        if method == "MOVE" and len(parts) == 2 and parts[0] == self.projects:  # a project renamed
            return self._rename(parts[1])
        place = self.place(path)
        try:
            getattr(self, f"_{method.lower()}")(place, path)
        except (shutil.Error, ET.ParseError) as e:  # a folder moved into itself; XML that is not
            raise ValueError(str(e)) from e

    def place(self, path: str) -> Place:
        """The place of a path of the drive's. Raises NotFound of a project the person sees not."""
        parts = _parts(path)
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
            # a page or an image of script, a browser asked for of a drive's password it keeps,
            # run in no origin, the app's least, nor shown: kept, as a file of the app's is
            "Content-Security-Policy": "sandbox",
            "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(file.name)}",
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
        with received(self.handler, target.parent) as partial:
            # new, of one there is not, or empty, as a lock, or Windows, makes one before it puts
            # it, and a document just moved aside, as Word saves one, is changed, not new
            new = (not target.exists() or not target.stat().st_size) and not _away(space, name)
            space.keep(name)  # if there is one, of content, as an empty one is kept not
            os.replace(partial, target)
        if not target.stat().st_size:  # an empty file, as Windows puts first, is no file yet
            return self._send(201, b"")
        action = "uploaded by the network drive" if new else "changed by the network drive"
        self._changed(place.project, action, name, new)
        self._send(201 if new else 204, b"")

    def _delete(self, place: Place, path: str) -> None:
        space, name = self._space(place)
        if not name:
            raise Refused("a project is deleted in Leat's app, never by its drive")
        space.delete(name, kept=not _passing(name))  # a program's file for a while, of no one's
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
        destination = self._destination()
        there = self.place(destination)
        into, renamed = self._space(there)
        if not renamed:
            raise Refused("a file moves into a folder of a space, not to its place")
        target = into.path(renamed)
        if not target.parent.is_dir():
            return self._send(409, b"")
        if target == source or target.is_relative_to(source):
            raise Refused("a file moves onto itself, or a folder into itself, never")
        existed = target.exists()
        if existed and self.handler.headers.get("Overwrite", "T").upper() == "F":
            return self._send(412, b"")
        if existed:
            into.delete(renamed)  # kept, as any file written over is
        if not copy and _passing(renamed) and not _passing(name):
            # a document moved aside, as Word saves one: kept as it was, by its own name, which it
            # is to be named again
            space.keep(name)
            _AWAY[(str(space.root), name)] = time.monotonic()
        if copy:
            (shutil.copytree if source.is_dir() else shutil.copy2)(source, target)
        else:
            shutil.move(source, target)
            self.agent.files_changed(place.project, self.person)
        new = not existed and not _away(into, renamed)
        if copy:
            action, detail = "copied by the network drive", f"{name} → {renamed}"
        elif _passing(name) and not _passing(renamed):  # as Word saves a document: its temporary
            action, detail = "changed by the network drive", renamed  # file named as it
            new = new and target.stat().st_size > 0
        else:
            action, detail = "moved by the network drive", f"{name} → {renamed}"
        self._changed(there.project, action, renamed, new, detail)
        self._send(204 if existed else 201, b"")

    def _rename(self, folder: str) -> None:
        # a project of the person's renamed, as Explorer and Finder name a new folder they made in
        # the projects: its name the Destination's folder's
        if (project := self._folders().get(folder)) is None:
            raise NotFound(f"there is no project {folder}")
        destination = self._destination()
        parts = _parts(destination)
        if len(parts) != 2 or parts[0] != self.projects:
            raise Refused("a project is renamed among the projects, never moved")
        if parts[1] in self._folders():
            return self._send(412, b"")
        self.agent.change_project(project, self.person, name=parts[1][:NAME])  # theirs, or Refused
        self.agent.note(self.person, "renamed the project by the network drive", project)
        self._send(201, b"")

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
        space = self._space(place)[0]
        folder, found = space.path(place.name), []
        for p in folder.iterdir():  # but a link out of the space, or to nothing, as code may make
            if not hidden(p.name) and p.resolve().is_relative_to(space.root) and (
                    p.is_file() or p.is_dir()):  # fmt: skip
                found.append(p.name)
        return sorted(found)

    def _folders(self) -> dict[str, str]:
        # the projects the person sees, by their folders' names: each its name, made a folder's,
        # and of two of one name, the one made later's its id's too, as a folder's path is never
        # another project's than it was, a project made or shared since
        found: dict[str, str] = {}
        seen = self.agent.store.projects(self.person)
        for p in sorted(seen, key=lambda p: (p["created"], p["id"])):
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

    def _destination(self) -> str:
        # the path a MOVE or COPY names, as its Destination header gives it whole
        return urllib.parse.unquote(
            urllib.parse.urlsplit(self.handler.headers.get("Destination", "")).path
        )

    def _body(self) -> bytes:
        return b"".join(chunks(self.handler))[: 1 << 20]

    def _xml(self, status: int, root: ET.Element, headers: dict[str, str] | None = None) -> None:
        data = b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root)
        kind = {"Content-Type": 'application/xml; charset="utf-8"'}
        self._send(status, data, kind | (headers or {}))

    def _send(self, status: int, data: bytes, headers: dict[str, str] | None = None) -> None:
        send(self.handler, status, data, headers)


def send(handler: BaseHTTPRequestHandler, status: int, data: bytes,
         headers: dict[str, str] | None = None) -> None:  # fmt: skip
    """Answers a WebDAV request, as a drive's server says it is one: of DAV's classes and Office's
    header, and `headers`."""
    handler.send_response(status)
    for key, value in {"DAV": "1, 2", "MS-Author-Via": "DAV", **(headers or {})}.items():
        handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class TooLarge(ValueError):
    """A request's body is larger than it may be."""


def chunks(handler: BaseHTTPRequestHandler) -> Iterator[bytes]:
    """A request's body, a chunk at a time, of its length, or in HTTP's chunks, as macOS sends a
    file of unknown length. Raises TooLarge of one past UPLOAD, ValueError of a length there cannot
    be, ConnectionError of one cut short, as of a laptop gone to sleep while it saved."""
    read, total = handler.rfile, 0
    if handler.headers.get("Transfer-Encoding", "").lower() == "chunked":
        while True:
            line = read.readline()
            if not line:  # the connection gone before the last chunk
                raise ConnectionError("the body broke off")
            if not (size := int(line.split(b";")[0].strip() or b"0", 16)):
                break
            if size < 0:
                raise ValueError("a chunk's length cannot be negative")
            if (total := total + size) > UPLOAD:
                raise TooLarge(f"a file may be {UPLOAD >> 20} MB at most")
            if len(chunk := read.read(size)) < size:
                raise ConnectionError("the body broke off")
            yield chunk
            read.readline()
        while read.readline().strip():  # the trailer, to its empty line
            pass
        return
    left = int(handler.headers.get("Content-Length") or 0)
    if left < 0:
        raise ValueError("a body's length cannot be negative")
    if left > UPLOAD:
        raise TooLarge(f"a file may be {UPLOAD >> 20} MB at most")
    while left > 0:
        if not (chunk := read.read(min(CHUNK, left))):  # gone before its length: no file
            raise ConnectionError("the body broke off")
        left -= len(chunk)
        yield chunk


@contextlib.contextmanager
def received(handler: BaseHTTPRequestHandler, folder: Path) -> Iterator[Path]:
    """A request's body, as a hidden file in `folder` written whole, to be moved to its place, and
    deleted if it was not; none, of a body cut short, which overwrites nothing."""
    partial = folder / f".upload-{uuid.uuid4().hex}"
    try:
        with partial.open("wb") as f:
            for chunk in chunks(handler):
                f.write(chunk)
        yield partial
    finally:
        partial.unlink(missing_ok=True)


def _parts(path: str) -> list[str]:
    # a path of the drive's, as its folders' and file's names
    return [p for p in path.removeprefix(PREFIX).split("/") if p]


def _away(space: Workspace, name: str) -> bool:
    # whether a document was moved aside just now, as Word saves one, to be named so again
    return time.monotonic() - _AWAY.pop((str(space.root), name), -AWAY) < AWAY


def _sub(parent: ET.Element, *names: str) -> ET.Element:
    # elements of DAV's tags, each in the one before, in `parent`: the last
    for name in names:
        parent = ET.SubElement(parent, f"{D}{name}")
    return parent


def _passing(name: str) -> bool:
    # whether a file is one a program makes for a while beside a document, as Word's ~$ owner file
    # and ~WRL0001.tmp, macOS's ._ files, and the document a Mac's app moves aside as it saves,
    # Brief.docx.sb-0d3df477-FBQRBZ: which the activity does not note, nor its versions keep
    base = name.rsplit("/", 1)[-1]
    return (base.startswith(("~", ".")) or base.lower().endswith((".tmp", "~"))
            or _SAFE_SAVE.search(base) is not None)  # fmt: skip


def _href(path: str) -> str:
    return urllib.parse.quote(path, safe="/")


def _modified(seconds: float) -> str:
    return email.utils.formatdate(seconds, usegmt=True)


def _created(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def _etag(stat: os.stat_result) -> str:
    return f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
