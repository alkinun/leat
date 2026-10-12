import base64
import http.client
import json
import os
import socket
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from leat.agent.agent import Agent
from leat.agent.client import Client
from leat.agent.store import Store
from leat.agent.workspace import Workspace
from tests.test_agent import cookie, request, serving

D = "{DAV:}"
Asking = Callable[..., tuple[int, dict[str, str], bytes]]


@pytest.fixture
def agent(tmp_path) -> Agent:
    workspace = Workspace(tmp_path / "workspace")
    return Agent(Store(tmp_path / "leat.db"), Client("http://127.0.0.1:9"), [], workspace)


@pytest.fixture
def server(agent) -> Iterator[str]:
    with serving(agent) as url:
        yield url


def drive(server: str, password: str | None = None) -> Asking:
    """A computer's network drive: a request of a method, of a password, Basic's, if any."""
    host = urllib.parse.urlsplit(server).netloc

    def ask(method: str, path: str, body: bytes = b"", **headers: str):
        connection = http.client.HTTPConnection(host, timeout=10)
        given = dict(headers)
        if password is not None:
            given["Authorization"] = (
                "Basic " + base64.b64encode(f"Alkın:{password}".encode()).decode()
            )
        connection.request(method, urllib.parse.quote(path), body, given)
        response = connection.getresponse()
        answer = (response.status, dict(response.getheaders()), response.read())
        connection.close()
        return answer

    return ask


def connected(server: str) -> tuple[Asking, str]:
    # a drive of the person serving() set up, of a password they made in the app
    status, body = request(f"{server}/api/drive", "POST", {})
    assert status == 200
    made = json.loads(body)
    assert made["user"] == "Alkın" and made["path"] == "/dav"
    return drive(server, made["password"]), made["password"]


def listed(answer: tuple[int, dict[str, str], bytes]) -> dict[str, dict[str, str | None]]:
    # a PROPFIND's answer: each path's properties, of text, collections' resourcetype "collection"
    status, _, body = answer
    assert status == 207, body
    found = {}
    for response in ET.fromstring(body).iter(f"{D}response"):
        href = urllib.parse.unquote(response.findtext(f"{D}href"))
        prop = response.find(f"{D}propstat/{D}prop")
        found[href] = {child.tag.removeprefix(D): (
            "collection" if child.find(f"{D}collection") is not None else child.text)
            for child in prop}  # fmt: skip
    return found


def test_signed_in(server):
    # the drive, by a drive's password alone: not of none, nor a wrong one, nor an app's cookie's
    # secret, which no drive's password is in the app either; OPTIONS of any path, of none
    status, headers, _ = drive(server)("PROPFIND", "/dav/", Depth="0")
    assert status == 401 and headers["WWW-Authenticate"].startswith('Basic realm="Leat"')
    assert drive(server, "wrong")("PROPFIND", "/dav/")[0] == 401
    secret = cookie().split("=", 1)[1]
    assert drive(server, secret)("PROPFIND", "/dav/")[0] == 401
    ask, password = connected(server)
    assert ask("PROPFIND", "/dav/", Depth="0")[0] == 207
    status, headers, _ = drive(server)("OPTIONS", "/")
    assert status == 200 and headers["DAV"] == "1, 2" and "PROPFIND" in headers["Allow"]
    connection = http.client.HTTPConnection(urllib.parse.urlsplit(server).netloc)
    connection.request("GET", "/api/me", headers={"Cookie": f"leat={password}"})
    assert connection.getresponse().status == 401


def test_files(server, agent):
    # a person's own files: put, listed, read, written over and deleted, each kept first; folders
    # made, and files moved and copied in them
    ask, _ = connected(server)
    top = listed(ask("PROPFIND", "/dav/", Depth="1"))
    assert set(top) == {"/dav/", "/dav/My files/", "/dav/Projects/"}
    assert top["/dav/My files/"]["resourcetype"] == "collection"
    assert ask("PUT", "/dav/My files/Brief.txt", b"one")[0] == 201
    assert ask("PUT", "/dav/My files/Brief.txt", b"two")[0] == 204
    status, headers, body = ask("GET", "/dav/My files/Brief.txt")
    assert (status, body, headers["Content-Length"]) == (200, b"two", "3")
    space = agent.space(None, 1)
    assert [v["name"] for v in space.versions()] == ["Brief.txt"]  # "one", kept
    assert ask("MKCOL", "/dav/My files/Akten")[0] == 201
    assert ask("MKCOL", "/dav/My files/Akten")[0] == 405  # there
    assert ask("PUT", "/dav/My files/Nowhere/x.txt", b"x")[0] == 409  # its folder first
    assert ask("MOVE", "/dav/My files/Brief.txt",
               Destination=f"{server}/dav/My files/Akten/Brief.txt")[0] == 201  # fmt: skip
    found = listed(ask("PROPFIND", "/dav/My files/Akten", Depth="1"))
    assert found["/dav/My files/Akten/Brief.txt"]["getcontentlength"] == "3"
    assert ask("COPY", "/dav/My files/Akten/Brief.txt",
               Destination=f"{server}/dav/My files/Kopie.txt")[0] == 201  # fmt: skip
    assert ask("COPY", "/dav/My files/Akten/Brief.txt", Destination=f"{server}/dav/My files/"
               "Kopie.txt", Overwrite="F")[0] == 412  # fmt: skip
    assert sorted(f["name"] for f in space.files()) == ["Akten/Brief.txt", "Kopie.txt"]
    assert ask("DELETE", "/dav/My files/Kopie.txt")[0] == 204
    assert ask("GET", "/dav/My files/Kopie.txt")[0] == 404
    assert "Kopie.txt" in {v["name"] for v in space.versions()}  # kept, to restore
    done = [(a["action"], a["detail"]) for a in agent.store.activity()][::-1]
    assert ("changed by the network drive", "Brief.txt") in done
    assert ("moved by the network drive", "Brief.txt → Akten/Brief.txt") in done
    assert ("deleted by the network drive", "Kopie.txt") in done


def test_projects(server, agent):
    # each project the person sees, and none other; a folder made there a project of theirs; a
    # project renamed by the drive, but neither deleted nor moved
    ask, _ = connected(server)
    agent.add_project("Müller Erbschaft", 1)
    ada = agent.store.add_person("Ada")["id"]
    agent.add_project("Ihres", ada)
    found = listed(ask("PROPFIND", "/dav/Projects", Depth="1"))
    assert set(found) == {"/dav/Projects/", "/dav/Projects/Müller Erbschaft/"}
    assert ask("PROPFIND", "/dav/Projects/Ihres", Depth="0")[0] == 404
    assert ask("PUT", "/dav/Projects/Ihres/x.txt", b"x")[0] == 404
    assert ask("PUT", "/dav/Projects/Müller Erbschaft/Testament.txt", b"Erbe")[0] == 201
    (project,) = [p["id"] for p in agent.projects(1)]
    assert agent.space(project, 1).path("Testament.txt").read_bytes() == b"Erbe"
    assert ask("MKCOL", "/dav/Projects/Weber GmbH")[0] == 201
    assert {p["name"] for p in agent.projects(1)} == {"Müller Erbschaft", "Weber GmbH"}
    assert ask("DELETE", "/dav/Projects/Weber GmbH")[0] == 403
    renamed = {"Destination": f"{server}/dav/Projects/Weber KG"}
    assert ask("MOVE", "/dav/Projects/Weber GmbH", **renamed)[0] == 201  # renamed, as its maker's
    out = {"Destination": f"{server}/dav/My files/Weber KG"}
    assert ask("MOVE", "/dav/Projects/Weber KG", **out)[0] == 403  # but moved, never
    assert ask("PUT", "/dav/Projects/x.txt", b"x")[0] == 404  # no project of that name


def test_word(server, agent):
    # a document as Word opens and saves it: locked, its owner file put, hidden and not noted, a
    # temporary file written and named as the document, noted as the document changed; a body
    # of HTTP's chunks, as macOS sends one
    ask, _ = connected(server)
    assert ask("PUT", "/dav/My files/Brief.docx", b"v1")[0] == 201
    status, headers, body = ask("LOCK", "/dav/My files/Brief.docx", b"<lockinfo/>")
    assert status == 200 and headers["Lock-Token"].startswith("<opaquelocktoken:")
    assert b"activelock" in body
    assert ask("PUT", "/dav/My files/~$Brief.docx", b"owner")[0] == 201
    assert "/dav/My files/~$Brief.docx" not in listed(ask("PROPFIND", "/dav/My files", Depth="1"))
    assert ask("PUT", "/dav/My files/~WRL0001.tmp", b"v2")[0] == 201
    assert ask("MOVE", "/dav/My files/~WRL0001.tmp",
               Destination=f"{server}/dav/My files/Brief.docx")[0] == 204  # fmt: skip
    assert ask("DELETE", "/dav/My files/~$Brief.docx")[0] == 204
    assert (
        ask("UNLOCK", "/dav/My files/Brief.docx", **{"Lock-Token": headers["Lock-Token"]})[0] == 204
    )
    assert ask("GET", "/dav/My files/Brief.docx")[2] == b"v2"
    notes = [(a["action"], a["detail"]) for a in agent.store.activity()][::-1][1:]
    assert notes == [("connected a network drive", None),
                     ("uploaded by the network drive", "Brief.docx"),
                     ("changed by the network drive", "Brief.docx")]  # fmt: skip
    assert [f["name"] for f in agent.space(None, 1).files()] == ["Brief.docx"]  # no ~$ in the app
    chunked = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
    assert (
        ask("PUT", "/dav/My files/macOS.txt", chunked, **{"Transfer-Encoding": "chunked"})[0] == 201
    )
    assert ask("GET", "/dav/My files/macOS.txt")[2] == b"hello world"
    status, _, _ = ask("LOCK", "/dav/My files/Neu.docx", b"<lockinfo/>")  # a lock that makes one
    assert status == 201 and agent.space(None, 1).path("Neu.docx").read_bytes() == b""


def test_german(server, agent):
    # the drive's folders in the box's language
    agent.change(language="de")
    ask, _ = connected(server)
    assert set(listed(ask("PROPFIND", "/dav/", Depth="1"))) == {
        "/dav/", "/dav/Meine Dateien/", "/dav/Projekte/"}  # fmt: skip
    assert ask("PUT", "/dav/Meine Dateien/Notiz.txt", b"x")[0] == 201
    assert Path(agent.space(None, 1).path("Notiz.txt")).read_bytes() == b"x"


def test_broken_off(server, agent):
    # a body cut short, as of a laptop gone to sleep while it saved, writes nothing over the file
    ask, password = connected(server)
    assert ask("PUT", "/dav/My files/Brief.txt", b"hello world")[0] == 201
    host, port = urllib.parse.urlsplit(server).netloc.split(":")
    basic = base64.b64encode(f"Alkın:{password}".encode()).decode()
    with socket.create_connection((host, int(port))) as s:
        s.sendall(
            b"PUT /dav/My%20files/Brief.txt HTTP/1.1\r\nHost: "
            + host.encode()
            + b"\r\nAuthorization: Basic "
            + basic.encode()
            + b"\r\nContent-Length: 11\r\n\r\nhel"
        )
        s.shutdown(socket.SHUT_WR)
        s.recv(1024)
    assert agent.space(None, 1).path("Brief.txt").read_bytes() == b"hello world"
    assert [n for n in agent.space(None, 1).path("").iterdir() if n.name.startswith(".dav")] == []


def test_locked_then_put(server, agent):
    # a file a lock made, empty, and then put, as Finder copies one: uploaded, no empty version
    ask, _ = connected(server)
    assert ask("LOCK", "/dav/My files/Neu.txt", b"<lockinfo/>")[0] == 201
    assert ask("PUT", "/dav/My files/Neu.txt", b"content")[0] == 201
    assert ask("PUT", "/dav/My files/Leer.txt", b"")[0] == 201  # as Windows puts first
    assert ask("PUT", "/dav/My files/Leer.txt", b"full")[0] == 201
    assert agent.space(None, 1).versions() == []
    done = [(a["action"], a["detail"]) for a in agent.store.activity()]
    assert ("uploaded by the network drive", "Neu.txt") in done
    assert ("uploaded by the network drive", "Leer.txt") in done
    assert not any(a == "changed by the network drive" for a, _ in done)


def test_word_saves_by_renames(server, agent):
    # Word's save of a document on a network drive: its new content put as a temporary file, the
    # document moved aside, the temporary file named as it, the aside one deleted: the document
    # changed, its earlier content kept by its own name, nothing new arrived
    ask, _ = connected(server)
    assert ask("PUT", "/dav/My files/Vertrag.docx", b"v1")[0] == 201
    arrived = []
    agent.arrived = lambda project, person, names: arrived.extend(names)
    assert ask("PUT", "/dav/My files/~WRD0000.tmp", b"v2")[0] == 201
    assert ask("MOVE", "/dav/My files/Vertrag.docx",
               Destination=f"{server}/dav/My files/~WRL0001.tmp")[0] == 201  # fmt: skip
    assert ask("MOVE", "/dav/My files/~WRD0000.tmp",
               Destination=f"{server}/dav/My files/Vertrag.docx")[0] == 201  # fmt: skip
    assert ask("DELETE", "/dav/My files/~WRL0001.tmp")[0] == 204
    space = agent.space(None, 1)
    assert space.path("Vertrag.docx").read_bytes() == b"v2" and arrived == []
    assert "Vertrag.docx" in {v["name"] for v in space.versions()}
    assert agent.store.activity()[0]["action"] == "changed by the network drive"


def test_drive_edges(server, agent):
    # two projects of one name: the older keeps its folder; a link out of a space listed not; a
    # file served as no page of the app's; a name of another encoding; a move onto itself, or into
    # itself, refused; XML not well formed refused; a project renamed among the projects
    ask, password = connected(server)
    older = agent.add_project("Akte", 1)["id"]
    agent.store._db.execute("UPDATE projects SET created = 1 WHERE id = ?", (older,))
    for _ in range(3):
        agent.add_project("Akte", 1)
        assert ask("PUT", "/dav/Projects/Akte/x.txt", b"x")[0] in (201, 204)
        assert agent.space(older, 1).path("x.txt").exists()
    os.symlink("/etc", agent.space(None, 1).path("etc"))
    assert ask("PUT", "/dav/My files/Seite.html", b"<script>alert(1)</script>")[0] == 201
    assert set(listed(ask("PROPFIND", "/dav/My files", Depth="1"))) == {
        "/dav/My files/", "/dav/My files/Seite.html"}  # fmt: skip
    _, headers, _ = ask("GET", "/dav/My files/Seite.html")
    assert headers["Content-Security-Policy"] == "sandbox"
    assert headers["Content-Disposition"].startswith("attachment")
    latin = drive(server)
    basic = base64.b64encode("Jürgen:".encode("latin-1") + password.encode()).decode()
    assert latin("PROPFIND", "/dav/", Depth="0", Authorization=f"Basic {basic}")[0] == 207
    assert ask("MKCOL", "/dav/My files/A")[0] == 201
    assert ask("MOVE", "/dav/My files/A", Destination=f"{server}/dav/My files/A/B")[0] == 403
    assert ask("MOVE", "/dav/My files/Seite.html",
               Destination=f"{server}/dav/My files/Seite.html")[0] == 403  # fmt: skip
    assert ask("PROPPATCH", "/dav/My files/Seite.html", b"<not xml")[0] == 400
    assert ask("MKCOL", "/dav/Projects/Neuer Ordner")[0] == 201
    assert ask("MOVE", "/dav/Projects/Neuer Ordner",
               Destination=f"{server}/dav/Projects/Weber GmbH")[0] == 201  # fmt: skip
    assert "Weber GmbH" in {p["name"] for p in agent.projects(1)}
