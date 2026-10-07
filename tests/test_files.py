"""The workspace, its tools and the sandbox. The sandbox's tests need bubblewrap, and those of the
documents it reads LEAT_SANDBOX, an environment made of leat/agent/sandbox.txt."""

import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from leat.agent.agent import Agent, NotFound
from leat.agent.client import Client
from leat.agent.context import message as _api
from leat.agent.server import Server
from leat.agent.store import Store
from leat.agent.tools import files
from leat.agent.workspace import Workspace


def _sandboxes() -> bool:
    # whether bubblewrap runs here: some systems forbid its user namespaces
    if shutil.which("bwrap") is None or shutil.which("prlimit") is None:
        return False
    return Workspace(Path("/tmp")).run("print(1)", timeout=30).output.strip() == "1"


SANDBOX = os.environ.get("LEAT_SANDBOX")
sandboxed = pytest.mark.skipif(not _sandboxes(), reason="needs bubblewrap, and its namespaces")
documents = pytest.mark.skipif(not SANDBOX, reason="needs LEAT_SANDBOX, the sandbox's environment")


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace", Path(SANDBOX) if SANDBOX else None)


def test_paths(workspace, tmp_path):
    # a name is the workspace's, an absolute one too; one out of it is refused, by a link too
    assert workspace.path("notes/a.txt") == workspace.root / "notes" / "a.txt"
    assert workspace.path("/etc/passwd") == workspace.root / "etc" / "passwd"
    with pytest.raises(ValueError, match="outside the workspace"):
        workspace.path("../leat.db")
    (workspace.root / "out").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="outside the workspace"):
        workspace.path("out/leat.db")


def test_files(workspace, tmp_path):
    (workspace.root / "notes").mkdir()
    (workspace.root / "notes" / "a.txt").write_text("a")
    (workspace.root / ".hidden").write_text("b")
    (workspace.root / "b.txt").write_text("bb")
    os.utime(workspace.root / "notes" / "a.txt", (0, 0))
    (tmp_path / "key").write_text("secret")
    (workspace.root / "key").symlink_to(tmp_path / "key")  # out of it
    assert [(f["name"], f["size"]) for f in workspace.files()] == [("b.txt", 2), ("notes/a.txt", 1)]
    # a name for an upload: at the top, free, and of no path
    assert workspace.free("b.txt") == "b (2).txt" and workspace.free("../../x.pdf") == "x.pdf"
    assert workspace.free(".bashrc") == "bashrc" and workspace.free("") == "file"
    workspace.delete("b.txt")
    with pytest.raises(FileNotFoundError):
        workspace.delete("b.txt")


def test_read_write_edit(workspace):
    files.write(workspace, "notes/plan.md", "# Plan\n\nBuy bread.\n")
    assert files.read(workspace, "notes/plan.md").content == "# Plan\n\nBuy bread.\n"
    assert files.read(workspace, ".").content == "notes/"
    assert files.read(workspace, "notes").content == "plan.md"
    result = files.edit(workspace, "notes/plan.md", "bread", "milk")
    assert result.info == {"files": ["notes/plan.md"]}
    assert (workspace.root / "notes" / "plan.md").read_text() == "# Plan\n\nBuy milk.\n"
    with pytest.raises(ValueError, match="is not in"):
        files.edit(workspace, "notes/plan.md", "bread", "milk")
    files.write(workspace, "twice.txt", "a a")
    with pytest.raises(ValueError, match="is in 2 places of"):
        files.edit(workspace, "twice.txt", "a", "b")
    # a long file in parts, saying where the next begins
    files.write(workspace, "long.txt", "x" * (files.READ + 10))
    first = files.read(workspace, "long.txt").content
    assert first.endswith(f"read on from start={files.READ})")
    assert files.read(workspace, "long.txt", files.READ).content == "x" * 10
    # what is not text
    (workspace.root / "photo.jpg").write_bytes(b"\xff\xd8")
    (workspace.root / "blob").write_bytes(b"\0\1\2")
    with pytest.raises(ValueError, match="an image, which you cannot see"):
        files.read(workspace, "photo.jpg")
    with pytest.raises(ValueError, match="not a file of text"):
        files.read(workspace, "blob")
    with pytest.raises(FileNotFoundError):
        files.read(workspace, "nothing.txt")


def test_skills(workspace):
    # each listed, and read from skills/, but nothing beside them
    found = dict(files.skills())
    assert set(found) == {f"skills/{k}/SKILL.md" for k in ("docx", "xlsx", "pptx", "pdf")}
    assert "python-docx" in files.read(workspace, "skills/docx/SKILL.md").content
    with pytest.raises(ValueError, match="outside the skills"):
        files.read(workspace, "skills/../tools/files.py")


@sandboxed
def test_run(workspace):
    # what the code prints, and the files it makes; the workspace alone, without the network
    code = """
import os, socket
open("made.txt", "w").write("hi")
print(sorted(os.listdir("/")), os.path.exists(os.path.expanduser("~/.ssh")))
try:
    socket.create_connection(("1.1.1.1", 80), timeout=2)
except OSError:
    print("no network")
"""
    (workspace.root / "json.py").write_text("raise SystemExit('a library taken')")
    result = files.run(workspace, code + "import json\n")
    assert "no network" in result.content and "False" in result.content
    assert result.info == {"files": ["made.txt"], "status": 0}
    assert result.content.endswith("Files made or changed: made.txt")
    assert "It failed, with exit status 1." in files.run(workspace, "1 / 0").content
    assert workspace.run("while True: pass", timeout=1).status is None


@documents
def test_documents(workspace):
    # what the libraries make, read back
    files.run(
        workspace,
        """
import docx, openpyxl
d = docx.Document(); d.add_heading("Plan", 0); d.add_paragraph("Buy milk."); d.save("plan.docx")
wb = openpyxl.Workbook(); wb.active.append(["Rent", 900, "=B1*12"]); wb.save("budget.xlsx")
""",
    )
    assert files.read(workspace, "plan.docx").content == "# Plan\n\nBuy milk.\n\n"
    assert files.read(workspace, "budget.xlsx").content == (
        "## Sheet\n\n\n| Rent | 900 | =B1*12 |\n| --- | --- | --- |\n\n")  # fmt: skip
    # a long one, in parts, whole
    files.run(workspace, """
import docx
d = docx.Document()
for i in range(400):
    d.add_paragraph(f"Line {i}: " + "words " * 10)
d.save("long.docx")
""")  # fmt: skip
    first = files.read(workspace, "long.docx").content
    assert first.endswith(f"read on from start={files.READ})")
    whole, read = "", first
    while "read on from start=" in read:
        whole += read.rsplit("\n\n(characters", 1)[0]
        read = files.read(workspace, "long.docx", int(read.rsplit("=", 1)[1][:-1])).content
    lines = (whole + read).split("\n\n")
    assert lines[:-1] == [f"Line {i}: {'words ' * 10}" for i in range(400)]
    with pytest.raises(ValueError, match="could not be read"):
        files.write(workspace, "broken.pdf", "not a PDF")
        files.read(workspace, "broken.pdf")


@pytest.fixture
def agent(workspace, tmp_path) -> Agent:
    return Agent(Store(tmp_path / "leat.db"), Client("http://127.0.0.1:9"), [], workspace)


def test_attached(agent, workspace):
    # the files attached to a message, named for the model, which must be in the workspace
    files.write(workspace, "plan.md", "Buy milk.")
    with pytest.raises(NotFound):
        agent.send(None, "Read it", attached=["nothing.md"])
    message = {"role": "user", "content": "Read it", "info": {"files": ["plan.md"]}}
    assert _api(message)["content"] == "Read it\n\n(Attached, in the workspace: plan.md)"
    with agent.events.watch() as events:
        agent.files_changed()
        assert events.get(timeout=1)["files"][0]["name"] == "plan.md"
        agent.files_changed()  # unchanged: nothing
        assert events.empty()


def test_system_prompt(agent, tmp_path):
    # the workspace and the skills, when the agent has a workspace
    from leat.agent.agent import _system

    with_workspace, without = (_system([], w)["content"] for w in (True, False))
    assert "- skills/docx/SKILL.md: Word documents" in with_workspace
    assert "workspace" not in without


@pytest.fixture
def server(agent) -> Iterator[str]:
    with Server(agent, port=0) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        yield f"http://127.0.0.1:{server.server_port}"
        server.shutdown()


def call(url: str, method: str = "GET", data: bytes | None = None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data, method=method)) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_api_files(server, workspace, monkeypatch):
    # uploads, by a free name, downloads, which a page cannot act in, and deletes
    status, _, body = call(f"{server}/api/files/My%20notes.txt", "PUT", b"Buy milk.")
    assert status == 200 and json.loads(body) == {"name": "My notes.txt"}
    assert json.loads(call(f"{server}/api/files/My%20notes.txt", "PUT", b"x")[2])["name"] == (
        "My notes (2).txt")  # fmt: skip
    status, headers, body = call(f"{server}/files/My%20notes.txt")
    assert status == 200 and body == b"Buy milk."
    assert headers["Content-Security-Policy"] == "sandbox"
    assert headers["Content-Disposition"] == "inline; filename*=UTF-8''My%20notes.txt"
    files.write(workspace, "page.html", "<script>fetch('/api/events')</script>")
    assert call(f"{server}/files/page.html")[1]["Content-Disposition"].startswith("attachment")
    for path in ("/files/..%2Fleat.db", "/files/nothing.txt"):
        assert call(f"{server}{path}")[0] == 404
    assert call(f"{server}/api/files/My%20notes.txt", "DELETE")[0] == 200
    assert call(f"{server}/api/files/My%20notes.txt", "DELETE")[0] == 404
    for length in ("-1", "many"):  # a length there cannot be
        bad = urllib.request.Request(f"{server}/api/files/x.txt", b"x", method="PUT")
        bad.add_unredirected_header("Content-Length", length)
        with pytest.raises(urllib.error.HTTPError, match="400"):
            urllib.request.urlopen(bad)
    monkeypatch.setattr("leat.agent.server.UPLOAD", 4)
    assert call(f"{server}/api/files/big.bin", "PUT", b"12345")[0] == 413
    assert call(f"{server}/files")[0] == 200  # the app's page of them
