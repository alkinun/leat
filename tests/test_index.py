"""The index: documents' places, passages, the matching of words, Turkish's too, reading files as
they change, and searching them, through search_files as the model does. Reading PDFs needs
LEAT_SANDBOX, as tests/test_files.py's documents do."""

import os
from pathlib import Path

import pytest

from leat.agent import documents
from leat.agent.agent import SEARCH, Agent, _system
from leat.agent.client import Client
from leat.agent.index import PASSAGE, Index, _match, fold, passages
from leat.agent.store import Store
from leat.agent.tools import Context, files
from leat.agent.workspace import Workspace
from tests.test_files import documents as needs_sandbox

SANDBOX = os.environ.get("LEAT_SANDBOX")


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace", Path(SANDBOX) if SANDBOX else None)


@pytest.fixture
def index(tmp_path, workspace) -> Index:
    return Index(tmp_path / "index.db", workspace)


def test_places():
    # a PDF's pages, a presentation's slides, a workbook's sheets, any other's sections
    pdf = "## Page 1\n\nTerms.\n\n## Page 2\n\nRent."
    assert documents.places("lease.pdf", pdf) == [(0, "page 1"), (19, "page 2")]
    assert documents.places("deck.pptx", "## Slide 3: Plan\n\nGrow.") == [(0, "slide 3")]
    assert documents.places("budget.xlsx", "## March\n\n| a |") == [(0, "sheet March")]
    notes = "Intro.\n\n# Notes\n\n## Termination\n\nNinety days."
    assert documents.places("notes.md", notes) == [(8, "Notes"), (17, "Termination")]
    assert documents.places("notes.txt", "Just text.") == []
    assert documents.places("a.md", "# " + "x" * 100)[0][1] == "x" * 59 + "…"


def test_passages():
    # lines within a place, PASSAGE characters at most, a longer line cut between its words
    text = "## Page 1\n\nShort.\n\n## Page 2\n\n" + "word " * 500
    found = passages("lease.pdf", text)
    assert found[0] == (0, "page 1", "## Page 1\n\nShort.")
    assert {place for _, place, _ in found[1:]} == {"page 2"}
    assert all(len(said) <= PASSAGE for _, _, said in found)
    assert " ".join(said for _, _, said in found[2:]).split() == ["word"] * 500
    for start, _, said in found:
        assert text[start : start + len(said)] == said
    lines = "\n".join(f"Line {i}." for i in range(400))
    packed = passages("notes.txt", lines)
    assert len(packed) == len(lines) // PASSAGE + 1 and all(p[1] == "" for p in packed)


def test_words():
    # dotted and dotless i one letter; a word found as it is begun, a long one by its stem
    assert fold("İSTANBUL Iğdır ılık") == "istanbul iğdir ilik"
    assert _match("Sözleşmesi feshi a ve kira") == '"sözleşm"* OR "feshi"* OR "ve" OR "kira"*'
    assert _match("termination") == '"terminat"*' and _match("?!") == ""


def test_search(workspace, index):
    # a space's files read, their passages found by their words, the likeliest first, of the
    # space's alone
    space, other = workspace.space("projects/a"), workspace.space("projects/b")
    lease = ("# Rent\n\nKira aylık 40.000 TL.\n\n# Termination\n\n"
             "Sözleşmenin feshi için 90 gün önceden bildirim gerekir.")  # fmt: skip
    files.write(space, "lease.md", lease)
    files.write(space, "notes.txt", "Yılmaz Ltd pays on the 5th of each month.")
    files.write(other, "secret.txt", "Another client's confidential terms.")
    (space.root / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    index.update("projects/a")
    index.update("projects/b")
    found = index.search(space, "sözleşmesi fesih")
    assert [(f["name"], f["place"]) for f in found] == [("lease.md", "Termination")]
    assert found[0]["text"].startswith("# Termination")
    assert found[0]["start"] == lease.index("# Termination")
    assert index.search(space, "YILMAZ")[0]["name"] == "notes.txt"  # dotless, in capitals
    assert index.search(space, "confidential") == index.search(space, "logo") == []
    assert index.states("projects/a") == {}


def test_changes(workspace, index, monkeypatch):
    # a file read once, its text kept until it changes; one changed read again, one gone
    # forgotten; one that cannot be read said to have failed, and why
    space = workspace.space("people/1")
    files.write(space, "a.txt", "apples")
    reads = []
    real = documents.text
    monkeypatch.setattr(documents, "text", lambda s, n: reads.append(n) or real(s, n))
    told = []
    index.changed = told.append
    index.update("people/1")
    assert index.text(space, "a.txt") == "apples" and reads == ["a.txt"]
    assert told == ["people/1", "people/1"]  # as it began to be read, and once it was
    files.write(space, "a.txt", "pears and plums")
    os.utime(space.path("a.txt"), (1, 1))
    index.update("people/1")
    assert index.search(space, "plums")[0]["name"] == "a.txt" and not index.search(space, "apples")
    space.path("a.txt").unlink()
    index.update("people/1")
    assert index.search(space, "plums") == []
    files.write(space, "broken.pdf", "not a PDF")
    index.update("people/1")
    assert index.states("people/1")["broken.pdf"]["state"] == "failed"
    index.forget("people/1")
    assert index.states("people/1") == {}


def test_versions(tmp_path, workspace):
    # an index of another version is made anew
    index = Index(tmp_path / "index.db", workspace)
    files.write(workspace.space("people/1"), "a.txt", "apples")
    index.update("people/1")
    index._db.execute("PRAGMA user_version = 0")
    again = Index(tmp_path / "index.db", workspace)
    assert again.search(workspace.space("people/1"), "apples") == []


def test_search_files(workspace, index):
    # the model's search: each passage numbered as a source, by its file and place, the same
    # number each time it is found
    space = workspace.space("people/1")
    files.write(space, "lease.md", "# Rent\n\nRent is 900 a month.\n\n# Deposit\n\nTwo months.")
    index.update("people/1")
    numbers: dict[str, int] = {}
    context = Context("c", lambda url, title: numbers.setdefault(url, len(numbers) + 1),
                      workspace=space)  # fmt: skip
    result = files.search(index, context, "rent")
    said = ("[1] lease.md, Rent (read on from start=0)\n# Rent\n\nRent is 900 a month.\n\n"
            "(Cite each passage you use by its number, as [1].)")  # fmt: skip
    assert result.content == said
    assert result.info["results"] == [{"n": 1, "url": "file:lease.md#Rent",
                                       "title": "lease.md, Rent", "file": "lease.md",
                                       "place": "Rent"}]  # fmt: skip
    assert files.search(index, context, "deposit").content.startswith("[2] lease.md, Deposit")
    assert files.search(index, context, "rent").info["results"][0]["n"] == 1
    assert files.search(index, context, "nothing").content == "No passage of the files says that."
    # and read reads the text the index keeps
    assert files.read(space, "lease.md", index=index).content.startswith("# Rent")


@needs_sandbox
def test_pdf_pages(workspace, index):
    # a PDF's passages, each of its page
    space = workspace.space("people/1")
    files.run(space, """
from fpdf import FPDF
pdf = FPDF()
pdf.set_font("Helvetica")
for said in ("The lease runs three years.", "The deposit is two months' rent."):
    pdf.add_page()
    pdf.cell(text=said)
pdf.output("lease.pdf")
""")  # fmt: skip
    index.update("people/1")
    found = index.search(space, "deposit")
    assert [(f["name"], f["place"]) for f in found] == [("lease.pdf", "page 2")]


def test_agent(engine, workspace, index, tmp_path):
    # the apps told of each file's state as it is read; the system prompt tells the model to
    # search the files, when it can
    agent = Agent(Store(tmp_path / "leat.db"), Client(engine.url), files.tools(index), workspace,
                  index)  # fmt: skip
    files.write(agent.space(), "broken.pdf", "not a PDF")
    with agent.events.watch() as events:
        agent.files_changed()
        index.update("people/0")
        states = [events.get(timeout=5)["files"][0].get("state") for _ in range(3)]
    assert states == [None, "reading", "failed"]
    assert SEARCH in _system(True, tools={"search_files"})["content"]
    assert "search_files" not in _system(True)["content"]
    assert "search_files" in agent.tools


class Seeing:
    """A transcriber whose model sees images, or not as `sees` says, which reads every image as
    `said`, and keeps the kinds it was given."""

    def __init__(self, said: str, sees: bool = True):
        self.said, self.sees, self.kinds = said, sees, []

    def available(self) -> bool:
        return self.sees

    def recheck(self) -> None:
        pass

    def __call__(self, image: bytes, kind: str) -> str:
        self.kinds.append(kind)
        return self.said


def test_scans():
    # a PDF's pages that hold next to no text, and its text with theirs read in their place
    text = "## Page 1\n\nThe lease runs three years.\n\n## Page 2\n\n\n\n## Page 3\n\n7\n"
    assert documents.scans("lease.pdf", text) == [2, 3]
    assert documents.scans("lease.txt", text) == []
    assert documents.transcribed(text, {2: "Deposit: two months."}) == (
        "## Page 1\n\nThe lease runs three years.\n\n## Page 2\n\nDeposit: two months.\n\n"
        "## Page 3\n\n7\n")  # fmt: skip


def test_images(workspace, tmp_path):
    # an image's text, as a model that sees images reads it, kept and searched; while none does,
    # images are not read, and are once one does
    seeing = Seeing("FATURA\n\nToplam: 1.200 TL", sees=False)
    index = Index(tmp_path / "index.db", workspace, seeing)  # type: ignore[arg-type]
    space = workspace.space("people/1")
    (space.root / "receipt.jpg").write_bytes(b"\xff\xd8\xff\x00 a photo")
    index.update("people/1")
    assert index.states("people/1") == {} and index.search(space, "toplam") == []
    seeing.sees = True
    index.update("people/1")
    assert index.search(space, "toplam")[0]["name"] == "receipt.jpg"
    assert seeing.kinds == ["image/jpeg"]


@needs_sandbox
def test_scanned_pdf(workspace, tmp_path):
    # a scanned page rendered, and read by a model that sees images; while none does, the PDF's
    # other pages are read, and it is said to be scanned, till one does
    seeing = Seeing("The deposit is two months' rent.", sees=False)
    index = Index(tmp_path / "index.db", workspace, seeing)  # type: ignore[arg-type]
    space = workspace.space("people/1")
    files.run(space, """
from fpdf import FPDF
pdf = FPDF()
pdf.set_font("Helvetica")
pdf.add_page()
pdf.cell(text="The lease runs three years.")
pdf.add_page()
pdf.rect(20, 20, 100, 60, style="F")
pdf.output("lease.pdf")
""")  # fmt: skip
    index.update("people/1")
    assert index.states("people/1") == {"lease.pdf": {"state": "scanned"}}
    assert index.search(space, "lease")[0]["place"] == "page 1"
    seeing.sees = True
    index.update("people/1")
    assert index.states("people/1") == {} and seeing.kinds == ["image/png"]
    assert [(f["place"], f["text"]) for f in index.search(space, "deposit")] == [
        ("page 2", "## Page 2\n\nThe deposit is two months' rent.")]  # fmt: skip


def test_transcriber(engine):
    # an image asked of the engine's model, as a data: URL before the request to read it; whether
    # the model sees images, as the engine says, asked again once rechecked
    from leat.agent.ocr import TRANSCRIBE, Transcriber

    transcriber = Transcriber(Client(engine.url))
    assert not transcriber.available()
    engine.vision = True
    assert not transcriber.available()  # as it was said, CHECK seconds ago
    transcriber.recheck()
    assert transcriber.available()
    engine.replies.put([{"content": " Total: 900 "}])
    assert transcriber(b"png", "image/png") == "Total: 900"
    ((image, text),) = [engine.requests[0]["messages"][0]["content"]]
    assert image == {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}
    assert text == {"type": "text", "text": TRANSCRIBE}
