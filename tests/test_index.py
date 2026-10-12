"""The index: documents' places, passages, the matching of words, German's too, reading files as
they change, and searching them, through search_files as the model does. Reading PDFs needs
LEAT_SANDBOX: see tests/agents.py."""

import functools
import os

import pytest

from leat.agent import documents
from leat.agent import index as index_module
from leat.agent.agent import Agent, _system
from leat.agent.client import Client
from leat.agent.embeddings import GEMMA2, QWEN3, Embedder
from leat.agent.index import COMMON, PASSAGE, Index, _match, fold, passages
from leat.agent.store import Store
from leat.agent.tools import Context, files
from tests.agents import call, ended, needs_sandbox, until

SANDBOX = os.environ.get("LEAT_SANDBOX")


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
    # ß as ss; a word found as it is begun, a long one by its stem; the words of every sentence
    # left out, German's and English's
    assert fold("Straße GRÜN") == "strasse grün"
    assert _match("Die Kündigung der Miete, bitte") == '"kündig"* OR "miete"*'
    assert _match("What is it?") == '"what"* OR "is" OR "it"'  # all common: all kept
    assert _match("termination") == '"terminat"*' and _match("?!") == ""


def test_common_words(workspace, index, monkeypatch):
    # in any language, the words in more than half of the index's passages, once it holds SEEN,
    # are left out of a search, as French's here; before, English's and German's alone
    space = workspace.space("people/1")
    said = {
        "a.txt": "Le loyer de la boutique est de 900 euros.",
        "b.txt": "La facture de mars.",
        "c.txt": "Le contrat de la société.",
        "d.txt": "Le délai de préavis est de trois mois.",
    }
    for name, text in said.items():
        files.write(space, name, text)
    index.update("people/1")
    question = "Quel est le loyer de la boutique ?"
    assert index.common(question) == COMMON  # too few passages to tell
    monkeypatch.setattr(index_module, "SEEN", 4)
    assert index.common(question) - COMMON == {"de", "la", "le"}  # "est" is in two of four
    # the rent's passage first, the one that shares "est" after, and none that shares only "le"
    assert [f["name"] for f in index.search(space, question)] == ["a.txt", "d.txt"]
    assert index._held("préavis") == 1  # its accents taken off, as the tokenizer does


def test_search(workspace, index):
    # a space's files read, their passages found by their words, the likeliest first, of the
    # space's alone
    space, other = workspace.space("projects/a"), workspace.space("projects/b")
    lease = ("# Miete\n\nDie Miete beträgt 2.500 EUR im Monat.\n\n# Kündigung\n\n"
             "Die Kündigungsfrist beträgt drei Monate zum Quartalsende.")  # fmt: skip
    files.write(space, "lease.md", lease)
    files.write(space, "notes.txt", "Weißbier GmbH pays on the 5th of each month.")
    files.write(other, "secret.txt", "Another client's confidential terms.")
    (space.root / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    index.update("projects/a")
    index.update("projects/b")
    found = index.search(space, "Wie lang ist die Kündigung?")
    assert [(f["name"], f["place"]) for f in found] == [("lease.md", "Kündigung")]
    assert found[0]["text"].startswith("# Kündigung")
    assert found[0]["start"] == lease.index("# Kündigung")
    assert index.search(space, "WEISSBIER")[0]["name"] == "notes.txt"  # ß as ss, in capitals
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
    assert files.search(index, context, "nothing").content == (
        "No passage of the files says that. The files:\n- lease.md")  # fmt: skip
    # and read reads the text the index keeps
    assert files.read(space, "lease.md", index=index).content.startswith("# Rent")
    # passages of some of a folder's files, the others said to be there, as a model would take
    # those found for all
    for n, total in (("1", "900,00 EUR"), ("2", "1.240,00 EUR"), ("3", "Wasser")):
        files.write(space, f"Rechnungen/{n}.txt", f"Rechnung {n}: {total}")
    index.update("people/1")
    found = files.search(index, context, "EUR").content
    assert found.endswith("(Rechnungen/ holds 3 files, of which these passages are of 2: for a "
                          "question of every one, call ask_files.)")  # fmt: skip


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


def test_private(workspace, index, tmp_path):
    # what of a web search's query must not leave the box: the names the conversation's files
    # give, of their passages that hold the query's words, and what its person wrote; the box's
    # people's names, and its office's
    agent = Agent(Store(tmp_path / "leat.db"), Client("http://127.0.0.1:9"), [], workspace, index)
    agent.store.add_person("Alkın Ünlü")
    agent.change(office="Kanzlei Weber")
    project = agent.add_project("Hofmann", 1)["id"]
    space = agent.space(project, 1)
    files.write(space, "letter.md", "Sehr geehrter Herr Hofmann,\n\nIhre Kündigung der Hofmann "
                "Textil GmbH ist eingegangen. Die Kündigung gilt zum 31. März.")  # fmt: skip
    files.write(workspace.space("people/1"), "other.md", "Herr Schulz kündigt.")
    index.update(f"projects/{project}")
    index.update("people/1")
    said = {"role": "user", "content": "Meine Mandantin Frau Yılmaz fragt nach der Frist."}
    id = agent.store.create("Frist", [said], 1, project)["id"]
    private = functools.partial(agent.private, id, space)
    assert private("Hofmann Textil Kündigung Frist") == ["Hofmann", "Textil"]
    assert private("Kündigung Frist Arbeitsvertrag") == []  # a noun, as "die Kündigung" has it
    assert private("Yılmaz Erbrecht") == ["Yılmaz"]
    assert private("Schulz Kündigung") == []  # of another space's files
    assert private("Kanzlei Weber Bewertungen") == ["Kanzlei Weber"]
    assert private("Alkın Ünlü Rechtsanwalt") == ["Alkın Ünlü"]
    # a page's address the model wrote, as of a search engine's; not one its person or a search gave
    assert private("https://duckduckgo.com/html/?q=Hofmann+Textil+K%C3%BCndigung") == [
        "Hofmann", "Textil"]  # fmt: skip
    said = {"role": "user", "content": "Lies https://hofmann-textil.de/impressum bitte."}
    searched = {"role": "tool", "content": "…", "info": {"results": [
        {"url": "https://news.example/hofmann-textil-insolvenz", "n": 1}]}}  # fmt: skip
    agent.store.append(id, said, searched)
    assert private("https://hofmann-textil.de/impressum") == []
    assert private("https://news.example/hofmann-textil-insolvenz") == []


def test_quotes_checked(engine, workspace, index, tmp_path):
    # an answer's quotes checked against the sources they cite, by code: of a passage, its file's
    # whole text, of a name that holds a "#" too; each found there, or not, in its info
    agent = Agent(Store(tmp_path / "leat.db"), Client(engine.url), files.tools(index), workspace,
                  index)  # fmt: skip
    files.write(agent.space(), "Akte #12.md", "# Kündigung\n\nDie Kündigungsfrist beträgt drei "
                "Monate zum Quartalsende.\n\n# Miete\n\nDie Miete beträgt 900 EUR.")  # fmt: skip
    index.update("people/0")
    engine.replies.put([{"tool_calls": [call("search_files", {"query": "Kündigungsfrist"})]}])
    answer = ("„Die Kündigungsfrist beträgt drei Monate zum Quartalsende.“ [1] Und “Die Miete "
              "beträgt fünf Euro im Jahr” [1].")  # fmt: skip
    engine.replies.put([{"content": answer}])
    with agent.events.watch() as events:
        id = agent.send(None, "Wie lang ist die Kündigungsfrist?")
        until(events, ended)
    assert agent.store.messages(id)[-1]["info"]["quotes"] == [
        {"quote": "Die Kündigungsfrist beträgt drei Monate zum Quartalsende.", "cited": [1],
         "found": True},
        {"quote": "Die Miete beträgt fünf Euro im Jahr", "cited": [1], "found": False},
    ]  # fmt: skip


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
    assert files.SEARCH in _system(True, tools=files.tools(index))["content"]
    assert "search_files" not in _system(True, tools=files.tools())["content"]
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


class Meaning:
    """An embedder whose embeddings point along each concept a text names, in English or German,
    or, for a text that names none, along one of seven other directions, by its checksum; that may
    fail, and whose model may change."""

    CONCEPTS = [("rent", "miete"), ("deposit", "kaution"), ("notice", "kündigung")]

    def __init__(self):
        self.embedded: list[str] = []
        self.fails = False
        self.model, self.form = "meaning", QWEN3

    def available(self) -> bool:
        return True

    def recheck(self) -> None:
        pass

    def _vector(self, text: str) -> bytes:
        import zlib

        from leat.agent.embeddings import _kept

        words, other = fold(text), [0.0] * 7
        vector = [float(any(w in words for w in c)) for c in self.CONCEPTS]
        if not any(vector):
            other[zlib.crc32(words.encode()) % 7] = 1.0
        return _kept(vector + other)

    def passages(self, texts: list[str]) -> list[bytes]:
        from leat.agent.client import EngineError

        if self.fails:
            raise EngineError("down")
        self.embedded += texts
        return [self._vector(t) for t in texts]

    def question(self, text: str) -> bytes:
        return self._vector(text)


def test_meaning(workspace, tmp_path, monkeypatch):
    # a passage found by what it means, as an English question finds a German lease's, beside
    # those found by their words; none that means too little; passages embedded once, in
    # batches, after their files are read, and again as their files change; those the embedder
    # failed to, at the next update
    from leat.agent import index as module

    monkeypatch.setattr(module, "EMBEDDED", 2)
    meaning = Meaning()
    index = Index(tmp_path / "index.db", workspace, embedder=meaning)  # type: ignore[arg-type]
    space = workspace.space("people/1")
    files.write(space, "miete.md", "# Miete\n\nMonatlich 2.500 EUR.\n\n# Kaution\n\nZwei Monate.")
    files.write(space, "notes.txt", "The deposit question came up on Monday.")
    files.write(space, "menu.txt", "Lunch is at noon.")
    index.update("people/1")
    assert len(meaning.embedded) == 4  # each passage, in batches of two
    found = [(f["name"], f["place"]) for f in index.search(space, "How much is the deposit?")]
    assert found == [("notes.txt", ""), ("miete.md", "Kaution")]  # by words and meaning first
    assert [f["name"] for f in index.search(space, "lunch")] == ["menu.txt"]  # by words alone
    assert index.search(space, "What is the weather?") == []  # nothing means it
    files.write(space, "miete.md", "# Kündigung\n\nDrei Monate zum Quartalsende.")
    meaning.fails = True
    index.update("people/1")
    assert index.search(space, "notice period") == []  # not yet embedded
    meaning.fails = False
    index.update("people/1")
    assert [f["place"] for f in index.search(space, "notice period")] == ["Kündigung"]
    count = index._db.execute("SELECT count(*) FROM vectors").fetchone()[0]
    assert count == 3  # the changed file's old passages' vectors gone
    meaning.model, meaning.embedded = "another", []
    assert index.search(space, "notice period") == []  # none of the new model's yet
    index.update("people/1")
    assert len(meaning.embedded) == 3  # each passage again, by the new model
    assert [f["place"] for f in index.search(space, "notice period")] == ["Kündigung"]


class Embeddings:
    """An engine whose embedding model is `model`, which embeds every text alike, and keeps the
    texts it was asked to embed."""

    def __init__(self, model: str):
        self.model, self.asked = model, []

    def embed(self, texts):
        self.asked += texts
        return self.model, [[1.0] + [0.0] * 1023 for _ in texts]


@pytest.mark.parametrize(
    "model, form",
    [("Qwen3-Embedding-0.6B-Q8_0", QWEN3), ("embeddinggemma-2-Q8_0", GEMMA2),
     ("text-embedding-3-small", QWEN3)],
)  # fmt: skip
def test_embedder_forms(model, form):
    # a question and a passage, each embedded as the engine's embedding model was taught, which
    # its name tells, Qwen3-Embedding's by default; the embeddings cut to DIMS dimensions
    engine = Embeddings(model)
    embedder = Embedder(engine)  # type: ignore[arg-type]
    assert embedder.available() and embedder.model == model and embedder.form == form
    engine.asked = []
    assert len(embedder.question("When is the rent due?")) == 4 * 512
    embedder.passages(["Die Miete ist zum 3. fällig."])
    assert engine.asked == [form.question.format("When is the rent due?"),
                            form.passage.format("Die Miete ist zum 3. fällig.")]  # fmt: skip
