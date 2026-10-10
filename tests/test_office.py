"""Word documents changed as an office does, in the sandbox: a template's fields filled. These need
LEAT_SANDBOX, as tests/test_files.py's documents do."""

import json
import os
from pathlib import Path

import pytest

from leat.agent.tools import Context, files, office
from leat.agent.workspace import Workspace
from tests.test_files import documents as needs_sandbox

pytestmark = needs_sandbox
SANDBOX = os.environ.get("LEAT_SANDBOX")


@pytest.fixture
def space(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace", Path(SANDBOX) if SANDBOX else None)


def make(space: Workspace, code: str) -> None:
    ran = files.run(space, f"from docx import Document\nfrom docx.shared import Pt\n{code}")
    assert ran.info["status"] == 0, ran.content


def runs(space: Workspace, name: str) -> list:
    # each paragraph's runs, as their text and whether they are bold, the body's, the tables'
    # and the header's
    ran = files.run(space, f"""
import json
from docx import Document
d = Document({name!r})
ps = [*d.paragraphs, *(p for t in d.tables for r in t.rows for c in r.cells for p in c.paragraphs),
      *d.sections[0].header.paragraphs]
print(json.dumps([[[r.text, bool(r.bold)] for r in p.runs] for p in ps if p.runs]))
""")  # fmt: skip
    return json.loads(ran.content.splitlines()[0])


def test_fill(space):
    # each field filled, however Word split it among runs, with the formatting of its first
    # character; in tables and headers too; a value's lines as lines; the template unchanged
    make(space, """
d = Document()
p = d.add_paragraph("Dear ")
p.add_run("{{Cli").bold = True
p.add_run("ent name}}")
p.add_run(", your fee is «Fee» a month.")
d.add_table(rows=1, cols=1).rows[0].cells[0].text = "Address: {{ address }}"
d.sections[0].header.paragraphs[0].text = "Ref {{REF}}"
d.add_paragraph("Signed: {{Partner}}")
d.save("Letter.docx")
""")  # fmt: skip
    values = {"client name": "Yılmaz Ltd", "Fee": "4.500 EUR", "Address": "Lindenstraße 5\nMünchen",
              "ref": "Y-12", "IBAN": "TR00"}  # fmt: skip
    result = office.fill(Context("c", workspace=space), "Letter.docx", values, "Letter - Yılmaz")
    assert result.content == (
        "Made Letter - Yılmaz.docx, of Letter.docx: 4 fields filled. Left as they were, without "
        "values: Partner. Values of no field there: IBAN.")  # fmt: skip
    assert result.info["files"] == ["Letter - Yılmaz.docx"]
    filled = runs(space, "Letter - Yılmaz.docx")
    assert filled[0] == [["Dear ", False], ["Yılmaz Ltd", True], ["", False],
                         [", your fee is 4.500 EUR a month.", False]]  # fmt: skip
    assert ["Address: Lindenstraße 5\nMünchen", False] in filled[2]
    assert filled[3] == [["Ref Y-12", False]]
    assert filled[1] == [["Signed: {{Partner}}", False]]
    assert runs(space, "Letter.docx")[0][1] == ["{{Cli", True]  # the template as it was


def test_refused(space):
    files.write(space, "notes.txt", "{{x}}")
    with pytest.raises(ValueError, match="not a Word document"):
        office.fill(Context("c", workspace=space), "notes.txt", {}, "x")
    with pytest.raises(FileNotFoundError):
        office.fill(Context("c", workspace=space), "nothing.docx", {}, "x")
    files.write(space, "broken.docx", "not a document")
    with pytest.raises(ValueError, match="it failed"):
        office.fill(Context("c", workspace=space), "broken.docx", {}, "x")
    assert not any(f["name"].startswith(".office") for f in space.files())


def versions(space: Workspace, name: str) -> dict:
    # each paragraph's text with every tracked change accepted, and rejected, its tracked
    # changes' authors, and its comments
    ran = files.run(space, f"""
import json
from docx import Document
from docx.oxml.ns import qn
d = Document({name!r})
def text(p, keep):
    out = []
    for e in p._p.iter():
        inside = {{a.tag for a in e.iterancestors()}}
        if e.tag == qn("w:t") and (keep == "accepted" or qn("w:ins") not in inside):
            out.append(e.text or "")
        if e.tag == qn("w:delText") and keep == "rejected":
            out.append(e.text or "")
    return "".join(out)
print(json.dumps({{
    "accepted": [text(p, "accepted") for p in d.paragraphs],
    "rejected": [text(p, "rejected") for p in d.paragraphs],
    "authors": sorted({{e.get(qn("w:author")) for e in d.element.iter(qn("w:ins"), qn("w:del"))}}),
    "comments": [c.text for c in d.comments],
}}))
""")  # fmt: skip
    return json.loads(ran.content.splitlines()[0])


def test_suggest(space):
    # each edit a tracked change of Leat's, over runs Word split, with its comment; one whose
    # passage is not there, or there twice, not made and said to be; the original unchanged
    make(space, """
d = Document()
p = d.add_paragraph("Either party may end it with ")
p.add_run("ninety").bold = True
p.add_run(" days' notice.")
d.add_paragraph("The rent is due monthly. The rent is fixed.")
d.add_paragraph("Disputes go to the courts of Bursa, in all cases.")
d.save("Lease.docx")
""")  # fmt: skip
    edits = [
        {
            "find": "ninety days'",
            "replace": "thirty days' written",
            "comment": "Shorter, and in writing.",
        },  # fmt: skip
        {"find": ", in all cases", "replace": ""},
        {"find": "The rent", "replace": "Rent"},
        {"find": "a clause not there", "replace": "x"},
    ]
    result = office.suggest(Context("c", workspace=space), "Lease.docx", edits, "Lease - suggested")
    assert result.content == (
        "Made Lease - suggested.docx, of Lease.docx: 2 of 4 edits suggested, as tracked changes, "
        "which the user accepts or rejects in Word.\n"
        "Not made: “The rent” is in it 2 times: give more of it.\n"
        "Not made: “a clause not there” is not in it.")  # fmt: skip
    seen = versions(space, "Lease - suggested.docx")
    assert seen["accepted"] == ["Either party may end it with thirty days' written notice.",
                                "The rent is due monthly. The rent is fixed.",
                                "Disputes go to the courts of Bursa."]  # fmt: skip
    assert seen["rejected"] == ["Either party may end it with ninety days' notice.",
                                "The rent is due monthly. The rent is fixed.",
                                "Disputes go to the courts of Bursa, in all cases."]  # fmt: skip
    assert seen["authors"] == ["Leat"] and seen["comments"] == ["Shorter, and in writing."]
    assert versions(space, "Lease.docx")["authors"] == []
    # the inserted text in the formatting of the first it replaces: bold, as "ninety" was
    assert ["thirty days' written", True] in runs_inserted(space, "Lease - suggested.docx")
    with pytest.raises(ValueError, match="edits must be"):
        office.suggest(Context("c", workspace=space), "Lease.docx", [{"replace": "x"}], "x")
    # an edit's parts by the names a model may give them, as "old" and "new"
    again = [{"old": "Disputes go to", "new": "Disputes are heard by", "reason": "Clearer."}]
    result = office.suggest(Context("c", workspace=space), "Lease.docx", again, "Lease 2")
    assert "1 of 1 edits suggested" in result.content
    assert versions(space, "Lease 2.docx")["comments"] == ["Clearer."]


def runs_inserted(space: Workspace, name: str) -> list:
    ran = files.run(space, f"""
import json
from docx import Document
from docx.oxml.ns import qn
d = Document({name!r})
found = []
for ins in d.element.iter(qn("w:ins")):
    for r in ins.iter(qn("w:r")):
        found.append(["".join(t.text for t in r.iter(qn("w:t"))), r.find(qn("w:rPr")) is not None
                      and r.find(qn("w:rPr")).find(qn("w:b")) is not None])
print(json.dumps(found))
""")  # fmt: skip
    return json.loads(ran.content.splitlines()[0])


def test_translate(space, engine, monkeypatch):
    # each paragraph translated in its place, its style and its first character's formatting
    # kept, in batches; a batch the model answers with too few, asked one by one; one it fails,
    # left as it was and said to be; how far it is told as each batch ends
    from leat.agent.client import Client

    monkeypatch.setattr(office, "READERS", 1)
    monkeypatch.setattr(office, "BATCH", 40)
    make(space, """
d = Document()
d.add_heading("Kira Sözleşmesi", level=1)
p = d.add_paragraph()
p.add_run("Kira").bold = True
p.add_run(" aylık 40.000 TL.")
d.add_paragraph("")
d.add_table(rows=1, cols=1).rows[0].cells[0].text = "Depozito iki aylık kiradır."
d.save("Kira.docx")
""")  # fmt: skip
    engine.replies.put([{"content": '["Lease Agreement"]'}])  # too few: asked one by one
    engine.replies.put([{"content": '["Lease Agreement"]'}])
    engine.replies.put([{"content": '["Rent is 40,000 TL a month."]'}])
    engine.replies.put([{"content": "Sorry, I cannot."}])  # the table's, failed
    told: list[dict] = []
    context = Context("c", workspace=space, progress=told.append)
    result = office.translate(
        Client(engine.url), context, "Kira.docx", "English", "Lease (English)"
    )
    assert result.content == ("Made Lease (English).docx, Kira.docx in English, 3 paragraphs "
                              "translated. 1 could not be, as the model failed, and are as they "
                              "were.")  # fmt: skip
    assert told == [{"done": 0, "total": 2}, {"done": 1, "total": 2}, {"done": 2, "total": 2}]
    asked = [json.loads(r["messages"][1]["content"]) for r in engine.requests]
    assert asked == [["Kira Sözleşmesi", "Kira aylık 40.000 TL."], ["Kira Sözleşmesi"],
                     ["Kira aylık 40.000 TL."], ["Depozito iki aylık kiradır."]]  # fmt: skip
    assert "into English" in engine.requests[0]["messages"][0]["content"]
    ran = files.run(space, """
import json
from docx import Document
d = Document("Lease (English).docx")
print(json.dumps([[p.style.name, [[r.text, bool(r.bold)] for r in p.runs]] for p in d.paragraphs]))
print(json.dumps(d.tables[0].rows[0].cells[0].text))
""")  # fmt: skip
    body, cell = (json.loads(line) for line in ran.content.splitlines()[:2])
    assert body[0] == ["Heading 1", [["Lease Agreement", False]]]
    assert body[1] == ["Normal", [["Rent is 40,000 TL a month.", True], ["", False]]]
    assert cell == "Depozito iki aylık kiradır."  # the batch it failed, as it was


def test_translate_stopped(space, engine):
    # a translation stopped saves nothing
    import threading

    from leat.agent.client import Client

    make(space, 'd = Document()\nd.add_paragraph("Merhaba")\nd.save("a.docx")')
    stopped = threading.Event()
    stopped.set()
    context = Context("c", workspace=space, stopped=stopped)
    result = office.translate(Client(engine.url), context, "a.docx", "English", "b")
    assert result.content.startswith("Stopped before the whole document was translated")
    assert [f["name"] for f in space.files()] == ["a.docx"]


def test_named():
    # a Word document's name, its kind added, or in place of another document's
    assert office._named("Letter") == "Letter.docx" and office._named("a.DOCX") == "a.DOCX"
    assert office._named("Letter - March.pdf") == "Letter - March.docx"
    assert office._named("Notes v1.2") == "Notes v1.2.docx"
