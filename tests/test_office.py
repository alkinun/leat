"""Documents made and changed as an office does, in the sandbox: made of markdown, a template's
fields filled, and edits suggested. Those made need LEAT_SANDBOX, as tests/test_files.py's
documents do."""

import json
import os
from pathlib import Path

import pytest

from leat.agent.tools import Context, files, office
from leat.agent.workspace import Workspace
from tests.test_files import documents as needs_sandbox

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


@needs_sandbox
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


@needs_sandbox
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


@needs_sandbox
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


def test_named():
    # a Word document's name, its kind added, or in place of another document's
    assert office._named("Letter") == "Letter.docx" and office._named("a.DOCX") == "a.DOCX"
    assert office._named("Letter - March.pdf") == "Letter - March.docx"
    assert office._named("Notes v1.2") == "Notes v1.2.docx"


def test_blocks():
    # a paragraph's lines kept, as an address's; headings, lists by their depths, labelled as
    # written, a quote, code, a rule and a table, its columns aligned as its divider says; a
    # document fenced whole read within its fence
    letter = """# Fees

Emma Hartley
14 Jersey Street

1. First
   still the first
2. Second
   - within it
- after

> Paid by transfer.

```
x = 1
```

---
| Item | Total |
|:--|--:|
| Fee | £1,850 |
| VAT |"""
    assert office.blocks(letter) == [
        ["heading", 1, [["Fees", ""]]],
        ["paragraph", [["Emma Hartley\n14 Jersey Street", ""]]],
        ["item", 0, "1.", [["First\nstill the first", ""]]],
        ["item", 0, "2.", [["Second", ""]]],
        ["item", 1, "–", [["within it", ""]]],
        ["item", 0, "•", [["after", ""]]],
        ["quote", [["Paid by transfer.", ""]]],
        ["code", "x = 1"],
        ["rule"],
        ["table", [[["Item", ""]], [["Total", ""]]], [[[["Fee", ""]], [["£1,850", ""]]],
                                                      [[["VAT", ""]]]], ["LEFT", "RIGHT"]],
    ]  # fmt: skip
    assert office.blocks("```markdown\n# A\n\nB<br>C\n```") == [
        ["heading", 1, [["A", ""]]], ["paragraph", [["B\nC", ""]]]]  # fmt: skip
    assert office.blocks("  \n\n") == []


def test_spans():
    # bold, italic and code, which a delimiter opens before a non-space and closes after one, an
    # _ not within a word; one that nothing closes, and arithmetic, as text; links as their text
    assert office.spans("Fee **£1,850** *net*, `code`") == [
        ["Fee ", ""], ["£1,850", "b"], [" ", ""], ["net", "i"], [", ", ""], ["code", "c"],
    ]  # fmt: skip
    assert office.spans("***both*** and _it_") == [["both", "bi"], [" and ", ""], ["it", "i"]]
    assert office.spans("5 * 3, snake_case_name, **open") == [
        ["5 * 3, snake_case_name, **open", ""]]  # fmt: skip
    assert office.spans(r"\*not\* [terms](https://x.io), [https://x.io](https://x.io)") == [
        ["*not* terms (https://x.io), https://x.io", ""]]  # fmt: skip


@needs_sandbox
def test_make(space):
    # a Word document in Leat's style, its headings styled, its bold bold, an address's lines as
    # lines and a table's heading bold; in a template's header and footer, its body left out; a
    # PDF of every letter; a PDF of a template, and a document of nothing, refused
    context = Context("c", workspace=space)
    letter = (
        "# Fees\n\nEmma Hartley\n14 Jersey Street\n\nThe fee is **£1,850** – Kündigung.\n\n"
        "| Item | Total |\n|---|--:|\n| Fee | £1,850 |"
    )
    assert office.make(context, "Fees - March", letter).content == "Made Fees - March.docx."
    made = files.run(space, """
import json
from docx import Document
d = Document("Fees - March.docx")
shown = [[p.style.name, [[r.text, bool(r.bold)] for r in p.runs]] for p in d.paragraphs]
cells = [[[r.text, bool(r.bold)] for r in c.paragraphs[0].runs] for c in d.tables[0]._cells]
print(json.dumps(shown + cells))
""")  # fmt: skip
    shown = json.loads(made.content.splitlines()[0])
    assert shown[:3] == [
        ["Heading 1", [["Fees", False]]],
        ["Normal", [["Emma Hartley\n", False], ["14 Jersey Street", False]]],  # a break
        ["Normal", [["The fee is ", False], ["£1,850", True], [" – Kündigung.", False]]],
    ]  # fmt: skip
    cells = [[["Item", True]], [["Total", True]], [["Fee", False]], [["£1,850", False]]]
    assert shown[-4:] == cells
    make(space, """
d = Document()
d.sections[0].header.paragraphs[0].text = "Whitfield & Partners"
d.sections[0].footer.paragraphs[0].text = "1 King Street"
d.add_paragraph("A letter of last year")
d.save("Letterhead.docx")
""")  # fmt: skip
    result = office.make(context, "Fees - letterhead.docx", letter, "Letterhead.docx")
    assert result.content == (
        "Made Fees - letterhead.docx, in the styles, headers and footers of Letterhead.docx."
    )
    assert result.info == {"files": ["Fees - letterhead.docx"]}
    headed = runs(space, "Fees - letterhead.docx")
    assert headed[-1] == [["Whitfield & Partners", False]]
    assert ["A letter of last year", False] not in [run for p in headed for run in p]
    assert office.make(context, "Fees.pdf", letter).content == "Made Fees.pdf."
    read = files.read(space, "Fees.pdf").content
    assert "£1,850" in read and "Kündigung" in read and "14 Jersey Street" in read
    with pytest.raises(ValueError, match="Word document alone"):
        office.make(context, "Fees.pdf", letter, "Letterhead.docx")
    with pytest.raises(ValueError, match="no text"):
        office.make(context, "Empty", "\n\n")
