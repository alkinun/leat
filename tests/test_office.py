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
    values = {"client name": "Yılmaz Ltd", "Fee": "4.500 TL", "Address": "Atatürk Cd. 5\nBursa",
              "ref": "Y-12", "IBAN": "TR00"}  # fmt: skip
    result = office.fill(Context("c", workspace=space), "Letter.docx", values, "Letter - Yılmaz")
    assert result.content == (
        "Made Letter - Yılmaz.docx, of Letter.docx: 4 fields filled. Left as they were, without "
        "values: Partner. Values of no field there: IBAN.")  # fmt: skip
    assert result.info["files"] == ["Letter - Yılmaz.docx"]
    filled = runs(space, "Letter - Yılmaz.docx")
    assert filled[0] == [["Dear ", False], ["Yılmaz Ltd", True], ["", False],
                         [", your fee is 4.500 TL a month.", False]]  # fmt: skip
    assert ["Address: Atatürk Cd. 5\nBursa", False] in filled[2]
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
