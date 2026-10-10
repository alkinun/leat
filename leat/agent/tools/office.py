"""Word documents changed as an office changes them, as tools: fill_template fills a template's
fields. Each works on a copy, in the sandbox, with python-docx, so that a document made to attack
its parser attacks nothing else, and keeps the document's formatting: a field's value takes the
formatting of the field's first character, however Word split the field among its runs.

A field is written "{{Client name}}", or as Word shows a merge field, "«Client name»"; values are
matched to fields whatever their case and spaces.
"""

import json
import uuid
from pathlib import PurePosixPath
from typing import Any

from leat.agent.tools import Context, Result, Tool
from leat.agent.workspace import Workspace

# the paragraphs of the document at sys.argv[2], its body's, its tables', its headers' and
# footers', in which a function `change` of each paragraph's text and of the spans of its runs
# changes the text; defined for the scripts below, which define `change`
_PARAGRAPHS = """
import json, re, sys
from docx import Document

def paragraphs(document):
    def within(parent):
        for paragraph in parent.paragraphs:
            yield paragraph
        for table in getattr(parent, "tables", []):
            for row in table.rows:
                for cell in row.cells:
                    yield from within(cell)
    yield from within(document)
    for section in document.sections:
        for part in (section.header, section.footer, section.first_page_header,
                     section.first_page_footer, section.even_page_header, section.even_page_footer):
            if not part.is_linked_to_previous:
                yield from within(part)

def replace(paragraph, start, end, new):
    # replaces the paragraph's text from start to end, over its runs, with new, in the run where
    # it starts, which keeps that run's formatting
    runs, at = paragraph.runs, 0
    spans = []
    for run in runs:
        spans.append((at, at + len(run.text)))
        at += len(run.text)
    first = next(i for i, (a, b) in enumerate(spans) if a <= start < b)
    last = next(i for i, (a, b) in enumerate(spans) if a < end <= b)
    head = runs[first].text[: start - spans[first][0]]
    tail = runs[last].text[end - spans[last][0]:]
    for i in range(first + 1, last):
        runs[i].text = ""
    if first == last:
        runs[first].text = head + new + tail
    else:
        runs[first].text, runs[last].text = head + new, tail
"""
# fills the fields of the document at sys.argv[2] with the values of the JSON at sys.argv[1], and
# saves it at sys.argv[3]; prints the fields filled, those without a value, and the values of no
# field, as JSON
_FILL = (
    _PARAGRAPHS
    + """
given = json.load(open(sys.argv[1]))
values = {" ".join(k.split()).casefold(): str(v) for k, v in given.items()}
document = Document(sys.argv[2])
FIELD = re.compile(r"\\{\\{\\s*([^{}]+?)\\s*\\}\\}|«\\s*([^«»]+?)\\s*»")
filled, missing = set(), set()
for paragraph in paragraphs(document):
    text = "".join(run.text for run in paragraph.runs)
    for match in reversed(list(FIELD.finditer(text))):
        name = " ".join((match[1] or match[2]).split())
        if name.casefold() in values:
            replace(paragraph, match.start(), match.end(), values[name.casefold()])
            filled.add(name)
        else:
            missing.add(name)
document.save(sys.argv[3])
used = {name.casefold() for name in filled}
unused = sorted(k for k in given if " ".join(k.split()).casefold() not in used)
print(json.dumps({"filled": sorted(filled), "missing": sorted(missing - filled), "unused": unused},
                 ensure_ascii=False))
"""
)


def tools() -> list[Tool]:
    """fill_template, in its call's conversation's workspace."""
    return [
        Tool(
            "fill_template",
            "Fill a Word template's fields, written {{Client name}} or «Client name», with "
            "values, keeping its formatting: a new document, the template unchanged; says which "
            "fields were filled, and which had no value. For many, call it once for each",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "the template, a .docx"},
                    "values": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "each field's value, by its name",
                    },
                    "name": {
                        "type": "string",
                        "description": "the new document's name, as 'Letter - Yılmaz Ltd.docx'",
                    },
                },
                "required": ["path", "values", "name"],
            },  # fmt: skip
            lambda context, path, values, name: fill(context, path, values, name),
        )
    ]


def fill(context: Context, path: str, values: dict[str, Any], name: str) -> Result:
    space = context.space()
    template = _docx(space, path)
    made = space.free(_named(name), folders=True)
    space.path(made).parent.mkdir(parents=True, exist_ok=True)
    said = _run(space, _FILL, values, template, made)
    filled, missing, unused = said["filled"], said["missing"], said["unused"]
    lines = [f"Made {made}, of {template}: {len(filled)} fields filled."]
    if missing:
        lines.append(f"Left as they were, without values: {', '.join(missing)}.")
    if unused:
        lines.append(f"Values of no field there: {', '.join(unused)}.")
    if not filled and not missing:
        lines.append("The template has no fields, written {{Name}} or «Name».")
    return Result(" ".join(lines), {"files": [made]} | said)


def _docx(space: Workspace, path: str) -> str:
    # a Word document's name in the space. Raises FileNotFoundError, or ValueError of another kind
    file = space.path(path)
    if not file.is_file():
        raise FileNotFoundError(f"there is no file {path}")
    if file.suffix.lower() != ".docx":
        raise ValueError(f"{path} is not a Word document, a .docx")
    return file.relative_to(space.root).as_posix()


def _named(name: str) -> str:
    # a name of a Word document, as given or with its kind added
    return name if PurePosixPath(name).suffix.lower() == ".docx" else f"{name}.docx"


def _run(space: Workspace, script: str, given: Any, *names: str) -> dict[str, Any]:
    # runs a script in the sandbox, of the JSON of `given`, in a hidden file of the space's, and of
    # files of the space's; returns what it printed last, as JSON. Raises ValueError if it fails
    inside = f".office-{uuid.uuid4().hex[:8]}.json"
    space.path(inside).write_text(json.dumps(given, ensure_ascii=False), encoding="utf-8")
    try:
        ran = space.run(script, f"/workspace/{inside}", *(f"/workspace/{n}" for n in names))
    finally:
        space.path(inside).unlink(missing_ok=True)
    lines = ran.output.strip().splitlines() or ["it printed nothing"]
    if ran.status != 0:
        raise ValueError(f"it failed: {lines[-1]}")
    return json.loads(lines[-1])
