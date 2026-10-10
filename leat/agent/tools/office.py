"""Word documents changed as an office changes them, as tools: fill_template fills a template's
fields, and suggest_edits suggests edits as tracked changes, which the user accepts or rejects in
Word. Each works on a copy, in the sandbox, with python-docx, so that a document made to attack its
parser attacks nothing else, and keeps the document's formatting: new text takes the formatting of
the first character of what it replaces, however Word split that among its runs.

A field is written "{{Client name}}", or as Word shows a merge field, "«Client name»"; values are
matched to fields whatever their case and spaces. An edit is of a passage written once in the
document, within a paragraph, which it deletes and puts its replacement after, both marked as
Leat's, with a comment if it has one.
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


# suggests the edits of the JSON at sys.argv[1], each {"find", "replace", "comment"}, to the
# document at sys.argv[2] as tracked changes, and saves it at sys.argv[3]; prints how many times
# each edit's passage was found, as JSON, those found once made
_SUGGEST = (
    _PARAGRAPHS
    + """
import copy, datetime
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.run import Run

edits = json.load(open(sys.argv[1]))
document = Document(sys.argv[2])
when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
taken = [int(v) for e in document.element.iter() if (v := e.get(qn("w:id"))) and v.isdigit()]
ids = iter(range(max(taken, default=0) + 1, 1 << 31))

def split(paragraph, at):
    # splits the run that holds the paragraph's character `at` there, so that a run begins at it
    position = 0
    for run in paragraph.runs:
        n = len(run.text)
        if position < at < position + n:
            text, twin = run.text, copy.deepcopy(run._r)
            run._r.addnext(twin)
            run.text, Run(twin, paragraph).text = text[: at - position], text[at - position:]
            return
        position += n

def covering(paragraph, start, end):
    # the runs that hold the paragraph's text from start to end, and no more
    split(paragraph, start)
    split(paragraph, end)
    runs, position = [], 0
    for run in paragraph.runs:
        if start <= position < end and run.text:
            runs.append(run)
        position += len(run.text)
    return runs

def change(tag):
    # a tracked change of Leat's, made now
    element = OxmlElement(tag)
    for key, value in (("w:id", str(next(ids))), ("w:author", "Leat"), ("w:date", when)):
        element.set(qn(key), value)
    return element

found = []
for edit in edits:
    find, new, note = edit["find"], edit.get("replace") or "", edit.get("comment")
    places = []
    for paragraph in paragraphs(document):
        text, at = "".join(run.text for run in paragraph.runs), 0
        while find and (at := text.find(find, at)) >= 0:
            places.append((paragraph, at))
            at += len(find)
    found.append(len(places))
    if len(places) != 1:
        continue
    paragraph, start = places[0]
    runs = covering(paragraph, start, start + len(find))
    deleted = change("w:del")
    runs[0]._r.addprevious(deleted)
    for run in runs:
        for t in run._r.findall(qn("w:t")):
            t.tag = qn("w:delText")
        deleted.append(run._r)
    marked = runs
    if new:
        inserted, r = change("w:ins"), copy.deepcopy(runs[0]._r)
        for child in [c for c in r if c.tag != qn("w:rPr")]:
            r.remove(child)
        inserted.append(r)
        deleted.addnext(inserted)
        Run(r, paragraph).text = new
        marked = [Run(r, paragraph)]
    if note:
        document.add_comment(marked, note, author="Leat", initials="L")
document.save(sys.argv[3])
print(json.dumps(found))
"""
)


def tools() -> list[Tool]:
    """fill_template and suggest_edits, in their call's conversation's workspace."""
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
        ),
        Tool(
            "suggest_edits",
            "Suggest edits to a Word document as tracked changes, which the user accepts or "
            "rejects in Word: each a passage of it and what replaces it, with a comment that says "
            "why if one; a new document, the original unchanged. Each passage must be written "
            "once in the document, within a paragraph, as it is there",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "the document, a .docx"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "find": {"type": "string", "description": "the passage"},
                                "replace": {
                                    "type": "string",
                                    "description": "what replaces it; nothing to delete it",
                                },
                                "comment": {"type": "string", "description": "why, if it says"},
                            },
                            "required": ["find", "replace"],
                        },
                    },
                    "name": {
                        "type": "string",
                        "description": "the new document's name, as 'Lease - suggested.docx'",
                    },
                },
                "required": ["path", "edits", "name"],
            },  # fmt: skip
            lambda context, path, edits, name: suggest(context, path, edits, name),
        ),
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


def suggest(context: Context, path: str, edits: list[dict[str, Any]], name: str) -> Result:
    space = context.space()
    document = _docx(space, path)
    if not isinstance(edits, list) or not all(isinstance(e, dict) and e.get("find") for e in edits):
        raise ValueError("edits must be a list of objects, each with the passage to find")
    made = space.free(_named(name), folders=True)
    space.path(made).parent.mkdir(parents=True, exist_ok=True)
    found = _run(space, _SUGGEST, edits, document, made)
    done = sum(n == 1 for n in found)
    lines = [f"Made {made}, of {document}: {done} of {len(edits)} edits suggested, as tracked "
             "changes, which the user accepts or rejects in Word."]  # fmt: skip
    for edit, n in zip(edits, found, strict=True):
        if n != 1:
            where = "is not in it" if n == 0 else f"is in it {n} times: give more of it"
            lines.append(f"Not made: “{edit['find'][:80]}” {where}.")
    return Result("\n".join(lines), {"files": [made], "suggested": done, "edits": len(edits)})


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
