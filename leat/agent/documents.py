"""Documents: the text of a workspace's files, and where in it each place is.

A PDF, or a Word, Excel or PowerPoint file, is read as markdown, which keeps its headings, lists
and tables, as markitdown reads one, in the sandbox, so that a file made to attack its parser
attacks nothing else; a text file is read as it is. A document's text marks its places, by which a
passage of it is cited: a PDF's pages, a presentation's slides, a workbook's sheets, and any
other's sections, by their headings.
"""

import re

from leat.agent.workspace import Workspace

# the kinds of file read by parsing them, in the sandbox
DOCUMENTS = (".pdf", ".docx", ".xlsx", ".pptx")
# reads the document at sys.argv[1] as markdown, which keeps its headings, lists and tables, as
# markitdown reads one, in the sandbox, with its libraries
_EXTRACT = """
import sys
path = sys.argv[1]
kind = path.rsplit(".", 1)[-1].lower()

def table(rows):
    rows = [["" if cell is None else " ".join(str(cell).split()) for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return
    width = max(len(row) for row in rows)
    print()
    for i, row in enumerate(rows):
        print("| " + " | ".join(row + [""] * (width - len(row))) + " |")
        if i == 0:
            print("|" + " --- |" * width)
    print()

if kind == "pdf":
    from pypdf import PdfReader
    for i, page in enumerate(PdfReader(path).pages, 1):
        print(f"## Page {i}\\n\\n{page.extract_text() or ''}\\n")
elif kind == "docx":
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    document = Document(path)
    for block in document.element.body.iterchildren():
        if block.tag.endswith("}tbl"):
            table([[cell.text for cell in row.cells] for row in Table(block, document).rows])
        elif block.tag.endswith("}p"):
            paragraph = Paragraph(block, document)
            style, text = paragraph.style.name if paragraph.style else "", paragraph.text
            if not text.strip():
                continue
            if style == "Title":
                print(f"# {text}\\n")
            elif style.startswith("Heading") and style[-1:].isdigit():
                print("#" * (int(style[-1]) + 1) + f" {text}\\n")
            elif style.startswith("List Number"):
                print(f"1. {text}")
            elif style.startswith("List"):
                print(f"- {text}")
            else:
                print(f"{text}\\n")
elif kind == "xlsx":  # each cell's value, or its formula if no program has computed it yet
    from openpyxl import load_workbook
    values, formulas = (load_workbook(path, read_only=True, data_only=d) for d in (True, False))
    for sheet, written in zip(values, formulas):
        print(f"## {sheet.title}\\n")
        table([[raw[i] if value is None else value for i, value in enumerate(row)]
               for row, raw in zip(sheet.iter_rows(values_only=True),
                                   written.iter_rows(values_only=True))])
elif kind == "pptx":
    from pptx import Presentation
    for i, slide in enumerate(Presentation(path).slides, 1):
        title = slide.shapes.title.text if slide.shapes.title is not None else ""
        print(f"## Slide {i}: {title}\\n" if title else f"## Slide {i}\\n")
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            if shape.has_table:
                table([[cell.text for cell in row.cells] for row in shape.table.rows])
            elif shape.has_text_frame and shape.text_frame.text.strip():
                print(shape.text_frame.text + "\\n")
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            print(f"Notes: {slide.notes_slide.notes_text_frame.text}\\n")
"""
SNIFF = 8192  # bytes of a file whose having no NUL makes it text
# where each kind of document's text begins a place, and what the place is called
_PLACES = {
    ".pdf": (re.compile(r"^## Page (\d+)$", re.M), "page {}"),
    ".pptx": (re.compile(r"^## Slide (\d+)(?:: .*)?$", re.M), "slide {}"),
    ".xlsx": (re.compile(r"^## (.+)$", re.M), "sheet {}"),
}
_HEADING = re.compile(r"^#{1,6} (.+)$", re.M)  # a section's, of any other text
PLACE = 60  # characters of a place's name at most, as a long heading's


def text(workspace: Workspace, name: str) -> str:
    """A file's text: a document's as markdown, read in the sandbox, or a text file's as it is.
    Raises FileNotFoundError if there is no such file, ValueError if it cannot be read as text."""
    file = workspace.path(name)
    if not file.is_file():
        raise FileNotFoundError(f"there is no file {name}")
    if file.suffix.lower() in DOCUMENTS:
        inside = f"/workspace/{file.relative_to(workspace.root).as_posix()}"
        ran = workspace.run(_EXTRACT, inside, kept=None)
        if ran.status != 0:
            why = (ran.output.strip().splitlines() or ["it stopped"])[-1]
            raise ValueError(f"{name} could not be read: {why}")
        return ran.output
    data = file.read_bytes()
    if b"\0" in data[:SNIFF]:
        raise ValueError(f"{name} is not a file of text")
    return data.decode("utf-8", "replace")


def readable(name: str, head: bytes) -> bool:
    """Whether a file has text to read, by its name and its first bytes: a document, or text."""
    return name.lower().endswith(DOCUMENTS) or b"\0" not in head[:SNIFF]


def places(name: str, text: str) -> list[tuple[int, str]]:
    """Where in a file's text each of its places begins, and its name, as "page 3": a document's
    pages, slides or sheets, or any other's sections; none of a text without them."""
    pattern, called = _PLACES.get(name[name.rfind(".") :].lower(), (_HEADING, "{}"))
    found = []
    for match in pattern.finditer(text):
        place = called.format(" ".join(match[1].split()))
        found.append((match.start(), place if len(place) <= PLACE else place[: PLACE - 1] + "…"))
    return found
