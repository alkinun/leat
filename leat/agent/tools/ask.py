"""Asking every file the same: ask_files has a reader, the model in a conversation of its own, read
each of the conversation's files, or those the call names, and answer for it, as leat.agent.tools.
web's reader answers for a page. The answers make a table, a row a file, each citing its file,
which the call returns and saves as a spreadsheet. A question of every file, as which contracts
renew themselves, or each invoice's date and total, is so answered whole, where a search finds
some of the files' passages alone.

The readers run READERS at once, which the engine batches, and the call tells how far it is as
each ends; once the turn is stopped, it reads no more. A file longer than a reader reads is read as
its start and the passages of it the index finds likeliest to answer.
"""

import base64
import concurrent.futures
import json
import mimetypes
import re
import uuid
from typing import Any

from leat.agent import documents
from leat.agent.client import Client, EngineError
from leat.agent.context import picture
from leat.agent.index import Index
from leat.agent.tools import Context, Result, Tool
from leat.agent.workspace import Workspace

READERS = 4  # files read at once, as many as leat serve's slots by default
FILES = 500  # files a call asks at most
READ = 24000  # characters of a file a reader reads at most
START = 4000  # of a longer one's, its start, before the passages likeliest to answer
ANSWER = 500  # tokens of a reader's answer at most
SHOWN = 50  # rows of the table the model reads; the spreadsheet holds every one
CELL = 300  # characters of an answer the model reads in the table, at most
# a reader's instructions, of the answers' keys
READING = """\
You read one file for Leat, an assistant, who asks the same of many files. Answer from this file \
alone, as a JSON object of these keys: {keys}. Each value is short: what the file says, a number \
or a date as it is written there, or "" where the file does not say. Write the JSON alone."""
# saves the table of the JSON file at sys.argv[1], a list of rows, the first its heading, as the
# spreadsheet sys.argv[2]: its heading bold, frozen and filtered, its columns as wide as they need
_SAVE = """
import json, sys
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
rows = json.load(open(sys.argv[1], encoding="utf-8"))
book = Workbook()
sheet = book.active
sheet.title = "Answers"
for row in rows:
    sheet.append(row)
for cell in sheet[1]:
    cell.font = Font(bold=True)
for row in sheet.iter_rows(min_row=2):
    for cell in row:
        cell.alignment = Alignment(vertical="top", wrap_text=True)
sheet.freeze_panes = "A2"
sheet.auto_filter.ref = sheet.dimensions
for column in sheet.columns:
    width = max(len(str(cell.value or "")) for cell in column)
    sheet.column_dimensions[column[0].column_letter].width = min(60, max(12, width + 2))
book.save(sys.argv[2])
"""


def tools(reader: Client, index: Index | None = None) -> list[Tool]:
    """ask_files, whose readers the model `reader` serves, reading the files' text as `index`
    keeps it, if given."""
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "what to find out of each file"},
            "columns": {
                "type": "array", "items": {"type": "string"},
                "description": "the table's columns, each a thing to find, as 'Date' and 'Total'; "
                "without them, one, the answer",
            },
            "files": {
                "type": "array", "items": {"type": "string"},
                "description": "the files to ask, by their paths, a folder's every file by "
                "its; without them, every file",
            },
            "name": {"type": "string", "description": "the spreadsheet's name, as 'Invoices.xlsx'"},
        },
        "required": ["question"],
    }  # fmt: skip
    return [
        Tool(
            "ask_files",
            "Ask each of the workspace's files the same question, one by one, for a question of "
            "every file, as each invoice's date and total, or which contracts renew themselves: "
            "a table of the answers, a row a file, each numbered, saved as a spreadsheet too",
            parameters,
            lambda context, question, columns=None, files=None, name=None: ask(
                reader, index, context, question, columns, files, name
            ),
        )
    ]


def ask(
    reader: Client, index: Index | None, context: Context, question: str,
    columns: list[str] | None = None, files: list[str] | None = None, name: str | None = None,
) -> Result:  # fmt: skip
    space = context.space()
    keys = [" ".join(c.split()) for c in columns or [] if c.strip()] or ["Answer"]
    names = _named_files(space, files)
    if not names:
        return Result("There are no files to ask.", {"question": question, "total": 0})
    if len(names) > FILES:
        raise ValueError(f"a call asks {FILES} files at most, not {len(names)}: name fewer")

    def read(named: str) -> dict[str, str] | None:  # a file's answers, none once stopped
        if context.stopped.is_set():
            return None
        return _answer(reader, index, space, named, question, keys, context.person)

    rows: list[dict[str, str] | None] = [None] * len(names)
    done = 0
    context.progress({"done": 0, "total": len(names)})
    with concurrent.futures.ThreadPoolExecutor(READERS) as pool:
        futures = {pool.submit(read, named): i for i, named in enumerate(names)}
        for future in concurrent.futures.as_completed(futures):
            if (row := future.result()) is not None:
                rows[futures[future]] = row
                done += 1
                context.progress({"done": done, "total": len(names)})
    answered = [(named, row) for named, row in zip(names, rows, strict=True) if row is not None]
    sources = []
    for named, _ in answered:
        n = context.cite(f"file:{named}", named)
        sources.append({"n": n, "url": f"file:{named}", "title": named, "file": named})
    table = [["File", *keys], *([named, *(row[k] for k in keys)] for named, row in answered)]
    saved, why = _save(space, table, name or _named(question))
    shown = [[f"{named} [{s['n']}]" if s["n"] else named, *(row[k][:CELL] for k in keys)]
             for (named, row), s in zip(answered[:SHOWN], sources, strict=False)]  # fmt: skip
    said = _markdown([["File", *keys], *shown])
    if len(answered) > SHOWN:
        said += f"\n\n(The first {SHOWN} rows of {len(answered)}.)"
    if len(answered) < len(names):
        said += f"\n\n(Stopped after {len(answered)} of the {len(names)} files.)"
    said += f"\n\nSaved as {saved}." if saved else f"\n\nIt could not be saved: {why}"
    info = {"question": question, "columns": keys, "done": done, "total": len(names)}
    return Result(said, info | {"results": sources} | ({"files": [saved]} if saved else {}))


def _answer(
    reader: Client, index: Index | None, space: Workspace, name: str, question: str,
    keys: list[str], person: int | None,
) -> dict[str, str]:  # fmt: skip
    # a file's answers, by their keys, as a reader reads it, an image as it sees it; why not, in
    # the first, if it cannot
    asked = f"The question: {question}\n\nThe file, {name}"
    content: str | list[dict[str, Any]]
    try:
        if picture(name):
            content = _image(space, name, f"{asked}, is this image.")
        else:
            text = index.text(space, name) if index else documents.text(space, name)
            if not documents.said(text):  # as a scanned PDF's, while no model reads images
                return _row(keys, "(It holds no text: a scan, which only a model that sees images "
                            "reads)")  # fmt: skip
            if len(text) > READ:
                text = _excerpt(index, space, name, text, f"{question} {' '.join(keys)}")
            content = f"{asked}:\n\n{text}"
    except (OSError, ValueError, RuntimeError) as e:
        return _row(keys, f"(It could not be read: {e})")
    system = {
        "role": "system",
        "content": READING.format(keys=json.dumps(keys, ensure_ascii=False)),
    }
    body = {
        "messages": [system, {"role": "user", "content": content}],
        "max_tokens": ANSWER,
        "temperature": 0,
        "reasoning_effort": "none",
    }
    try:
        said = reader.reply(body, person)["content"].strip()
    except EngineError as e:
        return _row(keys, f"(It could not be asked: {e})")
    return _parsed(said, keys)


def _row(keys: list[str], why: str) -> dict[str, str]:
    # a row of a file that could not be asked, why in its first column
    return {k: why if i == 0 else "" for i, k in enumerate(keys)}


def _image(space: Workspace, name: str, text: str) -> list[dict[str, Any]]:
    # a message's content of an image of the space's, as a data: URL, and text after it
    kind = mimetypes.guess_type(name)[0] or "image/png"
    url = f"data:{kind};base64,{base64.b64encode(space.path(name).read_bytes()).decode()}"
    return [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]


def _named_files(space: Workspace, named: list[str] | None) -> list[str]:
    # the files a call names, each of its folders' as their names' order has them, or without
    # any named every file of the space's to ask. Raises FileNotFoundError for one that is not
    every = [f["name"] for f in space.files() if _askable(space, f["name"])]
    if not named:
        return sorted(every)
    found: list[str] = []
    for name in named:
        path = space.path(name)
        if path.is_dir():
            folder = path.relative_to(space.root).as_posix() + "/"
            found += sorted(f for f in every if f.startswith(folder) or folder == "./")
        elif path.is_file():
            found.append(path.relative_to(space.root).as_posix())
        else:
            raise FileNotFoundError(f"there is no file {name}")
    return list(dict.fromkeys(found))


def _askable(space: Workspace, name: str) -> bool:
    # whether a file is one to ask: a document, text, or an image, which a reader sees
    return picture(name) or documents.readable(space, name)


def _excerpt(index: Index | None, space: Workspace, name: str, text: str, query: str) -> str:
    # a long file's start, and the passages after it the index finds likeliest to answer the
    # query, in the file's order, READ characters in all at most
    found = index.search(space, query, 20, name) if index else []
    parts, room = [text[:START]], READ - START
    for passage in sorted(found, key=lambda p: p["start"]):
        if passage["start"] >= START and len(passage["text"]) < room:
            parts.append(passage["text"])
            room -= len(passage["text"])
    return "\n\n[…]\n\n".join(parts)


def _parsed(said: str, keys: list[str]) -> dict[str, str]:
    # a reader's answers, by their keys, of the JSON object it wrote, found by its keys whatever
    # their case; of an answer that is not one, the whole of it in the first
    try:
        found = json.loads(said[said.index("{") : said.rindex("}") + 1])
        assert isinstance(found, dict)
    except (ValueError, AssertionError):
        return {k: " ".join(said.split()) if i == 0 else "" for i, k in enumerate(keys)}
    folded = {str(k).casefold(): v for k, v in found.items()}
    answers = {}
    for key in keys:
        value = folded.get(key.casefold(), "")
        answers[key] = " ".join(str(value if value is not None else "").split())
    return answers


def _save(space: Workspace, table: list[list[str]], name: str) -> tuple[str | None, str]:
    # saves a table as a spreadsheet of the space's, by a name free there, in the sandbox, which
    # has openpyxl; returns its name, or None and why not
    if space.environment is None:
        return None, "the sandbox has no spreadsheet library"
    name = space.free(name if name.lower().endswith(".xlsx") else f"{name}.xlsx")
    rows = f".ask-{uuid.uuid4().hex[:8]}.json"
    space.path(rows).write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    try:
        ran = space.run(_SAVE, f"/workspace/{rows}", f"/workspace/{name}")
    finally:
        space.path(rows).unlink(missing_ok=True)
    if ran.status != 0:
        return None, (ran.output.strip().splitlines() or ["it stopped"])[-1]
    return name, ""


def _named(question: str) -> str:
    # a spreadsheet's name of a question's first words
    words = re.sub(r"[^\w\s'-]", "", question).split()[:6]
    return " ".join(words).capitalize() or "Answers"


def _markdown(rows: list[list[str]]) -> str:
    # rows as a markdown table, the first its heading
    def cell(text: str) -> str:
        return " ".join(text.split()).replace("|", "\\|") or " "

    lines = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join([lines[0], "|" + " --- |" * len(rows[0]), *lines[1:]])
