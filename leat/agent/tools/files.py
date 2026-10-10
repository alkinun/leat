"""The workspace's files, as tools: reading them, searching them, writing and editing them, and
running Python among them in the sandbox. read reads the skills too, the guides to making
documents, at skills/. With an index, a file's text is read as the index keeps it, and the files
are searched there, each passage found numbered, across the conversation, for the model to cite."""

from pathlib import Path
from typing import Any

from leat.agent import documents
from leat.agent.context import picture
from leat.agent.index import Index
from leat.agent.tools import Context, Result, Tool, strings
from leat.agent.workspace import Workspace

READ = 12000  # characters of a file read at once
LISTED = 20  # files a search that finds nothing names, the first by name
SKILLS = Path(__file__).parent.parent / "skills"
LIBRARIES = "python-docx, openpyxl, python-pptx, fpdf2, pypdf, matplotlib, pandas"


def tools(index: Index | None = None) -> list[Tool]:
    """read, write, edit and run, each in its call's conversation's workspace, and with an index
    search_files, of the index's passages."""
    start = {
        "type": "integer",
        "description": "the character to start from, to read on in a long file",
    }
    path = {
        "type": "string",
        "description": "the file's or folder's path in the workspace, as 'notes.txt'",
    }
    reading = {"type": "object", "properties": {"path": path, "start": start}, "required": ["path"]}
    return [
        Tool(
            "read",
            "Read a file in the workspace as text, PDF, Word, Excel and PowerPoint files too, or "
            "list a folder's files, '.' for the whole workspace",
            reading,
            lambda context, path, start=0: read(context.space(), path, start, index),
        ),
        Tool(
            "write",
            "Write a text file in the workspace, replacing any file of its name",
            strings(path="the file's path in the workspace", content="the file's text"),
            lambda context, path, content: write(context.space(), path, content),
        ),
        Tool(
            "edit",
            "Replace a passage of a text file in the workspace; a Word document's are changed by "
            "suggest_edits",
            strings(
                path="the file's path in the workspace",
                old="the passage, exactly as it is in the file",
                new="the text that replaces it",
            ),  # fmt: skip
            lambda context, path, old, new: edit(context.space(), path, old, new),
        ),
        Tool(
            "run",
            "Run Python code in the workspace, in a sandbox without the internet. Returns what it "
            "prints, and the files it made or changed",
            strings(code=f"Python 3 code; its libraries are {LIBRARIES}"),
            lambda context, code: run(context.space(), code),
        ),
    ] + ([Tool(
        "search_files",
        "Search the workspace's files, documents too, by their words and what they mean. Returns "
        "the passages likeliest to answer, each numbered, with its file and where in it",
        strings(query="what to find, in the words a passage would use"),
        lambda context, query: search(index, context, query),
    )] if index is not None else [])  # fmt: skip


def read(
    workspace: Workspace, path: str, start: int | str = 0, index: Index | None = None
) -> Result:
    skill = path.startswith("skills/")
    file = _skill(path) if skill else workspace.path(path)
    start = max(0, int(start))
    if file.is_dir():
        names = sorted(f"{p.name}/" if p.is_dir() else p.name for p in file.iterdir())
        listed = "\n".join(name for name in names if not name.startswith("."))
        return Result(listed or "The folder is empty.", {"file": path})
    if not file.is_file():
        raise workspace.missing(path)
    if picture(path):  # which the agent shows the model, if it sees images
        return Result(f"The image {path}.", {"file": path, "images": [path]})
    if file.suffix.lower() in (".heic", ".heif"):
        raise ValueError(f"{path} is an image of a kind you cannot see: convert it to a PNG")
    if skill:
        text = file.read_text(encoding="utf-8")
    else:  # read on in parts, below
        text = index.text(workspace, path) if index else documents.text(workspace, path)
    part = text[start : start + READ]
    if (end := start + len(part)) < len(text):
        part += f"\n\n(characters {start} to {end} of {len(text)}; read on from start={end})"
    return Result(part or "The file is empty.", {"file": path})


def search(index: Index, context: Context, query: str) -> Result:
    # the passages of the conversation's files likeliest to say what the query asks, each
    # numbered as a source, by its file and place: "file:<name>#<place>"
    results, said = [], []
    for found in index.search(context.space(), query):
        name, place = found["name"], found["place"]
        where = f"{name}, {place}" if place else name
        url = f"file:{name}#{place}" if place else f"file:{name}"
        n = context.cite(url, where)
        results.append({"n": n, "url": url, "title": where, "file": name, "place": place})
        said.append(f"[{n}] {where} (read on from start={found['start']})\n{found['text']}")
    if not said:  # and the files there are, to read, as one in another language than the query
        names = sorted(f["name"] for f in context.space().files())
        listed = "\n".join(f"- {name}" for name in names[:LISTED])
        more = f"\n(and {len(names) - LISTED} more)" if len(names) > LISTED else ""
        there = f" The files:\n{listed}{more}" if names else " There are no files."
        return Result(f"No passage of the files says that.{there}", {"query": query, "results": []})
    content = "\n\n".join(said) + "\n\n(Cite each passage you use by its number, as [1].)"
    return Result(content, {"query": query, "results": results})


def write(workspace: Workspace, path: str, content: str) -> Result:
    file = workspace.path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    return Result(f"Wrote {path}, {len(content)} characters.", {"files": [_name(workspace, file)]})


def edit(workspace: Workspace, path: str, old: str, new: str) -> Result:
    file = workspace.path(path)
    if not file.is_file():
        raise workspace.missing(path)
    if file.suffix.lower() in documents.DOCUMENTS:
        raise ValueError(
            f"{path} is a document, not text, which edit cannot change: suggest_edits "
            "suggests edits to a Word document, and run changes any with its library"
        )
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path} is not a file of text") from e
    if (n := text.count(old)) != 1:
        found = "is not in" if n == 0 else f"is in {n} places of"
        raise ValueError(f"the passage {found} {path}: give it as it is, enough of it to be one")
    file.write_text(text.replace(old, new), encoding="utf-8")
    return Result(f"Edited {path}.", {"files": [_name(workspace, file)]})


def run(workspace: Workspace, code: str) -> Result:
    before = _state(workspace)
    ran = workspace.run(code)
    after = _state(workspace)
    changed = [name for name, state in after.items() if before.get(name) != state]
    lines = [ran.output.rstrip() or "It printed nothing."]
    if ran.status is None:
        lines.append("It ran out of time, and was stopped.")
    elif ran.status != 0:
        lines.append(f"It failed, with exit status {ran.status}.")
    if changed:
        lines.append(f"Files made or changed: {', '.join(changed)}")
    return Result("\n".join(lines), {"files": changed, "status": ran.status})


def skills() -> list[tuple[str, str]]:
    """Each skill's path, to read, and what it is for, from its SKILL.md's front matter."""
    found = []
    for file in sorted(SKILLS.glob("*/SKILL.md")):
        lines = file.read_text(encoding="utf-8").split("---")[1].strip().splitlines()
        about = dict(line.split(":", 1) for line in lines if ":" in line)
        found.append((f"skills/{file.parent.name}/SKILL.md", about["description"].strip()))
    return found


def _skill(path: str) -> Path:
    file = (SKILLS / path.removeprefix("skills/")).resolve()
    if not file.is_relative_to(SKILLS.resolve()):
        raise ValueError(f"{path} is outside the skills")
    return file


def _name(workspace: Workspace, file: Path) -> str:
    return file.relative_to(workspace.root).as_posix()


def _state(workspace: Workspace) -> dict[str, Any]:
    # each file's size and time of change, by name
    return {f["name"]: (f["size"], f["modified"]) for f in workspace.files()}
