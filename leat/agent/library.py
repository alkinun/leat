"""The library: German federal law, as gesetze-im-internet.de publishes it, kept on the box, which
search_law searches and cites, so that the model cites a law's paragraph as it reads it, rather
than as it remembers it, or makes it up.

A law is fetched as its XML, of gii-norm's DTD, and made a markdown file in the sandbox, as every
document from outside is read there: its title, then each of its norms under a heading of its own,
"## § 622 Kündigungsfristen bei Arbeitsverhältnissen", which the index takes for a place, and its
text, each numbered paragraph and list item a line. The library is a space of the workspace's,
which every conversation's search_law reads and none's other tools see; the laws it keeps, by
their abbreviations, are listed in it, of each its title, its address and when it was fetched.
Official texts of laws are free of copyright, § 5 UrhG.
"""

import datetime
import json
import re
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from leat.agent.index import LIBRARY, Index
from leat.agent.tools import Context, Result, Tool, strings
from leat.agent.workspace import Workspace

if TYPE_CHECKING:
    from leat.agent.agent import Agent

SITE = "https://www.gesetze-im-internet.de"
TIMEOUT = 120  # seconds a law's download may take
FOUND = 6  # paragraphs a search finds
LISTED = ".laws.json"  # the laws the library keeps, a hidden file of its own
_listing = threading.Lock()  # held while the list is changed, as two laws may be added at once
# the laws an office asks for most, by their abbreviations, and their addresses' names on the site
# where those differ from them
OFFERED = {
    "BGB": "bgb", "HGB": "hgb", "AO": "ao_1977", "UStG": "ustg_1980", "EStG": "estg",
    "KStG": "kstg_1977", "GmbHG": "gmbhg", "AktG": "aktg", "ZPO": "zpo", "InsO": "inso",
    "StGB": "stgb", "GG": "gg", "KSchG": "kschg", "BUrlG": "burlg", "ArbZG": "arbzg",
    "TzBfG": "tzbfg", "EntgFG": "entgfg", "BetrVG": "betrvg", "MuSchG": "muschg_2018",
    "BDSG": "bdsg_2018", "GewO": "gewo", "UWG": "uwg_2004", "WEG": "woeigg", "RVG": "rvg",
    "StBVV": "stbgebv",
}  # fmt: skip
# a paragraph's or article's number at the start of its place, as "§ 622a" or "Art 3"
_NUMBER = re.compile(r"^(§+|Art\.?)\s*(\d+[a-z]*)", re.I)
# makes the law of the zip at sys.argv[2], as gii-norm's XML, a markdown file at sys.argv[3]; prints
# its abbreviation, its title and how many norms it has
_CONVERT = """
import json, sys, zipfile
import xml.etree.ElementTree as ET

LINES = {"P", "LA", "BR", "row", "Title", "Subtitle"}  # each a line of its own
LEFT = {"FnR", "Footnotes", "FnArea", "img"}  # footnotes' marks and images
BREAK = "\\0"  # between lines, as the XML's own line breaks are spaces

def said(content):
    # a norm's text: each paragraph, list item and table's row a line, an item's term, as "1.",
    # before its text
    out, term = [], False
    def walk(e):
        nonlocal term
        if e.tag in LINES and not term or e.tag == "DT":
            out.append(BREAK)
        if e.tag in LINES:
            term = False
        out.append(e.text or "")
        for child in e:
            if child.tag not in LEFT:
                walk(child)
            out.append(child.tail or "")
        if e.tag == "DT":
            out.append(" ")
            term = True
        elif e.tag == "entry":
            out.append(" | ")
        elif e.tag in LINES:
            out.append(BREAK)
    walk(content)
    lines = (" ".join(line.split()).strip(" |") for line in "".join(out).split(BREAK))
    return "\\n\\n".join(line for line in lines if line)

with zipfile.ZipFile(sys.argv[2]) as z:
    root = ET.fromstring(z.read(next(n for n in z.namelist() if n.lower().endswith(".xml"))))
norms = root.findall("norm")
first = norms[0].find("metadaten")
abbreviation = " ".join((first.findtext("jurabk") or first.findtext("amtabk") or "").split())
title = " ".join((first.findtext("kurzue") or first.findtext("langue") or abbreviation).split())
parts, count = [f"# {title} ({abbreviation})"], 0
for norm in norms[1:]:
    meta, content = norm.find("metadaten"), norm.find("textdaten/text/Content")
    number = " ".join((meta.findtext("enbez") or "").split())
    if not number.startswith(("§", "Art", "Anlage")) or content is None:
        continue  # a part's heading, the table of contents, or a norm without text
    named = meta.find("titel")
    name = " ".join("".join(named.itertext()).split()) if named is not None else ""
    parts.append(f"## {number} {name}".rstrip() + "\\n\\n" + said(content))
    count += 1
with open(sys.argv[3], "w", encoding="utf-8") as made:
    made.write("\\n\\n".join(parts) + "\\n")
print(json.dumps({"abbreviation": abbreviation, "title": title, "norms": count}))
"""


class Library:
    """The library of an agent's workspace, as its owner's apps show it and add laws to it: each
    law fetched on a thread of its own, the apps told as it begins and how it went, and the index
    told to read it."""

    def __init__(self, agent: "Agent"):
        self.agent = agent
        self._adding: set[str] = set()  # the laws being added, as the owner named them
        self._error: str | None = None  # why the last law could not be added, if it could not
        self._lock = threading.Lock()

    def state(self) -> dict[str, Any]:
        """The library as the owner's apps show it: the laws it keeps, those being added, and why
        the last could not be, if it could not; and the laws an office asks for most."""
        kept = laws(self.agent.workspace) if self.agent.workspace else {}
        return {
            "type": "library", "laws": [{"abbreviation": a} | kept[a] for a in sorted(kept)],
            "adding": sorted(self._adding), "error": self._error, "offered": list(OFFERED),
        }  # fmt: skip

    def add(self, law: str, person: int | None) -> None:
        """Begins to add a law, by its abbreviation, as add() does. Raises ValueError while the
        agent is kept from the internet, or the law is being added."""
        law = law.strip()
        if self.agent.workspace is None:
            raise ValueError("this Leat keeps no files, nor laws")
        if self.agent.offline():
            raise ValueError("this Leat is kept from the internet: let it reach it to add a law")
        with self._lock:
            if law.lower() in {a.lower() for a in self._adding}:
                raise ValueError(f"{law} is being added")
            self._adding.add(law)
            self._error = None
        self._publish()
        threading.Thread(target=self._add, args=(law, person), daemon=True).start()

    def remove(self, law: str, person: int | None) -> None:
        """Removes a law, by its abbreviation, as remove() does."""
        remove(self.agent.workspace, law)
        self._read()
        self.agent.note(person, "removed the law", detail=law)
        self._publish()

    def _add(self, law: str, person: int | None) -> None:
        try:
            kept = add(self.agent.workspace, law)
        except (LookupError, OSError, ValueError) as e:
            self._error = f"Couldn't add {law}: {e}"
        else:
            self.agent.note(person, "added the law", detail=kept["abbreviation"])
            self._read()
        finally:
            with self._lock:
                self._adding.discard(law)
            self._publish()

    def _read(self) -> None:
        # has the index read the library again, as it changed
        if self.agent.index is not None:
            self.agent.index.refresh(LIBRARY)

    def _publish(self) -> None:
        self.agent.events.publish(self.state() | {"to": "owner"})


def laws(workspace: Workspace) -> dict[str, dict[str, Any]]:
    """The laws the library keeps, by their abbreviations, as "BGB": of each its title, its name on
    the site, how many norms it has, and when it was fetched."""
    return _listed(workspace.space(LIBRARY))


def add(workspace: Workspace, law: str) -> dict[str, Any]:
    """Fetches a law from gesetze-im-internet.de, by its abbreviation, as "BGB", or its name on the
    site, as "ustg_1980", and keeps it in the library, as anew if it was; returns it, as laws()
    lists it, its abbreviation too. Raises LookupError if the site has no such law, OSError if it
    cannot be reached, and ValueError if what it gives cannot be read."""
    space, law = workspace.space(LIBRARY), law.strip()
    name = next((v for k, v in OFFERED.items() if k.lower() == law.lower()), law.lower())
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise LookupError(f"{law} is not a law's abbreviation")
    zipped = space.path(f".{name}.zip")
    zipped.write_bytes(_fetch(law, name))
    try:
        return keep(space, zipped.name, name)
    finally:
        zipped.unlink(missing_ok=True)


def keep(space: Workspace, zipped: str, name: str) -> dict[str, Any]:
    """Makes the law of a zip of the library's, as gii-norm's XML, a markdown file of the library's,
    named by its abbreviation, in the sandbox, and lists it, of its name on the site; returns it.
    Raises ValueError if it cannot be read."""
    made = f".{name}.md"
    try:
        said = space.given(_CONVERT, None, zipped, made)
        if not said["abbreviation"] or not said["norms"]:
            raise ValueError("it holds no law's paragraphs")
        abbreviation = said["abbreviation"]
        if not re.fullmatch(r"[\w\-. ]+", abbreviation) or abbreviation.startswith("."):
            raise ValueError(f"its abbreviation, {abbreviation!r}, cannot name a file")
        space.path(made).replace(space.path(f"{abbreviation}.md"))
    finally:
        space.path(made).unlink(missing_ok=True)
    listed = {"title": said["title"], "name": name, "norms": said["norms"],
              "fetched": datetime.date.today().isoformat()}  # fmt: skip
    with _listing:
        _list(space, _listed(space) | {abbreviation: listed})
    return listed | {"abbreviation": abbreviation}


def remove(workspace: Workspace, law: str) -> None:
    """Removes a law from the library, by its abbreviation. Raises LookupError if it keeps none."""
    space = workspace.space(LIBRARY)
    with _listing:
        if law not in (kept := _listed(space)):
            raise LookupError(f"the library keeps no {law}")
        space.path(f"{law}.md").unlink(missing_ok=True)
        _list(space, {k: v for k, v in kept.items() if k != law})


def tools(index: Index) -> list[Tool]:
    """search_law, of the library the index keeps."""
    return [
        Tool(
            "search_law",
            "Search the German federal laws kept on this computer, as the BGB or HGB, for the "
            "paragraphs that say something, as their official text has them: the likeliest few, "
            "each numbered, with its law and §",
            strings(query="what to find, in the words the law would use"),
            lambda context, query: search(index, context, query),
        )
    ]


def search(index: Index, context: Context, query: str) -> Result:
    # the library's paragraphs likeliest to say what the query asks, each numbered as a source,
    # and linked to its page on the site
    kept, found = laws(index.workspace), []
    for passage in index.search(index.workspace.space(LIBRARY), query, FOUND):
        law = passage["name"].removesuffix(".md")
        cited = _cited(law, passage["place"], kept.get(law, {}).get("name"))
        n = context.cite(cited["url"], cited["title"])
        found.append((cited | {"n": n}, passage["text"]))
    if not found:
        listed = ", ".join(kept) or "none"
        return Result(f"No paragraph of the laws kept here says that. The laws kept: {listed}.",
                      {"query": query, "results": []})  # fmt: skip
    said = "\n\n".join(f"[{c['n']}] {c['title']}\n{text}" for c, text in found)
    content = said + "\n\n(Cite each paragraph you use by its number, as [1].)"
    return Result(content, {"query": query, "results": [c for c, _ in found]})


def _fetch(law: str, name: str) -> bytes:
    # a law's zip, by its name on the site
    try:
        with urllib.request.urlopen(f"{SITE}/{name}/xml.zip", timeout=TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise LookupError(f"gesetze-im-internet.de has no law {law}") from e
        raise OSError(f"gesetze-im-internet.de answered {e.code}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise OSError(f"gesetze-im-internet.de cannot be reached: {getattr(e, 'reason', e)}") from e


def _listed(space: Workspace) -> dict[str, dict[str, Any]]:
    listed = space.path(LISTED)
    return json.loads(listed.read_text(encoding="utf-8")) if listed.exists() else {}


def _list(space: Workspace, kept: dict[str, dict[str, Any]]) -> None:
    space.path(LISTED).write_text(json.dumps(kept, ensure_ascii=False, indent=1), "utf-8")


def _cited(law: str, place: str, name: str | None) -> dict[str, Any]:
    # a paragraph as a source: "§ 622 BGB", its heading after, and its page on the site
    number = _NUMBER.match(place)
    if number is None:
        return {"url": f"{SITE}/{name}/" if name else SITE, "title": f"{law}, {place}", "law": law}
    sign, n = number.groups()
    rest = place[number.end() :].strip()
    page = f"__{n}.html" if sign.startswith("§") else f"art_{n}.html"
    title = f"{sign} {n} {law}" + (f": {rest}" if rest else "")
    return {"url": f"{SITE}/{name}/{page}" if name else SITE, "title": title, "law": law}
