"""The web: searching it through the SearXNG on the box, and reading its pages.

Pages are the web's alone: an address on the box's own network, a router's or another machine's,
is refused, at every redirect too, so that a page cannot lead the model to read its devices.
The addresses checked are those connected to, so that a site's DNS cannot answer the check with
one and the connection with another.

A page is read as Hermes Agent reads one: its content alone, as markdown, which trafilatura finds in
the sandbox, where a page made to attack a parser attacks nothing else, or of its text without its
menus, without the sandbox's environment; whole up to a budget, or its start. The page is saved in
the workspace, to read on, or again once the conversation has cleared it to make room. Each source
is numbered, across the conversation, for the model to cite.

A page read for a question is read by a reader, as Claude's research and Hermes Agent's delegation
read theirs: the model, in a conversation of its own, which says what the page says of it, so that
the conversation takes those findings, not the page, and stays small. The model's calls run at once,
so its readers do, the engine batching them.
"""

import contextlib
import hashlib
import http.client
import ipaddress
import json
import re
import socket
import ssl
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from leat.agent.client import Client, EngineError
from leat.agent.tools import Context, Result, Tool, schema, strings
from leat.agent.workspace import Workspace

RESULTS = 6  # search results the model reads
PAGE = 8000  # characters of a page it reads at once, some 2,000 tokens
SAVED = ".web"  # the workspace's folder of the pages read, to read on
KEPT = 200  # pages it keeps, the latest
# finds a page's content as markdown, of its HTML at sys.argv[1], of the address sys.argv[2]: run
# in the sandbox, with trafilatura
_EXTRACT = """
import json, sys
import trafilatura
html = open(sys.argv[1], encoding="utf-8", errors="replace").read()
text = trafilatura.extract(
    html, url=sys.argv[2], output_format="markdown", include_links=True, include_tables=True,
    include_formatting=True, favor_recall=True,
)
meta = trafilatura.extract_metadata(html)
print(json.dumps({"title": (meta and meta.title) or "", "text": text or ""}))
"""
READ = 2 << 20  # bytes of a response read at most: a page's first characters are within them
TIMEOUT = 15  # seconds a request may take
# the types of what fetch reads: the web's text, and its pages' kinds beside text/*
TEXTS = ("application/xhtml+xml", "application/json", "application/xml")
# a page's charset as its <meta charset="..."> or <meta http-equiv content="...; charset=..."> says
_CHARSET = re.compile(rb"""<meta[^>]*charset\s*=\s*["']?([\w.:-]+)""", re.IGNORECASE)
LANGUAGE = "en"  # of the results: SearXNG's engines would answer in the box's country's otherwise
# elements whose text is not a page's content: code, and the menus around it
HIDDEN = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside"}
# elements that begin and end lines of a page's text
BLOCKS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
    "section", "article", "main", "table", "ul", "ol", "dl", "dt", "dd", "figcaption",
}  # fmt: skip
_AGENT = "Mozilla/5.0 (compatible; leat)"
# characters of a page read whole even for a question, as its findings would be no less
WHOLE = 4000
READER = 24000  # characters of a page a reader reads at most, its start
FOUND = 600  # tokens of a reader's findings at most
# a reader's instructions, of a page and a question
READING = """\
You read a web page for Leat, an assistant, who must answer a question. Say what the page says \
that answers it: the facts, numbers, names and dates, briefly, quoting what matters most word for \
word. If the page does not answer it, say so in a line. Add nothing the page does not say."""


def tools(searxng: str, reader: Client | None = None) -> list[Tool]:
    """search, through the SearXNG at `searxng`, and fetch, which reads pages in the sandbox of
    its call's conversation's workspace, and saves them there, if it has one, and reads them for
    a question by the model `reader` serves, if given."""
    return [
        Tool(
            "search",
            "Search the web: the top results' titles, links and snippets, each numbered",
            strings(query="what to search for"),
            lambda context, query: search(searxng, query, context),
        ),
        Tool(
            "fetch",
            "Read a web page, numbered: what it says of a question, or the page itself",
            schema(
                url=("string", "the page's address, as a search result's"),
                question=(
                    "string",
                    "what you want the page to answer; without one, the page "
                    "itself, for one the user asked you to read",
                ),
            ),  # fmt: skip
            lambda context, url, question=None: fetch(
                url, context.workspace, context, question, reader
            ),
        ),
    ]


def search(searxng: str, query: str, context: Context | None = None) -> Result:
    url = f"{searxng}/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "language": LANGUAGE}
    )
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            results = json.loads(response.read(READ))["results"][:RESULTS]
    except OSError as e:
        raise RuntimeError(f"search is not available: {e}") from e
    cite = context.cite if context else lambda url, title: 0
    found = [{"title": r.get("title") or r["url"], "url": r["url"]} for r in results]
    for f in found:
        f["n"] = cite(f["url"], f["title"])
    content = "\n\n".join(
        f"{f'[{n}] ' if (n := f['n']) else ''}{f['title']}\n{f['url']}\n{r.get('content') or ''}"
        .strip() for f, r in zip(found, results, strict=True)
    )  # fmt: skip
    return Result(content or "No results.", {"query": query, "results": found})


def fetch(
    url: str, workspace: Workspace | None = None, context: Context | None = None,
    question: str | None = None, reader: Client | None = None,
) -> Result:  # fmt: skip
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"not a web page's address: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    with _OPENER.open(request, timeout=TIMEOUT) as response:
        kind, final = response.headers.get_content_type(), response.url
        if not kind.startswith("text/") and kind not in TEXTS:
            raise ValueError(f"not a page of text but {kind}")
        data, charset = response.read(READ), response.headers.get_content_charset()
    if charset is None and "html" in kind and (meta := _CHARSET.search(data[:4096])):
        charset = meta[1].decode()  # as the page says it, if its headers do not
    try:
        text = data.decode(charset or "utf-8", "replace")
    except LookupError:  # a charset of no name Python knows
        text = data.decode("utf-8", "replace")
    title = ""
    if "html" in kind:
        title, text = _readable(text, final, workspace)
    text = _absolute(text.strip(), final) or "The page has no text."
    n = context.cite(final, title) if context else 0
    saved = _save(workspace, final, text) if workspace is not None else None
    head = f"[{n}] {title}" if n else title
    info = {"url": final, "title": title, "n": n} | ({"saved": saved} if saved else {})
    if question and reader is not None and len(text) > WHOLE:
        with contextlib.suppress(EngineError):  # read as it is, if the engine cannot
            person = context.person if context else None
            found = _read(reader, question, f"{title}, {final}", text, person)
            whole = f" The whole page is saved at {saved}." if saved else ""
            said = f"{found}\n\n(What the page says of: {question}.{whole})"
            return Result(f"{head}\n{final}\n\n{said}".strip(), info | {"question": question})
    if (rest := len(text) - PAGE) > 0:
        more = f": read {saved} from start={PAGE} for them" if saved else ""
        text = f"{text[:PAGE]}\n\n(The page goes on, {rest} characters more{more}.)"
    return Result(f"{head}\n{final}\n\n{text}".strip(), info)


def _read(reader: Client, question: str, page: str, text: str, person: int | None) -> str:
    # what a page says of a question, as the model reads it in a conversation of its own, for the
    # person who asked it, whose prompts alone may share its prefix: the question is theirs
    body = {
        "messages": [{"role": "system", "content": READING},
                     {"role": "user", "content": f"The question: {question}\n\nThe page, "
                      f"{page}:\n\n{text[:READER]}"}],
        "max_tokens": FOUND, "temperature": 0.3, "reasoning_effort": "none",
    }  # fmt: skip
    return reader.reply(body, person)["content"].strip() or "The page says nothing of it."


def _readable(html: str, url: str, workspace: Workspace | None) -> tuple[str, str]:
    # a page's title and content: as trafilatura finds them in the sandbox, if it can, or its text
    # without its menus
    if workspace is not None and workspace.environment is not None:
        raw = _save(workspace, url, html, ".html")
        try:
            ran = workspace.run(_EXTRACT, f"/workspace/{raw}", url, timeout=30, kept=None)
            found = json.loads(ran.output.strip().splitlines()[-1]) if ran.status == 0 else {}
            if found.get("text"):
                return " ".join(found["title"].split()), found["text"]
        except (ValueError, IndexError, RuntimeError):
            pass
        finally:
            workspace.path(raw).unlink(missing_ok=True)
    page = _Text()
    page.feed(html)
    page.close()
    return " ".join(page.title.split()), page.text()


def _absolute(text: str, base: str) -> str:
    # markdown's links, of the page at `base`, to the whole addresses
    def whole(link: re.Match) -> str:
        return f"]({urllib.parse.urljoin(base, link[1])}"

    return re.sub(r"\]\((?!https?:|mailto:|#)([^)\s]+)", whole, text)


def _save(workspace: Workspace, url: str, text: str, suffix: str = ".md") -> str:
    # saves a page's text in the workspace's SAVED folder, the oldest beyond KEPT deleted; returns
    # its name there
    folder = workspace.path(SAVED)
    folder.mkdir(exist_ok=True)
    name = f"{SAVED}/{hashlib.sha256(url.encode()).hexdigest()[:16]}{suffix}"
    workspace.path(name).write_text(text, encoding="utf-8")
    for old in sorted(folder.iterdir(), key=_changed)[:-KEPT]:
        old.unlink(missing_ok=True)
    return name


def _changed(path: Path) -> float:
    # when a saved page last changed; one another fetch, running at once, deleted meanwhile, as
    # the HTML it read, the oldest
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _connect(address: tuple[str, int], *options: Any) -> socket.socket:
    # a connection to a host of the web's, at one of its addresses, as socket.create_connection
    # makes one; raises ValueError if any is on this network
    host, port = address
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError(f"{host} was not found: {e}") from e
    addresses = list(dict.fromkeys(str(a[4][0]) for a in found))
    if any(not ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses):
        raise ValueError(f"{host} is on this network, not the web")
    for i, a in enumerate(addresses):  # as IPv6's, where the box has no route to it
        try:
            return socket.create_connection((a, port), *options)
        except OSError:
            if i == len(addresses) - 1:
                raise
    raise ValueError(f"{host} has no address")


class _HTTP(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _connect


class _HTTPS(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _connect


class _WebHTTP(urllib.request.HTTPHandler):
    # opens connections to the web alone, the first and every redirect's
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_HTTP, req)


class _WebHTTPS(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_HTTPS, req, context=_TLS)


_TLS = ssl.create_default_context()
# the web's alone: no handlers of ftp:, file: or data:, to which a page could redirect past the
# check of addresses, nor of proxies
_OPENER = urllib.request.OpenerDirector()
for _handler in (_WebHTTP(), _WebHTTPS(), urllib.request.HTTPRedirectHandler(),
                 urllib.request.HTTPDefaultErrorHandler(), urllib.request.HTTPErrorProcessor(),
                 urllib.request.UnknownHandler()):  # fmt: skip
    _OPENER.add_handler(_handler)


class _Text(HTMLParser):
    # an HTML page's title, and its text without its HIDDEN elements', a line to each block
    def __init__(self) -> None:
        super().__init__()
        self.title, self.parts = "", [""]
        self.hidden = 0  # HIDDEN elements open
        self.titled = False  # in the title

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in HIDDEN:
            self.hidden += 1
        elif tag == "title":
            self.titled = True
        elif tag in BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in HIDDEN:
            self.hidden = max(self.hidden - 1, 0)
        elif tag == "title":
            self.titled = False
        elif tag in BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.titled:
            self.title += data
        elif not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).split("\n"))
        return "\n".join(line for line in lines if line)
