"""The web: searching it through the SearXNG on the box, and reading its pages.

Pages are the web's alone: an address on the box's own network, a router's or another machine's,
is refused, at every redirect too, so that a page cannot lead the model to read the home's devices.
"""

import ipaddress
import json
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from leat.agent.tools import Result, Tool, strings

RESULTS = 6  # search results the model reads
PAGE = 8000  # characters of a page it reads at most, some 2,000 tokens
READ = 2 << 20  # bytes of a response read at most: a page's first characters are within them
TIMEOUT = 15  # seconds a request may take
# the types of what fetch reads: the web's text, and its pages' kinds beside text/*
TEXTS = ("application/xhtml+xml", "application/json", "application/xml")
LANGUAGE = "en"  # of the results: SearXNG's engines would answer in the box's country's otherwise
# elements whose text is not a page's content: code, and the menus around it
HIDDEN = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside"}
# elements that begin and end lines of a page's text
BLOCKS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
    "section", "article", "main", "table", "ul", "ol", "dl", "dt", "dd", "figcaption",
}  # fmt: skip
_AGENT = "Mozilla/5.0 (compatible; leat)"


def tools(searxng: str) -> list[Tool]:
    """search, through the SearXNG at `searxng`, and fetch."""
    return [
        Tool(
            "search",
            "Search the web: the top results' titles, links and snippets",
            strings(query="what to search for"),
            lambda query: search(searxng, query),
        ),
        Tool(
            "fetch",
            "Read a web page as text",
            strings(url="the page's address, as a search result's"),
            fetch,
        ),
    ]


def search(searxng: str, query: str) -> Result:
    url = f"{searxng}/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "language": LANGUAGE}
    )
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            results = json.loads(response.read(READ))["results"][:RESULTS]
    except OSError as e:
        raise RuntimeError(f"search is not available: {e}") from e
    found = [{"title": r.get("title") or r["url"], "url": r["url"]} for r in results]
    content = "\n\n".join(
        f"{i}. {f['title']}\n{f['url']}\n{r.get('content') or ''}".strip()
        for i, (f, r) in enumerate(zip(found, results, strict=True), 1)
    )
    return Result(content or "No results.", {"query": query, "results": found})


def fetch(url: str) -> Result:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"not a web page's address: {url}")
    _public(url)
    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    with _OPENER.open(request, timeout=TIMEOUT) as response:
        kind, final = response.headers.get_content_type(), response.url
        if not kind.startswith("text/") and kind not in TEXTS:
            raise ValueError(f"not a page of text but {kind}")
        charset = response.headers.get_content_charset() or "utf-8"
        text = response.read(READ).decode(charset, "replace")
    title = ""
    if "html" in kind:
        page = _Text()
        page.feed(text)
        page.close()
        title, text = " ".join(page.title.split()), page.text()
    text = text.strip()[:PAGE] or "The page has no text."
    return Result(f"{title}\n{final}\n\n{text}".strip(), {"url": final, "title": title})


def _public(url: str) -> None:
    # raises ValueError if the URL's host is on this network rather than the web
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        addresses = {str(a[4][0]) for a in socket.getaddrinfo(host, None)}
    except OSError as e:
        raise ValueError(f"{host} was not found: {e}") from e
    if any(not ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses):
        raise ValueError(f"{host} is on this network, not the web")


class _Redirects(urllib.request.HTTPRedirectHandler):
    # follows a redirect only to the web
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_Redirects)


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
