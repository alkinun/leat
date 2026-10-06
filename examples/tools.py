"""Web search and page reading, as tools for leat's chat app.

GET /tools lists the tools as OpenAI's API declares them; POST /call runs one, {"name": ...,
"arguments": {...}}, and answers {"content": text}, an error's message too, for the model to read.
Searches go to a SearXNG on this machine:

    docker run -d --name searxng -p 127.0.0.1:8888:8080 -e SEARXNG_SECRET=$(openssl rand -hex 32) \\
        -v $PWD/examples/searxng.yml:/etc/searxng/settings.yml:ro searxng/searxng
    python examples/tools.py

Then the chat app's settings take this server, http://127.0.0.1:8081, as their tools server. The
pages it reads may be any the machine reaches, its own network's too, so it answers no other
page in a browser than the chat app, of leat serve on this machine by default: with leat serving
another machine, `--host 0.0.0.0 --origin http://this-machine:8080`.
"""

import argparse
import contextlib
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEARXNG = "http://127.0.0.1:8888"
RESULTS, PAGE = 5, 8000  # search results a search gives, and characters of a page
# elements whose text is not a page's content: code, and the menus around it
HIDDEN = ("script", "style", "noscript", "nav", "header", "footer", "aside")


def _tool(name: str, description: str, argument: str, about: str) -> dict:
    # a function of one string, as OpenAI's API declares it
    schema = {"type": "string", "description": about}
    parameters = {"type": "object", "properties": {argument: schema}, "required": [argument]}
    function = {"name": name, "description": description, "parameters": parameters}
    return {"type": "function", "function": function}


TOOLS = [
    _tool("web_search", "Search the web: the top results' titles, links and snippets", "query",
          "what to search for"),
    _tool("fetch_page", "Read a web page as text: its first characters, without markup", "url",
          "the page's address, from a search result say"),
]  # fmt: skip


def web_search(query: str) -> str:
    url = f"{SEARXNG}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    results = json.loads(_get(url))["results"][:RESULTS]
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r.get('content', '')}" for r in results)


def fetch_page(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not a web page: {url}")
    parser = _Text()
    parser.feed(_get(url))
    return re.sub(r"\s*\n\s*", "\n", " ".join(parser.parts)).strip()[:PAGE]


class _Text(HTMLParser):
    # an HTML page's text, without its HIDDEN elements'
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in HIDDEN:
            self.skipping += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in HIDDEN and self.skipping:
            self.skipping -= 1

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(data)


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (leat tools)"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8", "replace")


class _Handler(BaseHTTPRequestHandler):
    origins: frozenset[str] = frozenset()  # the chat app's, whose pages alone may call the tools

    def do_GET(self) -> None:
        if not self._allowed():
            return self.send_error(403)
        if self.path != "/tools":
            return self.send_error(404)
        self._json(TOOLS)

    def do_POST(self) -> None:
        if not self._allowed():
            return self.send_error(403)
        if self.path != "/call":
            return self.send_error(404)
        call = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        run = {"web_search": web_search, "fetch_page": fetch_page}.get(call.get("name"))
        try:
            if run is None:
                raise ValueError(f"there is no tool {call.get('name')!r}")
            content = run(**call.get("arguments", {}))
        except Exception as e:  # for the model, which may try again
            content = f"error: {e}"
        self._json({"content": content})

    def _allowed(self) -> bool:
        # a browser's request from the chat app, or one of no browser, which sends no Origin: any
        # other page could read the network through fetch_page, or make it fetch
        origin = self.headers.get("Origin")
        return origin is None or origin in self.origins

    def _json(self, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if origin := self.headers.get("Origin"):  # the chat app, served by leat, reads it
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web search and page reading for leat's app")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 for other machines too")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--origin", action="append", help="the chat app's, as http://host:8080; "
                        "by default leat serve's on this machine")  # fmt: skip
    args = parser.parse_args()
    _Handler.origins = frozenset(args.origin or ("http://127.0.0.1:8080", "http://localhost:8080"))
    print(f"tools at http://{args.host}:{args.port}, for {', '.join(sorted(_Handler.origins))}")
    with contextlib.suppress(KeyboardInterrupt):
        ThreadingHTTPServer((args.host, args.port), _Handler).serve_forever()
