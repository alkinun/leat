import json
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from leat.agent.tools import web

PAGE = (
    "<html><head><title>A page</title><style>p {}</style></head><body><nav>Menu</nav>"
    "<p>Hello</p>world, <b>and</b> more<ul><li>An item</li></ul><script>x()</script>"
    "<footer>Foot</footer></body></html>"
)
RESULTS = [
    {"title": "One", "url": "https://one.example/", "content": "The first."},
    {"title": "", "url": "https://two.example/"},
]


class _Site(BaseHTTPRequestHandler):
    # a SearXNG's search, and pages
    queries: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/search":
            self.queries.append(urllib.parse.parse_qs(query))
            return self._send("application/json", json.dumps({"results": RESULTS}))
        if path == "/away":
            self.send_response(302)
            self.send_header("Location", "http://10.0.0.1/")
            return self.end_headers()
        kinds = {"/page": "text/html; charset=utf-8", "/text": "text/plain"}
        self._send(kinds.get(path, "application/pdf"), PAGE if path == "/page" else "Plain text.")

    def _send(self, kind: str, body: str) -> None:
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def site() -> Iterator[str]:
    with ThreadingHTTPServer(("127.0.0.1", 0), _Site) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        yield f"http://127.0.0.1:{server.server_port}"
        server.shutdown()


def allow(monkeypatch, site: str) -> None:
    # lets fetch read the site served here, as it would one on the web
    port, public = urllib.parse.urlsplit(site).port, web._public
    monkeypatch.setattr(web, "_public", lambda url: None if urllib.parse.urlsplit(url).port == port
                        else public(url))  # fmt: skip


def test_search(site):
    result = web.search(site, "strix halo")
    assert _Site.queries[-1] == {"q": ["strix halo"], "format": ["json"], "language": ["en"]}
    two = "https://two.example/"
    assert result.content == f"1. One\nhttps://one.example/\nThe first.\n\n2. {two}\n{two}"
    assert result.info == {"query": "strix halo", "results": [
        {"title": "One", "url": "https://one.example/"},
        {"title": "https://two.example/", "url": "https://two.example/"},
    ]}  # fmt: skip
    with pytest.raises(RuntimeError, match="search is not available"):
        web.search("http://127.0.0.1:9", "strix halo")


def test_fetch(site, monkeypatch):
    # a page's title and text, without its menus and code, a line to each block
    allow(monkeypatch, site)
    result = web.fetch(f"{site}/page")
    assert result.content == f"A page\n{site}/page\n\nHello\nworld, and more\nAn item"
    assert result.info == {"url": f"{site}/page", "title": "A page"}
    assert web.fetch(f"{site}/text").content == f"{site}/text\n\nPlain text."
    with pytest.raises(ValueError, match="not a page of text but application/pdf"):
        web.fetch(f"{site}/pdf")


def test_fetch_the_web_alone(site, monkeypatch):
    # an address on this network is refused, a redirect's too
    for url, error in [
        ("file:///etc/passwd", "not a web page's address"),
        ("http://127.0.0.1:9/", "127.0.0.1 is on this network"),
        ("http://192.168.1.1/admin", "192.168.1.1 is on this network"),
        ("http://[::1]/", "::1 is on this network"),
        ("http://100.111.0.1/", "100.111.0.1 is on this network"),  # a tailnet's
    ]:
        with pytest.raises(ValueError, match=error):
            web.fetch(url)
    allow(monkeypatch, site)
    with pytest.raises(ValueError, match="10.0.0.1 is on this network"):
        web.fetch(f"{site}/away")
