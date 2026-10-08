import json
import os
import socket
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from leat.agent.client import Client
from leat.agent.tools import Context, weather, web
from leat.agent.workspace import Workspace

PAGE = (
    "<html><head><title>A page</title><style>p {}</style></head><body><nav>Menu</nav>"
    "<p>Hello</p>world, <b>and</b> more<ul><li>An item</li></ul><script>x()</script>"
    "<footer>Foot</footer></body></html>"
)
ARTICLE = (
    "<html><head><title>The story - News</title></head><body><nav><a href='/'>Home</a> "
    "<a href='/a'>About</a> <a href='/b'>Contact</a></nav><main><article><h1>The story</h1><p>"
    "The council met on Tuesday to decide the fate of the old library, which has stood on the "
    "square since 1902 and which many in the town still visit every week, to read or to study."
    "</p><p>The council voted to restore it, from <a href='/spring'>the spring</a>.</p></article>"
    "</main><footer>Copyright 2026 News Ltd.</footer></body></html>"
)
# an article longer than the sandbox's output a run keeps
LONG = ARTICLE.replace("</article>", "".join(
    f"<p>Paragraph {i}: {'the library stays open late on Thursdays. ' * 25}</p>" for i in range(30)
) + "<p>The end.</p></article>")  # fmt: skip
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
        kinds = {"/page": "text/html", "/article": "text/html", "/long": "text/html",
                 "/text": "text/plain", "/odd": "text/plain; charset=klingon"}  # fmt: skip
        body = {"/page": PAGE, "/article": ARTICLE, "/long": LONG}.get(path, "Plain text.")
        self._send(kinds.get(path, "application/pdf"), body)

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
    port, connect = urllib.parse.urlsplit(site).port, web._connect

    def allowed(address: tuple[str, int], *options: Any) -> socket.socket:
        if address[1] == port:
            return socket.create_connection(address, *options)
        return connect(address, *options)

    monkeypatch.setattr(web, "_connect", allowed)


def numbering() -> Context:
    # a conversation's numbering of its sources, from 1, as a turn's
    numbers: dict[str, int] = {}
    return Context("c", lambda url, title: numbers.setdefault(url, len(numbers) + 1))


def test_search(site):
    context = numbering()
    context.cite("https://two.example/", "")  # read before
    result = web.search(site, "strix halo", context)
    assert _Site.queries[-1] == {"q": ["strix halo"], "format": ["json"], "language": ["en"]}
    two = "https://two.example/"
    assert result.content == f"[2] One\nhttps://one.example/\nThe first.\n\n[1] {two}\n{two}"
    assert result.info == {"query": "strix halo", "results": [
        {"title": "One", "url": "https://one.example/", "n": 2},
        {"title": "https://two.example/", "url": "https://two.example/", "n": 1},
    ]}  # fmt: skip
    assert web.search(site, "strix halo").content.startswith("One\n")  # nothing numbering
    with pytest.raises(RuntimeError, match="search is not available"):
        web.search("http://127.0.0.1:9", "strix halo")


def test_fetch(site, monkeypatch):
    # a page's title and text, without its menus and code, a line to each block
    allow(monkeypatch, site)
    result = web.fetch(f"{site}/page", context=numbering())
    assert result.content == f"[1] A page\n{site}/page\n\nHello\nworld, and more\nAn item"
    assert result.info == {"url": f"{site}/page", "title": "A page", "n": 1}
    assert web.fetch(f"{site}/text").content == f"{site}/text\n\nPlain text."
    assert web.fetch(f"{site}/odd").content.endswith("Plain text.")  # as UTF-8
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
    # a name of an address on this network, refused before any connection is made to it
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("10.0.0.2", 80))])
    with pytest.raises(ValueError, match="rebinding.example is on this network"):
        web.fetch("http://rebinding.example/")


class _OpenMeteo(BaseHTTPRequestHandler):
    # Open-Meteo's geocoding and forecast, of Paris, in France and in Texas
    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/search":
            paris = {"name": "Paris", "admin1": "Île-de-France", "country": "France"}
            texas = {"name": "Paris", "admin1": "Texas", "country": "United States"}
            found = [paris | {"latitude": 48.9, "longitude": 2.3}, texas | {"latitude": 33.7,
                     "longitude": -95.6}] if "Paris" in query else []  # fmt: skip
            body: dict[str, Any] = {"results": found} if found else {}
        else:
            hot = "latitude=33.7" in query  # Texas
            body = {
                "current": {"time": "2026-10-07T14:00", "temperature_2m": 30 if hot else 20.4,
                            "apparent_temperature": 21, "relative_humidity_2m": 60,
                            "weather_code": 2, "wind_speed_10m": 9.6},
                "daily": {"time": ["2026-10-07"], "weather_code": [61], "temperature_2m_max": [22],
                          "temperature_2m_min": [12], "precipitation_probability_max": [80],
                          "precipitation_sum": [4.2], "wind_speed_10m_max": [20]},
            }  # fmt: skip
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_weather(monkeypatch):
    with ThreadingHTTPServer(("127.0.0.1", 0), _OpenMeteo) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setattr(weather, "GEOCODING", f"{url}/search")
        monkeypatch.setattr(weather, "FORECAST", f"{url}/forecast")
        result = weather.weather("Paris")
        assert result.info == {"place": "Paris, Île-de-France, France"}
        assert result.content == (
            "Paris, Île-de-France, France, at 2026-10-07 14:00 local time: partly cloudy, "
            "20 °C (69 °F), feeling 21 °C (70 °F), humidity 60%, wind 10 km/h.\n"
            "Wednesday 7 October: light rain, 12 °C (54 °F) to 22 °C (72 °F), 80% chance of "
            "rain (4.2 mm), wind up to 20 km/h."
        )
        # a country or region after a comma chooses among places of a name
        assert weather.weather("Paris, Texas").info == {"place": "Paris, Texas, United States"}
        assert "30 °C" in weather.weather("Paris, Texas").content
        with pytest.raises(ValueError, match="there is no place called 'Atlantis'"):
            weather.weather("Atlantis")
        server.shutdown()


def test_fetch_long(site, monkeypatch, tmp_path):
    # a page's start, the rest saved in the workspace to read on; its links whole
    allow(monkeypatch, site)
    monkeypatch.setattr(web, "PAGE", 10)
    workspace = Workspace(tmp_path)
    content = web.fetch(f"{site}/page", workspace).content
    saved = content.split("read ")[1].split(" from")[0]
    assert content.endswith(f"(The page goes on, 19 characters more: read {saved} from start=10 "
                            "for them.)")  # fmt: skip
    assert workspace.path(saved).read_text() == "Hello\nworld, and more\nAn item"
    assert workspace.files() == []  # hidden, of the user's files
    assert web._absolute("[a](/b) [c](d) [e](https://f/) [g](#h)", "https://x.org/y/z") == (
        "[a](https://x.org/b) [c](https://x.org/y/d) [e](https://f/) [g](#h)")  # fmt: skip


def test_fetch_for_a_question(site, monkeypatch, tmp_path, engine):
    # a long page read for a question by a reader, the model in a conversation of its own, whose
    # findings the conversation takes, the page saved; a short one, or one the engine cannot
    # read, as it is
    allow(monkeypatch, site)
    workspace, reader = Workspace(tmp_path), Client(engine.url)
    engine.replies.put([{"content": "It stays open late on Thursdays."}])
    result = web.fetch(f"{site}/long", workspace, numbering(), "When is it open late?", reader)
    assert result.content.startswith(f"[1] The story - News\n{site}/long\n\nIt stays open late")
    assert result.content.endswith("(What the page says of: When is it open late?. The whole page "
                                   f"is saved at {result.info['saved']}.)")  # fmt: skip
    asked = engine.requests[-1]["messages"]
    assert asked[0]["content"] == web.READING and "Paragraph 0" in asked[1]["content"]
    assert len(asked[1]["content"]) < web.READER + 200  # the page's start
    assert result.info["question"] == "When is it open late?"
    assert web.fetch(f"{site}/page", workspace, None, "What?", reader).content.endswith("An item")
    engine.replies.put("the engine is busy")  # read as it is
    assert (
        "(The page goes on" in web.fetch(f"{site}/long", workspace, None, "When?", reader).content
    )
    assert len(engine.requests) == 2


@pytest.mark.skipif(not os.environ.get("LEAT_SANDBOX"), reason="needs LEAT_SANDBOX")
def test_fetch_readable(site, monkeypatch, tmp_path):
    # in the sandbox, trafilatura finds a page's content as markdown, without its menus
    allow(monkeypatch, site)
    workspace = Workspace(tmp_path, Path(os.environ["LEAT_SANDBOX"]))
    result = web.fetch(f"{site}/article", workspace)
    assert result.info["title"] == "The story"  # its site's name gone
    assert "Home" not in result.content and "Copyright" not in result.content
    assert "# The story\n\nThe council met on Tuesday" in result.content
    assert result.content.endswith(f"from [the spring]({site}/spring).")
    # the page kept, to read again, as result.info names it; the HTML gone
    assert [p.suffix for p in workspace.path(web.SAVED).iterdir()] == [".md"]
    assert workspace.path(result.info["saved"]).read_text().startswith("# The story")
    # a long one too, whole, its start read and the rest saved
    content = web.fetch(f"{site}/long", workspace).content
    assert "# The story" in content and "(The page goes on" in content
    saved = content.split("read ")[-1].split(" from")[0]
    assert workspace.path(saved).read_text().endswith("The end.")
