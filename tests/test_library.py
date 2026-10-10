"""The library: a law of gesetze-im-internet.de's XML made markdown, a heading of each paragraph,
kept and listed, found by search_law and cited by its § and law, linked to its page on the site.
Making it needs the sandbox, as tests/test_files.py's does."""

import functools
import http.server
import json
import os
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from leat.agent import library
from leat.agent.agent import LAW, Agent, _system
from leat.agent.client import Client
from leat.agent.index import LIBRARY, Index
from leat.agent.store import Store
from leat.agent.tools import Context
from leat.agent.workspace import Workspace
from tests.test_agent import request, serving
from tests.test_files import sandboxed

SANDBOX = os.environ.get("LEAT_SANDBOX")
# the Bundesurlaubsgesetz's start, as the site's XML has it: the law's own norm, a part's heading,
# paragraphs with numbered paragraphs, a list and a footnote's mark
BURLG = """<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE dokumente SYSTEM "http://www.gesetze-im-internet.de/dtd/1.01/gii-norm.dtd">
<dokumente builddate="20250101000000" doknr="BJNR000020963">
<norm builddate="20250101000000" doknr="BJNR000020963">
<metadaten><jurabk>BUrlG</jurabk><ausfertigung-datum manuell="ja">1963-01-08</ausfertigung-datum>
<kurzue>Bundesurlaubsgesetz</kurzue><langue>Mindesturlaubsgesetz für Arbeitnehmer</langue>
</metadaten><textdaten><text format="XML"><Content><P></P></Content></text></textdaten></norm>
<norm builddate="20250101000000" doknr="BJNR000020963BJNG000100314">
<metadaten><jurabk>BUrlG</jurabk><gliederungseinheit><gliederungskennzahl>010</gliederungskennzahl>
<gliederungsbez>Abschnitt 1</gliederungsbez><gliederungstitel>Allgemeines</gliederungstitel>
</gliederungseinheit></metadaten><textdaten/></norm>
<norm builddate="20250101000000" doknr="BJNR000020963BJNE000100314">
<metadaten><jurabk>BUrlG</jurabk><enbez>§ 1</enbez><titel format="parat">Urlaubsanspruch</titel>
</metadaten><textdaten><text format="XML"><Content><P>Jeder Arbeitnehmer hat in jedem
Kalenderjahr Anspruch auf bezahlten Erholungsurlaub.</P></Content></text></textdaten></norm>
<norm builddate="20250101000000" doknr="BJNR000020963BJNE000300314">
<metadaten><jurabk>BUrlG</jurabk><enbez>§ 3</enbez><titel format="parat">Dauer des Urlaubs</titel>
</metadaten><textdaten><text format="XML"><Content><P>(1) Der Urlaub beträgt jährlich mindestens
24 Werktage.<FnR ID="F1"/></P><P>(2) Als Werktage gelten alle Kalendertage, die nicht Sonn- oder
gesetzliche Feiertage sind.</P></Content></text></textdaten></norm>
<norm builddate="20250101000000" doknr="BJNR000020963BJNE000500314">
<metadaten><jurabk>BUrlG</jurabk><enbez>§ 5</enbez><titel format="parat">Teilurlaub</titel>
</metadaten><textdaten><text format="XML"><Content><P>(1) Anspruch auf ein Zwölftel des
Jahresurlaubs für jeden vollen Monat des Bestehens des Arbeitsverhältnisses hat der Arbeitnehmer
<DL Type="alpha"><DT>a)</DT><DD Font="normal"><LA Size="normal">für Zeiten eines Kalenderjahrs,
für die er wegen Nichterfüllung der Wartezeit keinen vollen Urlaubsanspruch erwirbt;</LA></DD>
<DT>b)</DT><DD Font="normal"><LA Size="normal">wenn er vor erfüllter Wartezeit aus dem
Arbeitsverhältnis ausscheidet.</LA></DD></DL></P></Content></text></textdaten></norm>
</dokumente>
"""


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace", Path(SANDBOX) if SANDBOX else None)


def test_cited():
    # a paragraph by its § and law, linked to its page; an article's too; another place by its law
    assert library._cited("BGB", "§ 622a Kündigungsfristen", "bgb") == {
        "url": "https://www.gesetze-im-internet.de/bgb/__622a.html",
        "title": "§ 622a BGB: Kündigungsfristen", "law": "BGB"}  # fmt: skip
    assert library._cited("GG", "Art 3", "gg")["url"].endswith("/gg/art_3.html")
    assert library._cited("GG", "Art 3", "gg")["title"] == "Art 3 GG"
    assert library._cited("BUrlG", "Anlage", "burlg")["title"] == "BUrlG, Anlage"


def test_told():
    # the model told to answer law from the law's text, never from memory, when it can search it
    assert LAW in _system(True, tools=["search_law"])["content"]
    assert LAW not in _system(True, tools=["search_files"])["content"]


@pytest.fixture
def site(tmp_path, monkeypatch) -> Iterator[None]:
    # gesetze-im-internet.de, as a server here of the Bundesurlaubsgesetz alone
    law = tmp_path / "site" / "burlg"
    law.mkdir(parents=True)
    with zipfile.ZipFile(law / "xml.zip", "w") as z:
        z.writestr("BJNR000020963.xml", BURLG)
    serve = functools.partial(http.server.SimpleHTTPRequestHandler, directory=law.parent)
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), serve) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        monkeypatch.setattr(library, "SITE", f"http://127.0.0.1:{server.server_port}")
        yield
        server.shutdown()


@sandboxed
def test_library(tmp_path, workspace, site):
    # a law fetched, made markdown in the sandbox, listed, found by search_law and removed; one the
    # site has not, or not of a law's name, refused
    kept = library.add(workspace, "BUrlG")
    assert kept["abbreviation"] == "BUrlG" and kept["norms"] == 3
    made = workspace.space(LIBRARY).path("BUrlG.md").read_text()
    assert made.startswith("# Bundesurlaubsgesetz (BUrlG)\n\n## § 1 Urlaubsanspruch\n\nJeder")
    assert "(1) Der Urlaub beträgt jährlich mindestens 24 Werktage.\n\n(2) Als Werktage" in made
    assert "\n\na) für Zeiten eines Kalenderjahrs, für die er wegen" in made
    assert "\n\nb) wenn er vor erfüllter Wartezeit" in made
    assert list(library.laws(workspace)) == ["BUrlG"]
    assert [f["name"] for f in workspace.space(LIBRARY).files()] == ["BUrlG.md"]

    index = Index(tmp_path / "index.db", workspace)
    index.update(LIBRARY)
    numbers: dict[str, int] = {}
    context = Context("c", lambda url, title: numbers.setdefault(url, len(numbers) + 1))
    found = library.search(index, context, "Wie viele Werktage Urlaub mindestens?")
    assert found.content.startswith("[1] § 3 BUrlG: Dauer des Urlaubs\n## § 3 Dauer des Urlaubs")
    assert found.info["results"][0]["url"] == f"{library.SITE}/burlg/__3.html"
    none = library.search(index, context, "Mehrwertsteuer")
    assert none.content == "No paragraph of the laws kept here says that. The laws kept: BUrlG."

    library.remove(workspace, "BUrlG")
    assert library.laws(workspace) == {} and workspace.space(LIBRARY).files() == []
    with pytest.raises(LookupError):
        library.remove(workspace, "BUrlG")
    with pytest.raises(LookupError, match="has no law KSchG"):
        library.add(workspace, "KSchG")
    with pytest.raises(LookupError):
        library.add(workspace, "../etc")


@sandboxed
def test_added(tmp_path, workspace, site):
    # by the owner, through the API: a law added on a thread, the apps told as it begins and how
    # it went, noted; refused while the box is kept from the internet; and removed
    agent = Agent(Store(tmp_path / "leat.db"), Client("http://127.0.0.1:9"), [], workspace,
                  Index(tmp_path / "index.db", workspace))  # fmt: skip
    with serving(agent) as url, agent.events.watch() as events:

        def told() -> dict:  # the library, as the apps are next told of it
            while (event := events.get(timeout=10))["type"] != "library":
                pass
            return event

        assert request(f"{url}/api/library", "POST", {"law": "BUrlG"})[0] == 200
        adding, added = told(), told()
        assert adding["adding"] == ["BUrlG"] and added["adding"] == [] and added["error"] is None
        assert [law["abbreviation"] for law in added["laws"]] == ["BUrlG"]
        assert added["laws"][0]["title"] == "Bundesurlaubsgesetz"
        agent.index.update(LIBRARY)  # read, told to no app, as no app lists the library's files
        assert agent.index.search(workspace.space(LIBRARY), "Teilurlaub")
        request(f"{url}/api/library", "POST", {"law": "KSchG"})
        assert told()["adding"] == ["KSchG"] and "has no law KSchG" in told()["error"]
        agent.set_offline(True)
        assert request(f"{url}/api/library", "POST", {"law": "BGB"})[0] == 400
        assert request(f"{url}/api/library/BUrlG", "DELETE")[0] == 200
        assert library.laws(workspace) == {}
        assert request(f"{url}/api/library/BUrlG", "DELETE")[0] == 404
        activity = json.loads(request(f"{url}/api/activity")[1])["activity"]
        assert [(a["action"], a["detail"]) for a in activity[:2]] == [
            ("removed the law", "BUrlG"), ("added the law", "BUrlG")]  # fmt: skip
