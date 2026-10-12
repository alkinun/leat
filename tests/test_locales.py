import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

from leat import locales
from leat.box import watch

ROOT = Path(__file__).parent.parent / "leat"
APP = ROOT / "agent" / "app"
# what is the same in every language: names, and examples of what to type
SAME = {"Leat", "K7X 2QM", "Kanzlei Weber"}
# the screen's words it says of values it is told, as a model's status, and of its link
SCREEN = {"ready", "loading", "Network", "Wired", "Wi-Fi", "{size} GB free", "Setup code",
          "or type {local}.local",
          "Recovery code, for {minutes} more minutes"}  # fmt: skip
# the box's warnings, as leat.box.watch says them, of their values
WARNINGS = {
    "Leat can't be found by its name",
    "The router hides addresses inside the office from the devices here. On a FRITZ!Box, add "
    "leatbox.net under {where}.",
    "The box can't reach the internet",
    "Leat works in the office, but web searches, updates and its certificate wait. Check the box's "
    "cable or Wi-Fi, and the office's firewall.",
    "Leat's certificate expires in {days} days",
    "Leat's certificate has expired",
    "Leat's certificate expires within a day",
    "The box renews it over the internet, and couldn't so far; browsers warn of Leat once it "
    "expires. Check the box reaches the internet.",
    "The box is nearly full",
    "{size} GB are left. Delete files Leat no longer needs.",
    "Leat can't reach its model",
    "Chats get no answer until it is back. Restart the box.",
    "An update failed",
    "The box went back to the version before, as {version} did not start. Leat works as it did.",
}

# the agent's errors a person may see, as the app shows them, of their values
ERRORS = {
    "a person needs a name",
    "a reply is already running",
    "a file may be {size} MB at most",
    "a file's name is {size} bytes at most",
    "a name is {size} characters at most",
    "an office's name is {size} characters at most",
    "a project's instructions are {size} characters at most",
    "a project's name is {size} characters at most",
    "a workflow's name is {name} characters at most, and its request {request}",
    "a device joins as someone with a device of theirs from one of theirs alone",
    "a device joins as you alone, of yours",
    "{name} could not be unpacked: {why}",
    "choose a folder of a name, not a whole disk",
    "{folder} holds more than {n} files: choose a folder in it",
    "{folder} is not a folder on this computer",
    "{folder} is not there: is its drive mounted?",
    "the folder must be named by its whole path, as /mnt/office",
    "the engine at {url} is not reachable: {why}",
    "the engine's reply broke off",
    "the engine's reply broke off: {why}",
    "there is no conversation {id}",
    "there is no project {id}",
    "there is no file {name}",
    "there is no workflow {id}",
    "the request was turned down, or waited too long: ask again",
    "this device has not joined this Leat: ask to join",
    "this Leat is set up already: ask to join",
    "this Leat is not set up: there is no one to recover",
    "this Leat keeps no files",
    "this Leat needs an admin: make someone else one first",
    "this Leat runs on no box of its own, so it cannot do that",
    "this Leat runs on no box of its own, so it cannot set its time zone",
    "that is not the recovery code the box's screen shows",
    "that is not the recovery code, and it holds no more: press the box's power button for another",
    "that is not the setup code the box's screen shows",
    "that is not the setup code: the box's screen shows a new one now",
    "no recovery code holds now: press the box's power button for one",
    "no such device asks to join",
    "only an admin may let someone new in",
    "only this Leat's admins may do that",
    "only whoever made a project may change whom it is shared with",
    "only whoever made a project, or an admin, may delete it",
    "too many devices ask to join now: try again in a few minutes",
    "you cannot remove yourself: ask another admin",
    "type {word} to erase everything on this Leat",
    "a passphrase is 8 to 63 letters, digits and signs",
    "a Wi-Fi network's name is 1 to 32 bytes",
    "the box's hotspot is closed",
}


def keys() -> set[str]:
    """Every text the app, its page, the box's screen and the agent's activity and warnings say,
    as each language's table must have it."""
    app = (APP / "app.mjs").read_text()
    found = {json.loads(f'"{k}"') for k in re.findall(r'\bt\(\s*"((?:[^"\\]|\\.)*)"', app)}
    found |= {k for pair in re.findall(r'\bcount\([^,]+,\s*"([^"]+)",\s*"([^"]+)"\)', app)
              for k in pair}  # fmt: skip
    for table in ("DID", "EVERY", "EFFORTS"):  # tables of words, each t()'d as it is shown
        block = re.search(rf"const {table} = [{{\[](.*?)[}}\]];", app, re.S)[1]
        literals = re.findall(r'"([^"]*)"', block)
        found |= set(literals[1::2] if table == "EVERY" else literals)
    found |= set(re.findall(r'\b(?:title|about|next|placeholder): "([^"]+)"', app))
    found |= {"used {name}"}  # what a turn did of a tool of no words of its own
    found |= _page()
    found |= _said() | SCREEN | WARNINGS | ERRORS | _actions()
    return found - SAME


def test_tables():
    # each language's table has every text, and no other, its values in the same places
    every = keys()
    for language in locales.LANGUAGES[1:]:
        table = json.loads((ROOT / "locales" / f"{language}.json").read_text())
        assert sorted(every - table.keys()) == [], f"{language} lacks these"
        assert sorted(table.keys() - every) == [], f"{language} has these, said nowhere"
        places = lambda text: set(re.findall(r"\{\w+\}", text))  # noqa: E731
        for text, translation in table.items():
            assert places(text) == places(translation), text


def test_translated():
    # a text of values the agent made, in German, its values in their places; one of none, as it is
    assert locales.translated("Leat's certificate expires in 9 days", "de") == (
        "Das Zertifikat von Leat läuft in 9 Tagen ab"
    )
    assert locales.translated("5 GB are left. Delete files Leat no longer needs.", "de").startswith(
        "5 GB"
    )
    assert locales.translated("notes.txt", "de") == "notes.txt"
    assert locales.translated("Leat can't reach its model", "en") == "Leat can't reach its model"
    assert locales.t("{size} GB free", "de", size=12) == "12 GB frei"


def _page() -> set[str]:
    # the texts of the app's page, and of its elements' placeholders, titles and labels
    found: set[str] = set()

    class Page(HTMLParser):
        def handle_starttag(self, tag, attrs):
            named = ("placeholder", "title", "aria-label", "data-empty")
            found.update(v for k, v in attrs if k in named and v)

        def handle_data(self, data):
            if data.strip() and self.lasttag not in ("script", "style", "title"):
                found.add(data.strip())

    Page().feed((APP / "index.html").read_text())
    return found


def _said() -> set[str]:
    # the screen's words: what said() returns, holds() and every t() of a text; and the hotspot
    # page's, of its every t()
    tree = ast.parse((ROOT / "box" / "screen.py").read_text() + "\n"
                     + (ROOT / "box" / "hotspot.py").read_text())  # fmt: skip
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("said", "holds"):
            for part in node.body[1:]:  # but its docstring
                formatted = [n for n in ast.walk(part) if isinstance(n, ast.JoinedStr)]
                inner = {id(v) for f in formatted for v in f.values}
                found |= {
                    n.value
                    for n in ast.walk(part)
                    if isinstance(n, ast.Constant)
                    and isinstance(n.value, str)
                    and n.value
                    and id(n) not in inner
                }
                found |= {s for f in formatted for s in _strings(f)}
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "t" and node.args:
            found |= _strings(node.args[0])
    return {f for f in found if not f.startswith("https://")}


def _actions() -> set[str]:
    # what the agent notes was done, as the activity says it: each note()'s action, and those of
    # the dicts and ternaries the server picks them of
    found = set()
    for path in (ROOT / "agent" / "server.py", ROOT / "agent" / "agent.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "note":
                found |= _strings(node.args[1]) if len(node.args) > 1 else set()
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in (
                    "said", "done"):  # fmt: skip
                found |= _strings(node.value)
    sharing = {
        "shared the project with",
        "stopped sharing the project with",
        "shared the project with everyone",
        "stopped sharing the project with everyone",
    }
    did = {"read", "searched the files", "searched the web", "read a page of", "made"}
    drive = {f"{done} by the network drive" for done in (
        "uploaded", "changed", "deleted", "moved", "copied", "made the project")}  # fmt: skip
    return found | sharing | did | drive


def _strings(node: ast.AST) -> set[str]:
    # the texts an expression may be: of a constant, a ternary's, a dict's values; an f-string's
    # as a text of values, of its places' names
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _strings(node.body) | _strings(node.orelse)
    if isinstance(node, ast.Dict):
        return set().union(*(_strings(v) for v in node.values if v is not None))
    if isinstance(node, ast.JoinedStr):
        return {"".join(v.value if isinstance(v, ast.Constant) else f"{{{_name(v)}}}"
                        for v in node.values)}  # fmt: skip
    return set()


def _name(value: ast.AST) -> str:
    # the name of an f-string's place: its variable's, or its key's
    held = getattr(value, "value", value)
    if isinstance(held, ast.Name):
        return held.id
    return held.slice.value if isinstance(held, ast.Subscript) else "value"  # type: ignore


def test_warnings_translated(tmp_path, monkeypatch):
    # every warning a box may give, as it gives it, in German, whole
    now = [0.0]
    monkeypatch.setattr(watch, "_found", lambda name: False)
    monkeypatch.setattr(watch, "_reached", lambda: False)
    monkeypatch.setattr(watch, "_days_left", lambda name: 3)
    monkeypatch.setattr(
        watch.shutil, "disk_usage", lambda path: SimpleNamespace(total=100e9, free=5e9)
    )
    monkeypatch.setattr(watch.system, "ask", lambda r: {"failed": "2026.10.12.0300-abc1234"})
    seen = watch.Watch("k7x2qm.leatbox.net", tmp_path, lambda: "down", clock=lambda: now[0])
    seen.look()
    now[0] = watch.HELD
    warned = seen.look()
    assert {w["key"] for w in warned} == {
        "name",
        "internet",
        "certificate",
        "disk",
        "model",
        "update",
    }
    for w in warned:
        for text in (w["title"], w["said"]):
            assert locales.translated(text, "de") != text, text
    monkeypatch.setattr(watch, "_days_left", lambda name: 0)
    assert (
        locales.translated(seen.look()[2]["title"], "de")
        == "Das Zertifikat von Leat ist abgelaufen"
    )
