"""ask_files: every file asked the same by a reader, the answers a table, each row citing its file,
saved as a spreadsheet; how far it is told as it goes, and a stop heeded. Saving the spreadsheet
needs LEAT_SANDBOX, as tests/test_files.py's documents do."""

import json
import os
import threading
from pathlib import Path

import pytest

from leat.agent.agent import ASK, Agent, _system
from leat.agent.client import Client
from leat.agent.index import Index
from leat.agent.store import Store
from leat.agent.tools import Context, ask, files
from leat.agent.workspace import Workspace
from tests.test_agent import call, ended, until
from tests.test_files import documents as needs_sandbox

SANDBOX = os.environ.get("LEAT_SANDBOX")


@pytest.fixture
def space(tmp_path) -> Workspace:
    root = Workspace(tmp_path / "workspace", Path(SANDBOX) if SANDBOX else None)
    return root.space("people/1")


@pytest.fixture(autouse=True)
def one_at_a_time(monkeypatch):
    # the readers in turn, so that the engine's scripted replies answer the files in order
    monkeypatch.setattr(ask, "READERS", 1)


def answers(*replies: dict) -> list[list[dict]]:
    return [[{"content": json.dumps(reply)}] for reply in replies]


def context(space: Workspace, told: list | None = None, **kwargs) -> Context:
    numbers: dict[str, int] = {}
    cite = lambda url, title: numbers.setdefault(url, len(numbers) + 1)  # noqa: E731
    progress = told.append if told is not None else lambda info: None
    return Context("c", cite, 1, space, progress, **kwargs)


def test_ask(engine, space):
    # each file read by a reader, the question and its file before it, the answers of the keys
    # asked, a row a file, each citing its file; how far it is told as each is read
    files.write(space, "a.txt", "Invoice 1, 5 March, total 900 TL")
    files.write(space, "b.txt", "Invoice 2, 9 March, total 1.200 TL")
    (space.root / "logo.png").write_bytes(b"\x89PNG\x00")  # not asked: an image
    for reply in answers({"date": "5 March", "Total": "900 TL"}, {"Date": "9 March"}):
        engine.replies.put(reply)
    told: list[dict] = []
    result = ask.ask(Client(engine.url), None, context(space, told), "Each invoice's date and "
                     "total?", ["Date", "Total"], ["a.txt", "b.txt"])  # fmt: skip
    assert result.content.startswith(
        "| File | Date | Total |\n| --- | --- | --- |\n"
        "| a.txt [1] | 5 March | 900 TL |\n| b.txt [2] | 9 March |   |")  # fmt: skip
    assert told == [{"done": 0, "total": 2}, {"done": 1, "total": 2}, {"done": 2, "total": 2}]
    asked = engine.requests[0]["messages"]
    assert '["Date", "Total"]' in asked[0]["content"]
    assert asked[1]["content"].startswith("The question: Each invoice's date and total?\n\n"
                                          "The file, a.txt:\n\nInvoice 1")  # fmt: skip
    assert engine.requests[0]["user"] == "person-1"
    assert [r["file"] for r in result.info["results"]] == ["a.txt", "b.txt"]
    assert result.info["results"][0] == {"n": 1, "url": "file:a.txt", "title": "a.txt",
                                         "file": "a.txt"}  # fmt: skip


def test_every_file(engine, space):
    # without files named, every one with text; without columns, one, the answer; a reply that
    # is not JSON taken whole
    files.write(space, "a.txt", "Renews each year.")
    engine.replies.put([{"content": "It renews itself\neach year."}])
    result = ask.ask(Client(engine.url), None, context(space), "Does it renew itself?")
    assert "| File | Answer |" in result.content
    assert "| a.txt [1] | It renews itself each year. |" in result.content
    assert ask.ask(Client(engine.url), None, context(Workspace(space.root / "x")), "?").content == (
        "There are no files to ask.")  # fmt: skip
    with pytest.raises(FileNotFoundError):
        ask.ask(Client(engine.url), None, context(space), "?", files=["nothing.txt"])


def test_unreadable(engine, space):
    # a file that cannot be read, or asked, says so in its row; the others are asked still
    files.write(space, "broken.pdf", "not a PDF")
    files.write(space, "fine.txt", "Fine.")
    engine.replies.put("the model is not loaded")
    result = ask.ask(
        Client(engine.url), None, context(space), "?", files=["broken.pdf", "fine.txt"]
    )
    assert "| broken.pdf [1] | (It could not be read:" in result.content
    assert "| fine.txt [2] | (It could not be asked: the model is not loaded) |" in result.content


def test_stop(space):
    # once the turn is stopped, no more files are read: the rows so far are kept, and said to be
    for name in ("a.txt", "b.txt", "c.txt"):
        files.write(space, name, name)
    stopped = threading.Event()

    class Stopping(Client):  # a reader whose first answer comes as the turn is stopped
        def reply(self, body, person=None):
            stopped.set()
            return {"content": '{"Answer": "A"}'}

    asking = Context("c", workspace=space, stopped=stopped)
    result = ask.ask(Stopping("http://127.0.0.1:9"), None, asking, "?")
    assert "| a.txt | A |" in result.content and "b.txt" not in result.content
    assert "(Stopped after 1 of the 3 files.)" in result.content


def test_long_file(engine, space, tmp_path):
    # a file longer than a reader reads: its start, and the passages likeliest to answer
    index = Index(tmp_path / "index.db", Workspace(tmp_path / "workspace"))
    text = "Lease.\n" + "\n".join(f"Clause {i}: nothing of note." for i in range(2000))
    files.write(space, "lease.txt", text + "\nClause 2000: the deposit is two months' rent.")
    index.update("people/1")
    engine.replies.put(answers({"Answer": "Two months"})[0])
    ask.ask(Client(engine.url), index, context(space), "What is the deposit?")
    read = engine.requests[0]["messages"][1]["content"]
    assert len(read) < ask.READ + 200 and "Lease.\nClause 0" in read
    assert "\n\n[…]\n\n" in read and read.endswith("Clause 2000: the deposit is two months' rent.")


def test_parsed():
    keys = ["Date", "Total"]
    assert ask._parsed('Here: {"date": "5 March", "total": 900}', keys) == {
        "Date": "5 March", "Total": "900"}  # fmt: skip
    assert ask._parsed('{"Date": null}', keys) == {"Date": "", "Total": ""}
    assert ask._parsed("[1, 2]", keys) == {"Date": "[1, 2]", "Total": ""}
    assert ask._named("Each invoice's date, and total?") == "Each invoice's date and total"


@needs_sandbox
def test_saved(engine, space):
    # the table saved as a spreadsheet, by a name free in the space, which the call says it made
    files.write(space, "a.txt", "Invoice 1")
    engine.replies.put(answers({"Total": "900 | TL"})[0])
    result = ask.ask(Client(engine.url), None, context(space), "Totals?", ["Total"], name="Sums")
    assert result.info["files"] == ["Sums.xlsx"] and "Saved as Sums.xlsx." in result.content
    assert "| a.txt [1] | 900 \\| TL |" in result.content
    read = files.read(space, "Sums.xlsx").content
    assert read.startswith("## Answers\n\n\n| File | Total |\n| --- | --- |\n| a.txt | 900 | TL |")
    assert not any(f["name"].startswith(".ask") for f in space.files())


def test_in_a_turn(engine, tmp_path):
    # the call's message tells the apps how far it is while it runs; the system prompt tells the
    # model when to call it
    workspace = Workspace(tmp_path / "workspace")
    agent = Agent(Store(tmp_path / "leat.db"), Client(engine.url), ask.tools(Client(engine.url)),
                  workspace)  # fmt: skip
    files.write(agent.space(), "a.txt", "A")
    engine.replies.put([{"tool_calls": [call("ask_files", {"question": "What?"})]}])
    engine.replies.put(answers({"Answer": "A"})[0])
    engine.replies.put([{"content": "Done."}])
    with agent.events.watch() as events:
        agent.send(None, "Ask every file")
        seen = until(events, ended)
    shown = [e["message"]["info"] for e in seen if e["type"] == "message"
             and e["message"]["role"] == "tool" and not e["message"]["content"]]  # fmt: skip
    assert [(i.get("done"), i.get("total")) for i in shown] == [(None, None), (0, 1), (1, 1)]
    assert ASK in _system(True, tools={"ask_files"})["content"]
    assert ASK in agent.store.messages(agent.conversations()[0]["id"])[0]["content"]


def test_image(engine, space):
    # an image asked as the reader sees it, among the files asked by default
    (space.root / "receipt.png").write_bytes(b"\x89PNG\x00")
    engine.replies.put(answers({"Total": "900"})[0])
    result = ask.ask(Client(engine.url), None, context(space), "Totals?", ["Total"])
    assert "| receipt.png [1] | 900 |" in result.content
    image, text = engine.requests[0]["messages"][1]["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert text["text"] == "The question: Totals?\n\nThe file, receipt.png, is this image."


def test_folders(engine, space):
    # a folder named asks each of its files, as their names' order has them
    for name in ("March/2.txt", "March/1.txt", "April/3.txt"):
        files.write(space, name, name)
    for _ in range(2):
        engine.replies.put(answers({"Answer": "x"})[0])
    result = ask.ask(Client(engine.url), None, context(space), "?", files=["March"])
    assert [r["file"] for r in result.info["results"]] == ["March/1.txt", "March/2.txt"]
