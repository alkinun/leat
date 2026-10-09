import contextlib
import http.client
import http.cookiejar
import json
import queue
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import ANY

import pytest

from leat.agent import background, context
from leat.agent.agent import LAST, Agent, Busy, NotFound
from leat.agent.client import Client, EngineError
from leat.agent.context import message as _api
from leat.agent.server import BODY, Server
from leat.agent.store import Store
from leat.agent.tools import Result, Tool, strings

HOLD = None  # in a scripted reply: wait there until the test releases it


class FakeEngine(ThreadingHTTPServer):
    """leat serve's API, streaming scripted replies in turn: each a list of deltas, or the message
    of an error to answer with."""

    daemon_threads = True

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []  # each completion's body
        self.replies: queue.SimpleQueue[list[dict[str, Any] | None] | str] = queue.SimpleQueue()
        self.released = threading.Event()
        self.loads: list[str] = []
        self.context: int | None = None  # the model's, said only if set
        self.status = "loaded"  # the model's
        self.vision = False  # whether it sees images
        self.reasoning: dict[str, Any] | None = None  # the efforts it takes, if it is told any
        self.authorized: list[str | None] = []  # each request's Authorization header
        super().__init__(("127.0.0.1", 0), _FakeHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _FakeHandler(BaseHTTPRequestHandler):
    server: FakeEngine

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.server.authorized.append(self.headers.get("Authorization"))
        model = {"id": "fake", "status": self.server.status}
        if self.server.vision:
            model |= {"vision": True, "image_tokens": 300}
        if self.server.reasoning:
            model["reasoning"] = self.server.reasoning
        if self.server.context:
            model["max_context"] = self.server.context
        self._json(200, {"object": "list", "data": [model]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/models/load":
            self.server.loads.append(body["model"])
            return self._json(200, {"id": body["model"], "status": "loaded"})
        self.server.requests.append(body)
        reply = self.server.replies.get(timeout=5)
        if isinstance(reply, str):
            return self._json(400, {"error": {"message": reply}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        finish = "stop"  # unless a delta says otherwise, as {"finish_reason": "length"}
        with contextlib.suppress(OSError):  # a client that stopped the reply
            for delta in reply:
                if delta is HOLD:
                    self.server.released.wait(timeout=5)
                elif "finish_reason" in delta:
                    finish = delta["finish_reason"]
                else:
                    self._chunk({"choices": [{"index": 0, "delta": delta}]})
            # the prompt's tokens, three characters each, none cached
            read = len(json.dumps(body["messages"])) // 3
            timings = {"predicted_n": 5, "predicted_ms": 100.0, "cache_n": 0, "prompt_n": read}
            choice = {"index": 0, "delta": {}, "finish_reason": finish}
            self._chunk({"choices": [choice], "timings": timings})
            self.wfile.write(b"data: [DONE]\n\n")

    def _chunk(self, chunk: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps({'model': 'fake'} | chunk)}\n\n".encode())

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def echo(context, text: str) -> Result:
    return Result(f"echo: {text}", {"echoed": text})


@pytest.fixture
def agent(engine, tmp_path) -> Agent:
    tool = Tool("echo", "Says the text again", strings(text="what to say"), echo)
    return Agent(Store(tmp_path / "leat.db"), Client(engine.url), [tool])


@pytest.fixture
def events(agent) -> Iterator[queue.SimpleQueue]:
    with agent.events.watch() as events:
        yield events


def until(events: queue.SimpleQueue, done: Callable[[dict], bool]) -> list[dict]:
    """The events up to the first that is `done`."""
    seen: list[dict] = []
    while not seen or not done(seen[-1]):
        seen.append(events.get(timeout=5))
    return seen


def stamped(agent: Agent, id: str, index: int) -> str:
    """The time a stored message of the user's begins with, as the model reads it."""
    return context.stamp(agent.store.messages(id)[index]["info"]["at"]) + "\n"


def ended(event: dict) -> bool:
    # the end of a turn: its conversation no longer running, or gone
    c = event.get("conversation")
    return event["type"] == "deleted" or event["type"] == "conversation" and not c["running"]


REPLY = [{"reasoning_content": "Hmm."}, {"content": "Hello"}, {"content": " there."}]


def call(name: str, arguments: Any, id: str = "call_1") -> dict:
    """A reply's call of a tool, as leat serve sends it, its arguments as JSON text."""
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"index": 0, "id": id, "type": "function", "function": {"name": name, "arguments": text}}


def test_turn(agent, engine, events):
    # a message starts a conversation, whose reply streams into it and stays
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi\nand more")
    seen = until(events, ended)
    assert [e["type"] for e in seen[:3]] == ["conversation", "message", "message"]
    summary = {"id": id, "title": "Hi", "updated": 0, "running": True}
    assert seen[0]["conversation"] | {"updated": 0} == summary
    user = {"role": "user", "content": "Hi\nand more",
            "info": {"at": pytest.approx(time.time(), abs=5)}}  # fmt: skip
    assert seen[1]["message"] == user
    assert seen[2]["index"] == 2 and seen[2]["message"]["content"] == ""
    messages = agent.store.messages(id)
    system, user, reply = messages
    assert system["role"] == "system"
    assert "Each of the user's messages begins with the date and time" in system["content"]
    assert (reply["reasoning_content"], reply["content"]) == ("Hmm.", "Hello there.")
    info = reply["info"]
    assert info["model"] == "fake" and info["tokens"] == 5 and info["rate"] == pytest.approx(40.0)
    read = len(json.dumps(engine.requests[0]["messages"])) // 3
    assert info["first"] >= 0 and (info["cached"], info["read"]) == (0, read)
    assert seen[-2] == {"type": "message", "conversation": id, "index": 2, "message": reply,
                        "to": None}  # fmt: skip
    # the model read the system prompt and the message, at its own effort, of a model told none,
    # sampled as leat serve has the model, of which the request says nothing
    (request,) = engine.requests
    assert request["messages"] == [_api(m) for m in messages[:2]] and request["stream"] is True
    assert request["messages"][1]["content"] == stamped(agent, id, 1) + "Hi\nand more"
    assert request["chat_template_kwargs"] == {"preserve_thinking": True}
    assert "reasoning_effort" not in request
    assert not {"temperature", "top_k", "top_p", "min_p", "presence_penalty"} & request.keys()


def test_effort(agent, engine, events):
    # a message's effort, which the engine gives as near as the model can, or else the model's
    # default
    engine.reasoning = {"efforts": ["none", "high"], "default": "high"}
    for effort in (None, "none", "medium"):
        engine.replies.put(REPLY)
        id = agent.send(None, "Prove it.", effort)
        until(events, ended)
        assert engine.requests[-1].get("reasoning_effort") == effort
        assert agent.store.messages(id)[1]["info"].get("effort") == effort


def test_deltas(agent, engine, events):
    # each delta says where its text goes, so that an app that has some already, read with the
    # conversation, takes only what it lacks
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    deltas = [e for e in until(events, ended) if e["type"] == "delta"]
    reply = {"reasoning_content": "", "content": ""}
    for delta in deltas + deltas[1:]:  # some twice
        reply[delta["key"]] = reply[delta["key"]][: delta["at"]] + delta["text"]
    assert reply == {k: agent.store.messages(id)[2][k] for k in reply}


def test_next_turn(agent, engine, events):
    # the next turn sends the conversation so far, the reply's reasoning too but not its info
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    until(events, ended)
    engine.replies.put([{"content": "Fine."}])
    assert agent.send(id, "How are you?") == id
    until(events, ended)
    sent = engine.requests[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1] == {"role": "user", "content": stamped(agent, id, 1) + "Hi"}
    assert sent[2] == {"role": "assistant", "content": "Hello there.", "reasoning_content": "Hmm."}
    assert [m["content"] for m in agent.store.messages(id)][3:] == ["How are you?", "Fine."]
    # and the conversations, the latest updated first, are kept for the next run
    other = agent.send(None, "Another")
    engine.replies.put(REPLY)
    until(events, ended)
    again = Agent(agent.store, agent.engine)
    assert [c["id"] for c in again.conversations()] == [other, id]


def test_tools(agent, engine, events):
    # a reply's calls run, and the model replies again, reading their answers, until it calls none
    engine.replies.put([{"content": "Let me see."}, {"tool_calls": [call("echo", {"text": "hi"})]}])
    engine.replies.put([{"content": "It said hi."}])
    id = agent.send(None, "Echo hi")
    until(events, ended)
    _, _, step, answer, reply = agent.store.messages(id)
    function = {"name": "echo", "arguments": '{"text": "hi"}'}
    assert step["content"] == "Let me see."
    assert step["tool_calls"] == [{"id": "call_1", "type": "function", "function": function}]
    assert answer == {
        "role": "tool", "tool_call_id": "call_1", "name": "echo", "content": "echo: hi",
        "info": {"arguments": {"text": "hi"}, "echoed": "hi"},
    }  # fmt: skip
    assert reply["content"] == "It said hi."
    first, second = engine.requests
    assert [tool["function"]["name"] for tool in first["tools"]] == ["echo"]
    assert second["messages"][-1] == _api(answer)


def test_empty_reply(agent, engine, events):
    # a reply of nothing, neither text nor a call, as gpt-oss ends its reasoning without a word at
    # times, is asked again, and not kept; twice at most, after which the turn ends as it is
    engine.replies.put([{"reasoning_content": "Let me search again."}])
    engine.replies.put([{"content": "  "}])
    engine.replies.put([{"tool_calls": [call("echo", {"text": "hi"})]}])
    engine.replies.put([{"content": "It said hi."}])
    id = agent.send(None, "Echo hi")
    until(events, ended)
    assert len(engine.requests) == 4
    assert engine.requests[0]["messages"] == engine.requests[2]["messages"]  # the same prompt
    roles = [(m["role"], m.get("content")) for m in agent.store.messages(id)[2:]]
    assert roles == [("assistant", ""), ("tool", "echo: hi"), ("assistant", "It said hi.")]
    for _ in range(3):
        engine.replies.put([{"reasoning_content": "Hmm."}])
    other = agent.send(None, "Again")
    until(events, ended)
    last = agent.store.messages(other)[-1]
    assert len(engine.requests) == 7 and last["reasoning_content"] == "Hmm."


def test_bad_calls(agent, engine, events):
    # a call the tools cannot answer is answered with why, for the model to try again
    calls = [call("nothing", {}, "a"), call("echo", "{not JSON", "b"), call("echo", [1], "c"),
             call("echo", {"words": "hi"}, "d")]  # fmt: skip
    engine.replies.put([{"tool_calls": calls}])
    engine.replies.put([{"content": "Sorry."}])
    id = agent.send(None, "Hi")
    until(events, ended)
    answers = [m["content"] for m in agent.store.messages(id) if m["role"] == "tool"]
    assert answers[0] == "error: there is no tool 'nothing'"
    assert answers[1] == "error: the arguments must be a JSON object, not '{not JSON'"
    assert answers[2] == "error: the arguments must be a JSON object, not [1]"
    assert answers[3].startswith("error: ") and "'words'" in answers[3]


def test_calls_at_once(agent, engine, events):
    # a reply's calls run at once: each of these waits for the other
    both = threading.Barrier(2, timeout=5)
    meet = Tool("meet", "Waits for another", strings(), lambda c: Result(str(both.wait())))
    agent.tools["meet"] = meet
    engine.replies.put([{"tool_calls": [call("meet", {}, "a"), call("meet", {}, "b")]}])
    engine.replies.put([{"content": "Met."}])
    id = agent.send(None, "Meet")
    until(events, ended)
    answers = [m["content"] for m in agent.store.messages(id) if m["role"] == "tool"]
    assert sorted(answers) == ["0", "1"]


def test_stop_in_call(agent, engine, events):
    # a stop while a tool runs answers its call so, and ends the turn
    released = threading.Event()
    slow = Tool("slow", "Takes its time", strings(), lambda c: Result(str(released.wait(5))))
    agent.tools["slow"] = slow
    engine.replies.put([{"tool_calls": [call("slow", {})]}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "message" and e["message"]["role"] == "tool")
    assert agent.conversation(id)["messages"][-1]["content"] == ""  # running
    agent.stop(id)
    until(events, ended)
    released.set()
    answer = agent.store.messages(id)[-1]
    assert answer["content"] == "Stopped before it answered." and answer["info"]["stopped"]
    assert len(engine.requests) == 1


def test_rounds(agent, engine, events, monkeypatch):
    # a turn's last reply is told to answer, the tools still declared so that its prompt extends
    # the last; one that calls them anyway is redone without them
    monkeypatch.setattr("leat.agent.agent.ROUNDS", 3)
    for _ in range(2):
        engine.replies.put([{"tool_calls": [call("echo", {"text": "again"})]}])
    engine.replies.put([{"content": "Done."}])
    id = agent.send(None, "Hi")
    until(events, ended)
    assert ["tools" in request for request in engine.requests] == [True, True, True]
    assert engine.requests[-1]["messages"][-1] == {"role": "user", "content": LAST}
    previous = engine.requests[-2]["messages"]
    assert engine.requests[-1]["messages"][: len(previous)] == previous
    for _ in range(3):
        engine.replies.put([{"tool_calls": [call("echo", {"text": "again"})]}])
    engine.replies.put([{"content": "Done at last."}])
    agent.send(id, "Again")
    until(events, ended)
    assert ["tools" in request for request in engine.requests[3:]] == [True, True, True, False]
    *_, calls, answer = agent.store.messages(id)
    assert calls["role"] == "tool" and answer["content"] == "Done at last."


def test_people(agent, engine, events):
    # all that was no one's becomes the first person's, the owner's; then each person's
    # conversations are their own, and each event says whose it is
    engine.replies.put(REPLY)
    before = agent.send(None, "Hi")
    until(events, ended)
    owner, ada = agent.store.add_person("Alkın"), agent.store.add_person("Ada")
    assert (owner["owner"], ada["owner"]) == (1, 0)
    assert [c["id"] for c in agent.conversations(owner["id"])] == [before]
    assert agent.conversations(ada["id"]) == [] and agent.conversations() == []
    engine.replies.put(REPLY)
    with agent.events.watch() as told:
        mine = agent.send(None, "Hello", person=ada["id"])
        assert {e.get("to") for e in until(told, ended)} == {ada["id"]}
    assert " The user is Ada. " in agent.store.messages(mine)[0]["content"]
    for other in (owner["id"], None):
        assert agent.conversation(mine, other) is None
        with pytest.raises(NotFound):
            agent.send(mine, "Hi", person=other)
        with pytest.raises(NotFound):
            agent.delete(mine, other)
        with pytest.raises(NotFound):
            agent.stop(mine, other)


def test_person_to_the_engine(agent, engine, events):
    # each request of a person's conversation names them to the engine, which keeps the prefixes
    # it caches of their prompts to theirs: their turns', and the naming of their conversations;
    # one of no one's names no one
    engine.replies.put(REPLY)
    agent.send(None, "Hi")
    until(events, ended)
    agent.store.add_person("Alkın")
    ada = agent.store.add_person("Ada")
    engine.replies.put([{"tool_calls": [call("echo", {"text": "hi"})]}])
    engine.replies.put(REPLY)
    id = agent.send(None, "Hello", person=ada["id"])
    until(events, ended)
    engine.replies.put([{"content": "Greetings"}])
    background.name(agent, id)
    assert "user" not in engine.requests[0]
    assert [r["user"] for r in engine.requests[1:]] == [f"person-{ada['id']}"] * 3


def test_name(agent, engine, events):
    # a conversation named after its first exchange, once
    engine.replies.put(REPLY)
    id = agent.send(None, "When should I plant tulip bulbs?")
    until(events, ended)
    engine.replies.put([{"content": "“Planting tulip bulbs.”\nMore"}])
    background.name(agent, id)
    asked = engine.requests[-1]["messages"]
    assert asked[0]["content"] == background.NAME
    assert engine.requests[-1]["reasoning_effort"] == "none"  # the least the model takes
    said = "When should I plant tulip bulbs?\n\nAssistant: Hello there."
    assert asked[1]["content"] == f"User: {stamped(agent, id, 1)}{said}"
    assert agent.conversations()[0]["title"] == "Planting tulip bulbs"
    assert agent.store.unnamed() == []
    assert until(events, lambda e: e["type"] == "conversation")[-1]["conversation"]["title"] == (
        "Planting tulip bulbs")  # fmt: skip
    # one the model gives no name keeps its own, and is not named again
    engine.replies.put(REPLY)
    other = agent.send(None, "Hi")
    until(events, ended)
    engine.replies.put([{"content": "  "}])
    background.name(agent, other)
    assert agent.conversation(other)["title"] == "Hi" and agent.store.unnamed() == []


def test_background_failures(agent, engine, events, monkeypatch):
    # a naming that fails leaves the others done
    ids = []
    for text in ("One", "Two"):
        engine.replies.put([{"content": "Hi."}])
        ids.append(agent.send(None, text))
        until(events, ended)
    named = []

    def name(agent, id):
        if id == ids[1]:  # the latest updated, named first
            raise RuntimeError("a bug")
        named.append(id)

    monkeypatch.setattr(background, "name", name)
    agent.background = background.Background(agent)
    agent.background.start()
    deadline = time.time() + 5
    while not named and time.time() < deadline:
        time.sleep(0.05)
    assert named == [ids[0]]


def test_background(agent, engine, events):
    # once started, the agent names the conversations it has not, and each after its first turn
    engine.replies.put(REPLY)
    agent.send(None, "Hello")
    until(events, ended)
    engine.replies.put([{"content": "Hello again"}])  # its name, at the first look
    agent.background = background.Background(agent)
    agent.background.start()
    until(
        events,
        lambda e: e["type"] == "conversation" and e["conversation"]["title"] == "Hello again",
    )
    engine.replies.put(REPLY)
    engine.replies.put([{"content": "Greetings"}])  # its name
    agent.send(None, "Hi")
    until(
        events, lambda e: e["type"] == "conversation" and e["conversation"]["title"] == "Greetings"
    )
    assert len(engine.requests) == 4


def test_migration(tmp_path):
    # a state of the first version, conversations alone, is brought up to date, keeping them
    db = sqlite3.connect(tmp_path / "leat.db")
    db.executescript("""
        CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL NOT NULL,
                                    updated REAL NOT NULL);
        CREATE TABLE messages (conversation TEXT NOT NULL REFERENCES conversations (id)
                               ON DELETE CASCADE, position INTEGER NOT NULL, message TEXT NOT NULL,
                               PRIMARY KEY (conversation, position));
        INSERT INTO conversations VALUES ('0123456789ab', 'Bulbs', 0, 0);
        INSERT INTO messages VALUES ('0123456789ab', 0, '{"role": "system", "content": "tulips"}'),
                                    ('0123456789ab', 1, '{"role": "user", "content": "Tulips?"}');
    """)  # fmt: skip
    db.close()
    store = Store(tmp_path / "leat.db")
    assert [m["content"] for m in store.messages("0123456789ab")] == ["tulips", "Tulips?"]
    assert Store(tmp_path / "leat.db").conversations()[0]["id"] == "0123456789ab"
    # of the household's features gone, nothing is left
    tables = {name for (name,) in store._db.execute("SELECT name FROM sqlite_schema")}
    assert {t for t in tables if not t.startswith("sqlite_")} == {
        "conversations", "messages", "people", "devices"}  # fmt: skip
    columns = [row[1] for row in store._db.execute("PRAGMA table_info(conversations)")]
    assert columns == ["id", "title", "created", "updated", "context", "named", "person"]
    # and a conversation deleted takes its messages with it, as the state's references hold
    store.delete("0123456789ab")
    assert store._db.execute("SELECT count(*) FROM messages").fetchone() == (0,)


def test_clearing(agent, engine, events):
    # past COMPACT of the context, a tool's long answer before the latest messages is cleared
    engine.context = 3000
    page = "x" * 6000  # some 2,000 tokens
    agent.tools["page"] = Tool("page", "A long page", strings(), lambda c: Result(page))
    engine.replies.put([{"tool_calls": [call("page", {})]}])
    engine.replies.put([{"content": "Read it."}])
    id = agent.send(None, "Read the page")
    until(events, ended)
    assert engine.requests[1]["messages"][-1]["content"] == page  # read whole while it is new
    engine.replies.put([{"content": "Sure."}])
    agent.send(id, "Thanks!")
    until(events, ended)
    cleared = [m for m in engine.requests[2]["messages"] if m["role"] == "tool"]
    assert cleared[0]["content"] == context.CLEARED
    assert agent.store.messages(id)[3]["content"] == page  # kept whole, for the app
    assert agent.store.context(id)["cleared"] == 4 and "summary" not in agent.store.context(id)


def test_summary(agent, engine, events):
    # what clearing cannot make small enough, the model summarizes, of a transcript where the
    # conversation and its summary would not fit the context, and the prompt goes on from it
    engine.context = 3000
    engine.replies.put([{"content": "Noted."}])
    id = agent.send(None, "a" * 3600)
    until(events, ended)
    engine.replies.put([{"content": "Goal: the user writes long."}])  # the summary
    engine.replies.put([{"content": "Noted again."}])
    with agent.events.watch() as seen:
        agent.send(id, "b" * 3600)
        compacted = until(seen, lambda e: e["type"] == "compacted")[-1]
        until(seen, ended)
    summarizing, reply = engine.requests[1:]
    assert summarizing["messages"][0]["content"] == context.SUMMARIZE
    assert f"User: {stamped(agent, id, 1)}" + "a" * 3600 in summarizing["messages"][1]["content"]
    system, last = reply["messages"]
    assert system["content"].endswith("go on from the summary:\n\nGoal: the user writes long.")
    assert last == {"role": "user", "content": stamped(agent, id, 3) + "b" * 3600}
    assert compacted["summarized"] == 3 == agent.conversation(id)["summarized"]


def test_stop_while_summarizing(agent, engine, events):
    # a stop ends the turn as the model summarizes, keeping no summary it cut short, nor asking
    # for the reply
    engine.context = 3000
    engine.replies.put([{"content": "Noted."}])
    id = agent.send(None, "a" * 3000)
    until(events, ended)
    engine.replies.put([{"content": "Goal: the user"}, HOLD, {"content": " writes long."}])
    agent.send(id, "b" * 3000)
    while len(engine.requests) < 2:  # the summary asked for
        time.sleep(0.01)
    agent.stop(id)
    until(events, ended)
    engine.released.set()
    assert len(engine.requests) == 2 and "summary" not in agent.store.context(id)
    assert agent.store.messages(id)[-1]["info"]["stopped"]


def test_summary_in_place(agent, engine, events):
    # a summary asked at the end of the conversation's prompt, as the engine's cache holds it,
    # when the prompt and the summary fit the context; its tools declared, that it extends the last
    engine.context = 4400
    engine.replies.put([{"content": "Noted."}])
    id = agent.send(None, "a" * 3600)
    until(events, ended)
    engine.replies.put([{"content": "Goal: the user writes long."}])  # the summary
    engine.replies.put([{"content": "Noted again."}])
    agent.send(id, "b" * 3600)
    until(events, ended)
    first, summarizing, reply = engine.requests
    asked = summarizing["messages"]
    assert asked[-1] == {"role": "user", "content": context.IN_PLACE}
    assert asked[:2] == first["messages"] and summarizing["tools"] == first["tools"]
    assert reply["messages"][0]["content"].endswith("Goal: the user writes long.")


def test_cut_off(agent, engine, events):
    # a reply the context cuts off is redone in its place, once the prompt is smaller
    engine.context = 3000
    agent.tools["page"] = Tool("page", "A long page", strings(), lambda c: Result("x" * 2400))
    engine.replies.put([{"tool_calls": [call("page", {})]}])
    engine.replies.put([{"content": "Read it."}])
    id = agent.send(None, "Read the page")
    until(events, ended)
    engine.replies.put([{"content": "Th"}, {"finish_reason": "length"}])
    engine.replies.put([{"content": "There."}])
    agent.send(id, "And?")
    until(events, ended)
    *_, reply = agent.store.messages(id)
    assert reply["content"] == "There." and "cut" not in reply["info"]
    assert engine.requests[-1]["messages"][3]["content"] == context.CLEARED
    # one cut off again after that is said to be cut off
    engine.replies.put([{"content": "Th"}, {"finish_reason": "length"}])
    engine.replies.put([{"content": "Goal: the page."}])  # the summary that makes it smaller
    engine.replies.put([{"content": "Th"}, {"finish_reason": "length"}])
    agent.send(id, "And then?")
    until(events, ended)
    assert agent.store.messages(id)[-1]["info"]["cut"] is True


def test_citations(agent, engine, events):
    # sources numbered across the conversation, from 1: one read again keeps its number
    def read(context, url):
        n = context.cite(url, "A page")
        return Result(f"[{n}] A page", {"url": url, "n": n})

    agent.tools["read"] = Tool("read", "Reads a page", strings(url="its address"), read)
    id = None
    for urls in (["a", "b"], ["b", "c"]):
        for i, url in enumerate(urls):
            engine.replies.put([{"tool_calls": [call("read", {"url": url}, f"{url}{i}")]}])
        engine.replies.put([{"content": "Read."}])
        id = agent.send(id, f"Read {' and '.join(urls)}")
        until(events, ended)
    assert [m["info"]["n"] for m in agent.store.messages(id) if m["role"] == "tool"] == [1, 2, 2, 3]


def test_live_reply(agent, engine, events):
    # while a reply streams, the conversation shows it, and another message must wait for it
    engine.replies.put([{"content": "Hel"}, HOLD, {"content": "lo"}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "delta")
    c = agent.conversation(id)
    assert c["running"] and c["messages"][-1]["content"] == "Hel"
    with pytest.raises(Busy):
        agent.send(id, "Hello?")
    engine.released.set()
    until(events, ended)
    assert agent.conversation(id)["messages"][-1]["content"] == "Hello"
    with pytest.raises(NotFound):
        agent.send("0" * 12, "Hi")


def test_stop(agent, engine, events):
    # a stopped reply keeps what it had, and says it was stopped
    engine.replies.put([{"content": "Hel"}, HOLD, {"content": "lo"}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "delta")
    agent.stop(id)
    until(events, ended)
    reply = agent.store.messages(id)[-1]
    assert reply["content"] == "Hel" and reply["info"]["stopped"] is True


def test_stop_in_prefill(agent, engine, events):
    # a stop before the first token ends the reply too, as one waiting for a long prompt's prefill
    engine.replies.put([HOLD, {"content": "Hello"}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "message" and e["message"]["role"] == "assistant")
    time.sleep(0.1)  # into the completion
    agent.stop(id)
    until(events, ended)
    assert agent.store.messages(id)[-1]["content"] == ""


def test_delete(agent, engine, events):
    # a conversation deleted while its reply streams is gone, the reply too
    engine.replies.put([{"content": "Hel"}, HOLD])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "delta")
    agent.delete(id)
    assert until(events, ended)[-1] == {"type": "deleted", "id": id, "to": None}
    assert agent.conversations() == [] and agent.store.messages(id) == []


def test_failure(agent, engine, events):
    # a turn that fails is taken back whole, saying why, with the message to send again; the
    # conversation it began too
    engine.replies.put("the prompt is too long")
    id = agent.send(None, "Hi")
    error, deleted = until(events, ended)[-2:]
    assert error == {"type": "error", "conversation": id, "error": "the prompt is too long",
                     "start": 1, "content": "Hi", "to": None}  # fmt: skip
    assert deleted == {"type": "deleted", "id": id, "to": None} and agent.conversations() == []
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    until(events, ended)
    engine.replies.put("the prompt is too long")
    agent.send(id, "More")
    assert until(events, ended)[-2]["start"] == 3
    assert len(agent.store.messages(id)) == 3 and not agent.conversation(id)["running"]


def test_failure_after_summary(agent, engine, events):
    # a turn taken back leaves the prompt's state as it was before it, its summary gone too
    engine.context = 3000
    engine.replies.put([{"content": "Noted."}])
    id = agent.send(None, "a" * 3000)
    until(events, ended)
    before = agent.store.context(id)
    engine.replies.put([{"content": "Goal: the user writes long."}])  # the summary
    engine.replies.put("the engine broke")
    agent.send(id, "b" * 3000)
    until(events, ended)
    assert agent.store.context(id) == before and agent.conversation(id)["summarized"] is None


def test_unreachable(tmp_path, engine):
    # an engine that is not there fails the turn, saying so
    engine.shutdown()
    engine.server_close()
    agent = Agent(Store(tmp_path / "leat.db"), Client(engine.url))
    with agent.events.watch() as events:
        agent.send(None, "Hi")
        error = until(events, ended)[-2]
    assert error["type"] == "error" and "is not reachable" in error["error"]
    models = agent.models_event()
    assert models["models"] == [] and "is not reachable" in models["error"]


def test_models(agent, engine, events):
    assert agent.models_event()["models"] == [{"id": "fake", "status": "loaded"}]
    agent.load("other")
    assert engine.loads == ["other"]
    assert [e["type"] for e in until(events, lambda e: e["type"] == "models")] == [
        "loading", "models"]  # fmt: skip
    # the apps are told of the models as they change, as when the engine loads one or comes up
    agent.models_changed()
    assert events.empty()  # as they were told
    engine.status = "loading"
    agent.models_changed()
    assert events.get(timeout=1)["models"] == [{"id": "fake", "status": "loading"}]


def test_engine_key(engine):
    # the engine's API key, if it asks for one, with every request
    Client(engine.url, "leat-secret").models()
    Client(engine.url).models()
    assert engine.authorized == ["Bearer leat-secret", None]


def test_engine_stuck(tmp_path, monkeypatch):
    # an engine that takes the request but never answers is as good as away, once LIST has passed
    monkeypatch.setattr("leat.agent.client.LIST", 0.2)
    with socket.create_server(("127.0.0.1", 0)) as stuck:
        client = Client(f"http://127.0.0.1:{stuck.getsockname()[1]}")
        start = time.monotonic()
        with pytest.raises(EngineError, match="timed out"):
            client.models()
        assert time.monotonic() - start < 2


def test_hang_up_is_quiet(agent, capsys):
    # a page closed before its answer was written is no error to print; any other error is
    with Server(agent, port=0) as server:
        for error in (BrokenPipeError(), ConnectionResetError(), ValueError("a bug")):
            try:
                raise error
            except Exception:
                server.handle_error(None, ("127.0.0.1", 1))
    printed = capsys.readouterr().err
    assert "ValueError: a bug" in printed and "Broken" not in printed and "Reset" not in printed


@contextlib.contextmanager
def serving(agent: Agent) -> Iterator[str]:
    """The agent's server, its owner set up, whose cookie urllib's requests send from
    then on, as a browser's do."""
    with Server(agent, port=0) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        processor = urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        urllib.request.install_opener(urllib.request.build_opener(processor))
        try:
            assert request(f"{url}/api/setup", "POST", {"name": "Alkın"})[0] == 200
            yield url
        finally:
            urllib.request.install_opener(None)  # type: ignore[arg-type]
            server.shutdown()


def cookie() -> str:
    """The Cookie header of the device serving() set up."""
    opener = urllib.request._opener  # type: ignore[attr-defined]
    jar = next(h.cookiejar for h in opener.handlers if hasattr(h, "cookiejar"))
    return "; ".join(f"{c.name}={c.value}" for c in jar)


@pytest.fixture
def server(agent) -> Iterator[str]:
    with serving(agent) as url:
        yield url


def request(url: str, method: str = "GET", body: Any = None, **headers: str) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data, headers, method=method)) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_app(server):
    for path in ("/", "/c/0123456789ab"):
        with urllib.request.urlopen(f"{server}{path}") as response:
            assert response.headers["Content-Type"] == "text/html; charset=utf-8"
            assert b"<title>leat</title>" in response.read()
    for path in ("/app.mjs", "/markdown.mjs"):
        with urllib.request.urlopen(f"{server}{path}") as response:
            assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
    with urllib.request.urlopen(f"{server}/vendor/temml/Temml.woff2") as response:
        assert response.headers["Content-Type"] == "font/woff2"
    with urllib.request.urlopen(f"{server}/logo.svg") as response:
        assert response.headers["Content-Type"] == "image/svg+xml"
    for path in ("/server.py", "/vendor/temml/LICENSE", "/../store.py", "/api/nothing"):
        assert request(f"{server}{path}")[0] == 404


def test_api(server, engine, agent, events):
    engine.replies.put(REPLY)
    status, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    id = json.loads(body)["id"]
    assert status == 200 and agent.conversation(id, 1) is not None  # the owner's
    until(events, ended)
    # what is wrong with a request, said
    assert request(f"{server}/api/conversations", "POST", {"content": " "})[0] == 400
    assert request(f"{server}/api/conversations", "POST", [1])[0] == 400
    asked = {"content": "Hi", "effort": "max"}
    assert request(f"{server}/api/conversations", "POST", asked)[0] == 400
    assert request(f"{server}/api/conversations/0123456789ab/messages", "POST",
                   {"content": "Hi"})[0] == 404  # fmt: skip
    assert request(f"{server}/api/conversations/0123456789ab")[0] == 404
    status, body = request(f"{server}/api/conversations/{id}")
    assert status == 200 and json.loads(body)["messages"][1]["content"] == "Hi"
    assert request(f"{server}/api/conversations/{id}", "DELETE")[0] == 200
    assert agent.conversations(1) == []


def test_api_busy(server, engine, events):
    engine.replies.put([{"content": "Hel"}, HOLD])
    _, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    id = json.loads(body)["id"]
    until(events, lambda e: e["type"] == "delta")
    url = f"{server}/api/conversations/{id}"
    assert request(f"{url}/messages", "POST", {"content": "Hello?"})[0] == 409
    assert request(f"{url}/stop", "POST", {})[0] == 200
    until(events, ended)


def test_joining(server, agent, engine, events):
    # a device asks to join, showing a code the owner's device shows too; let in as a new person,
    # it sees their own alone, may not do what the owner alone may, and is unpaired by the owner
    jar = urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    phone = urllib.request.build_opener(jar)

    def ask(path: str, method: str = "GET", body: Any = None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        try:
            with phone.open(urllib.request.Request(f"{server}{path}", data, method=method)) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    assert ask("/api/me") == (401, {"error": {"message": ANY}, "empty": False})
    assert ask("/api/setup", "POST", {"name": "Eve"})[0] == 400  # the box has its owner
    status, asked = ask("/api/pairings", "POST", {"name": "Ada"})
    assert status == 200 and len(asked["code"]) == 6
    connection = http.client.HTTPConnection(server.replace("http://", ""))
    connection.request("GET", "/api/events", headers={"Cookie": cookie()})
    response = connection.getresponse()
    while not (line := response.readline()).startswith(b'data: {"type": "accounts"'):
        pass
    (request_,) = json.loads(line[6:])["requests"]
    assert (request_["name"], request_["code"]) == ("Ada", asked["code"])
    connection.close()
    assert ask(f"/api/pairings/{asked['id']}")[0] == 401  # asked after by a POST alone
    assert ask(f"/api/pairings/{asked['id']}", "POST")[0] == 202  # waiting
    status, ada = request(f"{server}/api/pairings/{asked['id']}/allow", "POST", {})
    assert status == 200 and json.loads(ada)["name"] == "Ada"
    assert ask(f"/api/pairings/{asked['id']}", "POST")[0] == 200  # its cookie, taken once
    assert ask(f"/api/pairings/{asked['id']}", "POST")[0] == 404
    assert ask("/api/me") == (200, {"person": 2, "name": "Ada", "owner": False, "device": 2})
    # Ada's own alone: her events tell of her conversations, not the owner's
    adas = "; ".join(f"{c.name}={c.value}" for c in jar.cookiejar)
    connection = http.client.HTTPConnection(server.replace("http://", ""))
    connection.request("GET", "/api/events", headers={"Cookie": adas})
    stream = connection.getresponse()
    engine.replies.put(REPLY)
    _, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    until(events, ended)
    assert ask(f"/api/conversations/{json.loads(body)['id']}")[0] == 404
    engine.replies.put(REPLY)
    _, mine = ask("/api/conversations", "POST", {"content": "Hello"})
    until(events, ended)
    told = []
    while not told or told[-1].get("type") != "conversation":
        if (line := stream.readline()).startswith(b"data: "):
            told.append(json.loads(line[6:]))
    assert told[0] == {"type": "conversations", "conversations": []}
    assert told[-1]["conversation"]["id"] == mine["id"]  # the owner's never told
    connection.close()
    for path, body in [("/api/models/load", {"model": "fake"}),
                       ("/api/pairings/0123456789abcdef/allow", {})]:  # fmt: skip
        assert ask(path, "POST", body)[0] in (403, 404)
    assert ask("/api/models/load", "POST", {"model": "fake"})[0] == 403
    assert ask("/api/people/1", "DELETE")[0] == 403
    # unpaired by the owner, it must ask again; and removed, with all that is hers
    (device,) = [d["id"] for d in agent.store.devices() if d["person"] == 2]
    connection = http.client.HTTPConnection(server.replace("http://", ""), timeout=5)
    connection.request("GET", "/api/events", headers={"Cookie": adas})
    stream = connection.getresponse()
    assert request(f"{server}/api/devices/{device}", "DELETE")[0] == 200
    assert ask("/api/me")[0] == 401
    time.sleep(1.1)  # the stream open before ends, at the next event past a second
    agent.events.publish({"type": "files", "files": [], "to": "owner"})
    assert stream.read().endswith(b"\n\n")  # to its end, which the server closed
    connection.close()
    assert request(f"{server}/api/people/2", "DELETE")[0] == 200
    assert agent.store.person(2) is None and agent.conversations(2) == []
    assert request(f"{server}/api/people/1", "DELETE")[0] == 400  # the owner


def test_trust(server, engine):
    # a request naming the box by an address or a local name, and a write from its own page
    engine.replies.put(REPLY)
    url, own = f"{server}/api/conversations", server.replace("http://", "")
    assert request(f"{server}/", Host="evil.example")[0] == 403  # DNS rebinding
    assert request(f"{server}/", Host="leat.local:8000")[0] == 200
    assert request(f"{server}/", Host="[::1]:8000")[0] == 200
    hi = {"content": "Hi"}
    assert request(url, "POST", hi, Origin="http://evil.example")[0] == 403
    assert request(url, "POST", hi, Origin=f"http://{own}")[0] == 200
    # nor a body larger than any request's, which it does not read: it may come from anyone
    connection = http.client.HTTPConnection(own)
    connection.putrequest("POST", "/api/pairings")
    connection.putheader("Content-Length", str(BODY + 1))
    connection.endheaders()
    assert connection.getresponse().status == 413


def test_events(server, agent, engine):
    # what there is, then what changes
    connection = http.client.HTTPConnection(server.replace("http://", ""))
    connection.request("GET", "/api/events", headers={"Cookie": cookie()})
    response = connection.getresponse()
    assert response.headers["Content-Type"] == "text/event-stream"

    def event() -> dict:
        while not (line := response.readline()).startswith(b"data: "):
            pass
        return json.loads(line[6:])

    assert event() == {"type": "conversations", "conversations": []}
    assert event() == {"type": "files", "files": []}
    assert event()["type"] == "accounts"  # the owner's
    assert event()["type"] == "models"
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi", person=1)
    assert event()["conversation"]["id"] == id
    connection.close()
