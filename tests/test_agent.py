import contextlib
import datetime
import http.client
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from leat.agent.agent import Agent, Busy, NotFound, _api
from leat.agent.client import Client
from leat.agent.server import Server
from leat.agent.store import Store

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
        super().__init__(("127.0.0.1", 0), _FakeHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _FakeHandler(BaseHTTPRequestHandler):
    server: FakeEngine

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self._json(200, {"object": "list", "data": [{"id": "fake", "status": "loaded"}]})

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
        with contextlib.suppress(OSError):  # a client that stopped the reply
            for delta in reply:
                if delta is HOLD:
                    self.server.released.wait(timeout=5)
                    continue
                self._chunk({"choices": [{"index": 0, "delta": delta}]})
            timings = {"predicted_n": 5, "predicted_ms": 100.0}
            self._chunk({"choices": [{"index": 0, "delta": {}}], "timings": timings})
            self.wfile.write(b"data: [DONE]\n\n")

    def _chunk(self, chunk: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps({'model': 'fake'} | chunk)}\n\n".encode())

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def engine() -> Iterator[FakeEngine]:
    engine = FakeEngine()
    threading.Thread(target=engine.serve_forever, args=(0.01,), daemon=True).start()
    yield engine
    engine.released.set()
    engine.shutdown()
    engine.server_close()


@pytest.fixture
def agent(engine, tmp_path) -> Agent:
    return Agent(Store(tmp_path / "leat.db"), Client(engine.url))


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


def ended(event: dict) -> bool:
    # the end of a turn: its conversation no longer running, or gone
    c = event.get("conversation")
    return event["type"] == "deleted" or event["type"] == "conversation" and not c["running"]


REPLY = [{"reasoning_content": "Hmm."}, {"content": "Hello"}, {"content": " there."}]


def test_turn(agent, engine, events):
    # a message starts a conversation, whose reply streams into it and stays
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi\nand more")
    seen = until(events, ended)
    assert [e["type"] for e in seen[:3]] == ["conversation", "message", "message"]
    assert seen[0]["conversation"] | {"updated": 0} == {
        "id": id, "title": "Hi", "updated": 0, "running": True}  # fmt: skip
    user = {"role": "user", "content": "Hi\nand more", "info": {"think": False}}
    assert seen[1]["message"] == user
    assert seen[2]["index"] == 2 and seen[2]["message"]["content"] == ""
    messages = agent.store.messages(id)
    system, user, reply = messages
    today = datetime.date.today()
    assert system["role"] == "system"
    assert f"Today is {today:%A}, {today.day} {today:%B %Y}." in system["content"]
    assert (reply["reasoning_content"], reply["content"]) == ("Hmm.", "Hello there.")
    assert reply["info"]["model"] == "fake" and reply["info"]["tokens"] == 5
    assert reply["info"]["rate"] == pytest.approx(40.0) and reply["info"]["first"] >= 0
    assert seen[-2] == {"type": "message", "conversation": id, "index": 2, "message": reply}
    # the model read the system prompt and the message, not thinking, sampled as Qwen3.6
    # recommends then
    (request,) = engine.requests
    assert request["messages"] == [_api(m) for m in messages[:2]] and request["stream"] is True
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["temperature"] == 0.7 and request["presence_penalty"] == 1.5


def test_think(agent, engine, events):
    # a message may ask the model to think first, sampled as Qwen3.6 recommends for thinking
    engine.replies.put(REPLY)
    id = agent.send(None, "Prove it.", think=True)
    until(events, ended)
    assert engine.requests[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert engine.requests[-1]["temperature"] == 1.0
    assert agent.store.messages(id)[1]["info"] == {"think": True}


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
    assert sent[1] == {"role": "user", "content": "Hi"}
    assert sent[2] == {"role": "assistant", "content": "Hello there.", "reasoning_content": "Hmm."}
    assert [m["content"] for m in agent.store.messages(id)][3:] == ["How are you?", "Fine."]
    # and the conversations, the latest updated first, are kept for the next run
    other = agent.send(None, "Another")
    engine.replies.put(REPLY)
    until(events, ended)
    again = Agent(agent.store, agent.engine)
    assert [c["id"] for c in again.conversations()] == [other, id]


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
    assert until(events, ended)[-1] == {"type": "deleted", "id": id}
    assert agent.conversations() == [] and agent.store.messages(id) == []


def test_failure(agent, engine, events):
    # a turn that fails is taken back whole, saying why, with the message to send again; the
    # conversation it began too
    engine.replies.put("the prompt is too long")
    id = agent.send(None, "Hi")
    error, deleted = until(events, ended)[-2:]
    assert error == {"type": "error", "conversation": id, "error": "the prompt is too long",
                     "start": 1, "content": "Hi"}  # fmt: skip
    assert deleted == {"type": "deleted", "id": id} and agent.conversations() == []
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    until(events, ended)
    engine.replies.put("the prompt is too long")
    agent.send(id, "More")
    assert until(events, ended)[-2]["start"] == 3
    assert len(agent.store.messages(id)) == 3 and not agent.conversation(id)["running"]


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


@pytest.fixture
def server(agent) -> Iterator[str]:
    with Server(agent, port=0) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        yield f"http://127.0.0.1:{server.server_port}"
        server.shutdown()


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
    with urllib.request.urlopen(f"{server}/app.mjs") as response:
        assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
    with urllib.request.urlopen(f"{server}/vendor/temml/Temml.woff2") as response:
        assert response.headers["Content-Type"] == "font/woff2"
    for path in ("/server.py", "/vendor/temml/LICENSE", "/../store.py", "/api/nothing"):
        assert request(f"{server}{path}")[0] == 404


def test_api(server, engine, agent, events):
    engine.replies.put(REPLY)
    status, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    id = json.loads(body)["id"]
    assert status == 200 and agent.conversation(id) is not None
    until(events, ended)
    # what is wrong with a request, said
    assert request(f"{server}/api/conversations", "POST", {"content": " "})[0] == 400
    assert request(f"{server}/api/conversations", "POST", [1])[0] == 400
    assert request(f"{server}/api/conversations", "POST", {"content": "Hi", "think": 1})[0] == 400
    assert request(f"{server}/api/conversations/0123456789ab/messages", "POST",
                   {"content": "Hi"})[0] == 404  # fmt: skip
    assert request(f"{server}/api/conversations/0123456789ab")[0] == 404
    status, body = request(f"{server}/api/conversations/{id}")
    assert status == 200 and json.loads(body)["messages"][1]["content"] == "Hi"
    assert request(f"{server}/api/conversations/{id}", "DELETE")[0] == 200
    assert agent.conversations() == []


def test_api_busy(server, engine, events):
    engine.replies.put([{"content": "Hel"}, HOLD])
    _, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    id = json.loads(body)["id"]
    until(events, lambda e: e["type"] == "delta")
    url = f"{server}/api/conversations/{id}"
    assert request(f"{url}/messages", "POST", {"content": "Hello?"})[0] == 409
    assert request(f"{url}/stop", "POST", {})[0] == 200
    until(events, ended)


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


def test_events(server, agent, engine):
    # what there is, then what changes
    connection = http.client.HTTPConnection(server.replace("http://", ""))
    connection.request("GET", "/api/events")
    response = connection.getresponse()
    assert response.headers["Content-Type"] == "text/event-stream"

    def event() -> dict:
        while not (line := response.readline()).startswith(b"data: "):
            pass
        return json.loads(line[6:])

    assert event() == {"type": "conversations", "conversations": []}
    assert event()["type"] == "models"
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    assert event()["conversation"]["id"] == id
    connection.close()
