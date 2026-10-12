"""What leat agent's tests share: a fake of leat serve, scripted reply by reply; the agent's
server, set up, and requests to it; and the mark of tests that need the sandbox's environment,
LEAT_SANDBOX, made of leat/agent/sandbox.txt, as its comment says."""

import contextlib
import copy
import http.cookiejar
import json
import os
import queue
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from leat.agent.agent import Agent
from leat.agent.server import Server

SANDBOX = os.environ.get("LEAT_SANDBOX")
needs_sandbox = pytest.mark.skipif(
    not SANDBOX, reason="needs LEAT_SANDBOX, the sandbox's environment"
)

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
        self.models: list[dict[str, Any]] | None = None  # listed as they are, if set, as another's
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
        if self.server.models is not None:
            return self._json(200, {"object": "list", "data": copy.deepcopy(self.server.models)})
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
                else:  # a delta's calls numbered as leat serve numbers them, but for parts'
                    if calls := delta.get("tool_calls"):
                        numbered = [{"index": i} | c for i, c in enumerate(calls)]
                        delta = delta | {"tool_calls": numbered}
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


def call(name: str, arguments: Any, id: str = "call_1") -> dict:
    """A reply's call of a tool, as leat serve sends it, whole, its arguments as JSON text."""
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": id, "type": "function", "function": {"name": name, "arguments": text}}


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


def request(url: str, method: str = "GET", body: Any = None, **headers: str) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data, headers, method=method)) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
