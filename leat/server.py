"""OpenAI-compatible HTTP server: chat completions, whole or streamed, the model list, loading a
model, and a chat app at the root.

Handler threads parse requests, render prompts and write responses. One worker thread owns the
engine: it runs completions together, one per slot, a token of each per batched step, and loads a
model once the completions before have finished; the rest wait their turn in the order they arrive.
"""

import collections
import contextlib
import gc
import itertools
import json
import queue
import select
import socket
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import jinja2

from leat.chat import ChatTemplate, Reply, parse_tool_calls, split_reply, tool_call_start
from leat.engine import Engine, Sequence
from leat.sampler import Sampling
from leat.tokenizer import Tokenizer

# the chat app's files, each served at its path in leat/ and the app at /, and their types
_APP = (
    "app.html",
    "markdown.mjs",
    "vendor/temml/temml.mjs",
    "vendor/temml/Temml-Latin-Modern.css",
    "vendor/temml/Temml.woff2",
    "vendor/temml/latinmodernmath.woff2",
)
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
}
# seconds between a handler's checks that its client is still there, while it waits for text
_HANG_UP_CHECK = 0.25


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _between(low: float, high: float) -> Callable[[Any], bool]:
    return lambda v: (_integer(v) or isinstance(v, float)) and low <= v <= high


def _tool(v: Any) -> bool:
    # a function to call, named; its parameters, a JSON schema, the template reads as it may
    return (
        isinstance(v, dict)
        and isinstance(f := v.get("function"), dict)
        and isinstance(f.get("name"), str)
    )


# the request fields leat reads, when present: what makes them valid, and how to say so
_FIELDS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "messages": (lambda v: isinstance(v, list) and bool(v) and all(isinstance(m, dict) for m in v),
                 "a non-empty list of objects"),
    "temperature": (_between(0, 2), "a number from 0 to 2"),
    "top_k": (_integer, "an integer"),
    "top_p": (_between(0, 1), "a number from 0 to 1"),
    "min_p": (_between(0, 1), "a number from 0 to 1"),
    "presence_penalty": (_between(-2, 2), "a number from -2 to 2"),
    "max_tokens": (lambda v: _integer(v) and v > 0, "a positive integer"),
    "max_completion_tokens": (lambda v: _integer(v) and v > 0, "a positive integer"),
    "seed": (_integer, "an integer"),
    "stop": (lambda v: isinstance(v, str) or isinstance(v, list)
             and all(isinstance(s, str) for s in v), "a string or a list of strings"),
    "stream": (lambda v: isinstance(v, bool), "a boolean"),
    "stream_options": (lambda v: isinstance(v, dict), "an object"),
    "tools": (lambda v: isinstance(v, list) and all(_tool(t) for t in v),
              'a list of objects, each {"type": "function", "function": {"name": ...}}'),
    "chat_template_kwargs": (lambda v: isinstance(v, dict), "an object"),
}  # fmt: skip

# what leat does not implement, each with the value that asks for nothing more
_UNSUPPORTED = {
    "n": 1, "frequency_penalty": 0, "repetition_penalty": 1, "logit_bias": {}, "logprobs": False,
    "response_format": {"type": "text"},
}  # fmt: skip


@dataclass(frozen=True)
class _Finish:
    reason: str  # "stop" or "length"
    cached: int  # prompt tokens the cache held
    tokens: int  # tokens generated
    prefill_time: float  # seconds from the start of generation to the first token
    decode_time: float  # seconds from the first token to the last


@dataclass(frozen=True)
class _Loaded:
    # the loaded model: its id, its engine, and its chat template
    name: str
    engine: Engine
    chat: ChatTemplate


@dataclass
class _Load:
    # a model to load: the worker answers None once it is ready, or the exception it raised
    name: str
    done: queue.SimpleQueue[Exception | None] = field(default_factory=queue.SimpleQueue)


@dataclass
class _Completion:
    """A chat completion: what the worker generates, and what the handler answers with."""

    prompt: list[int]
    model: str  # whose chat template rendered the prompt
    max_tokens: int
    sampling: Sampling
    seed: int | None
    stop: list[str]
    tools: list[dict[str, Any]] | None
    stream: bool
    stream_usage: bool
    form: str | None = None  # how the reply marks its reasoning, as ChatTemplate.form
    thinking: bool = False  # the prompt opened a <think> block
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    created: int = field(default_factory=lambda: int(time.time()))
    # from the worker: pieces of text, then how generation finished or the exception it raised
    out: queue.SimpleQueue[str | _Finish | Exception] = field(default_factory=queue.SimpleQueue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    finish: _Finish = field(init=False)  # once pieces() is exhausted

    def pieces(self, hung_up: Callable[[], bool] = lambda: False) -> Iterator[str]:
        """The reply's text as the worker produces it; then `finish` says how it ended. Raises
        ConnectionAbortedError once `hung_up()`, checked at each piece and while none comes, as
        for a reply that sends nothing before it ends, which would never find its client gone."""
        while True:
            try:
                item: str | _Finish | Exception | None = self.out.get(timeout=_HANG_UP_CHECK)
            except queue.Empty:
                item = None
            if hung_up():
                raise ConnectionAbortedError("the client hung up")
            if isinstance(item, str):
                yield item
            elif item is not None:
                break
        if isinstance(item, Exception):
            raise RuntimeError(f"generation failed: {item!r}") from item
        self.finish = item


class Server(ThreadingHTTPServer):
    """Serves GGUF models at http://host:port, the API at /v1, until shut down.

    One model is loaded at a time, none until load(), as Engine(path, **options); it answers every
    request, whatever model it names. A model's id is its file name without .gguf.
    """

    def __init__(
        self, models: Iterable[str | Path], host: str = "127.0.0.1", port: int = 8080,
        **options: Any,
    ):  # fmt: skip
        self.models = {Path(path).stem: Path(path) for path in models}
        self.options, self.created = options, int(time.time())
        self.loaded: _Loaded | None = None
        self.loading: str | None = None  # the id of the model the worker is loading
        self.ready = threading.Event()  # set but while a model loads, which completions wait for
        self.ready.set()
        self.requests: queue.SimpleQueue[_Completion | _Load | None] = queue.SimpleQueue()
        super().__init__((host, port), _Handler)
        threading.Thread(target=self._work, name="leat engine", daemon=True).start()

    def load(self, name: str) -> None:
        """Loads a model in place of the loaded one, once the completions before have finished,
        and compiles its graphs; returns when it is ready. Loading the loaded model does nothing."""
        if name not in self.models:
            raise ValueError(f"there is no model {name!r}")
        self.requests.put(load := _Load(name))
        if (error := load.done.get()) is not None:
            raise RuntimeError(f"loading {name} failed: {error!r}") from error

    def server_close(self) -> None:
        super().server_close()
        self.requests.put(None)  # the worker stops after the requests before it

    def _work(self) -> None:
        # starts waiting requests in turn, then steps every running completion; waits for a
        # request only when none is running or waiting, and stops after the requests that came
        # before shutdown
        waiting: collections.deque[_Completion | _Load] = collections.deque()
        running: dict[Sequence, _Writer] = {}
        stopping = False
        while not stopping or waiting or running:
            try:
                for request in self._arrivals(wait=not (waiting or running)):
                    if request is None:
                        stopping = True
                    else:
                        waiting.append(request)
                while waiting and self._start(waiting[0], running):
                    waiting.popleft()
                if running:
                    self._step(running)
            except Exception as e:  # a bug's, for every client, rather than a worker gone
                self._fail(e, waiting, running)

    def _fail(
        self, error: Exception, waiting: collections.deque[_Completion | _Load],
        running: dict[Sequence, "_Writer"],
    ) -> None:  # fmt: skip
        # ends every request with the error, the running completions' sequences too
        for sequence, writer in running.items():
            if self.loaded is not None:
                self.loaded.engine.cancel(sequence)
            writer.c.out.put(error)
        for request in waiting:
            (request.done if isinstance(request, _Load) else request.out).put(error)
        running.clear()
        waiting.clear()

    def _step(self, running: dict[Sequence, "_Writer"]) -> None:
        # steps every running completion, after ending those whose clients hung up. The engine
        # is local to this call, which a load waits out, so that the load can free it.
        assert self.loaded is not None  # a load waits for running completions to finish
        engine = self.loaded.engine
        for sequence, writer in list(running.items()):
            if writer.c.cancelled.is_set():  # the client hung up
                engine.cancel(sequence)
                writer.finish("stop")
                del running[sequence]
        if not running:
            return
        try:
            stepped = engine.step()
        except Exception as e:  # for every running completion's client
            for sequence, writer in running.items():
                engine.cancel(sequence)
                writer.c.out.put(e)
            running.clear()
            return
        # a speculative step gives a sequence several tokens, the last of which may end it
        last = {sequence: i for i, (sequence, _) in enumerate(stepped)}
        for i, (sequence, token) in enumerate(stepped):
            if sequence not in running:  # a stop string ended it before
                continue
            writer = running[sequence]
            if writer.take(token):  # end of generation or a stop string
                engine.cancel(sequence)
                writer.finish("stop")
            elif sequence.done and i == last[sequence]:  # max_tokens, or the context full
                writer.finish("length")
            else:
                continue
            del running[sequence]

    def _start(self, request: _Completion | _Load, running: dict[Sequence, "_Writer"]) -> bool:
        # starts a request, or ends it if it never can: a completion while a slot is free, a load
        # once no completion runs. False if it must wait.
        if isinstance(request, _Load):
            if running:
                return False
            self._load(request)
            return True
        if request.cancelled.is_set():
            return True
        if (loaded := self.loaded) is None or loaded.name != request.model:
            request.out.put(
                RuntimeError(f"{request.model} was unloaded before the completion started")
            )
            return True
        engine = loaded.engine
        if len(engine.active) == engine.slots:
            return False
        try:  # timed from before start(), which copies a prefix in or restores a kept state
            started, cached = time.perf_counter(), engine.cached_prefix(request.prompt)
            sequence = engine.start(
                request.prompt, request.max_tokens, request.sampling, request.seed
            )
        except Exception as e:  # for the client; the server carries on
            request.out.put(e)
            return True
        running[sequence] = _Writer(request, engine.tokenizer, cached, started)
        return True

    def _load(self, load: _Load) -> None:
        # replaces the loaded model, whose memory is freed first: a GPU holds one model at most
        if self.loaded is not None and self.loaded.name == load.name:
            return load.done.put(None)
        self.ready.clear()
        self.loaded, self.loading = None, load.name
        gc.collect()  # an engine's graphs refer back to it
        error: Exception | None = None
        try:
            engine = Engine(self.models[load.name], **self.options)
            chat = ChatTemplate(engine.gguf.metadata, engine.tokenizer)
            engine.warm_up()
            self.loaded = _Loaded(load.name, engine, chat)
        except Exception as e:  # for the client; the server carries on with no model
            error = e
        self.loading = None
        self.ready.set()
        load.done.put(error)

    def _arrivals(self, wait: bool) -> Iterator[_Completion | _Load | None]:
        # the requests queued so far, after waiting for one if `wait`
        with contextlib.suppress(queue.Empty):
            yield self.requests.get(block=wait)
            while True:
                yield self.requests.get_nowait()


class _Writer:
    """Puts a completion's reply into its out queue piece by piece as tokens come: decoded, and
    holding back any end of the text that may begin a stop string."""

    def __init__(self, c: _Completion, tokenizer: Tokenizer, cached: int, started: float):
        self.c, self.tokenizer, self.cached = c, tokenizer, cached
        self.decode, self.text, self.sent, self.count = tokenizer.stream(), "", 0, 0
        self.stopped = False  # by a stop string, the text cut where it begins
        # when generation started, after any wait for a slot, and when the first token came
        self.started, self.first = started, 0.0

    def take(self, token: int) -> bool:
        """Takes the next token; True if it ends the reply: end of generation or a stop string."""
        self.count += 1
        if self.count == 1:
            self.first = time.perf_counter()
        if token in self.tokenizer.eog_ids:
            return True
        self.text += self.decode(token)
        # sent text holds no start of a stop string, so one can only begin after it
        if found := [i for s in self.c.stop if (i := self.text.find(s, self.sent)) >= 0]:
            self.text, self.stopped = self.text[: min(found)], True
            return True
        if (end := len(self.text) - _partial_stop(self.text, self.c.stop)) > self.sent:
            self.c.out.put(self.text[self.sent : end])
            self.sent = end
        return False

    def finish(self, reason: str) -> None:
        """Puts the rest of the text, then how the reply ended."""
        if not self.stopped:
            self.text += self.decode(None)
        if len(self.text) > self.sent:
            self.c.out.put(self.text[self.sent :])
        end = time.perf_counter()
        first = self.first if self.count else end
        finish = _Finish(reason, self.cached, self.count, first - self.started, end - first)
        self.c.out.put(finish)


class _Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        self.path = urllib.parse.urlsplit(self.path).path  # without a query, ?v=2 say
        if (name := "app.html" if self.path == "/" else self.path[1:]) in _APP:
            file = Path(__file__).parent / name
            return self._send(200, _TYPES[file.suffix], file.read_bytes())
        if self.path != "/v1/models":
            return self._error(404, f"there is no GET {self.path}")
        self._json(200, {"object": "list", "data": [self._model(m) for m in self.server.models]})

    def do_POST(self) -> None:
        self.path = urllib.parse.urlsplit(self.path).path
        routes = {"/v1/chat/completions": self._complete, "/v1/models/load": self._load}
        if (route := routes.get(self.path)) is None:
            return self._error(404, f"there is no POST {self.path}")
        if not self._same_origin():
            return self._error(403, "requests from other sites' pages are refused")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except (ValueError, RecursionError) as e:  # not JSON, or nested too deep to parse
            return self._error(400, str(e))
        with contextlib.suppress(OSError):  # the client hung up, while a model loaded say
            route(body)

    def _load(self, body: Any) -> None:
        name = body.get("model") if isinstance(body, dict) else None
        if not isinstance(name, str) or name not in self.server.models:
            return self._error(404, f"there is no model {name!r}")
        try:
            self.server.load(name)
        except RuntimeError as e:
            return self._error(500, str(e))
        self._json(200, self._model(name))

    def _complete(self, body: Any) -> None:
        self.server.ready.wait()  # for the model loading, if one is, whose template renders it
        try:
            completion = _completion(body, self.server)
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            return self._error(400, str(e))
        self.server.requests.put(completion)
        try:
            if completion.stream:
                self._stream(completion)
            else:
                self._reply(completion)
        except OSError:  # the client hung up
            completion.cancelled.set()

    def _reply(self, c: _Completion) -> None:
        try:
            reply = split_reply("".join(c.pieces(self._hung_up)), c.form, c.thinking, done=True)
        except RuntimeError as e:
            return self._error(500, str(e))
        message: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.reasoning:
            message["reasoning_content"] = reply.reasoning
        reason = c.finish.reason
        content, calls = _calls(reply, c.tools)
        if calls:
            message |= {"content": content or None, "tool_calls": _tool_calls(calls)}
            reason = "tool_calls"
        choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": reason}
        body = {"choices": [choice], "usage": _usage(c), "timings": _timings(c)}
        self._json(200, self._head(c, "chat.completion") | body)

    def _stream(self, c: _Completion) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self._chunk(c, {"role": "assistant", "content": ""})
        # what may yet be part of a tool call is held back; the reply is split once more when
        # done, the end that may have begun a marker then its own
        text, reasoned, sent, reply = "", 0, 0, Reply()
        try:
            for piece in itertools.chain(c.pieces(self._hung_up), [None]):
                text += piece or ""
                reply = split_reply(text, c.form, c.thinking, done=piece is None)
                if len(reply.reasoning) > reasoned:
                    self._chunk(c, {"reasoning_content": reply.reasoning[reasoned:]})
                    reasoned = len(reply.reasoning)
                content = reply.content
                if (end := tool_call_start(content) if c.tools else len(content)) > sent:
                    self._chunk(c, {"content": content[sent:end]})
                    sent = end
        except RuntimeError as e:
            return self._event({"error": {"message": str(e), "type": "server_error"}})
        reason = c.finish.reason
        content, calls = _calls(reply, c.tools)
        if len(content) > sent:
            self._chunk(c, {"content": content[sent:]})
        if calls:
            self._chunk(
                c,
                {"tool_calls": [{"index": i} | call for i, call in enumerate(_tool_calls(calls))]},
            )
            reason = "tool_calls"
        # the timings on the last chunk, as llama.cpp's server sends them
        timings = {"timings": _timings(c)}
        self._chunk(c, {}, reason, {} if c.stream_usage else timings)
        if c.stream_usage:
            usage = {"choices": [], "usage": _usage(c)} | timings
            self._event(self._head(c, "chat.completion.chunk") | usage)
        self._event("[DONE]")

    def _same_origin(self) -> bool:
        # a browser's request from a page of this server, the app's, or one of no browser, which
        # sends no Origin: another site's page may not make it generate or load models
        origin = self.headers.get("Origin")
        return origin is None or urllib.parse.urlsplit(origin).netloc == self.headers.get("Host")

    def _hung_up(self) -> bool:
        # whether the client closed the connection: it reads as ready, with nothing to read
        try:
            ready, _, _ = select.select([self.connection], [], [], 0)
            return bool(ready) and not self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            return True

    def _model(self, name: str) -> dict[str, Any]:
        s = self.server
        loaded = s.loaded is not None and s.loaded.name == name
        status = "loaded" if loaded else "loading" if s.loading == name else "unloaded"
        model = {"id": name, "object": "model", "created": s.created, "owned_by": "leat"}
        return model | {"status": status}

    def _head(self, c: _Completion, kind: str) -> dict[str, Any]:
        return {"id": c.id, "object": kind, "created": c.created, "model": c.model}

    def _chunk(
        self, c: _Completion, delta: dict[str, Any], reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:  # fmt: skip
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": reason}
        self._event(self._head(c, "chat.completion.chunk") | {"choices": [choice]} | (extra or {}))

    def _event(self, data: dict[str, Any] | str) -> None:
        text = data if isinstance(data, str) else json.dumps(data)
        self.wfile.write(f"data: {text}\n\n".encode())

    def _json(self, status: int, body: dict[str, Any]) -> None:
        self._send(status, "application/json", json.dumps(body).encode())

    def _send(self, status: int, kind: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        kind = "invalid_request_error" if status < 500 else "server_error"
        error = {"message": message, "type": kind, "param": None, "code": None}
        self._json(status, {"error": error})


def _completion(body: Any, server: Server) -> _Completion:
    # what a request body asks for, or a ValueError saying what is wrong with it
    if not isinstance(body, dict) or body.get("messages") is None:
        raise ValueError("a chat completion needs messages")
    for key, (valid, expected) in _FIELDS.items():
        if body.get(key) is not None and not valid(body[key]):
            raise ValueError(f"{key} must be {expected}")
    for key, neutral in _UNSUPPORTED.items():
        if body.get(key) not in (None, neutral):
            raise ValueError(f"{key}={body[key]!r} is not supported")
    if (choice := body.get("tool_choice") or "auto") not in ("auto", "none"):
        raise ValueError(f"tool_choice={choice!r} is not supported, only 'auto' and 'none'")
    if (loaded := server.loaded) is None:
        raise ValueError("no model is loaded")
    tools = (body.get("tools") or None) if choice == "auto" else None
    # options for the template too, such as Qwen3's enable_thinking, as llama.cpp and vLLM take
    options = (body.get("chat_template_kwargs") or {}) | ({"tools": tools} if tools else {})
    text = loaded.chat.render(body["messages"], **options)
    prompt = loaded.chat.tokens(text)
    if len(prompt) >= (context := loaded.engine.max_context):
        raise ValueError(f"the prompt has {len(prompt)} tokens, too many for {context} of context")
    stop, given = body.get("stop") or [], {k: v for k, v in body.items() if v is not None}
    sampling = Sampling(
        temperature=given.get("temperature", 1.0),  # OpenAI's default
        top_k=max(given.get("top_k", 0), 0),  # vLLM's -1 keeps every token too
        top_p=given.get("top_p", 1.0),
        min_p=given.get("min_p", 0.0),
        presence_penalty=given.get("presence_penalty", 0.0),
    )
    return _Completion(
        prompt,
        loaded.name,
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens") or context,
        sampling=sampling,
        seed=body.get("seed"),
        stop=[s for s in ([stop] if isinstance(stop, str) else stop) if s],
        tools=tools,
        stream=bool(body.get("stream")),
        stream_usage=bool((body.get("stream_options") or {}).get("include_usage")),
        form=loaded.chat.form,
        thinking=loaded.chat.opens_thinking(text),
    )


def _calls(reply: Reply, tools: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
    # a reply's text and its calls to the tools: those of the harmony format whose arguments are
    # JSON objects and name a tool, or those of the text
    if not tools:
        return reply.content, []
    if not reply.calls:
        return parse_tool_calls(reply.content, tools)
    names, calls = {tool.get("function", {}).get("name") for tool in tools}, []
    for call in reply.calls:
        with contextlib.suppress(ValueError):
            arguments = json.loads(call["arguments"])
            if call["name"] in names and isinstance(arguments, dict):
                calls.append({"name": call["name"], "arguments": arguments})
    return reply.content, calls


def _partial_stop(text: str, stops: list[str]) -> int:
    # the length of the longest end of `text` that a stop string begins with
    return max((n for s in stops for n in range(1, len(s)) if text.endswith(s[:n])), default=0)


def _tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # as OpenAI's API has them, with the arguments as JSON text
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
            },
        }
        for c in calls
    ]


def _usage(c: _Completion) -> dict[str, Any]:
    prompt, generated = len(c.prompt), c.finish.tokens
    return {
        "prompt_tokens": prompt,
        "completion_tokens": generated,
        "total_tokens": prompt + generated,
        "prompt_tokens_details": {"cached_tokens": c.finish.cached},
    }


def _timings(c: _Completion) -> dict[str, Any]:
    # as llama.cpp's server reports them: the prompt's tokens past those cached, timed from the
    # start of generation, after any wait for a slot, to the first token; and the reply's tokens,
    # timed from the first to the last: the first comes of the prompt's last step, so that time
    # holds one step fewer than there are tokens, and of a reply of one token, no rate
    f = c.finish
    timings: dict[str, Any] = {"cache_n": f.cached}
    parts = (("prompt", len(c.prompt) - f.cached, f.prefill_time, True),
             ("predicted", f.tokens, f.decode_time, f.tokens > 1))  # fmt: skip
    for name, n, seconds, timed in parts:
        timings |= {
            f"{name}_n": n,
            f"{name}_ms": 1e3 * seconds,
            f"{name}_per_token_ms": 1e3 * seconds / n if n and timed else 0.0,
            f"{name}_per_second": n / seconds if seconds and timed else 0.0,
        }
    return timings
