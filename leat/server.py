"""OpenAI-compatible HTTP server: chat completions, whole or streamed, and the model list.

Handler threads parse requests, render prompts and write responses. One worker thread owns the
engine and runs completions together, one per slot, a token of each per batched step; the rest
wait their turn in the order they arrive.
"""

import collections
import contextlib
import json
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import jinja2

from leat.chat import ChatTemplate, Reply, parse_tool_calls, split_reply, tool_call_start
from leat.engine import Engine, Sequence
from leat.tokenizer import Tokenizer


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# the request fields leat reads, when present: what makes them valid, and how to say so
_FIELDS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "messages": (lambda v: isinstance(v, list) and bool(v) and all(isinstance(m, dict) for m in v),
                 "a non-empty list of objects"),
    "temperature": (lambda v: (_integer(v) or isinstance(v, float)) and 0 <= v <= 2,
                    "a number from 0 to 2"),
    "max_tokens": (lambda v: _integer(v) and v > 0, "a positive integer"),
    "max_completion_tokens": (lambda v: _integer(v) and v > 0, "a positive integer"),
    "seed": (_integer, "an integer"),
    "stop": (lambda v: isinstance(v, str) or isinstance(v, list)
             and all(isinstance(s, str) for s in v), "a string or a list of strings"),
    "stream": (lambda v: isinstance(v, bool), "a boolean"),
    "stream_options": (lambda v: isinstance(v, dict), "an object"),
    "tools": (lambda v: isinstance(v, list) and all(isinstance(t, dict) for t in v),
              "a list of objects"),
    "chat_template_kwargs": (lambda v: isinstance(v, dict), "an object"),
}  # fmt: skip

# what leat does not implement, each with the value that asks for nothing more
_UNSUPPORTED = {
    "n": 1, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0, "logit_bias": {},
    "logprobs": False, "response_format": {"type": "text"},
}  # fmt: skip


@dataclass(frozen=True)
class _Finish:
    reason: str  # "stop" or "length"
    cached: int  # prompt tokens the cache held
    tokens: int  # tokens generated


@dataclass
class _Completion:
    """A chat completion: what the worker generates, and what the handler answers with."""

    prompt: list[int]
    max_tokens: int
    temperature: float
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

    def pieces(self) -> Iterator[str]:
        """The reply's text as the worker produces it; then `finish` says how it ended."""
        while isinstance(item := self.out.get(), str):
            yield item
        if isinstance(item, Exception):
            raise RuntimeError(f"generation failed: {item!r}") from item
        self.finish = item


class Server(ThreadingHTTPServer):
    """Serves an engine's model at http://host:port/v1 until shut down.

    The model's id is its file name without .gguf; requests may name any model.
    """

    def __init__(self, engine: Engine, host: str = "127.0.0.1", port: int = 8080):
        self.engine, self.model, self.created = engine, engine.gguf.path.stem, int(time.time())
        self.chat = ChatTemplate(engine.gguf.metadata, engine.tokenizer)
        self.completions: queue.SimpleQueue[_Completion | None] = queue.SimpleQueue()
        super().__init__((host, port), _Handler)
        threading.Thread(target=self._work, name="leat engine", daemon=True).start()

    def server_close(self) -> None:
        super().server_close()
        self.completions.put(None)  # the worker stops after the completions before it

    def _work(self) -> None:
        # starts waiting completions while slots are free, then steps every running one; waits
        # for a completion only when none is running or waiting, and stops after the completions
        # that came before shutdown
        engine = self.engine
        waiting: collections.deque[_Completion] = collections.deque()
        running: dict[Sequence, _Writer] = {}
        stopping = False
        while not stopping or waiting or running:
            for c in self._arrivals(wait=not (waiting or running)):
                if c is None:
                    stopping = True
                else:
                    waiting.append(c)
            while waiting and len(engine.active) < engine.slots:
                c = waiting.popleft()
                if c.cancelled.is_set():
                    continue
                try:
                    cached = engine.cached_prefix(c.prompt)
                    sequence = engine.start(c.prompt, c.max_tokens, c.temperature, c.seed)
                except Exception as e:  # for the client; the server carries on
                    c.out.put(e)
                    continue
                running[sequence] = _Writer(c, engine.tokenizer, cached)
            for sequence, writer in list(running.items()):
                if writer.c.cancelled.is_set():  # the client hung up
                    engine.cancel(sequence)
                    writer.finish("stop")
                    del running[sequence]
            if not running:
                continue
            try:
                stepped = engine.step()
            except Exception as e:  # for every running completion's client
                for sequence, writer in running.items():
                    engine.cancel(sequence)
                    writer.c.out.put(e)
                running.clear()
                continue
            for sequence, token in stepped:
                writer = running[sequence]
                if writer.take(token):  # end of generation or a stop string
                    engine.cancel(sequence)
                    writer.finish("stop")
                elif sequence.done:  # max_tokens, or the context full
                    writer.finish("length")
                else:
                    continue
                del running[sequence]

    def _arrivals(self, wait: bool) -> Iterator[_Completion | None]:
        # the completions queued so far, after waiting for one if `wait`
        with contextlib.suppress(queue.Empty):
            yield self.completions.get(block=wait)
            while True:
                yield self.completions.get_nowait()


class _Writer:
    """Puts a completion's reply into its out queue piece by piece as tokens come: decoded, and
    holding back any end of the text that may begin a stop string."""

    def __init__(self, c: _Completion, tokenizer: Tokenizer, cached: int):
        self.c, self.tokenizer, self.cached = c, tokenizer, cached
        self.decode, self.text, self.sent, self.count = tokenizer.stream(), "", 0, 0
        self.stopped = False  # by a stop string, the text cut where it begins

    def take(self, token: int) -> bool:
        """Takes the next token; True if it ends the reply: end of generation or a stop string."""
        self.count += 1
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
        self.c.out.put(_Finish(reason, self.cached, self.count))


class _Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            return self._error(404, f"there is no GET {self.path}")
        s = self.server
        model = {"id": s.model, "object": "model", "created": s.created, "owned_by": "leat"}
        self._json(200, {"object": "list", "data": [model]})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            return self._error(404, f"there is no POST {self.path}")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            completion = _completion(body, self.server)
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            return self._error(400, str(e))
        self.server.completions.put(completion)
        try:
            if completion.stream:
                self._stream(completion)
            else:
                self._reply(completion)
        except OSError:  # the client hung up
            completion.cancelled.set()

    def _reply(self, c: _Completion) -> None:
        try:
            reply = split_reply("".join(c.pieces()), c.form, c.thinking)
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
        body = self._head(c, "chat.completion") | {"choices": [choice], "usage": _usage(c)}
        self._json(200, body)

    def _stream(self, c: _Completion) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self._chunk(c, {"role": "assistant", "content": ""})
        # what may yet be part of a tool call is held back
        text, reasoned, sent, reply = "", 0, 0, Reply()
        try:
            for piece in c.pieces():
                text += piece
                reply = split_reply(text, c.form, c.thinking)
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
        self._chunk(c, {}, reason)
        if c.stream_usage:
            usage = {"choices": [], "usage": _usage(c)}
            self._event(self._head(c, "chat.completion.chunk") | usage)
        self._event("[DONE]")

    def _head(self, c: _Completion, kind: str) -> dict[str, Any]:
        return {"id": c.id, "object": kind, "created": c.created, "model": self.server.model}

    def _chunk(self, c: _Completion, delta: dict[str, Any], reason: str | None = None) -> None:
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": reason}
        self._event(self._head(c, "chat.completion.chunk") | {"choices": [choice]})

    def _event(self, data: dict[str, Any] | str) -> None:
        text = data if isinstance(data, str) else json.dumps(data)
        self.wfile.write(f"data: {text}\n\n".encode())

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
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
    tools = (body.get("tools") or None) if choice == "auto" else None
    # options for the template too, such as Qwen3's enable_thinking, as llama.cpp and vLLM take
    options = (body.get("chat_template_kwargs") or {}) | ({"tools": tools} if tools else {})
    text = server.chat.render(body["messages"], **options)
    prompt = server.chat.tokens(text)
    if len(prompt) >= (context := server.engine.max_context):
        raise ValueError(f"the prompt has {len(prompt)} tokens, too many for {context} of context")
    stop, temperature = body.get("stop") or [], body.get("temperature")
    return _Completion(
        prompt,
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens") or context,
        temperature=1.0 if temperature is None else temperature,  # OpenAI's default
        seed=body.get("seed"),
        stop=[s for s in ([stop] if isinstance(stop, str) else stop) if s],
        tools=tools,
        stream=bool(body.get("stream")),
        stream_usage=bool((body.get("stream_options") or {}).get("include_usage")),
        form=server.chat.form,
        thinking=server.chat.opens_thinking(text),
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
