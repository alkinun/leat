import contextlib
import json
import socket
import statistics
import threading
import time
import urllib.error
import urllib.request
import weakref
from collections.abc import Iterator
from pathlib import Path

import openai
import pytest

from leat.chat import ChatTemplate
from leat.engine import Engine
from leat.sampler import Sampling
from leat.server import Server, _completion
from tests.helpers import CONTEXT, chat_template

WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "The weather in a city now",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@contextlib.contextmanager
def serving(*models: Path, **options) -> Iterator[Server]:
    # a server of the models on a free port, none loaded
    with Server(models, port=0, **options) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield server
        server.shutdown()


def connect(server: Server) -> openai.OpenAI:
    url = f"http://127.0.0.1:{server.server_port}/v1"
    return openai.OpenAI(base_url=url, api_key="unused", max_retries=0)


def load(client: openai.OpenAI, model: str) -> dict:
    return client.post("/models/load", cast_to=object, body={"model": model})


@pytest.fixture(scope="module")
def server(tiny_model) -> Iterator[Server]:
    with serving(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2) as server:
        server.load("tiny")
        yield server


@pytest.fixture(scope="module")
def engine(server) -> Engine:
    return server.loaded.engine


@pytest.fixture(scope="module")
def client(server) -> openai.OpenAI:
    return connect(server)


@pytest.fixture(scope="module")
def expected(tiny_model):
    # the greedy reply to a message, from an engine of its own
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)

    def reply(content: str, max_tokens: int) -> str:
        prompt = engine.tokenizer.encode(content)  # the tiny model's template is the content
        return engine.tokenizer.decode(list(engine.generate(prompt, max_tokens)))

    return reply


def chat(client: openai.OpenAI, content: str, **kwargs):
    messages = [{"role": "user", "content": content}]
    return client.chat.completions.create(model="tiny", messages=messages, **kwargs)


def complete(client: openai.OpenAI, content: str, **kwargs) -> tuple[str, str]:
    # a reply's text and finish reason, streamed or whole
    response = chat(client, content, **kwargs)
    if not kwargs.get("stream"):
        return response.choices[0].message.content, response.choices[0].finish_reason
    choices = [chunk.choices[0] for chunk in response if chunk.choices]
    return "".join(c.delta.content or "" for c in choices), choices[-1].finish_reason


def test_models(client):
    assert [(model.id, model.status) for model in client.models.list()] == [("tiny", "loaded")]


def test_load(tiny_model, tmp_path):
    # models load on request, each in place of the last, and answer whatever model a request names
    (other := tmp_path / "other.gguf").symlink_to(tiny_model[0])
    with serving(tiny_model[0], other, max_context=CONTEXT) as server:
        client = connect(server)
        with pytest.raises(openai.BadRequestError, match="no model is loaded"):
            chat(client, "hello")
        engines = []
        for name, unloaded in (("tiny", "other"), ("other", "tiny"), ("other", "tiny")):
            assert load(client, name)["status"] == "loaded"
            statuses = {model.id: model.status for model in client.models.list()}
            assert statuses == {name: "loaded", unloaded: "unloaded"}
            assert chat(client, "hello", max_tokens=2).model == name
            engines.append(weakref.ref(server.loaded.engine))
        # the first is freed before the second loads, as a GPU holds one model at most; loading
        # the second again keeps it
        first, second, again = (engine() for engine in engines)
        assert first is None and second is again is server.loaded.engine
        with pytest.raises(openai.NotFoundError, match="there is no model 'tinier'"):
            load(client, "tinier")


def test_app(server):
    url = f"http://127.0.0.1:{server.server_port}"
    with urllib.request.urlopen(f"{url}/") as response:
        assert response.headers["Content-Type"] == "text/html; charset=utf-8"
        assert b"<title>leat</title>" in response.read()
    with urllib.request.urlopen(f"{url}/markdown.mjs") as response:
        assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
        assert b"export function markdown(" in response.read()
    with urllib.request.urlopen(f"{url}/vendor/temml/Temml.woff2") as response:
        assert response.headers["Content-Type"] == "font/woff2"
    for path in ("/server.py", "/vendor/temml/LICENSE", "/../pyproject.toml"):
        with pytest.raises(urllib.error.HTTPError, match="404"):
            urllib.request.urlopen(f"{url}{path}")


def check_timings(timings: dict, usage) -> None:
    # llama.cpp's timings of a reply, as its usage counts its tokens
    prompt = (timings["cache_n"], timings["prompt_n"])
    assert prompt == (usage.prompt_tokens_details.cached_tokens, usage.prompt_tokens - prompt[0])
    assert timings["predicted_n"] == usage.completion_tokens
    for name in ("prompt", "predicted"):
        n, ms = timings[f"{name}_n"], timings[f"{name}_ms"]
        assert ms > 0 and timings[f"{name}_per_token_ms"] == pytest.approx(ms / n)
        assert timings[f"{name}_per_second"] == pytest.approx(1e3 * n / ms)


def test_reply(client, expected):
    response = chat(client, "hello", max_tokens=8, temperature=0)
    assert response.choices[0].message.content == expected("hello", 8)
    assert response.choices[0].finish_reason == "length"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (5, 8)
    check_timings(response.model_extra["timings"], response.usage)


def test_stream(client, expected):
    usage = {"include_usage": True}
    stream = chat(client, "hello", max_tokens=8, temperature=0, stream=True, stream_options=usage)
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks[:-1]) == expected("hello", 8)
    assert chunks[-2].choices[0].finish_reason == "length"
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 8
    # the timings on the last chunk, as llama.cpp's: the usage's, or else the finish's
    assert all("timings" not in c.model_extra for c in chunks[:-1])
    check_timings(chunks[-1].model_extra["timings"], chunks[-1].usage)
    *_, last = chat(client, "hello", max_tokens=8, temperature=0, stream=True)
    assert last.choices[0].finish_reason == "length"
    assert last.model_extra["timings"]["predicted_n"] == 8


def test_timings_leave_out_the_wait(client):
    # a request that waits for a slot, both taken by longer replies, is timed from when it gets
    # one: the wait is in the time its client takes to see the first token, not in its timings
    contents = ("a long reply", "another long reply")
    streams = [chat(client, content, max_tokens=30, stream=True) for content in contents]
    for stream in streams:  # both are generating
        next(c for c in stream if c.choices and c.choices[0].delta.content)
    start = time.perf_counter()
    response = chat(client, "a short one", max_tokens=2)
    elapsed = 1e3 * (time.perf_counter() - start)
    for stream in streams:
        assert [c for c in stream if c.choices][-1].choices[0].finish_reason == "length"
    timings = response.model_extra["timings"]
    assert timings["prompt_ms"] + timings["predicted_ms"] < elapsed / 4


@pytest.mark.parametrize("stream", [False, True])
def test_stop(client, expected, stream):
    full = expected("stop me", 16)
    stops = ["never there", full[6:9]]
    reply = complete(client, "stop me", max_tokens=16, temperature=0, stop=stops, stream=stream)
    assert reply == (full[: full.index(stops[1])], "stop")


def test_completion(server):
    # what a request asks for: OpenAI's fields and vLLM's, top_k of -1 keeping every token as 0
    messages = [{"role": "user", "content": "hi"}]
    body = {"messages": messages, "temperature": 0.5, "top_k": -1, "top_p": 0.9, "min_p": 0.05,
            "presence_penalty": 1.5, "max_tokens": 9, "max_completion_tokens": 7, "stop": "x",
            "seed": 3}  # fmt: skip
    asked = _completion(body, server)
    assert asked.sampling == Sampling(0.5, 0, 0.9, 0.05, 1.5)
    hi = server.loaded.engine.tokenizer.encode("hi")
    assert (asked.prompt, asked.max_tokens, asked.stop, asked.seed) == (hi, 7, ["x"], 3)
    # and by default, OpenAI's temperature of 1, up to the context and without empty stops
    asked = _completion({"messages": messages, "stop": ["", "y"], "top_p": None}, server)
    assert (asked.sampling, asked.max_tokens, asked.stop) == (Sampling(1.0), CONTEXT, ["y"])


def test_end_of_generation(client, engine, monkeypatch):
    # a reply ends at an end-of-generation token, which its text leaves out
    tokens = list(engine.generate(engine.tokenizer.encode("hello"), 8))
    monkeypatch.setattr(engine.tokenizer, "eog_ids", {tokens[3]})
    text = engine.tokenizer.decode(tokens[: tokens.index(tokens[3])])
    assert complete(client, "hello", max_tokens=8, temperature=0) == (text, "stop")


def test_seed(client):
    # sampled, as temperature is 1 unless given
    first, again = (complete(client, "seeded", max_tokens=8, seed=3) for _ in range(2))
    assert first == again
    # and cut, as Qwen3.6 recommends: top_k and min_p in OpenAI's client's extra_body
    options = {"top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0}}
    first, again = (complete(client, "cut", max_tokens=8, seed=3, **options) for _ in range(2))
    assert first == again


def test_cached_tokens(client):
    chat(client, "a shared start, then one end", max_tokens=2)
    response = chat(client, "a shared start, then another", max_tokens=2)
    assert response.usage.prompt_tokens_details.cached_tokens == len("a shared start, then ")


def test_chat_template_kwargs(client):
    # options for the template, which the tiny model's writes before the last message
    response = chat(
        client, "hello", max_tokens=1, extra_body={"chat_template_kwargs": {"prefix": "ab"}}
    )
    assert response.usage.prompt_tokens == len("abhello")


@pytest.fixture
def replies_with(engine, monkeypatch):
    # makes the engine reply with the given text, which the tiny model would never write: each
    # step gives every active sequence its next token, and ends it at the last
    def reply_with(text: str) -> None:
        tokens = engine.tokenizer.encode(text)

        def step():
            out = []
            for sequence in list(engine.active):
                sequence.tokens.append(token := tokens[len(sequence.tokens)])
                out.append((sequence, token))
                if len(sequence.tokens) == len(tokens):
                    engine.cancel(sequence)
            return out

        monkeypatch.setattr(engine, "step", step)

    return reply_with


# as Llama 3, Qwen3, Gemma 4 and Qwen3.5 call tools; the text before a call is the content
CALLS = [
    (' {"name": "weather", "parameters": {"city": "Paris"}}', None),
    ('Checking.\n<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>',
     "Checking."),
    ('<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>', None),
    ("Checking.\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n"
     "</function>\n</tool_call>", "Checking."),
]  # fmt: skip


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply, content", CALLS)
def test_tool_call(client, replies_with, stream, reply, content):
    replies_with(reply)
    response = chat(client, "Weather in Paris?", tools=[WEATHER], stream=stream)
    if stream:
        choices = [chunk.choices[0] for chunk in response]
        assert "".join(c.delta.content or "" for c in choices) == (content or "")
        (call,), reason = choices[-2].delta.tool_calls, choices[-1].finish_reason
    else:
        message, reason = response.choices[0].message, response.choices[0].finish_reason
        assert message.content == content
        (call,) = message.tool_calls
    assert call.type == "function" and call.id.startswith("call_") and reason == "tool_calls"
    assert call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "form, reply",
    [("think", "<think>Sunny, I recall.</think>It is sunny."),
     ("think", "\n<think>Sunny, I recall.</think>It is sunny."),  # its newline not the text's
     ("harmony", "<|channel|>analysis<|message|>Sunny, I recall.<|end|><|start|>assistant"
                 "<|channel|>final<|message|>It is sunny.")],
)  # fmt: skip
def test_reasoning(client, replies_with, monkeypatch, stream, form, reply):
    # Qwen3's and gpt-oss's reasoning, apart from the text
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: form))
    replies_with(reply)
    response = chat(client, "Weather in Paris?", stream=stream)
    if stream:
        deltas = [chunk.choices[0].delta for chunk in response if chunk.choices]
        reasoning = "".join(getattr(d, "reasoning_content", None) or "" for d in deltas)
        content = "".join(d.content or "" for d in deltas)
    else:
        message = response.choices[0].message
        reasoning, content = getattr(message, "reasoning_content", None), message.content
    assert (reasoning, content) == ("Sunny, I recall.", "It is sunny.")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "form, reply, content",
    [(None, "if a <", "if a <"), ("think", "<think>b</think>a </", "a </"),
     ("harmony", "<|channel|>final<|message|>a <|", "a <|")],
)  # fmt: skip
def test_reply_ends_as_a_marker_begins(
    client, replies_with, monkeypatch, stream, form, reply, content
):
    # an end that may begin a marker waits for more only until the reply is done
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: form))
    replies_with(reply)
    assert complete(client, "Weather in Paris?", stream=stream)[0] == content


@pytest.mark.parametrize("stream", [False, True])
def test_harmony_tool_call(client, replies_with, monkeypatch, stream):
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: "harmony"))
    replies_with(
        '<|channel|>commentary to=functions.weather <|constrain|>json<|message|>{"city": "Paris"}'
    )
    response = chat(client, "Weather in Paris?", tools=[WEATHER], stream=stream)
    if stream:
        choices = [chunk.choices[0] for chunk in response]
        (call,), reason = choices[-2].delta.tool_calls, choices[-1].finish_reason
    else:
        (call,), reason = response.choices[0].message.tool_calls, response.choices[0].finish_reason
    assert reason == "tool_calls" and call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "text, choice",
    [
        ("It is sunny.", "auto"),
        ('{"name": "unknown", "parameters": {}}', "auto"),
        ('{"name": "weather", "parameters": {}}', "none"),
    ],
)
def test_text_is_not_a_tool_call(client, replies_with, stream, text, choice):
    replies_with(text)
    kwargs = {"tools": [WEATHER], "tool_choice": choice, "stream": stream}
    assert complete(client, "Weather in Paris?", **kwargs) == (text, "length")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n": 2}, "n=2 is not supported"),
        ({"frequency_penalty": 0.5}, "frequency_penalty=0.5 is not supported"),
        ({"top_p": 1.5}, "top_p must be a number from 0 to 1"),
        ({"temperature": 3}, "temperature must be a number from 0 to 2"),
        ({"temperature": True}, "temperature must be a number from 0 to 2"),
        ({"extra_body": {"top_k": 1.5}}, "top_k must be an integer"),
        ({"max_tokens": 0}, "max_tokens must be a positive integer"),
        ({"tools": [WEATHER], "tool_choice": "required"}, "tool_choice='required' is not"),
        ({"messages": []}, "messages must be a non-empty list of objects"),
        ({"messages": [{"role": "user", "content": "x" * CONTEXT}]}, "the prompt has 64 tokens"),
        ({"extra_body": {"chat_template_kwargs": "x"}}, "chat_template_kwargs must be an object"),
        # what a message or tool holds, of the shapes templates read
        ({"messages": [{"role": "user", "content": [None]}]}, "only text content"),
        ({"messages": [{"role": "user", "content": [{"type": "text"}]}]}, "text must be a string"),
        ({"messages": [{"role": "assistant", "tool_calls": [{"id": "x"}]}]}, "each with its"),
        ({"tools": [{"function": "weather"}]}, "tools must be a list of objects, each"),
        ({"tools": [{"function": {"name": []}}]}, "tools must be a list of objects, each"),
    ],
)
def test_bad_request(client, kwargs, error):
    request = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]} | kwargs
    with pytest.raises(openai.BadRequestError, match=error):
        client.chat.completions.create(**request)


def test_unknown_route(client, server):
    with pytest.raises(openai.NotFoundError):
        client.completions.create(model="tiny", prompt="hi")
    url = f"http://127.0.0.1:{server.server_port}"
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(f"{url}/v1/nothing")
    with pytest.raises(urllib.error.HTTPError, match="400"):  # not JSON
        urllib.request.urlopen(urllib.request.Request(f"{url}/v1/chat/completions", b"{"))


def test_concurrent_requests(client, server, engine, expected, monkeypatch):
    # more requests at once than the engine's 2 slots, whole and streamed: each gets the reply it
    # would alone, two in batched steps, the third once a slot is free
    batches, decode_step, step = [], engine._decode_step, engine.step
    monkeypatch.setattr(engine, "_decode_step", lambda s: batches.append(len(s)) or decode_step(s))
    contents, replies, first = ["hello", "a b c", "the third one"], {}, threading.Event()

    def step_once_queued() -> list:
        # the first request's first step lets the others in and waits until they are queued, so
        # that they come before it finishes however fast the engine is: taking them off the
        # queue waits for them, and they go back in order
        if not first.is_set():
            first.set()
            queued = [server.requests.get() for _ in contents[1:]]
            for request in queued:
                server.requests.put(request)
        return step()

    monkeypatch.setattr(engine, "step", step_once_queued)

    def ask(content: str, stream: bool) -> None:
        if content != contents[0]:
            first.wait()
        replies[content] = complete(client, content, max_tokens=12, temperature=0, stream=stream)

    threads = [threading.Thread(target=ask, args=(c, i > 0)) for i, c in enumerate(contents)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert replies == {c: (expected(c, 12), "length") for c in contents}
    assert max(batches) == 2


def test_hang_up_frees_the_slot(client, engine):
    # a client that hangs up mid-reply ends its sequence, while another's carries on
    other = chat(client, "a long reply", max_tokens=30, stream=True)
    gone = chat(client, "another long reply", max_tokens=30, stream=True)
    next(iter(gone))
    gone.close()
    choices = [chunk.choices[0] for chunk in other if chunk.choices]
    assert choices[-1].finish_reason == "length"
    assert not engine.active


def test_hang_up_before_the_reply_frees_the_slot(server, engine, monkeypatch):
    # a client of a whole reply that hangs up while it waits, which no write would find, ends its
    # sequence too, steps of 50 ms long before its 40 tokens
    step, steps = engine.step, []
    monkeypatch.setattr(engine, "step", lambda: steps.append(time.sleep(0.05)) or step())
    request = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 40}
    body = json.dumps(request).encode()
    head = f"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n"
    with socket.create_connection(("127.0.0.1", server.server_port)) as connection:
        connection.sendall(head.encode() + body)
        while not engine.active:
            time.sleep(0.01)
    while engine.active:
        time.sleep(0.01)
    assert len(steps) < 20


def test_engine_error(client, engine, monkeypatch):
    # a failing step fails every running completion, and the server serves the next
    def fail():
        monkeypatch.undo()
        raise RuntimeError("out of memory")

    monkeypatch.setattr(engine, "step", fail)
    with pytest.raises(openai.InternalServerError, match="out of memory"):
        chat(client, "hello", max_tokens=4)
    assert complete(client, "hello", max_tokens=2)[1] == "length"


@pytest.fixture(scope="module")
def served(model_path) -> Iterator[openai.OpenAI]:
    with serving(model_path, max_context=4096, slots=4) as server:
        server.load(model_path.stem)
        yield connect(server)


@pytest.mark.gpu
@pytest.mark.model
def test_calls_tools(served, model_path):
    if "tools" not in chat_template(model_path):
        pytest.skip("the model's chat template takes no tools")
    messages = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    response = served.chat.completions.create(
        model="real", messages=messages, tools=[WEATHER], temperature=0
    )
    (call,) = response.choices[0].message.tool_calls
    assert call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@pytest.mark.gpu
@pytest.mark.model
def test_shared_system_prompt(served, model_path):
    # a system prompt that another conversation cached cuts the time to the first token by 10x or
    # more: from 449 to 22 ms for these 2141 tokens on the 3090
    if "system" not in chat_template(model_path):
        pytest.skip("the model's chat template takes no system prompt")

    def system(name: str) -> str:
        rule = "{} {}: answer plainly, cite source {}, and keep replies under {} words."
        return " ".join(rule.format(name, i, i, i + 50) for i in range(100))

    def first_token(system: str, question: str) -> float:
        # the time to a reply of one token: text, or for a model that thinks first, reasoning
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        start = time.perf_counter()
        served.chat.completions.create(model="real", messages=messages, max_tokens=1, temperature=0)
        return time.perf_counter() - start

    for question in ("Hi", "Hello"):  # captures the graphs, the copy's too
        first_token(system("Warm"), question)
    cold, warm = [], []
    for name in ("Rule", "Law", "Step"):
        cold.append(first_token(system(name), "What is rule 3?"))
        warm.append(first_token(system(name), "And rule 7?"))
    assert statistics.median(cold) > 10 * statistics.median(warm)
