import collections
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

from leat.chat import ChatTemplate, Reply
from leat.engine import Engine
from leat.keys import Keys
from leat.sampler import Sampling
from leat.server import Server, _calls, _completion, _Load, _Writer
from tests.helpers import CONTEXT, Oracle, chat_template

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
    (model,) = client.models.list()
    assert (model.id, model.status, model.max_context) == ("tiny", "loaded", CONTEXT)


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


def test_completion_waits_for_a_load(tiny_model, tmp_path, monkeypatch):
    # a completion asked for once a model is to load waits for it, rather than finding none
    # loaded, and the models asked for meanwhile say it is loading, as when leat serve starts
    (other := tmp_path / "other.gguf").symlink_to(tiny_model[0])
    warm_up = Engine.warm_up
    monkeypatch.setattr(Engine, "warm_up", lambda self: time.sleep(0.5) or warm_up(self))
    with serving(tiny_model[0], other, max_context=CONTEXT) as server:
        client = connect(server)
        loading = threading.Thread(target=server.load, args=("tiny",))
        loading.start()
        while server.ready.is_set():
            time.sleep(0.001)
        assert chat(client, "hello", max_tokens=2).model == "tiny"
        loading.join()
        loading = threading.Thread(target=load, args=(client, "other"))
        loading.start()
        while server.loading is None:
            time.sleep(0.01)
        statuses = {model.id: model.status for model in client.models.list()}
        assert statuses == {"tiny": "unloaded", "other": "loading"}
        loading.join()


def test_api_alone(server):
    # the API and nothing else: the app is leat agent's
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/")


def test_hang_up_is_quiet(server, capsys):
    # a client that hung up before its answer was written, as the agent's wait for the models
    # does while the engine compiles, is no error to print; any other error is
    for error in (BrokenPipeError(), ConnectionResetError(), ValueError("a bug")):
        try:
            raise error
        except Exception:
            server.handle_error(None, ("127.0.0.1", 1))
    printed = capsys.readouterr().err
    assert "ValueError: a bug" in printed and "Broken" not in printed and "Reset" not in printed


def check_timings(timings: dict, usage) -> None:
    # llama.cpp's timings of a reply, as its usage counts its tokens
    prompt = (timings["cache_n"], timings["prompt_n"])
    assert prompt == (usage.prompt_tokens_details.cached_tokens, usage.prompt_tokens - prompt[0])
    assert timings["predicted_n"] == usage.completion_tokens
    for name in ("prompt", "predicted"):
        n, ms = timings[f"{name}_n"], timings[f"{name}_ms"]
        if name == "predicted" and n == 1:  # one token takes no step, of no rate
            assert timings["predicted_per_token_ms"] == timings["predicted_per_second"] == 0
            continue
        assert ms > 0 and timings[f"{name}_per_token_ms"] == pytest.approx(ms / n)
        assert timings[f"{name}_per_second"] == pytest.approx(1e3 * n / ms)


def test_reply(client, expected):
    response = chat(client, "hello", max_tokens=8, temperature=0)
    assert response.choices[0].message.content == expected("hello", 8)
    assert response.choices[0].finish_reason == "length"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (5, 8)
    check_timings(response.model_extra["timings"], response.usage)
    response = chat(client, "hello", max_tokens=1)
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


def test_timings_leave_out_the_wait(client, engine, monkeypatch):
    # a request that waits for a slot, both taken by longer replies, is timed from when it gets
    # one: the wait is in the time its client takes to see the first token, not in its timings.
    # Each step takes 10 ms more, so that the wait is long beside a GPU's few milliseconds.
    step = engine.step
    monkeypatch.setattr(engine, "step", lambda: time.sleep(0.01) or step())
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


@pytest.mark.parametrize("stream", [False, True])
def test_speculative_steps(tiny, tiny_assistant, stream):
    # a step that gives several tokens ends the reply at max_tokens or a stop string within them
    path = tiny("gemma4")[0]
    plain = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    prompt = plain.tokenizer.encode("go on")
    tokens = list(plain.generate(prompt, 16))
    with serving(path, max_context=CONTEXT, prefill_chunk=8, draft=tiny_assistant[0]) as server:
        server.load("tiny")
        server.loaded.engine.drafter = Oracle(prompt + tokens)  # every step gives 4 tokens
        client = connect(server)
        reply = complete(client, "go on", max_tokens=6, temperature=0, stream=stream)
        assert reply == (plain.tokenizer.decode(tokens[:6]), "length")
        full = plain.tokenizer.decode(tokens)
        cut = len(plain.tokenizer.decode(tokens[:3]))  # within the first speculative step
        stop = full[cut : cut + 3]
        reply = complete(client, "go on", max_tokens=16, temperature=0, stop=[stop], stream=stream)
        assert reply == (full[: full.index(stop)], "stop")


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
     ("gemma4", "<|channel>thought\nSunny, I recall.<channel|>It is sunny."),
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
        ({"stop": "x" * 257}, "stop must be a string or a list of 16 strings at most"),
        ({"stop": ["x"] * 17}, "stop must be a string or a list of 16 strings at most"),
        ({"tools": [WEATHER], "tool_choice": "required"}, "tool_choice='required' is not"),
        ({"messages": []}, "messages must be a non-empty list of objects"),
        ({"messages": [{"role": "user", "content": "x" * CONTEXT}]}, "the prompt has 64 tokens"),
        ({"extra_body": {"chat_template_kwargs": "x"}}, "chat_template_kwargs must be an object"),
        # what a message or tool holds, of the shapes templates read
        ({"messages": [{"role": "user", "content": [None]}]}, "text, image or image_url"),
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


def test_web_pages_are_refused(server):
    # a page may not make the server generate or load models, even one whose name was made this
    # address's, as DNS rebinding does; a client of no browser may
    url = f"http://127.0.0.1:{server.server_port}"
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    rebound = {"Origin": "http://evil.example:8080", "Host": "evil.example:8080"}
    for path, data in (("/v1/chat/completions", body), ("/v1/models/load", b'{"model": "tiny"}')):
        for headers in ({"Origin": "https://evil.example"}, rebound):
            with pytest.raises(urllib.error.HTTPError, match="403"):
                urllib.request.urlopen(urllib.request.Request(url + path, data, headers))
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body)) as r:
        assert r.status == 200


def test_unknown_route(client, server):
    with pytest.raises(openai.NotFoundError):
        client.completions.create(model="tiny", prompt="hi")
    url = f"http://127.0.0.1:{server.server_port}"
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(f"{url}/v1/nothing")
    with pytest.raises(urllib.error.HTTPError, match="400"):  # not JSON
        urllib.request.urlopen(urllib.request.Request(f"{url}/v1/chat/completions", b"{"))
    with urllib.request.urlopen(f"{url}/v1/models?x=1") as response:  # the path, its query aside
        assert response.status == 200


def metrics(server: Server, key: str | None = None) -> dict[str, float]:
    """/metrics' samples, by their names and labels as written."""
    url, headers = f"http://127.0.0.1:{server.server_port}/metrics", {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, None, headers)
    with urllib.request.urlopen(request) as response:
        assert response.headers["Content-Type"].startswith("text/plain; version=0.0.4")
        lines = response.read().decode().splitlines()
    return {k: float(v) for k, v in (line.rsplit(" ", 1) for line in lines if line[0] != "#")}


def test_metrics(client, server):
    # each completion's tokens and time counted, its requests by path and status, and the slots
    # and model of now; with no keys, of no key's
    before = metrics(server)
    _, reason = complete(client, "hello", max_tokens=3, temperature=0)
    after = metrics(server)
    grown = lambda name: after.get(name, 0) - before.get(name, 0)  # noqa: E731
    assert grown('leat_completion_tokens_total{key=""}') == 3
    assert grown('leat_completions_total{finish="length",key=""}') == 1 and reason == "length"
    assert grown('leat_prompt_tokens_total{key=""}') > 0
    assert grown('leat_decode_seconds_total{key=""}') > 0
    ok = 'leat_requests_total{key="",path="/v1/chat/completions",status="200"}'
    assert grown(ok) == 1
    assert grown('leat_requests_total{key="",path="/metrics",status="200"}') == 1
    assert after["leat_slots"] == 2 and after["leat_slots_busy"] == 0
    assert after["leat_completions_waiting"] == 0
    assert after['leat_model_loaded{model="tiny"}'] == CONTEXT


def test_keys(tiny_model, tmp_path):
    # given keys, every request must hold one, as OpenAI's clients send it or Anthropic's, and
    # each key's use is counted apart; a key removed is refused from the next request on
    keys = Keys(tmp_path / "keys.json")
    alkin, agent = keys.add("alkin"), keys.add("agent")
    with serving(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, keys=keys) as server:
        server.load("tiny")
        url = f"http://127.0.0.1:{server.server_port}"
        for headers in ({}, {"Authorization": "Bearer leat-wrong"}, {"x-api-key": alkin[:-1]}):
            for path in ("/v1/models", "/metrics"):
                with pytest.raises(urllib.error.HTTPError) as refused:
                    urllib.request.urlopen(urllib.request.Request(url + path, None, headers))
                assert refused.value.code == 401
                assert refused.value.headers["WWW-Authenticate"].startswith("Bearer")
                error = json.loads(refused.value.read())["error"]
                assert error["code"] == "invalid_api_key"
        with pytest.raises(openai.AuthenticationError):
            complete(connect(server), "hello", max_tokens=1)
        client = openai.OpenAI(base_url=f"{url}/v1", api_key=alkin, max_retries=0)
        assert complete(client, "hello", max_tokens=2, temperature=0)[1] == "length"
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
        anthropic = {"x-api-key": agent, "Content-Type": "application/json"}
        request = urllib.request.Request(f"{url}/v1/chat/completions", body.encode(), anthropic)
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
        counted = metrics(server, agent)
        assert counted['leat_completion_tokens_total{key="alkin"}'] == 2
        assert counted['leat_completion_tokens_total{key="agent"}'] == 1
        assert counted['leat_requests_total{key="",path="/v1/models",status="401"}'] == 3
        keys.remove("alkin")
        with pytest.raises(openai.AuthenticationError):
            client.models.list()
        server.shutdown()


def test_keys_share_their_own_prefixes(tiny_model, tmp_path):
    # a key's prompts start from the cached prefixes of its own alone: another key's that shares
    # them is prefilled whole, so that it cannot tell from its time to the first token, or its
    # cached tokens, what the first asked
    keys = Keys(tmp_path / "keys.json")
    alkin, agent = keys.add("alkin"), keys.add("agent")
    options = {"max_context": CONTEXT, "prefill_chunk": 8, "slots": 2, "keys": keys}
    with serving(tiny_model[0], **options) as server:
        server.load("tiny")
        url = f"http://127.0.0.1:{server.server_port}/v1"
        first, second = (openai.OpenAI(base_url=url, api_key=key, max_retries=0)
                         for key in (alkin, agent))  # fmt: skip

        def cached(client: openai.OpenAI, content: str) -> int:
            return chat(client, content, max_tokens=2).usage.prompt_tokens_details.cached_tokens

        assert cached(first, "a shared start, then one end") == 0
        assert cached(second, "a shared start, then another") == 0
        assert cached(first, "a shared start, then another") == len("a shared start, then ")
        assert cached(second, "a shared start, then more") == len("a shared start, then ")
        server.shutdown()


@pytest.mark.parametrize("keyed", [True, False])
@pytest.mark.parametrize("field", ["user", "safety_identifier"])
def test_users_share_their_own_prefixes(tiny_model, tmp_path, keyed, field):
    # an end user a request names, as an app serving several people does, shares cached prefixes
    # with that user's prompts alone, under one key or without keys; prompts that name no one
    # share with each other
    keys = Keys(tmp_path / "keys.json") if keyed else None
    key = keys.add("agent") if keys else "unused"
    options = {"max_context": CONTEXT, "prefill_chunk": 8, "slots": 3, "keys": keys}
    with serving(tiny_model[0], **options) as server:
        server.load("tiny")
        url = f"http://127.0.0.1:{server.server_port}/v1"
        client = openai.OpenAI(base_url=url, api_key=key, max_retries=0)

        def cached(content: str, user: str | None = None) -> int:
            named = {"extra_body": {field: user}} if user else {}
            response = chat(client, content, max_tokens=2, **named)
            return response.usage.prompt_tokens_details.cached_tokens

        shared = len("a shared start, then ")
        assert cached("a shared start, then one end", "person-1") == 0
        assert cached("a shared start, then another", "person-2") == 0
        assert cached("a shared start, then more") == 0
        assert cached("a shared start, then yet more", "person-1") == shared
        assert cached("a shared start, then the last", "person-2") == shared
        assert cached("a shared start, then no one's") == shared
        with pytest.raises(openai.BadRequestError, match=f"{field} must be a string"):
            chat(client, "hi", max_tokens=1, extra_body={field: 1})
        server.shutdown()


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


def test_worker_error(client, monkeypatch):
    # a failure past the engine's step, a bug's, fails the completions rather than the worker
    def fail(self, token):
        monkeypatch.undo()
        raise KeyError("a bug")

    monkeypatch.setattr(_Writer, "take", fail)
    with pytest.raises(openai.InternalServerError, match="a bug"):
        chat(client, "hello", max_tokens=4)
    assert complete(client, "hello", max_tokens=2)[1] == "length"


def test_worker_error_ends_a_waiting_load(server):
    # completions waiting for a load the error ended go on, rather than waiting forever
    server.ready.clear()
    load = _Load("tiny")
    server._fail(KeyError("a bug"), collections.deque([load]), {})
    assert isinstance(load.done.get(), KeyError) and server.ready.is_set()


def test_loading_the_loaded_model_waits_for_nothing(client, server, monkeypatch):
    # not even for the completions running, which a load of another model waits out
    going, take = threading.Event(), _Writer.take
    monkeypatch.setattr(_Writer, "take", lambda self, token: going.wait(5) and take(self, token))
    reply = threading.Thread(target=chat, args=(client, "hello"), kwargs={"max_tokens": 2})
    reply.start()
    loading = threading.Thread(target=server.load, args=("tiny",))
    loading.start()
    loading.join(2)
    assert not loading.is_alive()
    going.set()
    reply.join()


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


def test_cut_off_call():
    # a call the context cut off is no text: the reply's text ends where it began
    reply = Reply(content="Here it is.\n<tool_call>\n<function=weather>\n<parameter=city>\nPar")
    assert _calls(reply, [WEATHER], "length") == ("Here it is.", [])
    assert _calls(reply, [WEATHER], "stop")[0] == reply.content
