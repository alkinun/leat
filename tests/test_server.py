import contextlib
import json
import statistics
import threading
import time
from collections.abc import Iterator

import openai
import pytest

from leat.engine import Engine
from leat.server import Server
from tests.helpers import CONTEXT

WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "The weather in a city now",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@contextlib.contextmanager
def serving(engine: Engine) -> Iterator[openai.OpenAI]:
    # a client of a server of the engine, on a free port
    with Server(engine, port=0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/v1"
        yield openai.OpenAI(base_url=url, api_key="unused", max_retries=0)
        server.shutdown()


@pytest.fixture(scope="module")
def engine(tiny_model) -> Engine:
    return Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)


@pytest.fixture(scope="module")
def client(engine) -> Iterator[openai.OpenAI]:
    with serving(engine) as client:
        yield client


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
    assert [model.id for model in client.models.list()] == ["tiny"]


def test_reply(client, expected):
    response = chat(client, "hello", max_tokens=8, temperature=0)
    assert response.choices[0].message.content == expected("hello", 8)
    assert response.choices[0].finish_reason == "length"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (5, 8)


def test_stream(client, expected):
    usage = {"include_usage": True}
    stream = chat(client, "hello", max_tokens=8, temperature=0, stream=True, stream_options=usage)
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks[:-1]) == expected("hello", 8)
    assert chunks[-2].choices[0].finish_reason == "length"
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 8


@pytest.mark.parametrize("stream", [False, True])
def test_stop(client, expected, stream):
    full = expected("stop me", 16)
    stops = ["never there", full[6:9]]
    reply = complete(client, "stop me", max_tokens=16, temperature=0, stop=stops, stream=stream)
    assert reply == (full[: full.index(stops[1])], "stop")


def test_seed(client):
    # sampled, as temperature is 1 unless given
    first, again = (complete(client, "seeded", max_tokens=8, seed=3) for _ in range(2))
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
    # makes the engine reply with the given text, which the tiny model would never write
    def reply_with(text: str) -> None:
        tokens = engine.tokenizer.encode(text)
        monkeypatch.setattr(engine, "generate", lambda *args: (t for t in tokens))

    return reply_with


# as Llama 3, Qwen3 and Gemma 4 call tools; Qwen3's text before its call is the content
CALLS = [
    (' {"name": "weather", "parameters": {"city": "Paris"}}', None),
    ('Checking.\n<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>',
     "Checking."),
    ('<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>', None),
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
        ({"top_p": 0.5}, "top_p=0.5 is not supported"),
        ({"temperature": 3}, "temperature must be a number from 0 to 2"),
        ({"max_tokens": 0}, "max_tokens must be a positive integer"),
        ({"tools": [WEATHER], "tool_choice": "required"}, "tool_choice='required' is not"),
        ({"messages": []}, "messages must be a non-empty list of objects"),
        ({"messages": [{"role": "user", "content": "x" * CONTEXT}]}, "the prompt has 64 tokens"),
        ({"extra_body": {"chat_template_kwargs": "x"}}, "chat_template_kwargs must be an object"),
    ],
)
def test_bad_request(client, kwargs, error):
    request = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]} | kwargs
    with pytest.raises(openai.BadRequestError, match=error):
        client.chat.completions.create(**request)


def test_unknown_route(client):
    with pytest.raises(openai.NotFoundError):
        client.completions.create(model="tiny", prompt="hi")


def test_client_hangs_up(client):
    stream = chat(client, "a long reply", max_tokens=40, stream=True)
    next(iter(stream))
    stream.close()
    assert complete(client, "and the next", max_tokens=2)[1] == "length"


@pytest.fixture(scope="module")
def served(model_path) -> Iterator[openai.OpenAI]:
    with serving(Engine(model_path, max_context=4096, slots=4)) as client:
        yield client


@pytest.mark.gpu
@pytest.mark.model
def test_calls_tools(served):
    messages = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    response = served.chat.completions.create(
        model="real", messages=messages, tools=[WEATHER], temperature=0
    )
    (call,) = response.choices[0].message.tool_calls
    assert call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@pytest.mark.gpu
@pytest.mark.model
def test_shared_system_prompt(served):
    # a system prompt that another conversation cached cuts the time to the first token by 5x or
    # more: from 549 to 55 ms for these 2141 tokens on the 3090
    def system(name: str) -> str:
        rule = "{} {}: answer plainly, cite source {}, and keep replies under {} words."
        return " ".join(rule.format(name, i, i, i + 50) for i in range(100))

    def first_token(system: str, question: str) -> float:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        start = time.perf_counter()
        with served.chat.completions.create(
            model="real", messages=messages, max_tokens=1, temperature=0, stream=True
        ) as stream:
            next(chunk for chunk in stream if chunk.choices and chunk.choices[0].delta.content)
        return time.perf_counter() - start

    for question in ("Hi", "Hello"):  # captures the graphs, the copy's too
        first_token(system("Warm"), question)
    cold, warm = [], []
    for name in ("Rule", "Law", "Step"):
        cold.append(first_token(system(name), "What is rule 3?"))
        warm.append(first_token(system(name), "And rule 7?"))
    assert statistics.median(cold) > 5 * statistics.median(warm)
