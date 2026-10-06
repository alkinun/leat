import jinja2
import pytest

from leat.chat import ChatTemplate, Reply, parse_tool_calls, split_reply, tool_call_start
from leat.gguf import GGUF
from leat.tokenizer import Tokenizer
from tests.helpers import ids, tiny_metadata

TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}{{ eos_token }}"
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
)


def chat(template: str = TEMPLATE) -> tuple[ChatTemplate, Tokenizer]:
    metadata = tiny_metadata(**{"tokenizer.chat_template": template})
    tok = Tokenizer(metadata)
    return ChatTemplate(metadata, tok), tok


def test_render():
    c, _ = chat()
    assert c.render([{"role": "user", "content": "hi"}]) == "<s>user:hi<|eot|>assistant:"
    assert (
        c.render([{"role": "user", "content": "hi"}], add_generation_prompt=False)
        == "<s>user:hi<|eot|>"
    )


def test_bos_exactly_once():
    msgs = [{"role": "user", "content": "ab"}]
    for template in (TEMPLATE, TEMPLATE.removeprefix("{{ bos_token }}")):
        c, tok = chat(template)
        encoded = c.encode(msgs)
        assert encoded[0] == tok.bos_id and encoded.count(tok.bos_id) == 1
        assert ids(tok, "<|eot|>")[0] in encoded  # control text from the template is parsed


def test_template_helpers():
    c, _ = chat("{{ raise_exception('no tools') if tools else messages | tojson }}")
    assert c.render([{"content": "<é>"}]) == '[{"content": "<é>"}]'  # no HTML escaping
    with pytest.raises(jinja2.TemplateError, match="no tools"):
        c.render([], tools=[{"name": "f"}])


def test_openai_messages():
    # text parts are joined, and tool calls are left out where there are none and otherwise have
    # their JSON arguments decoded, as templates expect
    c, _ = chat(
        "{% for m in messages %}{{ m.content or '' }}"
        "{% if 'tool_calls' in m %}{{ m.tool_calls[0].function.arguments.city }}{% endif %};"
        "{% endfor %}"
    )
    function = {"name": "f", "arguments": '{"city": "Oslo"}'}
    call = {"id": "1", "type": "function", "function": function}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "assistant", "content": "c", "tool_calls": None},
    ]
    assert c.render(messages) == "a\nb;Oslo;c;"
    with pytest.raises(ValueError, match="only text"):
        c.render([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])


WEATHER = [{"type": "function", "function": {"name": "weather"}}]
PARIS = {"name": "weather", "arguments": {"city": "Paris"}}


@pytest.mark.parametrize(
    "reply, text",
    [
        (' {"name": "weather", "parameters": {"city": "Paris"}} ', ""),  # Llama 3
        ('\n{"name": "weather", "arguments": {"city": "Paris"}}\n', ""),
        ('Let me see.\n<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n'
         "</tool_call>", "Let me see."),  # Qwen3
        ("<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
         "</tool_call>", ""),  # Qwen3.5
        ('<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>', ""),  # Gemma 4
    ],
)  # fmt: skip
def test_tool_calls(reply, text):
    assert parse_tool_calls(reply, WEATHER) == (text, [PARIS])


def test_tool_call_arguments():
    # Gemma 4's syntax: quoted strings, which may hold its other marks, bare or quoted keys,
    # nested objects and lists, and JSON's other values; several calls in a row
    q = '<|"|>'
    args = f"{{a:{q}x, y: {{z}}{q},b:[1,-2.5,true,null],c:{{{q}d{q}:false}}}}"
    reply = f"<|tool_call>call:weather{args}<tool_call|><|tool_call>call:weather{{}}<tool_call|>"
    arguments = {"a": "x, y: {z}", "b": [1, -2.5, True, None], "c": {"d": False}}
    assert parse_tool_calls(reply, WEATHER) == (
        "",
        [PARIS | {"arguments": arguments}, PARIS | {"arguments": {}}],
    )


def test_tool_call_parameters():
    # Qwen3.5's syntax: values of several lines, as text where the tool takes a string and else
    # as JSON, or as text where that fails; several calls in a row
    properties = {"q": {"type": "string"}, "n": {"type": "integer"}, "tags": {"type": "array"}}
    tools = [{"type": "function", "function": {"name": "search", "parameters": {
        "type": "object", "properties": properties}}}]  # fmt: skip
    call = (
        "<tool_call>\n<function=search>\n<parameter=q>\n12\nmore lines\n</parameter>\n"
        '<parameter=n>\n5\n</parameter>\n<parameter=tags>\n["a"]\n</parameter>\n'
        "<parameter=other>\nnot json\n</parameter>\n</function>\n</tool_call>"
    )
    arguments = {"q": "12\nmore lines", "n": 5, "tags": ["a"], "other": "not json"}
    assert parse_tool_calls("Searching.\n\n" + call + "\n" + call, tools) == (
        "Searching.",
        [{"name": "search", "arguments": arguments}] * 2,
    )


@pytest.mark.parametrize(
    "reply",
    [
        "It is sunny.",
        '{"name": "news", "parameters": {}}',
        '{"name": "weather"}',
        '<tool_call>{"name": "weather", "arguments": {}</tool_call>',
        "<|tool_call>call:news{}<tool_call|>",
        "<tool_call>\n<function=weather>\nParis\n</function>\n</tool_call>",
        "<|tool_call>call:weather{city:Paris}<tool_call|>",
    ],
)
def test_text_is_not_a_tool_call(reply):
    assert parse_tool_calls(reply, WEATHER) == (reply, [])


def test_tool_call_start():
    # what follows may yet be a tool call: all of a reply that may be JSON, from a marker on, or
    # the start of a marker at the end, and whitespace before those
    replies = ("", ' {"na', " It", "Hi\n<tool_call>{", "Hi <|tool", "Hi ")
    assert [tool_call_start(t) for t in replies] == [0, 0, 3, 2, 2, 2]


def test_no_template():
    with pytest.raises(ValueError, match="no chat template"):
        ChatTemplate(tiny_metadata(), Tokenizer(tiny_metadata()))


@pytest.mark.model
def test_llama3(model_path):
    metadata = GGUF.open(model_path).metadata
    if "<|start_header_id|>" not in metadata.get("tokenizer.chat_template", ""):
        pytest.skip("checks Llama 3's template")
    c = ChatTemplate(metadata, Tokenizer(metadata))
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    # Llama 3.1's template dates the system prompt 26 Jul 2024, 3.2's today, unless told
    text = c.render(messages, date_string="26 Jul 2024")
    assert text == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe brief.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


HARMONY = (
    "<|channel|>analysis<|message|>The user wants weather.<|end|><|start|>assistant"
    "<|channel|>final<|message|>It is sunny."
)
HARMONY_CALL = (
    "<|channel|>analysis<|message|>Need a tool.<|end|><|start|>assistant<|channel|>commentary "
    'to=functions.weather <|constrain|>json<|message|>{"city": "Paris"}'
)


@pytest.mark.parametrize(
    "text, form, thinking, reply",
    [
        ("<think>\nhmm</think>\n\nHi.", "think", False, Reply("hmm", "Hi.")),
        ("hmm</think>Hi.", "think", True, Reply("hmm", "Hi.")),  # the prompt opened it
        ("Hi.", "think", False, Reply(content="Hi.")),
        ("<think>still", "think", False, Reply("still")),
        ("a <think>b", None, False, Reply(content="a <think>b")),
        (HARMONY, "harmony", False, Reply("The user wants weather.", "It is sunny.")),
        (HARMONY_CALL, "harmony", False,
         Reply("Need a tool.", "", [{"name": "weather", "arguments": '{"city": "Paris"}'}])),
    ],
)  # fmt: skip
def test_split_reply(text, form, thinking, reply):
    assert split_reply(text, form, thinking) == reply


def test_split_reply_done():
    # an end that may begin a marker waits for more, but not once the reply is done
    assert split_reply("<think>a <", "think") == Reply("a ")
    assert split_reply("<think>a <", "think", done=True) == Reply("a <")
    assert split_reply("x </", None) == Reply(content="x ")
    assert split_reply("x </", None, done=True) == Reply(content="x </")


@pytest.mark.parametrize(
    "text, form",
    [(HARMONY, "harmony"), ("<think>a b</think> c d", "think"), ("\n<think>a</think> b", "think")],
)
def test_split_reply_streams(text, form):
    # every prefix splits into prefixes of the whole reply's parts: what a stream sent stays
    whole = split_reply(text, form)
    for n in range(len(text)):
        part = split_reply(text[:n], form)
        assert whole.reasoning.startswith(part.reasoning) and whole.content.startswith(part.content)
