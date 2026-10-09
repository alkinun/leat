import json

import jinja2
import pytest

from leat.chat import (
    EFFORTS,
    IMAGE,
    ChatTemplate,
    Reply,
    images,
    parse_tool_calls,
    split_reply,
    tool_call_start,
)
from leat.gguf import GGUF
from leat.tokenizer import USER_DEFINED, Tokenizer
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
    with pytest.raises(ValueError, match="text, image or image_url"):
        c.render([{"role": "user", "content": [{"type": "input_audio"}]}])
    # a call's content of None, as OpenAI's clients send it, as text to templates that take text,
    # as Qwen3's slices it
    sliced = chat("{% for m in messages %}{{ m.content[:3] }}|{{ m.content is none }};{% endfor %}")
    assert sliced[0].render(messages[1:2]) == "|False;"


def test_undefined_joins_as_no_text():
    # a tool's description, which OpenAI's API leaves out at will, joined to text as gpt-oss's
    # template joins it; any other use of what is not there fails as before
    c, _ = chat("{% for t in tools %}{{ '// ' + t.description + '!' }}{% endfor %}")
    assert c.render([], tools=[{"name": "f"}, {"name": "g", "description": "G"}]) == "// !// G!"
    with pytest.raises(jinja2.UndefinedError):
        chat("{{ tools[0].description + 1 }}")[0].render([], tools=[{"name": "f"}])


def test_images():
    # an image part stands as IMAGE in the text, joined to its neighbours without a newline, and
    # its tokens take its place; data: URLs give its bytes, other URLs none
    c, tok = chat("{% for m in messages %}{{ m.content }};{% endfor %}")
    png = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0="}}
    parts = [{"type": "text", "text": "a"}, png, {"type": "text", "text": f"b{IMAGE}"},
             {"type": "text", "text": "c"}]  # fmt: skip
    text = c.render([{"role": "user", "content": parts}])
    assert text == f"a{IMAGE}b\nc;"
    # transformers' image parts stand for images too, and text of its own holds no IMAGE
    other = [{"type": "image"}, {"type": "text", "text": "d"}]
    assert c.render([{"content": f"e{IMAGE}"}, {"content": other}]) == f"e;{IMAGE}d;"
    # nor do a tool call's arguments, or what else the template is given
    called = {"function": {"name": "f", "arguments": json.dumps({"x": IMAGE})}}
    shown = chat("{{ messages[0].tool_calls[0].function.arguments.x }}{{ tools[0] }}")[0]
    assert shown.render([{"tool_calls": [called]}], tools=[IMAGE]) == ""
    assert c.tokens(text, [[-5, -5]]) == tok.encode("a") + [-5, -5] + tok.encode("b\nc;", bos=False)
    with pytest.raises(ValueError, match="1 images, 0 are given"):
        c.tokens(text)
    assert images([{"role": "user", "content": parts}, {"content": "d"}]) == [b"\x89PNG\r"]
    for url in ("https://example.com/a.png", "data:image/png;base64,not base64!"):
        with pytest.raises(ValueError, match="data: URL|not base64"):
            images([{"content": [{"type": "image_url", "image_url": {"url": url}}]}])


def with_marker(marker: str) -> ChatTemplate:
    # chat()'s template, of a vocab that holds a reasoning marker, user-defined as in those models
    metadata = tiny_metadata(**{"tokenizer.chat_template": TEMPLATE})
    metadata["tokenizer.ggml.tokens"] = [*metadata["tokenizer.ggml.tokens"], marker]
    metadata["tokenizer.ggml.token_type"] = [*metadata["tokenizer.ggml.token_type"], USER_DEFINED]
    return ChatTemplate(metadata, Tokenizer(metadata))


def test_form():
    # how replies mark their reasoning, by the vocab's markers or the template's, whether a prompt
    # opened a block of it, and what opens an answer
    think = chat(TEMPLATE + "<think>")[0]
    assert chat()[0].form is None and think.form == "think" and think.opens_thinking("a<think>\n")
    harmony = with_marker("<|channel|>")
    assert harmony.form == "harmony" and harmony.answer == "<|channel|>final<|message|>"
    assert chat()[0].answer == ""  # of a format that marks no answer
    gemma = with_marker("<|channel>")
    assert gemma.form == "gemma4" and gemma.opens_thinking("<|turn>model\n<|channel>thought\n")
    assert not gemma.opens_thinking("<|turn>model\n<|channel>thought\n<channel|>")


def test_thinking():
    # reasoning as gpt-oss's template reads it too, of a message without text, which it would
    # take for the reasoning and refuse both
    c, _ = chat("{% for m in messages %}{{ m.thinking }}|{% endfor %}")
    call = {"type": "function", "function": {"name": "f", "arguments": "{}"}}
    messages = [
        {"role": "assistant", "content": "", "reasoning_content": "Hm.", "tool_calls": [call]},
        {"role": "assistant", "content": "Hi.", "reasoning_content": "Hm."},
        {"role": "assistant", "content": "", "reasoning_content": "Hm.", "thinking": "Own."},
    ]
    assert c.render(messages, add_generation_prompt=False) == "Hm.||Own.|"


GPT_OSS = (
    '{%- if reasoning_effort is not defined %}{%- set reasoning_effort = "medium" %}{%- endif %}'
)
MISTRAL = (
    "{%- set reasoning_effort = reasoning_effort | default('none') %}"
    "{%- if reasoning_effort not in ['none', 'high'] %}{{ raise_exception('no') }}{% endif %}"
)
LEVELS = (
    "{%- if reasoning_effort == 'xhigh' %}x{% elif 'low' == reasoning_effort %}l{% endif %}"
    '{%- set reasoning_effort = reasoning_effort | default("medium") %}'
)
GEMMA = "{%- set enable_thinking = enable_thinking | default(false) -%}"
QWEN = "{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}"


@pytest.mark.parametrize(
    "template, efforts, default",
    [
        (TEMPLATE, (), None),  # never reasons
        (TEMPLATE + "<think>", (), None),  # always does, and cannot be told
        (GPT_OSS + TEMPLATE, ("low", "medium", "high"), "medium"),  # compares it with nothing
        (MISTRAL + TEMPLATE, ("none", "high"), "none"),  # of the values it compares it with
        (LEVELS + TEMPLATE, ("low", "medium", "xhigh"), "medium"),
        (GEMMA + TEMPLATE, ("none", "high"), "none"),  # off by default, as it says
        (QWEN + TEMPLATE, ("none", "high"), "high"),  # on, unless turned off
    ],
)
def test_efforts(template, efforts, default):
    # the efforts a template's model reasons at, and its default, of what the template reads
    c = chat(template)[0]
    assert (c.efforts, c.default_effort) == (efforts, default)


def test_effort():
    # a request's effort, as the template is told it: the nearest of its levels, the lesser of two
    # as near; on or off; or nothing
    levels, toggle = chat(LEVELS + TEMPLATE)[0], chat(QWEN + TEMPLATE)[0]
    told = {e: levels.effort(e)["reasoning_effort"] for e in EFFORTS}
    assert told == {"none": "low", "minimal": "low", "low": "low", "medium": "medium",
                    "high": "medium", "xhigh": "xhigh"}  # fmt: skip
    turned = [toggle.effort(e)["enable_thinking"] for e in ("none", "minimal", "xhigh")]
    assert turned == [False, True, True]
    assert chat()[0].effort("high") == {}
    shown = chat(GPT_OSS + "Reasoning: {{ reasoning_effort }}")[0]
    assert shown.render([], **shown.effort("xhigh")) == "Reasoning: high"
    assert shown.render([]) == "Reasoning: medium"


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
        ('<|tool_call>call:weather{ city : <|"|>Paris<|"|> }<tool_call|>', ""),  # with spaces
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


def test_tool_call_schema_as_written():
    # a tool's parameters as its client wrote them, a JSON schema or not: values as JSON but where
    # the schema says string
    call = "<tool_call>\n<function=f>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>"
    for parameters in ("x", {"properties": "x"}, {"properties": {"n": "x"}}):
        tools = [{"type": "function", "function": {"name": "f", "parameters": parameters}}]
        assert parse_tool_calls(call, tools) == ("", [{"name": "f", "arguments": {"n": 5}}])


def test_tool_call_parameters():
    # Qwen3.5's syntax: values of several lines, as text where the tool takes a string and else
    # as JSON, or as text where that fails; several calls in a row
    properties = {
        "q": {"type": "string"},
        "n": {"type": "integer"},
        "tags": {"type": "array"},
        "id": {"type": ["string", "null"]},
        "k": {"type": ["integer", "string"]},
    }
    tools = [{"type": "function", "function": {"name": "search", "parameters": {
        "type": "object", "properties": properties}}}]  # fmt: skip
    call = (
        "<tool_call>\n<function=search>\n<parameter=q>\n12\nmore lines\n</parameter>\n"
        '<parameter=n>\n5\n</parameter>\n<parameter=tags>\n["a"]\n</parameter>\n'
        "<parameter=other>\nnot json\n</parameter>\n<parameter=id>\n123\n</parameter>\n"
        "<parameter=k>\n7\n</parameter>\n</function>\n</tool_call>"
    )
    # of several types, one of them string, JSON only of another: as llama.cpp's parser
    arguments = {
        "q": "12\nmore lines",
        "n": 5,
        "tags": ["a"],
        "other": "not json",
        "id": "123",
        "k": 7,
    }
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
        # what is no call however malformed, nested too deep to parse even
        '{"name": [], "parameters": {}}',
        "[" * 100000,
        "<|tool_call>call:weather{city:" + "[" * 100000 + "<tool_call|>",
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
        # Gemma 4's thought channel
        ("<|channel>thought\nhmm<channel|>Hi.", "gemma4", False, Reply("hmm", "Hi.")),
        ("Hi.", "gemma4", False, Reply(content="Hi.")),
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
    [
        (HARMONY, "harmony"),
        ("<think>a b</think> c d", "think"),
        ("\n<think>a</think> b", "think"),
        ("<|channel>thought\na b<channel|> c d", "gemma4"),
    ],
)
def test_split_reply_streams(text, form):
    # every prefix splits into prefixes of the whole reply's parts: what a stream sent stays
    whole = split_reply(text, form)
    for n in range(len(text)):
        part = split_reply(text[:n], form)
        assert whole.reasoning.startswith(part.reasoning) and whole.content.startswith(part.content)
