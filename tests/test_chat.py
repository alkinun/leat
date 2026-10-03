import json

import jinja2
import pytest

from leat.chat import ChatTemplate, may_call_tool, parse_tool_call
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


def test_tool_calls():
    tools = [{"type": "function", "function": {"name": "weather"}}]
    paris = {"name": "weather", "arguments": {"city": "Paris"}}
    for key, space in (("parameters", " "), ("arguments", "\n")):
        reply = space + json.dumps({"name": "weather", key: {"city": "Paris"}}) + space
        assert parse_tool_call(reply, tools) == paris
    for text in ("It is sunny.", '{"name": "news", "parameters": {}}', '{"name": "weather"}'):
        assert parse_tool_call(text, tools) is None
    assert may_call_tool("") and may_call_tool(' {"na') and not may_call_tool(" It")


def test_no_template():
    with pytest.raises(ValueError, match="no chat template"):
        ChatTemplate(tiny_metadata(), Tokenizer(tiny_metadata()))


@pytest.mark.model
def test_llama3(model_path):
    metadata = GGUF.open(model_path).metadata
    c = ChatTemplate(metadata, Tokenizer(metadata))
    text = c.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
    assert text == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe brief.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
