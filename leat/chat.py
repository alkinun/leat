"""Chat prompts, rendered with the model's own Jinja template from GGUF metadata, and replies
split into their reasoning, text and tool calls, in each supported model's syntax."""

import contextlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from leat.tokenizer import Tokenizer


class ChatTemplate:
    """Turns OpenAI-style messages into prompt tokens.

    Rendering matches transformers' `apply_chat_template` once text parts are joined and the JSON
    arguments of tool calls decoded, as templates expect. As there and in llama.cpp, control-token
    text inside message content is parsed as control tokens.
    """

    def __init__(self, metadata: dict[str, Any], tokenizer: Tokenizer):
        if (source := metadata.get("tokenizer.chat_template")) is None:
            raise ValueError("the model has no chat template")
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(source)
        self._tokenizer = tokenizer
        tokens = metadata["tokenizer.ggml.tokens"]
        self._bos = "" if tokenizer.bos_id is None else tokens[tokenizer.bos_id]
        self._eos = "" if tokenizer.eos_id is None else tokens[tokenizer.eos_id]
        # how replies mark their reasoning: gpt-oss's harmony channels, or the <think> blocks of
        # Qwen3 and DeepSeek-R1's distillations, whose templates write them
        self._form = (
            "harmony" if "<|channel|>" in tokens else "think" if "<think>" in source else None
        )

    @property
    def form(self) -> str | None:
        """How replies mark their reasoning: "harmony", "think", or None."""
        return self._form

    def render(
        self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **kwargs
    ) -> str:
        return self._template.render(
            messages=[_message(m) for m in messages],
            add_generation_prompt=add_generation_prompt,
            bos_token=self._bos,
            eos_token=self._eos,
            **kwargs,
        )

    def encode(
        self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **kwargs
    ) -> list[int]:
        return self.tokens(self.render(messages, add_generation_prompt, **kwargs))

    def tokens(self, text: str) -> list[int]:
        """A rendered prompt's tokens."""
        # most templates write the BOS text themselves; add it only when they don't
        bos = self._tokenizer.add_bos and not (self._bos and text.startswith(self._bos))
        return self._tokenizer.encode(text, bos=bos, special=True)

    def opens_thinking(self, text: str) -> bool:
        """Whether a rendered prompt ends inside a <think> block, which the reply then continues,
        as DeepSeek-R1's distillations' templates open it."""
        return self.form == "think" and text.rstrip().endswith("<think>")


@dataclass
class Reply:
    """A reply split: its reasoning, its text, and the tool calls it makes in the harmony format,
    gpt-oss's, each {"name": ..., "arguments": JSON text}. Other formats' calls are in the text,
    for parse_tool_calls."""

    reasoning: str = ""
    content: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)


# the harmony format's markers, and those of <think> blocks
_REPLY_MARKERS = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|constrain|>",
                      "<think>", "</think>")  # fmt: skip


def split_reply(text: str, form: str | None, thinking: bool = False) -> Reply:
    """A reply, or as much of it as there is so far: what is surely reasoning, surely text, and
    the calls. An end that may be the start of a marker waits for more, so that what a stream
    has split stays so. `thinking` says the prompt opened a <think> block."""
    text = text[: len(text) - _partial(text, _REPLY_MARKERS)]
    if form == "harmony":
        return _harmony(text)
    if form == "think":
        start = text.find("<think>")
        if start >= 0 and not text[:start].strip():
            text, thinking = text[start + len("<think>") :], True
        if thinking:
            reasoning, _, content = text.partition("</think>")
            return Reply(reasoning.lstrip(), content.lstrip() if _ else "")
    return Reply(content=text)


def _harmony(text: str) -> Reply:
    # messages separated by <|start|> or <|end|>, each a header, as <|channel|>analysis or
    # <|channel|>commentary to=functions.weather <|constrain|>json, then <|message|> and its text:
    # analysis is reasoning, a message to a function a call, and any other text
    reply = Reply()
    for message in re.split(r"<\|end\|>|<\|start\|>", text):
        header, marked, body = message.partition("<|message|>")
        if not marked:  # a header, so far
            continue
        if to := re.search(r"to=functions\.([^\s<]+)", header):
            reply.calls.append({"name": to.group(1), "arguments": body.strip()})
        elif re.search(r"<\|channel\|>analysis", header):
            reply.reasoning += body
        else:
            reply.content += body
    return reply


def _partial(text: str, markers: tuple[str, ...]) -> int:
    # the length of the longest end of `text` that a marker begins with, short of the marker
    return max((n for m in markers for n in range(1, len(m)) if text.endswith(m[:n])), default=0)


# Qwen and Gemma 4 mark their tool calls, which may follow other text
_MARKERS = ("<tool_call>", "<|tool_call>")
_CALLS = re.compile(
    r"<tool_call>(.*?)</tool_call>|<\|tool_call>call:([^{]+)(\{.*?\})<tool_call\|>", re.S
)
_QUOTE = '<|"|>'  # Gemma 4's string quotes


def parse_tool_calls(reply: str, tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The text of a reply before its calls to `tools`, and those calls, each {"name": ...,
    "arguments": {...}}; or the whole reply and no calls, if one is malformed or names no tool.

    Llama 3 calls a tool by replying with nothing but {"name": ..., "parameters": {...}}, Qwen
    with <tool_call>{"name": ..., "arguments": {...}}</tool_call> after any text, and Gemma 4
    with <|tool_call>call:name{key:value,...}<tool_call|>.
    """
    names = {tool.get("function", {}).get("name") for tool in tools}
    if not (blocks := list(_CALLS.finditer(reply))):
        call = _json_call(reply, names)
        return ("", [call]) if call else (reply, [])
    calls = [
        _json_call(b[1], names) if b[1] is not None else _gemma_call(b[2], b[3], names)
        for b in blocks
    ]
    if any(call is None for call in calls):
        return reply, []
    return reply[: blocks[0].start()].rstrip(), [call for call in calls if call]


def tool_call_start(text: str) -> int:
    """How much of a reply that begins with `text` surely is text before any tool call: not the
    whitespace before a call, which parse_tool_calls drops."""
    if text.lstrip()[:1] in ("", "{"):  # Llama 3's calls are all of a reply
        return 0
    if starts := [i for m in _MARKERS if (i := text.find(m)) >= 0]:
        return len(text[: min(starts)].rstrip())
    partial = (n for m in _MARKERS for n in range(1, len(m)) if text.endswith(m[:n]))
    return len(text[: len(text) - max(partial, default=0)].rstrip())


def _json_call(text: str, names: set[str]) -> dict[str, Any] | None:
    try:
        call = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or call.get("name") not in names:
        return None
    arguments = call.get("parameters", call.get("arguments"))
    return {"name": call["name"], "arguments": arguments} if isinstance(arguments, dict) else None


def _gemma_call(name: str, text: str, names: set[str]) -> dict[str, Any] | None:
    try:
        arguments, end = _gemma_value(text, 0)
    except (ValueError, IndexError):
        return None
    called = name in names and end == len(text) and isinstance(arguments, dict)
    return {"name": name, "arguments": arguments} if called else None


def _gemma_value(text: str, i: int) -> tuple[Any, int]:
    # the value at text[i:] in Gemma 4's call syntax, and where it ends: strings between quotes,
    # objects of bare or quoted keys, lists, and JSON's numbers, true, false and null
    if text.startswith(_QUOTE, i):
        end = text.index(_QUOTE, i + len(_QUOTE))
        return text[i + len(_QUOTE) : end], end + len(_QUOTE)
    if text[i] in "{[":
        close, items, keys, i = "}" if text[i] == "{" else "]", [], [], i + 1
        while text[i] != close:
            if close == "}":
                if text.startswith(_QUOTE, i):
                    key, i = _gemma_value(text, i)
                else:
                    key, i = text[i : text.index(":", i)], text.index(":", i)
                keys.append(key)
                i += 1  # the colon
            value, i = _gemma_value(text, i)
            items.append(value)
            i += text[i] == ","
        return (dict(zip(keys, items, strict=True)) if close == "}" else items), i + 1
    end = min((j for c in ",}]" if (j := text.find(c, i)) >= 0), default=len(text))
    return json.loads(text[i:end]), end


def _message(message: dict[str, Any]) -> dict[str, Any]:
    # an OpenAI message as templates read it: content as one string, and tool calls only where
    # there are some, with their arguments as objects rather than JSON text
    message = dict(message)
    if isinstance(parts := message.get("content"), list):
        if any(part.get("type") != "text" for part in parts):
            raise ValueError("only text content is supported")
        message["content"] = "\n".join(part["text"] for part in parts)
    if calls := message.pop("tool_calls", None):
        message["tool_calls"] = [_decoded(call) for call in calls]
    return message


def _decoded(call: dict[str, Any]) -> dict[str, Any]:
    # arguments that are already an object, or not JSON, reach the template as they are
    function = dict(call["function"])
    with contextlib.suppress(TypeError, ValueError):
        function["arguments"] = json.loads(function["arguments"])
    return {**call, "function": function}


def _tojson(value: Any, ensure_ascii=False, indent=None, separators=None, sort_keys=False) -> str:
    # transformers' version: unlike Jinja's own, it does not HTML-escape
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _raise(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)
