"""Chat prompts, rendered with the model's own Jinja template from GGUF metadata, and replies
split into their reasoning, text and tool calls, in each supported model's syntax."""

import base64
import binascii
import contextlib
import json
import re
from collections.abc import Container, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from leat.tokenizer import Tokenizer

# where an image of a message's content stands in the rendered text, which tokens() replaces with
# the image's own tokens: the object replacement character, which text parts never hold
IMAGE = "\ufffc"


class ChatTemplate:
    """Turns OpenAI-style messages into prompt tokens.

    Rendering matches transformers' `apply_chat_template` once text parts are joined and the JSON
    arguments of tool calls decoded, as templates expect. As there and in llama.cpp, control-token
    text inside message content is parsed as control tokens. Images, `image_url` parts as images()
    reads them, stand as IMAGE in the text, as llama.cpp's markers do, whatever the template does
    with images: tokens() puts each image's own there.
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
        # how replies mark their reasoning: gpt-oss's harmony channels, Gemma 4's thought channel,
        # or the <think> blocks of Qwen3 and DeepSeek-R1's distillations, whose templates write them
        self._form = (
            "harmony" if "<|channel|>" in tokens
            else "gemma4" if "<|channel>" in tokens
            else "think" if "<think>" in source
            else None
        )  # fmt: skip

    @property
    def form(self) -> str | None:
        """How replies mark their reasoning: "harmony", "gemma4", "think", or None."""
        return self._form

    def render(
        self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **kwargs
    ) -> str:
        return self._template.render(
            messages=[_message(_plain(m)) for m in messages],
            add_generation_prompt=add_generation_prompt,
            bos_token=self._bos,
            eos_token=self._eos,
            **_plain(kwargs),
        )

    def encode(
        self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **kwargs
    ) -> list[int]:
        return self.tokens(self.render(messages, add_generation_prompt, **kwargs))

    def tokens(self, text: str, images: Sequence[list[int]] = ()) -> list[int]:
        """A rendered prompt's tokens, each IMAGE in it the tokens that show an image, in turn."""
        # most templates write the BOS text themselves; add it only when they don't
        bos = self._tokenizer.add_bos and not (self._bos and text.startswith(self._bos))
        first, *rest = text.split(IMAGE)
        if len(rest) != len(images):
            raise ValueError(f"the prompt shows {len(rest)} images, {len(images)} are given")
        ids = self._tokenizer.encode(first, bos=bos, special=True)
        for image, piece in zip(images, rest, strict=True):
            ids += image + self._tokenizer.encode(piece, bos=False, special=True)
        return ids

    def opens_thinking(self, text: str) -> bool:
        """Whether a rendered prompt ends inside a block of reasoning, which the reply then
        continues, as DeepSeek-R1's distillations' templates open their <think>."""
        return self.form in _THINKING and text.rstrip().endswith(_THINKING[self.form][0])


@dataclass
class Reply:
    """A reply split: its reasoning, its text, and the tool calls it makes in the harmony format,
    gpt-oss's, each {"name": ..., "arguments": JSON text}. Other formats' calls are in the text,
    for parse_tool_calls."""

    reasoning: str = ""
    content: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)


# where a form's block of reasoning opens and closes, before the text: <think> blocks, and Gemma
# 4's thought channel
_THINKING = {"think": ("<think>", "</think>"), "gemma4": ("<|channel>thought", "<channel|>")}
# the harmony format's markers, and those blocks'
_REPLY_MARKERS = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|constrain|>",
                  *(marker for markers in _THINKING.values() for marker in markers))  # fmt: skip


def split_reply(text: str, form: str | None, thinking: bool = False, done: bool = False) -> Reply:
    """A reply, or as much of it as there is so far: what is surely reasoning, surely text, and
    the calls. Until the reply is `done`, an end that may be the start of a marker waits for
    more, so that what a stream has split stays so. `thinking` says the prompt opened a block of
    reasoning."""
    if not done:
        text = text[: len(text) - _partial(text, _REPLY_MARKERS)]
    if form == "harmony":
        return _harmony(text)
    if form in _THINKING:
        opens, closes = _THINKING[form]
        start = text.find(opens)
        if start >= 0 and not text[:start].strip():
            text, thinking = text[start + len(opens) :], True
        elif not (thinking or done or text.strip()):  # what may yet come before the block
            return Reply()
        if thinking:
            reasoning, _, content = text.partition(closes)
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
# Qwen3.5's calls inside <tool_call>: a function, and a parameter on each value's lines
_FUNCTION = re.compile(r"\s*<function=([^>\s]+)>\n?(.*?)</function>\s*", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>\s*", re.S)


def parse_tool_calls(reply: str, tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The text of a reply before its calls to `tools`, and those calls, each {"name": ...,
    "arguments": {...}}; or the whole reply and no calls, if one is malformed or names no tool.

    Llama 3 calls a tool by replying with nothing but {"name": ..., "parameters": {...}}, Qwen
    with <tool_call>{"name": ..., "arguments": {...}}</tool_call> after any text, Qwen3.5 with
    <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>, values
    as text where the tool takes a string and else as JSON, and Gemma 4 with
    <|tool_call>call:name{key:value,...}<tool_call|>.
    """
    functions = {f.get("name"): f for tool in tools if (f := tool.get("function"))}
    if not (blocks := list(_CALLS.finditer(reply))):
        call = _json_call(reply, functions)
        return ("", [call]) if call else (reply, [])
    calls = [
        (_json_call(b[1], functions) or _xml_call(b[1], functions))
        if b[1] is not None
        else _gemma_call(b[2], b[3], functions)
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


def _json_call(text: str, names: Container[str]) -> dict[str, Any] | None:
    try:
        call = json.loads(text)
    except (ValueError, RecursionError):  # not JSON, or nested too deep for Python's parser
        return None
    name = call.get("name") if isinstance(call, dict) else None
    if not isinstance(name, str) or name not in names:
        return None
    arguments = call.get("parameters", call.get("arguments"))
    return {"name": call["name"], "arguments": arguments} if isinstance(arguments, dict) else None


def _xml_call(text: str, functions: dict[str, Any]) -> dict[str, Any] | None:
    if not (match := _FUNCTION.fullmatch(text)) or match[1] not in functions:
        return None
    if _PARAMETER.sub("", match[2]).strip():  # anything but parameters
        return None
    arguments: dict[str, Any] = {}
    for key, value in _PARAMETER.findall(match[2]):
        kind = _schema(functions[match[1]], "parameters", "properties", key, "type")
        types = {kind} if isinstance(kind, str) else set(kind) if isinstance(kind, list) else set()
        arguments[key] = _typed(value, types)
    return {"name": match[1], "arguments": arguments}


def _typed(text: str, types: set[str]) -> Any:
    # a parameter's value as llama.cpp's Qwen3.5 parser reads it: as text where its schema takes
    # strings alone, else as JSON, of one of its other types where it takes strings too; as text
    # where that fails
    if types == {"string"}:
        return text
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return text
    return text if "string" in types and not types & _JSON_TYPES[type(value)] else value


# the JSON schema types a decoded value is of
_JSON_TYPES = {type(None): {"null"}, bool: {"boolean"}, int: {"integer", "number"},
               float: {"number"}, str: {"string"}, list: {"array"}, dict: {"object"}}  # fmt: skip


def _schema(value: Any, *keys: str) -> Any:
    # value[key][key]..., or None where one is missing: a tool's JSON schema, as a client wrote it
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _gemma_call(name: str, text: str, names: Container[str]) -> dict[str, Any] | None:
    try:
        arguments, end = _gemma_value(text, 0)
    except (ValueError, IndexError, RecursionError):
        return None
    called = name in names and end == len(text) and isinstance(arguments, dict)
    return {"name": name, "arguments": arguments} if called else None


def _gemma_value(text: str, i: int) -> tuple[Any, int]:
    # the value at text[i:] in Gemma 4's call syntax, and where it ends: strings between quotes,
    # objects of bare or quoted keys, lists, and JSON's numbers, true, false and null, with
    # whitespace about their parts, as llama.cpp's grammar takes
    i = _blank(text, i)
    if text.startswith(_QUOTE, i):
        end = text.index(_QUOTE, i + len(_QUOTE))
        return text[i + len(_QUOTE) : end], end + len(_QUOTE)
    if text[i] in "{[":
        close, items, keys, i = "}" if text[i] == "{" else "]", [], [], _blank(text, i + 1)
        while text[i] != close:
            if close == "}":
                if text.startswith(_QUOTE, i):
                    key, i = _gemma_value(text, i)
                else:
                    key, i = text[i : text.index(":", i)].rstrip(), text.index(":", i)
                if text[i := _blank(text, i)] != ":":
                    raise ValueError(f"no colon after key {key!r}")
                keys.append(key)
                i += 1
            value, i = _gemma_value(text, i)
            items.append(value)
            i = _blank(text, i)
            i = _blank(text, i + 1) if text[i] == "," else i
        return (dict(zip(keys, items, strict=True)) if close == "}" else items), i + 1
    end = min((j for c in ",}]" if (j := text.find(c, i)) >= 0), default=len(text))
    return json.loads(text[i:end]), end


def _blank(text: str, i: int) -> int:
    # past any whitespace at text[i:]
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def images(messages: list[dict[str, Any]]) -> list[bytes]:
    """The images of messages' content in turn, `image_url` parts of data: URLs, as bytes: a
    ValueError for one of another URL, which leat does not fetch."""
    out = []
    for message in messages:
        parts = message.get("content")
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part["image_url"].get("url") if isinstance(part.get("image_url"), dict) else None
            if not isinstance(url, str) or not re.match(r"data:[^,]*;base64,", url):
                raise ValueError("an image_url's url must be a data: URL of base64")
            try:
                out.append(base64.b64decode(url.partition(",")[2], validate=True))
            except binascii.Error as e:
                raise ValueError(f"an image_url's data is not base64: {e}") from None
    return out


def _message(message: dict[str, Any]) -> dict[str, Any]:
    # an OpenAI message as templates read it: content as one string, its text parts joined by
    # newlines and its images, `image_url` parts or transformers' `image` ones, IMAGE; and tool
    # calls only where there are some, with their arguments as objects rather than JSON text
    message = dict(message)
    if isinstance(content := message.get("content"), list):
        kinds = ("text", "image", "image_url")
        if any(not isinstance(part, dict) or part.get("type") not in kinds for part in content):
            raise ValueError("content parts must be text, image or image_url")
        if any(p["type"] == "text" and not isinstance(p.get("text"), str) for p in content):
            raise ValueError("a text part's text must be a string")
        pieces = [p["text"] if p["type"] == "text" else IMAGE for p in content]
        message["content"] = "".join(
            p if i == 0 or IMAGE in (p, pieces[i - 1]) else "\n" + p for i, p in enumerate(pieces)
        )
    if calls := message.pop("tool_calls", None):
        if not isinstance(calls, list) or not all(
            isinstance(call, dict) and isinstance(call.get("function"), dict) for call in calls
        ):
            raise ValueError("tool_calls must be a list of objects, each with its function")
        message["tool_calls"] = [_decoded(call) for call in calls]
    return message


def _plain(value: Any) -> Any:
    # a value as given, with no IMAGE in its strings: there, only images stand for it
    if isinstance(value, str):
        return value.replace(IMAGE, "")
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    return value


def _decoded(call: dict[str, Any]) -> dict[str, Any]:
    # arguments that are already an object, or not JSON, reach the template as they are
    function = dict(call["function"])
    with contextlib.suppress(TypeError, ValueError, RecursionError):
        function["arguments"] = _plain(json.loads(function["arguments"]))  # "\ufffc" in JSON
    return {**call, "function": function}


def _tojson(value: Any, ensure_ascii=False, indent=None, separators=None, sort_keys=False) -> str:
    # transformers' version: unlike Jinja's own, it does not HTML-escape
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _raise(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)
