"""Chat prompts, rendered with the model's own Jinja template from GGUF metadata, and the tool
calls in replies."""

import contextlib
import json
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
        text = self.render(messages, add_generation_prompt, **kwargs)
        # most templates write the BOS text themselves; add it only when they don't
        bos = self._tokenizer.add_bos and not (self._bos and text.startswith(self._bos))
        return self._tokenizer.encode(text, bos=bos, special=True)


def parse_tool_call(reply: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The call a reply makes to one of `tools`, as {"name": ..., "arguments": {...}}, or None.

    Llama 3 calls a tool by replying with nothing but {"name": ..., "parameters": {...}}.
    """
    try:
        call = json.loads(reply)
    except json.JSONDecodeError:
        return None
    names = {tool.get("function", {}).get("name") for tool in tools}
    if not isinstance(call, dict) or call.get("name") not in names:
        return None
    arguments = call.get("parameters", call.get("arguments"))
    return {"name": call["name"], "arguments": arguments} if isinstance(arguments, dict) else None


def may_call_tool(start: str) -> bool:
    """Whether a reply that begins with `start` may yet turn out to call a tool."""
    return start.lstrip()[:1] in ("", "{")


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
