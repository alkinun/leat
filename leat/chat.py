"""Chat prompts, rendered with the model's own Jinja template from GGUF metadata."""

import json
from datetime import datetime
from typing import Any

import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from leat.tokenizer import Tokenizer


class ChatTemplate:
    """Turns OpenAI-style messages into prompt tokens.

    Rendering matches transformers' `apply_chat_template` once text parts are joined into one
    string, as templates expect. As there and in llama.cpp, control-token text inside message
    content is parsed as control tokens.
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


def _message(message: dict[str, Any]) -> dict[str, Any]:
    # an OpenAI message as templates read it: content as one string
    message = dict(message)
    if isinstance(parts := message.get("content"), list):
        if any(part.get("type") != "text" for part in parts):
            raise ValueError("only text content is supported")
        message["content"] = "\n".join(part["text"] for part in parts)
    return message


def _tojson(value: Any, ensure_ascii=False, indent=None, separators=None, sort_keys=False) -> str:
    # transformers' version: unlike Jinja's own, it does not HTML-escape
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _raise(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)
