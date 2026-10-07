"""The tools the model calls. Each takes a few arguments, and answers with text for the model to
read and info for people to see of it, such as the pages it read."""

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Context:
    """What a call is made in: its conversation, and the numbering of the sources read in it."""

    conversation: str  # the id of the conversation
    # a source's number, of its address and title, the same each time it is read, for the model to
    # cite and the app to link; 0 where nothing numbers them
    cite: Callable[[str, str], int] = lambda url, title: 0


@dataclass(frozen=True)
class Result:
    content: str  # what the model reads
    info: dict[str, Any] = field(default_factory=dict)  # what people see of it


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # its arguments, as a JSON schema
    run: Callable[..., Result]  # of the call's Context and the arguments, by name

    def declaration(self) -> dict[str, Any]:
        """The tool as OpenAI's API declares it."""
        function = {"name": self.name, "description": self.description}
        return {"type": "function", "function": function | {"parameters": self.parameters}}


def arguments(raw: Any) -> dict[str, Any]:
    """A call's arguments as an object, of the JSON text leat serve gives them as. Raises
    ValueError if they are not one."""
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"the arguments must be a JSON object, not {raw!r}")
    return raw


def schema(**parameters: tuple) -> dict[str, Any]:
    """The schema of arguments, each given by name as its type, its description, and its values if
    they are few; the first is required, the rest not."""
    properties = {}
    for name, (kind, description, *values) in parameters.items():
        properties[name] = {"type": kind, "description": description}
        if values:
            properties[name]["enum"] = values[0]
    return {"type": "object", "properties": properties, "required": list(parameters)[:1]}


def strings(**arguments: str) -> dict[str, Any]:
    """The schema of arguments that are strings, each required, given by name and description."""
    properties = {name: {"type": "string", "description": d} for name, d in arguments.items()}
    return {"type": "object", "properties": properties, "required": list(arguments)}
