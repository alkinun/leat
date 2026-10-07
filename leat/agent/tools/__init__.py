"""The tools the model calls. Each takes a few arguments, and answers with text for the model to
read and info for people to see of it, such as the pages it read."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Context:
    """What a call is made in."""

    conversation: str  # the id of the conversation


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


def strings(**arguments: str) -> dict[str, Any]:
    """The schema of arguments that are strings, each required, given by name and description."""
    properties = {name: {"type": "string", "description": d} for name, d in arguments.items()}
    return {"type": "object", "properties": properties, "required": list(arguments)}
