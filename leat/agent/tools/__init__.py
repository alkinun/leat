"""The tools the model calls. Each takes a few arguments, and answers with text for the model to
read and info for people to see of it, such as the pages it read.

A tool carries its rules, what the system prompt tells the model of when and how to call it, in a
section of the prompt's, the web's or the files', as a bundle of a feature's tools and instructions
does: the prompt is made of the tools offered, and a tool taken away takes its rules with it."""

import contextlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from leat.agent.workspace import Workspace

if TYPE_CHECKING:
    from leat.agent.client import Client
    from leat.agent.index import Index


@dataclass(frozen=True)
class Context:
    """What a call is made in: its conversation, the numbering of the sources read in it, whose
    the conversation is, the files it works among, how a long call tells how far it is, and
    whether the turn is stopped, at which a long call ends."""

    conversation: str  # the id of the conversation
    # a source's number, of its address and title, the same each time it is read, for the model to
    # cite and the app to link; 0 where nothing numbers them
    cite: Callable[[str, str], int] = lambda url, title: 0
    person: int | None = None  # the id of the person whose conversation it is, if anyone's
    workspace: Workspace | None = None  # the conversation's space: its project's, or its person's
    # tells the apps how far the call is, as info its message shows while it runs
    progress: Callable[[dict[str, Any]], None] = lambda info: None
    stopped: threading.Event = field(default_factory=threading.Event)  # set once the turn stops

    def space(self) -> Workspace:
        """The conversation's workspace. Raises ValueError if it has none."""
        if self.workspace is None:
            raise ValueError("there is no workspace")
        return self.workspace


@dataclass(frozen=True)
class Result:
    content: str  # what the model reads
    info: dict[str, Any] = field(default_factory=dict)  # what people see of it


WEB, FILES = "web", "files"  # the sections of the system prompt a tool's rules are in


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # its arguments, as a JSON schema
    run: Callable[..., Result]  # of the call's Context and the arguments, by name
    rules: tuple[str, ...] = ()  # the system prompt's rules of it, each a line of its section
    section: str = FILES  # WEB, of a tool that reaches the internet, which offline takes away

    def declaration(self) -> dict[str, Any]:
        """The tool as OpenAI's API declares it."""
        function = {"name": self.name, "description": self.description}
        return {"type": "function", "function": function | {"parameters": self.parameters}}


def numbered(messages: list[dict[str, Any]]) -> dict[str, int]:
    """The sources the calls of a conversation's messages numbered, by their addresses."""
    found: dict[str, int] = {}
    for m in messages:
        info = m.get("info", {}) if m["role"] == "tool" else {}
        for source in [info, *info.get("results", [])]:
            if source.get("n") and source.get("url"):
                found.setdefault(source["url"], source["n"])
    return found


def arguments(raw: Any) -> dict[str, Any]:
    """A call's arguments as an object, of the JSON text leat serve gives them as. Raises
    ValueError if they are not one."""
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"the arguments must be a JSON object, not {raw!r}")
    return raw


def schema(required: int = 1, **parameters: tuple) -> dict[str, Any]:
    """The schema of arguments, each given by name as its type, its description, and its values if
    they are few; the first `required` are required, the rest not."""
    properties = {}
    for name, (kind, description, *values) in parameters.items():
        properties[name] = {"type": kind, "description": description}
        if values:
            properties[name]["enum"] = values[0]
    return {"type": "object", "properties": properties, "required": list(parameters)[:required]}


def strings(**arguments: str) -> dict[str, Any]:
    """The schema of arguments that are strings, each required, given by name and description."""
    properties = {name: {"type": "string", "description": d} for name, d in arguments.items()}
    return {"type": "object", "properties": properties, "required": list(arguments)}


def every(searxng: str, engine: "Client", index: "Index | None" = None) -> list[Tool]:
    """The tools leat agent offers, in the order the model is told of them, those made for a task
    before the general ones, files.GENERAL: the web's, through the SearXNG at `searxng`, the
    files', ask_files and the office's, whose readers and pages the model `engine` serves reads,
    and which find the files' passages by `index`."""
    from leat.agent.tools import ask, files, office, web

    tools = [
        *web.tools(searxng, engine), *files.tools(index), *ask.tools(engine, index),
        *office.tools(),
    ]  # fmt: skip
    return sorted(tools, key=lambda tool: tool.name in files.GENERAL)
