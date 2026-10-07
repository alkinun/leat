"""Memory: remembering what the user tells of themselves, forgetting it, and recalling earlier
conversations."""

import datetime
from typing import TYPE_CHECKING

from leat.agent.tools import Context, Result, Tool, strings

if TYPE_CHECKING:
    from leat.agent.agent import Agent

FOUND = 8  # passages of earlier conversations recall finds at most


def tools(agent: "Agent") -> list[Tool]:
    """remember, forget and recall, of the agent's memories and conversations."""
    number = {"type": "integer", "description": "the memory's number"}
    return [
        Tool(
            "remember",
            "Remember one fact about the user, in the conversations after this one",
            strings(memory="the fact, in a short sentence, as 'The user's daughter is 7.'"),
            lambda context, memory: remember(agent, memory),
        ),
        Tool(
            "forget",
            "Forget one of the memories",
            {"type": "object", "properties": {"number": number}, "required": ["number"]},
            lambda context, number: forget(agent, number),
        ),
        Tool(
            "recall",
            "Search the earlier conversations with the user: the passages that match",
            strings(query="words to look for"),
            lambda context, query: recall(agent, context, query),
        ),
    ]


def remember(agent: "Agent", memory: str) -> Result:
    if not memory.strip():
        raise ValueError("there is nothing to remember")
    m = agent.remember(memory.strip())
    return Result(f"Remembered, as [{m['id']}].", {"memory": m})


def forget(agent: "Agent", number: int | str) -> Result:
    m = agent.forget(int(number))
    return Result(f"Forgot [{m['id']}]: {m['text']}", {"memory": m})


def recall(agent: "Agent", context: Context, query: str) -> Result:
    found = agent.store.search(query, exclude=context.conversation, limit=FOUND)
    conversations: dict[str, dict] = {}  # each with the passages found in it, the best first
    for f in found:
        c = conversations.setdefault(f["conversation"], f | {"passages": []})
        c["passages"].append(f"{f['role']}: {f['text']}")
    content = "\n\n".join(
        f"“{c['title']}”, {_day(c['updated'])}\n" + "\n".join(c["passages"])
        for c in conversations.values()
    )
    info = {"conversations": [{"id": id, "title": c["title"]} for id, c in conversations.items()]}
    return Result(content or "No earlier conversation matches.", info)


def _day(timestamp: float) -> str:
    day = datetime.date.fromtimestamp(timestamp)
    return f"{day.day} {day:%B %Y}"
