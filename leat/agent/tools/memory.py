"""Memory, as the proven agents keep it: facts about the user, each of a category, which every
conversation begun after knows, in a bounded room; remembered, replaced and forgotten by the model
as it talks, and by its review of each conversation once idle; and the earlier conversations,
recalled by their words or by their time."""

import datetime
import re
import time
from typing import TYPE_CHECKING, Any

from leat.agent.tools import Context, Result, Tool, schema

if TYPE_CHECKING:
    from leat.agent.agent import Agent

ROOM = 3000  # characters every memory takes at most, together, so that they stay few and good
LONGEST = 300  # characters of a memory
FOUND = 8  # messages recall finds at most
RECENT = 7  # days back recall lists the conversations of, by default
# the categories, as people read them: who the user is, how they like things, the people in their
# life, their work, studies and projects, and their plans and dates
CATEGORIES = {
    "about": "About them", "preferences": "Preferences", "people": "People", "work": "Work",
    "plans": "Plans",
}  # fmt: skip
# characters that show nothing, which an instruction smuggled into a memory hides in
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")


def tools(agent: "Agent") -> list[Tool]:
    """remember, forget and recall, of the agent's memories and conversations."""
    return [
        Tool(
            "remember",
            "Remember a fact about the user, in the conversations after this one; or change one",
            schema(
                memory=("string", "the fact, in a short sentence, as 'The user's daughter is 7.'"),
                category=("string", "what it is of", list(CATEGORIES)),
                replaces=("integer", "the number of the memory it changes, if it does"),
            ),
            lambda context, memory, category="about", replaces=None: remember(
                agent, memory, category, replaces
            ),
        ),
        Tool(
            "forget",
            "Forget one of the memories",
            schema(number=("integer", "the memory's number")),
            lambda context, number: forget(agent, number),
        ),
        Tool(
            "recall",
            "Find what was said in earlier conversations with the user, by words, or the latest "
            "conversations by time",
            schema(
                query=("string", "words to look for; '' for the latest conversations"),
                days=("integer", f"how many days back to look; for the latest, {RECENT}"),
            ),
            lambda context, query, days=None: recall(agent, context, query, days),
        ),
    ]


def remember(agent: "Agent", memory: str, category: str, replaces: int | str | None) -> Result:
    replaced = None if replaces in (None, "") else int(replaces)
    m = agent.remember(memory, category, replaced)
    said = f"Changed [{m['id']}]." if replaced is not None else f"Remembered, as [{m['id']}]."
    return Result(said, {"memory": m, "replaced": replaced is not None})


def forget(agent: "Agent", number: int | str) -> Result:
    m = agent.forget(int(number))
    return Result(f"Forgot [{m['id']}]: {m['text']}", {"memory": m})


def recall(agent: "Agent", context: Context, query: str, days: int | str | None) -> Result:
    # the messages that match, each with the one it answers or that answers it, by conversation;
    # or, of no words, the latest conversations, each by its first message
    store, back = agent.store, int(days) if days not in (None, "") else None
    since = time.time() - 86400 * back if back else 0
    if not re.search(r"\w", query):
        found = store.recent(since or time.time() - 86400 * RECENT, exclude=context.conversation)
        lines = [f"“{f['title']}”, {_day(f['updated'])}: the user began, {_clip(f['text'], 300)}"
                 for f in found]  # fmt: skip
        conversations = [{"id": f["conversation"], "title": f["title"]} for f in found]
        return Result("\n".join(lines) or "No conversation then.", {"conversations": conversations})
    found = store.search(query, exclude=context.conversation, since=since, limit=FOUND)
    passages: dict[str, list[tuple[int, dict[str, Any]]]] = {}  # by conversation, in order
    titles = {f["conversation"]: (f["title"], f["updated"]) for f in found}
    for f in found:
        messages = store.messages(f["conversation"])
        for j in _exchange(messages, f["position"]):
            passage = passages.setdefault(f["conversation"], [])
            if j not in [k for k, _ in passage]:
                passage.append((j, messages[j]))
    parts = []
    for id, passage in passages.items():
        title, updated = titles[id]
        said = "\n".join(f"{m['role']}: {_clip(m['content'], 600)}" for _, m in sorted(passage))
        parts.append(f"“{title}”, {_day(updated)}\n{said}")
    conversations = [{"id": id, "title": titles[id][0]} for id in passages]
    content = "\n\n".join(parts) or "No earlier conversation matches."
    return Result(content, {"conversations": conversations})


def _exchange(messages: list[dict[str, Any]], i: int) -> list[int]:
    # the positions of the user's message of the turn that message i is in, and of i, or of the
    # turn's answer, its last reply with text, if i is the user's
    asked = next(j for j in range(i, 0, -1) if messages[j]["role"] == "user")
    if i != asked:
        return [asked, i]
    end = next((j for j in range(i + 1, len(messages)) if messages[j]["role"] == "user"), None)
    replies = [j for j, m in enumerate(messages[i + 1 : end], i + 1)
               if m["role"] == "assistant" and m.get("content")]  # fmt: skip
    return [asked, *replies[-1:]]


def checked(text: str, category: str) -> str:
    """A memory's text, stripped, if it may be one. Raises ValueError, saying why not."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("there is nothing to remember")
    if len(text) > LONGEST:
        raise ValueError(f"a memory is a fact in {LONGEST} characters at most: say it shorter")
    if _INVISIBLE.search(text):
        raise ValueError("a memory may not hold characters that show nothing")
    if category not in CATEGORIES:
        raise ValueError(f"the category must be one of {', '.join(CATEGORIES)}")
    return text


def listing(memories: list[dict[str, Any]]) -> str:
    """The memories, each by its number, under their categories, with how full their room is."""
    used = sum(len(m["text"]) for m in memories)
    lines = [f"({len(memories)} memories, {100 * used // ROOM}% of their room)"]
    for category, name in CATEGORIES.items():
        if of := [m for m in memories if m["category"] == category]:
            lines += [f"{name}:", *(f"[{m['id']}] {m['text']}" for m in of)]
    return "\n".join(lines) if memories else "Nothing yet."


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _day(timestamp: float) -> str:
    day = datetime.date.fromtimestamp(timestamp)
    return f"{day.day} {day:%B %Y}"
