"""Memory, as the proven agents keep it: facts about the user, each of a category, which every
conversation begun after knows, in a bounded room; remembered, replaced and forgotten by the model
as it talks, and by its review of each conversation once idle; and the earlier conversations,
recalled by their words or by their time.

What keeps it right, as ChatGPT's, Claude's and Hermes Agent's failures taught them:
- a memory rests on the user's own words, which the model quotes and the code finds in what they
  said, so that nothing it read, as a page or a file, becomes what the user is;
- each is dated when it was last made or said again, and a plan has its last day, so that the
  model sees what is old and what has passed;
- when the room is full, the model is told which memories were least recently confirmed, to change
  or forget;
- what is forgotten or changed is kept as it was, to restore;
- passwords and the numbers of IDs, cards and accounts are never kept.
"""

import datetime
import re
import time
from collections.abc import Iterable
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
# what a memory never holds: secrets, and the numbers of identity documents, cards and accounts
_SECRET = re.compile(
    r"\b(passwords?|passcodes?|pins?|pin codes?|şifre\w*|parola\w*|ssn|social security|iban|"
    r"passport numbers?|tc kimlik|tckn|credit cards?|card numbers?|cvv|cvc)\b",
    re.IGNORECASE,
)
# characters that show nothing, which an instruction smuggled into a memory hides in
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")


def tools(agent: "Agent") -> list[Tool]:
    """remember, forget and recall, of the agent's memories and conversations."""
    return [
        Tool(
            "remember",
            "Remember a fact about the user, in the conversations after this one; or change one",
            schema(
                required=2,
                memory=("string", "the fact, in a short sentence, as 'The user's daughter is 7.'"),
                evidence=("string", "the user's own words it rests on, quoted exactly"),
                category=("string", "what it is of", list(CATEGORIES)),
                replaces=("integer", "the number of the memory it changes, if it does"),
                until=("string", "of a plan, its last day, as 'YYYY-MM-DD'"),
            ),
            lambda context, memory, evidence, category="about", replaces=None, until=None: remember(
                agent, context, memory, evidence, category, replaces, until
            ),
        ),
        Tool(
            "forget",
            "Forget one of the memories",
            schema(number=("integer", "the memory's number")),
            lambda context, number: forget(agent, context, number),
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


def remember(
    agent: "Agent", context: Context, memory: str, evidence: str, category: str,
    replaces: int | str | None, until: str | None,
) -> Result:  # fmt: skip
    replaced = None if replaces in (None, "") else int(replaces)
    was = [m["text"] for m in agent.memories() if m["id"] == replaced]  # its evidence, once
    if not said(evidence, agent.store.messages(context.conversation), was):
        raise ValueError(
            "the evidence must be the user's own words, quoted exactly from their messages: "
            "remember only what they told you of themselves, not what you read"
        )
    m = agent.remember(memory, category, replaced, until or None, context.by)
    done = f"Changed [{m['id']}]." if replaced is not None else f"Remembered, as [{m['id']}]."
    return Result(done, {"memory": m, "replaced": replaced is not None})


def forget(agent: "Agent", context: Context, number: int | str) -> Result:
    m = agent.forget(int(number), context.by)
    return Result(f"Forgot [{m['id']}]: {m['text']}", {"memory": m})


def said(evidence: str, messages: list[dict[str, Any]], also: Iterable[str] = ()) -> bool:
    """Whether the user said these words, whatever their case and punctuation, in their messages,
    not a scheduled task's, or in the texts `also`, as a memory that was evidenced once."""
    told = [
        m["content"] for m in messages if m["role"] == "user" and "task" not in m.get("info", {})
    ]
    words = " ".join(re.findall(r"\w+", evidence.lower()))
    return bool(words) and any(
        words in " ".join(re.findall(r"\w+", text.lower())) for text in [*told, *also]
    )


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


def checked(text: str, category: str, until: str | None = None) -> tuple[str, str | None]:
    """A memory's text, stripped, and its last day, as YYYY-MM-DD, if they may be one's. Raises
    ValueError, saying why not."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("there is nothing to remember")
    if len(text) > LONGEST:
        raise ValueError(f"a memory is a fact in {LONGEST} characters at most: say it shorter")
    if _INVISIBLE.search(text):
        raise ValueError("a memory may not hold characters that show nothing")
    if _SECRET.search(text) or _card(text):
        raise ValueError("memories never hold passwords, PINs, or the numbers of IDs, cards or "
                         "accounts")  # fmt: skip
    if category not in CATEGORIES:
        raise ValueError(f"the category must be one of {', '.join(CATEGORIES)}")
    if until:
        try:
            until = datetime.date.fromisoformat(until.strip()).isoformat()
        except ValueError:
            raise ValueError(f"until must be a day, as 'YYYY-MM-DD', not {until!r}") from None
    return text, until or None


def listing(memories: list[dict[str, Any]], today: datetime.date | None = None) -> str:
    """The memories, each by its number, under their categories, with how full their room is, and
    each's date: when it was last made or said again, or of a plan the last day it holds, or that
    it has passed."""
    today = today or datetime.date.today()
    used = sum(len(m["text"]) for m in memories)
    lines = [f"({len(memories)} memories, {100 * used // ROOM}% of their room)"]
    for category, name in CATEGORIES.items():
        if of := [m for m in memories if m["category"] == category]:
            lines += [f"{name}:", *(f"[{m['id']}] {m['text']} ({_dated(m, today)})" for m in of)]
    return "\n".join(lines) if memories else "Nothing yet."


def _dated(m: dict[str, Any], today: datetime.date) -> str:
    # a memory's date as the model reads it: "8 Oct 2026", "until 17 Oct 2026", "passed 2 Oct 2026"
    if until := m.get("until"):
        day = datetime.date.fromisoformat(until)
        return f"{'until' if day >= today else 'passed'} {day.day} {day:%b %Y}"
    day = datetime.date.fromtimestamp(m["confirmed"] or m["created"])
    return f"{day.day} {day:%b %Y}"


def _card(text: str) -> bool:
    # whether a text holds a card's number: 13 to 19 digits, whose Luhn checksum holds
    for match in re.finditer(r"\d(?:[ -]?\d){12,18}", text):
        digits = [int(d) for d in re.sub(r"\D", "", match[0])][::-1]
        total = sum(
            d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(digits)
        )
        if total % 10 == 0:
            return True
    return False


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _day(timestamp: float) -> str:
    day = datetime.date.fromtimestamp(timestamp)
    return f"{day.day} {day:%B %Y}"
