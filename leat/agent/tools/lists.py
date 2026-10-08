"""The household's lists, as tools: its shopping, its chores, what to pack; everyone's, which the
model adds to and checks off as anyone asks, from the app or a chat, and the app shows, to check
off at the shop."""

import re
import threading
from typing import TYPE_CHECKING, Any

from leat.agent.tools import Result, Tool

if TYPE_CHECKING:
    from leat.agent.agent import Agent

MOST = 20  # lists at most
ITEMS = 60  # items a list holds at most
LONGEST = 120  # characters of a list's name, or an item
_lock = threading.Lock()  # a list's items read and changed at once


def tools(agent: "Agent") -> list[Tool]:
    """add_to_list, check_off and lists, of the household's lists."""
    named = {"type": "string", "description": "the list's name, as 'shopping'"}
    things = {
        "type": "array",
        "items": {"type": "string"},
        "description": "each a thing, as 'milk'",
    }
    both = {"type": "object", "properties": {"list": named, "items": things},
            "required": ["list", "items"]}  # fmt: skip
    return [
        Tool(
            "add_to_list",
            "Add things to one of the household's lists, as its shopping list, which everyone at "
            "home shares; a list not there yet is made",
            both,
            lambda context, list, items: add(agent, list, items),
        ),
        Tool(
            "check_off",
            "Take things off one of the household's lists, as bought or done",
            both,
            lambda context, list, items: check_off(agent, list, items),
        ),
        Tool(
            "lists",
            "Show one of the household's lists, or all of them",
            {
                "type": "object",
                "properties": {"list": named | {"description": "its name; none for all"}},
            },  # fmt: skip
            lambda context, list=None: show(agent, list),
        ),
    ]


def add(agent: "Agent", name: str, items: list[str] | str) -> Result:
    things = _things(items)
    with _lock:
        found = find(agent, name) or make(agent, name)
        have, new = {i["text"].lower() for i in found["items"]}, []
        for thing in things:  # each once, whatever its case
            if thing.lower() not in have:
                have.add(thing.lower())
                new.append(thing)
        if len(found["items"]) + len(new) > ITEMS:
            raise ValueError(f"a list holds {ITEMS} things at most: check some off first")
        for thing in new:
            agent.store.add_item(found["id"], thing)
    changed(agent)
    count = len(found["items"]) + len(new)
    said = f"Added to {found['name']}: {', '.join(new)}." if new else "All of it is there already."
    return Result(f"{said} It has {count} things.", {"list": found["name"], "added": new})


def check_off(agent: "Agent", name: str, items: list[str] | str) -> Result:
    with _lock:
        if (found := find(agent, name)) is None:
            raise LookupError(f"there is no list {name!r}: {_names(agent)}")
        done, missing = [], []
        for thing in _things(items):
            if (item := _match(found["items"], thing)) is None:
                missing.append(thing)
            else:
                agent.store.remove_item(item["id"])
                found["items"].remove(item)
                done.append(item["text"])
    changed(agent)
    said = f"Checked off {', '.join(done)}." if done else "Nothing was checked off."
    if missing:
        said += f" Not on {found['name']}: {', '.join(missing)}."
    return Result(said, {"list": found["name"], "done": done})


def show(agent: "Agent", name: str | None = None) -> Result:
    if name:
        if (found := find(agent, name)) is None:
            raise LookupError(f"there is no list {name!r}: {_names(agent)}")
        items = "\n".join(f"- {i['text']}" for i in found["items"]) or "It is empty."
        return Result(f"{found['name']}:\n{items}", {"list": found["name"]})
    return Result(_names(agent), {})


def find(agent: "Agent", name: str) -> dict[str, Any] | None:
    """A list by its name, whatever its case and spaces, and "list" after, as "shopping list"."""
    key = _key(name)
    return next((li for li in agent.store.lists() if _key(li["name"]) == key), None)


def make(agent: "Agent", name: str) -> dict[str, Any]:
    """A new list, of a name. Raises ValueError if there are too many, or the name is none."""
    if not (name := _checked(re.sub(r"\s+list$", "", name.strip(), flags=re.I))):
        raise ValueError("a list needs a name")
    if len(agent.store.lists()) >= MOST:
        raise ValueError(f"the household has {MOST} lists, the most: remove one")
    return agent.store.add_list(name[:1].upper() + name[1:])


def changed(agent: "Agent") -> None:
    """Tells every app of the lists."""
    agent.events.publish(event(agent))


def event(agent: "Agent") -> dict[str, Any]:
    return {"type": "lists", "lists": agent.store.lists()}


def _things(items: list[str] | str) -> list[str]:
    # the things to add or check off, of a list of them, or of one text of them, said with commas
    things = items if isinstance(items, list) else re.split(r"[,\n]", str(items))
    return [checked for t in things if (checked := _checked(str(t)))]


def _checked(text: str) -> str:
    return " ".join(text.split())[:LONGEST]


def _key(name: str) -> str:
    return re.sub(r"\s+list$", "", " ".join(name.lower().split()))


def _match(items: list[dict[str, Any]], thing: str) -> dict[str, Any] | None:
    # the item a thing names: the one of its text, whatever its case, or the one alone that holds
    # it, as "milk" for "2 litres of milk"
    thing = thing.lower()
    if exact := [i for i in items if i["text"].lower() == thing]:
        return exact[0]
    holding = [i for i in items if thing in i["text"].lower()]
    return holding[0] if len(holding) == 1 else None


def _names(agent: "Agent") -> str:
    lists = agent.store.lists()
    names = ", ".join(f"{li['name']} ({len(li['items'])})" for li in lists)
    return f"The lists: {names}." if lists else "There are no lists yet."
