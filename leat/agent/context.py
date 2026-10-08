"""The prompt a conversation sends the model, fit to its context, as Hermes Agent fits it.

A conversation's messages are kept whole; what the model reads of them is derived of the
conversation's state, which changes only when the prompt outgrows a share of the context:
- first, the tools' answers before the recent tail, if long, are cleared, a line said in place of
  each, as a search's pages fill a context fastest;
- then, if it is still too long, the messages before the tail are summarized, with the summary
  before, into the system prompt, of which the model goes on.

Each change rewrites the prompt's start, which the engine's cache then holds again: so it is rare,
and between changes every prompt extends the last. The tokens a prompt takes are those the engine
counted for the last reply, and an estimate of the messages since.
"""

import datetime
import json
from collections.abc import Callable
from typing import Any

COMPACT = 0.6  # of the context, past which a prompt is made smaller
TAIL = 0.2  # of the context, of the latest messages kept whole, as Hermes Agent keeps its
KEEP = 200  # characters of a tool's answer kept whole, however old
CHARS = 3  # characters to a token, as an estimate that errs long
IMAGE = 300  # tokens of an image the model sees, where the engine says no more: Gemma 4's take 280
CLEARED = "[This old answer was cleared to make room; call the tool again if you need it.]"
# of an answer saved in the workspace, as a page read
SAVED = "[This old answer was cleared to make room; read {saved} if you need it again.]"
SUMMARY = """

Earlier parts of this conversation were summarized to make room; go on from the summary:

{summary}"""
# a user's message where the summary leaves the prompt to the model's, as templates that want the
# roles to alternate need one
GOING_ON = "(The conversation goes on from the summary.)"
# the system prompt of a summary, and the tokens it may take
SUMMARIZE = """\
You summarize a conversation between a user and their assistant, for the assistant to go on from \
it without it. Keep what it will need, and leave out what it will not:
- Goal: what the user wants, and what they asked last.
- The user: what they told of themselves, and how they like things done.
- Done and found: what the assistant did and found, with the facts, numbers, names and web \
addresses that matter.
- Files: the files made or read, by name.
- Next: what is left to do.
Write these sections, short, in the conversation's language, and nothing else."""
SUMMARY_TOKENS = 1500
# the note that asks for a summary at the end of the conversation, as the engine holds it
IN_PLACE = """\
(Write a summary of this conversation so far, for yourself to go on from it without it, keeping \
what the summary above, if there is one, says that still matters. In these sections, short:
- Goal: what the user wants, and what they asked last.
- The user: what they told of themselves, and how they like things done.
- Done and found: what you did and found, with the facts, numbers, names and web addresses that \
matter.
- Files: the files made or read, by name.
- Next: what is left to do.
Write the summary alone, in the conversation's language, and call no tools.)"""

State = dict[str, Any]  # "cleared": tools' answers before this message's index cleared;
# "summary", "summarized": the summary of the messages before that index; "used", "at": the tokens
# the prompt and reply took, by the engine's count, of the messages before that index


def message(m: dict[str, Any]) -> dict[str, Any]:
    """A message as the model reads it: without what only people see, nor an empty reasoning, but
    with the files the user attached named, a scheduled task's said to be one, and a user's
    begun with when it was sent, which is how the model knows the time as it answers; and with
    `images`, the names of the workspace's images it shows, those the user attached or a tool
    read, which the agent shows the model if it sees images. The reasoning is sent back, which
    templates such as Qwen3.5's show the steps of an agent's turn."""
    api = {k: v for k, v in m.items() if k != "info" and (v or k != "reasoning_content")}
    info = m.get("info", {})
    if images := [*info.get("images", []), *(f for f in info.get("files", []) if picture(f))]:
        api["images"] = images
    if (task := info.get("task")) and (quiet := info.get("quiet")) and info.get("trial"):
        check = api["content"].rstrip(" .")
        api["content"] = (
            f"(Your scheduled check [{task}] is run now, for the user to try it: {check}. Tell "
            f"them what you find, and whether {quiet}.)"
        )
    elif task and quiet:
        check = api["content"].rstrip(" .")
        api["content"] = (
            f"(Your scheduled check [{task}] is due now: {check}. Tell the user only if {quiet}; "
            "if not, reply NOTHING alone.)"
        )
    elif task:
        api["content"] = f"(Your scheduled task [{task}] is due now: {api['content']})"
    if attached := info.get("files"):
        api["content"] += f"\n\n(Attached, in the workspace: {', '.join(attached)})"
    if m["role"] == "user" and (at := info.get("at")):
        api["content"] = f"{stamp(at)}\n{api['content']}"
    return api


def picture(name: str) -> bool:
    """Whether a file is an image, of a kind the engine reads, by its name."""
    return name.lower().endswith(
        (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")
    )


def stamp(at: float) -> str:
    """A time as a user's message begins with it: "[Thursday 8 October 2026, 14:05]"."""
    t = datetime.datetime.fromtimestamp(at)
    return f"[{t:%A} {t.day} {t:%B %Y}, {t:%H:%M}]"


def prompt(messages: list[dict[str, Any]], state: State) -> list[dict[str, Any]]:
    """The messages the model reads of a conversation's, in its state."""
    view, start = _view(messages, state), state.get("summarized", 1)
    system = view[0]
    if summary := state.get("summary"):
        system["content"] += SUMMARY.format(summary=summary)
    going_on = summary and view[start:] and view[start]["role"] != "user"
    return [system, *([{"role": "user", "content": GOING_ON}] if going_on else []), *view[start:]]


def estimate(
    messages: list[dict[str, Any]], state: State, extra: int = 0, image: int = IMAGE
) -> int:
    """The tokens the prompt of a conversation's messages takes, with `extra` characters more, as
    the tools' declarations, each image `image` tokens: the engine's count of the last, and an
    estimate of what is new."""
    if state.get("used") is not None:
        return state["used"] + sum(_tokens(message(m), image) for m in messages[state["at"] :])
    return sum(_tokens(m, image) for m in prompt(messages, state)) + extra // CHARS


def compact(
    messages: list[dict[str, Any]], state: State, limit: int, extra: int,
    summarize: Callable[[str | None, list[dict[str, Any]], int], str], force: bool = False,
    image: int = IMAGE,
) -> State:  # fmt: skip
    """The state in which a conversation's prompt, each image `image` tokens, takes COMPACT of
    the context `limit` at most,
    if clearing old answers or summarizing old messages can make it, or the most they can; one
    smaller anyway if `force`, as for a reply the context cut off. `summarize` gives a summary of
    messages, with the last one, in some tokens at most. Messages are summarized only if they take
    twice those tokens: the summary must save more than it costs.

    What came before the turn running is made smaller first, and its own answers only if that is
    not enough, so that a turn reading many pages keeps what it found to answer from."""
    view = _view(messages, state)
    tail = _tail(view, limit, image)
    turn = next((i for i in range(len(view) - 1, 0, -1) if view[i]["role"] == "user"), tail)
    smaller = state
    for edge in dict.fromkeys((min(turn, tail), tail)):
        smaller = _compact_to(messages, smaller, limit, extra, summarize, edge, force, image)
        if smaller is not state and (
            force or estimate(messages, smaller, extra, image) <= COMPACT * limit
        ):
            return smaller
    return smaller


def _compact_to(
    messages: list[dict[str, Any]], state: State, limit: int, extra: int,
    summarize: Callable[[str | None, list[dict[str, Any]], int], str], edge: int, force: bool,
    image: int,
) -> State:  # fmt: skip
    # the state made smaller up to the message at `edge`: the tools' answers before it cleared,
    # and if that is not enough, the messages before it summarized
    cleared = state | {"cleared": max(state.get("cleared", 0), edge), "used": None}
    small = estimate(messages, cleared, extra, image) <= COMPACT * limit
    if cleared["cleared"] > state.get("cleared", 0) and (small or force):
        return cleared
    start, tokens = state.get("summarized", 1), summary_tokens(limit)
    spanned = sum(_tokens(m, image) for m in _view(messages, cleared)[start:edge])
    if edge <= start or spanned <= 2 * tokens:
        return cleared if cleared["cleared"] > state.get("cleared", 0) else state
    summary = summarize(state.get("summary"), messages[start:edge], tokens)
    return cleared | {"summary": summary, "summarized": edge}


def summary_tokens(limit: int) -> int:
    """The tokens a summary may take, of a context of `limit`."""
    return min(SUMMARY_TOKENS, limit // 10)


def transcript(messages: list[dict[str, Any]], shown: int = 1000) -> str:
    """Messages as a summary's model reads them: the tools' answers to `shown` characters."""
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"User: {message(m)['content']}")
        elif m["role"] == "assistant":
            if m.get("content"):
                lines.append(f"Assistant: {m['content']}")
            for call in m.get("tool_calls", []):
                f = call["function"]
                lines.append(f"Assistant called {f['name']} with {f['arguments']}")
        elif m["role"] == "tool":
            answer = m["content"] if len(m["content"]) <= shown else m["content"][:shown] + " …"
            lines.append(f"{m['name']} answered: {answer}")
    return "\n\n".join(lines)


def _view(messages: list[dict[str, Any]], state: State) -> list[dict[str, Any]]:
    # each message as the model reads it, but for a summary: the tools' old long answers cleared,
    # saying where one is saved, to read again
    view = [message(m) for m in messages]
    for m, v in zip(messages[: state.get("cleared", 0)], view, strict=False):
        if v["role"] == "tool" and (len(v["content"]) > KEEP or v.get("images")):
            v.pop("images", None)
            saved = m.get("info", {}).get("saved")
            v["content"] = SAVED.format(saved=saved) if saved else CLEARED
    return view


def _tail(view: list[dict[str, Any]], limit: int, image: int) -> int:
    # where the latest messages begin that take TAIL of the context, the last one at least, and
    # never at a tool's answer, which goes with the reply that called it
    start, tokens = len(view) - 1, _tokens(view[-1], image)
    while start > 1 and tokens + _tokens(view[start - 1], image) <= TAIL * limit:
        start, tokens = start - 1, tokens + _tokens(view[start - 1], image)
    while start > 1 and view[start]["role"] == "tool":
        start -= 1
    return start


def _tokens(m: dict[str, Any], image: int = IMAGE) -> int:
    # of a message as the model reads it, each of its images `image` tokens
    return len(json.dumps(m)) // CHARS + image * len(m.get("images", []))
