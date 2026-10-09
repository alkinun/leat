"""The agent's work in the background: running the scheduled tasks as each is due, naming each
conversation after its first turn, once a conversation is idle, reviewing what is new in it for
memories, tidying each person's memory once a day, at night, and telling the apps when the engine
comes up or goes away, as it does when the box starts.

The review is ChatGPT's "dreaming" and Hermes Agent's background review, which both have as
models do not save every memory they should as they talk: small ones say they noted a fact and call
nothing. The model reads what was said since the last review, with every memory, and changes them
with remember and forget alone: what is new about the user, what changed or passed, and what breaks
the rules of what to remember.
"""

import dataclasses
import datetime
import threading
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from leat.agent import context
from leat.agent.client import EngineError
from leat.agent.tools import Context, Tool, arguments, memory
from leat.agent.tools import tasks as scheduling

if TYPE_CHECKING:
    from leat.agent.agent import Agent

IDLE = 120  # seconds after its last message a conversation is reviewed
TIDIED = "tidied"  # of the settings: the day each person's memory was last tidied
CHECK = 30  # seconds between looks for idle conversations, and tasks due, at most
ROUNDS = 4  # replies a review takes at most
READ = 24000  # characters of what was said that a review reads at most, the latest
NIGHT = 3  # the hour from which each day's tidying of the memory runs, or at the first look after
FORGETS = 4  # of a person's memories, the share a tidying may forget at most, a quarter, or 3
NAME = (
    "Name the conversation below in 2 to 6 words, as a title, in its language: the name alone, "
    "without quotes or a full stop."
)
REVIEW = """\
You keep the memory of Leat, an assistant, about its user; today is {date}. Change the memory with \
remember and forget, then reply "Done.".

First look over the memory, below. Forget each memory that is no lasting fact about the user, as \
"The user asked about the weather", and change, with remember's replaces, each that time has made \
wrong, as a plan that has passed, quoting the old memory as its evidence.

Then read what the user and Leat said since you last looked, and remember what will matter in \
later conversations: who the user is, the people in their life, their work and plans, how they \
like things done. Each fact a memory of its own, written of "the user", as "The user's cat is \
called Pamuk.", with the user's own words it rests on as its evidence, and a plan with its last \
day. Not what they asked about or wondered, what Leat said or found, or a task's details. What is \
remembered already, leave; what changed, change with replaces.

If nothing is to change, reply "Done." alone.

The memory, each by its number and dated when it was last confirmed:
{memories}"""
TIDY = """\
You tidy the memory of Leat, an assistant, about its user; today is {date}. Change the memory with \
remember's replaces and with forget, then reply "Done.".

Merge memories that say the same, or nearly, into one: remember the one with replaces, quoting \
one of them as its evidence, and forget the others. Change a plan that has passed into what \
happened, as "The user went to Rome in May 2026", quoting it as its evidence, or forget it if it \
no longer matters. Forget each memory that is no lasting fact about the user. Leave every other \
memory as it is.

If nothing is to change, reply "Done." alone.

The memory, each by its number and dated when it was last confirmed:
{memories}"""


class Background:
    """The background's work of an agent, on a thread of its own once started."""

    def __init__(self, agent: "Agent", idle: float = IDLE):
        self.agent, self.idle = agent, idle
        self._woken = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._work, name="leat background", daemon=True).start()

    def wake(self) -> None:
        """Has the work done now, not at the next look: as a turn ends, after which its
        conversation is named and the tasks waiting for it run, or as the tasks change."""
        self._woken.set()

    def _work(self) -> None:
        # tells the apps of the engine's models if they changed, as it came up, runs the tasks due,
        # names the conversations not named, as those whose first turns ended, and reviews the
        # idle ones, one at a time, then waits for a wake, the next task or the next look. Each
        # piece of work that fails is tried again at the next look, the others done meanwhile: a
        # task due that waits, for its conversation or the engine, and any while the engine is
        # away. A bug's error is said, not the thread's end.
        while True:
            self._woken.clear()
            soonest = float("inf")
            try:
                self.agent.models_changed()
                due = self.agent.store.due(time.time())
                for task in due:
                    _attempt(run, self.agent, task)
                for id in self.agent.store.unnamed():
                    if not self.agent.running(id):
                        _attempt(name, self.agent, id)
                for id in self.agent.store.idle(time.time() - self.idle):
                    if not self.agent.running(id):
                        _attempt(review, self.agent, id)
                now, tidied = datetime.datetime.now(), self.agent.store.setting(TIDIED) or {}
                for person in [p["id"] for p in self.agent.store.people()] or [None]:
                    if now.hour >= NIGHT and tidied.get(str(person)) != now.date().isoformat():
                        _attempt(tidy, self.agent, person)
                # the next task's time, of all but those due at this look that wait, as they were
                tasks = [t["next"] for t in self.agent.store.tasks(everyone=True) if t not in due]
                soonest = min(tasks, default=soonest)
            except Exception:
                traceback.print_exc()
            self._woken.wait(max(0, min(CHECK, soonest - time.time())))


def _attempt(work: Callable[..., None], *args: Any) -> None:
    # does a piece of the work, which, if it fails, is tried at the next look: one the engine
    # failed quietly, as it may be away, and one a bug failed saying so
    try:
        work(*args)
    except EngineError:
        pass
    except Exception:
        traceback.print_exc()


def run(agent: "Agent", task: dict[str, Any]) -> None:
    """Sends a task that is due, in its conversation, or a new one if it is gone, and sets when it
    runs next: after now, so that one missed while the box was off runs once. A task whose
    conversation is busy waits for the next look; one the engine has no model to run, as the box
    starts, raises EngineError, and waits too."""
    from leat.agent.agent import Busy  # which imports this module

    if not any(m.get("status") == "loaded" for m in agent.models()):
        raise EngineError("no model is loaded")
    conversation = task["conversation"]
    if conversation is not None and agent.store.conversation(conversation) is None:
        conversation = None  # deleted: a new one, the task's person's
    try:
        sent = agent.send(
            conversation, task["prompt"], task=task["id"], person=task["person"],
            quiet=task["condition"],
        )  # fmt: skip
    except Busy:
        return
    now, day = datetime.datetime.now(), datetime.datetime.fromtimestamp(task["first"]).day
    due: datetime.datetime | None = datetime.datetime.fromtimestamp(task["next"])
    while due is not None and due <= now:
        due = scheduling.following(due, task["repeat"], day)
    agent.ran(task["id"], due.timestamp() if due else None, sent)


def name(agent: "Agent", id: str) -> None:
    """Names a conversation after its first exchange, as the model does; one the model gives no
    name keeps its own, its first message's start."""
    if (c := agent.store.conversation(id)) is None:  # deleted
        return
    messages = agent.store.messages(id)
    said = context.transcript([m for m in messages[1:] if m["role"] != "tool"][:2])[:2000]
    body = {
        "messages": [{"role": "system", "content": NAME}, {"role": "user", "content": said}],
        "max_tokens": 24, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False},
    }  # fmt: skip
    lines = agent.engine.reply(body)["content"].strip().splitlines()
    title = lines[0].strip(" \"'“”.") if lines else ""
    agent.rename(id, title[:80] or c["title"])


def review(agent: "Agent", id: str) -> None:
    """Reviews what is new in a conversation for memories, its person's and the household's,
    changing them by the model's calls; not one with a character, whose roleplay is no fact."""
    if (c := agent.store.conversation(id)) is None:  # deleted
        return
    messages = agent.store.messages(id)
    new = [m for m in messages[agent.store.reviewed(id) :] if m["role"] in ("user", "assistant")]
    # a roleplay's words are no facts of the user's
    if c["character"] is None and any(m.get("content") for m in new):
        today = datetime.date.today()
        system = REVIEW.format(
            date=f"{today:%A}, {today.day} {today:%B %Y}",
            memories=memory.listing(agent.memories(c["person"])),
        )
        # the latest of it, as much as half the model's context holds
        read = READ if (limit := agent.limit()) is None else min(READ, limit * context.CHARS // 2)
        said = context.transcript([m for m in new if m.get("content")])[-read:]
        conversation = [{"role": "system", "content": system}, {"role": "user", "content": said}]
        tools = {t.name: t for t in memory.tools(agent) if t.name in ("remember", "forget")}
        _work_on(agent, Context(id, by="review", person=c["person"]), conversation, tools)
    agent.store.mark_reviewed(id, len(messages))


def tidy(agent: "Agent", person: int | None) -> None:
    """Tidies a person's memory, and the household's, as ChatGPT's "dreaming" does: merges what
    says the same, turns plans passed into what happened, and forgets what is no lasting fact. It
    adds nothing, and forgets a quarter of the memories at most, or 3, as OpenClaw's consolidation
    is bounded, so that one bad night loses little; all it changes is kept, to undo."""
    if len(memories := agent.memories(person)) > 1:
        today, forgot = datetime.date.today(), list[int | str]()
        own = {t.name: t for t in memory.tools(agent) if t.name in ("remember", "forget")}

        def changed(context: Context, **called: Any) -> Any:
            if called.get("replaces") in (None, ""):
                raise ValueError("tidying changes memories, with replaces, and adds none")
            return own["remember"].run(context, **called)

        def forget(context: Context, number: int | str) -> Any:
            if len(forgot) >= (most := max(3, len(memories) // FORGETS)):
                raise ValueError(f"a tidying forgets {most} memories at most")
            forgot.append(number)
            return own["forget"].run(context, number=number)

        tools = {"remember": dataclasses.replace(own["remember"], run=changed),
                 "forget": dataclasses.replace(own["forget"], run=forget)}  # fmt: skip
        system = TIDY.format(
            date=f"{today:%A}, {today.day} {today:%B %Y}",
            memories=memory.listing(memories, today),
        )
        conversation = [{"role": "system", "content": system},
                        {"role": "user", "content": "Tidy the memory."}]  # fmt: skip
        _work_on(agent, Context("", by="tidy", person=person), conversation, tools)
    tidied = agent.store.setting(TIDIED) or {}
    agent.store.set_setting(TIDIED, tidied | {str(person): datetime.date.today().isoformat()})


def _work_on(
    agent: "Agent", context: Context, conversation: list[dict[str, Any]], tools: dict[str, Tool]
) -> None:
    # the model's replies to a conversation of its own, each after the calls it made of `tools`
    body = {
        "tools": [t.declaration() for t in tools.values()], "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }  # fmt: skip
    for _ in range(ROUNDS):
        reply = agent.engine.reply(body | {"messages": conversation})
        conversation.append(reply)
        if not (calls := reply.get("tool_calls")):
            return
        for call in calls:
            f = call["function"]
            try:
                if (tool := tools.get(f["name"])) is None:
                    raise ValueError(f"there is no tool {f['name']!r}")
                answer = tool.run(context, **arguments(f["arguments"])).content
            except Exception as e:  # for the model, which may try again
                answer = f"error: {e}"
            conversation.append({"role": "tool", "tool_call_id": call["id"], "content": answer})
