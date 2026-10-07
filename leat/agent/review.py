"""The agent's work in the background: naming each conversation after its first turn, and, once a
conversation is idle, reviewing what is new in it for memories.

The review is ChatGPT's "dreaming" and Hermes Agent's background review, which both have as
models do not save every memory they should as they talk: small ones say they noted a fact and call
nothing. The model reads what was said since the last review, with every memory, and changes them
with remember and forget alone: what is new about the user, what changed or passed, and what breaks
the rules of what to remember.
"""

import datetime
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

from leat.agent import context
from leat.agent.client import EngineError
from leat.agent.tools import Context, arguments, memory

if TYPE_CHECKING:
    from leat.agent.agent import Agent

IDLE = 120  # seconds after its last message a conversation is reviewed
CHECK = 30  # seconds between looks for idle conversations
ROUNDS = 4  # replies a review takes at most
READ = 24000  # characters of what was said that a review reads at most, the latest
NAME = (
    "Name the conversation below in 2 to 6 words, as a title, in its language: the name alone, "
    "without quotes or a full stop."
)
REVIEW = """\
You keep the memory of Leat, an assistant, about its user; today is {date}. Change the memory with \
remember and forget, then reply "Done.".

First look over the memory, below. Forget each memory that is no lasting fact about the user, as \
"The user asked about the weather", and change, with remember's replaces, each that time has made \
wrong, as a plan whose date has passed.

Then read what the user and Leat said since you last looked, and remember what will matter in \
later conversations: who the user is, the people in their life, their work and plans, how they \
like things done. Each fact a memory of its own, written of "the user", as "The user's cat is \
called Pamuk." Not what they asked about or wondered, what a search finds again, or a task's \
details. What is remembered already, leave; what changed, change with replaces.

If nothing is to change, reply "Done." alone.

The memory, each by its number:
{memories}"""


class Background:
    """The background's work of an agent, on a thread of its own once started."""

    def __init__(self, agent: "Agent", idle: float = IDLE):
        self.agent, self.idle = agent, idle
        self.ended: queue.SimpleQueue[str] = queue.SimpleQueue()  # conversations whose turn ended

    def start(self) -> None:
        threading.Thread(target=self._work, name="leat background", daemon=True).start()

    def _work(self) -> None:
        # names the conversations whose turns end, and reviews the idle ones, one at a time; an
        # engine that is away is tried again later, and a bug's error said, not the thread's end
        while True:
            try:
                ended = self.ended.get(timeout=CHECK)
            except queue.Empty:
                ended = None
            try:
                if ended is not None and not self.agent.store.named(ended):
                    name(self.agent, ended)
                for id in self.agent.store.idle(time.time() - self.idle):
                    if not self.agent.running(id):
                        review(self.agent, id)
            except EngineError:
                pass
            except Exception:
                traceback.print_exc()


def name(agent: "Agent", id: str) -> None:
    """Names a conversation after its first exchange, as the model does."""
    messages = agent.store.messages(id)
    said = context.transcript([m for m in messages[1:] if m["role"] != "tool"][:2])[:2000]
    body = {
        "messages": [{"role": "system", "content": NAME}, {"role": "user", "content": said}],
        "max_tokens": 24, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False},
    }  # fmt: skip
    lines = agent.engine.reply(body)["content"].strip().splitlines()
    if title := (lines[0].strip(" \"'“”.") if lines else ""):
        agent.rename(id, title[:80])


def review(agent: "Agent", id: str) -> None:
    """Reviews what is new in a conversation for memories, changing them by the model's calls."""
    messages = agent.store.messages(id)
    new = [m for m in messages[agent.store.reviewed(id) :] if m["role"] in ("user", "assistant")]
    if any(m.get("content") for m in new):
        today = datetime.date.today()
        system = REVIEW.format(
            date=f"{today:%A}, {today.day} {today:%B %Y}", memories=memory.listing(agent.memories())
        )
        said = context.transcript([m for m in new if m.get("content")])[-READ:]
        _work_on(
            agent, id, [{"role": "system", "content": system}, {"role": "user", "content": said}]
        )
    agent.store.mark_reviewed(id, len(messages))


def _work_on(agent: "Agent", id: str, conversation: list[dict[str, Any]]) -> None:
    # the model's replies to a conversation of its own, each after the memory's calls it made
    tools = {t.name: t for t in memory.tools(agent) if t.name in ("remember", "forget")}
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
                result = tools[f["name"]].run(Context(id), **arguments(f["arguments"]))
                answer = result.content
            except Exception as e:  # for the model, which may try again
                answer = f"error: {e}"
            conversation.append({"role": "tool", "tool_call_id": call["id"], "content": answer})
