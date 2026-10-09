"""The agent's work in the background: naming each conversation after its first turn, and telling
the apps when the engine comes up or goes away, as it does when the box starts."""

import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from leat.agent import context
from leat.agent.client import EngineError

if TYPE_CHECKING:
    from leat.agent.agent import Agent

CHECK = 30  # seconds between looks at the engine and the conversations not named
NAME = (
    "Name the conversation below in 2 to 6 words, as a title, in its language: the name alone, "
    "without quotes or a full stop."
)
# tokens a name may take, its reasoning's first where the model cannot be told not to reason, as
# gpt-oss, whose least effort spent all of 24 on its analysis, naming nothing; a model that does
# not reason stops after the name
NAME_TOKENS = 160


class Background:
    """The background's work of an agent, on a thread of its own once started."""

    def __init__(self, agent: "Agent"):
        self.agent = agent
        self._woken = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._work, name="leat background", daemon=True).start()

    def wake(self) -> None:
        """Has the work done now, not at the next look: as a turn ends, after which its
        conversation is named."""
        self._woken.set()

    def _work(self) -> None:
        # tells the apps of the engine's models if they changed, as it came up, and names the
        # conversations not named, as those whose first turns ended, one at a time, then waits
        # for a wake or the next look. Each naming that fails is tried again at the next look,
        # the others done meanwhile, as any while the engine is away. A bug's error is said, not
        # the thread's end.
        while True:
            self._woken.clear()
            try:
                self.agent.models_changed()
                for id in self.agent.store.unnamed():
                    if not self.agent.running(id):
                        _attempt(name, self.agent, id)
            except Exception:
                traceback.print_exc()
            self._woken.wait(CHECK)


def _attempt(work: Callable[..., None], *args: Any) -> None:
    # does a piece of the work, which, if it fails, is tried at the next look: one the engine
    # failed quietly, as it may be away, and one a bug failed saying so
    try:
        work(*args)
    except EngineError:
        pass
    except Exception:
        traceback.print_exc()


def name(agent: "Agent", id: str) -> None:
    """Names a conversation after its first exchange, as the model does, for the person whose
    it is, as their own prompts are; one the model gives no name keeps its own, its first
    message's start."""
    if (c := agent.store.conversation(id)) is None:  # deleted
        return
    messages = agent.store.messages(id)
    said = context.transcript([m for m in messages[1:] if m["role"] != "tool"][:2])[:2000]
    body = {
        "messages": [{"role": "system", "content": NAME}, {"role": "user", "content": said}],
        "max_tokens": NAME_TOKENS, "temperature": 0.3, "reasoning_effort": "none",
    }  # fmt: skip
    lines = agent.engine.reply(body, c["person"])["content"].strip().splitlines()
    title = lines[0].strip(" \"'“”.") if lines else ""
    agent.rename(id, title[:80] or c["title"])
