"""The agent: conversations, the turns that run in them, and the events that tell every app watching
what changed.

A turn is a job of its own, on a thread of the box: the user's message, then the model's replies,
streamed into the conversation, each after the tools the last one called, until one calls none. The
app that sent the message watches it as any other does, and closing it stops nothing. A turn that
fails is taken back whole, the user's message too, to send again.

Every change is made, and its event published, under one lock: an app that reads a conversation
and watches the events after misses nothing, and an event it gets twice changes nothing more.
"""

import contextlib
import copy
import datetime
import json
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

from leat.agent.client import Client, Completion, EngineError
from leat.agent.store import Store
from leat.agent.tools import Context, Result, Tool, memory

# sampling as Qwen3.6 recommends for general tasks, thinking first, then not
SAMPLING = {
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
}
# the system prompt, fixed when a conversation starts so that every prompt after extends the last
SYSTEM = """\
You are Leat, an assistant that runs on a computer in the user's home: private, and theirs. Today \
is {date}.

When a question needs facts you may not know, or that may have changed since you learned them, \
call search, then fetch the most promising pages before you answer. Link the pages you used.

When the user tells you something about themselves worth knowing in later conversations, such as \
their name, work, family, plans or tastes, call remember, once for each fact, written of "the \
user", as "The user's cat is called Pamuk." Never say you will remember something without calling \
remember. When they ask you to forget something, call forget; to change a memory, forget it and \
remember the new one. To find what you talked about in earlier conversations, call recall.

What you remember of the user, each by its number:
{memories}"""
TITLE = 60  # characters of a conversation's title at most: its first message's start
ROUNDS = 12  # replies a turn takes at most; the last may call no tools, and answers

Event = dict[str, Any]


class NotFound(Exception):
    """There is no such conversation."""


class Busy(Exception):
    """A reply is already running in the conversation."""


class Events:
    """Every change, for each app watching: each has a queue of its own."""

    def __init__(self) -> None:
        self._queues: set[queue.SimpleQueue[Event]] = set()
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def watch(self) -> Iterator[queue.SimpleQueue[Event]]:
        """A queue of the events published from now on, until the block ends."""
        q: queue.SimpleQueue[Event] = queue.SimpleQueue()
        with self._lock:
            self._queues.add(q)
        try:
            yield q
        finally:
            with self._lock:
                self._queues.discard(q)

    def publish(self, event: Event) -> None:
        with self._lock:
            for q in self._queues:
                q.put(event)


class Agent:
    """The conversations and memories in `store`, the turns run by the model `engine` serves,
    which calls the memory's tools and `tools`."""

    def __init__(self, store: Store, engine: Client, tools: list[Tool] | None = None):
        self.store, self.engine, self.events = store, engine, Events()
        self.tools = {tool.name: tool for tool in [*memory.tools(self), *(tools or [])]}
        self._turns: dict[str, _Turn] = {}  # the running ones, by their conversation's id
        self._lock = threading.Lock()

    def conversations(self) -> list[dict[str, Any]]:
        """Every conversation, without its messages, the latest updated first."""
        with self._lock:
            return [self._summary(c) for c in self.store.conversations()]

    def conversation(self, id: str) -> dict[str, Any] | None:
        """A conversation and its messages, with the one a turn is writing."""
        with self._lock:
            if (c := self.store.conversation(id)) is None:
                return None
            messages = self.store.messages(id)
            if turn := self._turns.get(id):
                messages += copy.deepcopy(turn.live)
            return self._summary(c) | {"messages": messages}

    def send(self, id: str | None, content: str, think: bool = False) -> str:
        """Starts a turn of the user's message, in a new conversation without an id; returns the
        conversation's id. The model thinks before it replies if `think`, which takes longer.
        Raises NotFound if there is no such conversation, Busy if a turn runs in it."""
        message = {"role": "user", "content": content, "info": {"think": think}}
        with self._lock:
            if id is None:
                system = _system(self.store.memories())
                id = self.store.create(_title(content), [system, message])["id"]
                start = 1
            else:
                if self.store.conversation(id) is None:
                    raise NotFound(f"there is no conversation {id}")
                if id in self._turns:
                    raise Busy("a reply is already running")
                start = len(self.store.messages(id))
                self.store.append(id, message)
            turn = self._turns[id] = _Turn(self, id, start, content, think)
            self._publish_summary(id)
            self._publish_message(id, start, message)
        threading.Thread(target=turn.run, name=f"turn {id}", daemon=True).start()
        return id

    def stop(self, id: str) -> None:
        """Stops a conversation's turn, if one runs: its reply so far is kept."""
        with self._lock:
            turn = self._turns.get(id)
        if turn is not None:
            turn.stop()

    def delete(self, id: str) -> None:
        with self._lock:
            if (turn := self._turns.pop(id, None)) is not None:
                turn.stop()
            self.store.delete(id)
            self.events.publish({"type": "deleted", "id": id})

    def memories(self) -> list[dict[str, Any]]:
        """What the agent remembers of the user, the oldest first."""
        return self.store.memories()

    def remember(self, text: str) -> dict[str, Any]:
        """Remembers something of the user, in the conversations begun from now on."""
        with self._lock:
            m = self.store.add_memory(text)
            self.events.publish(self.memories_event())
        return m

    def forget(self, id: int) -> dict[str, Any]:
        """Forgets a memory, and returns it. Raises NotFound if there is no such memory."""
        with self._lock:
            if (m := self.store.delete_memory(id)) is None:
                raise NotFound(f"there is no memory {id}")
            self.events.publish(self.memories_event())
        return m

    def memories_event(self) -> Event:
        return {"type": "memories", "memories": self.store.memories()}

    def models(self) -> list[dict[str, Any]]:
        """The engine's models, as it lists them. Raises EngineError."""
        return self.engine.models()

    def load(self, model: str) -> None:
        """Loads a model in the engine, telling every app as it starts and once it is done."""
        self.events.publish({"type": "loading", "model": model})
        try:
            self.engine.load(model)
        finally:
            self.events.publish(self.models_event())

    def models_event(self) -> Event:
        """The engine's models, as an event, or why there are none."""
        try:
            return {"type": "models", "models": self.models()}
        except EngineError as e:
            return {"type": "models", "models": [], "error": str(e)}

    def _summary(self, c: dict[str, Any]) -> dict[str, Any]:
        running = c["id"] in self._turns
        return {"id": c["id"], "title": c["title"], "updated": c["updated"], "running": running}

    def _publish_summary(self, id: str) -> None:
        if (c := self.store.conversation(id)) is not None:
            self.events.publish({"type": "conversation", "conversation": self._summary(c)})

    def _publish_message(self, id: str, index: int, message: dict[str, Any]) -> None:
        event = {"type": "message", "conversation": id, "index": index}
        self.events.publish(event | {"message": copy.deepcopy(message)})


class _Turn:
    """A turn running in a conversation, of the user's message at `start`."""

    def __init__(self, agent: Agent, id: str, start: int, content: str, think: bool):
        self.agent, self.id, self.start, self.content, self.think = agent, id, start, content, think
        self.kept = start + 1  # the conversation's messages kept
        self.live: list[dict[str, Any]] = []  # those after, to keep: a reply, or its calls running
        self.stopped = threading.Event()
        self.completion: Completion | None = None

    def stop(self) -> None:
        self.stopped.set()
        if self.completion is not None:
            self.completion.close()

    def run(self) -> None:
        try:
            for n in range(ROUNDS):
                calls = self._reply(tools=n < ROUNDS - 1)
                if not calls or self.stopped.is_set():
                    break
                self._call(calls)
                if self.stopped.is_set():
                    break
            self._end()
        except _Deleted:
            pass
        except EngineError as e:
            self._take_back(str(e))
        except Exception as e:  # a bug's: taken back, rather than left running for ever
            self._take_back(f"the turn failed: {e!r}")
            raise

    def _reply(self, tools: bool) -> list[dict[str, Any]]:
        # streams a reply of the model's into the conversation and keeps it; returns the calls it
        # makes of the agent's tools, if offered them
        a = self.agent
        messages = a.store.messages(self.id)
        reply: dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": ""}
        reply["info"] = info = {}
        (index,) = self._show(reply)
        body = {"messages": [_api(m) for m in messages], **SAMPLING[self.think]}
        # as Qwen3's templates take it, and others ignore
        body["chat_template_kwargs"] = {"enable_thinking": self.think}
        if tools and a.tools:
            body["tools"] = [tool.declaration() for tool in a.tools.values()]
        started = time.monotonic()
        self.completion = a.engine.complete(body)
        if self.stopped.is_set():  # before the completion was there to close
            self.completion.close()
        for chunk in self.completion:
            delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
            with a._lock:
                info["model"] = chunk.get("model")
                for key in ("reasoning_content", "content"):
                    if text := delta.get(key):
                        info.setdefault("first", time.monotonic() - started)
                        event = {"type": "delta", "conversation": self.id, "index": index}
                        a.events.publish(event | {"key": key, "at": len(reply[key]), "text": text})
                        reply[key] += text
                if calls := delta.get("tool_calls"):  # leat serve sends them whole, at the end
                    keys = ("id", "type", "function")
                    reply["tool_calls"] = [{k: call[k] for k in keys} for call in calls]
                if timings := chunk.get("timings"):
                    info |= _speed(timings)
        with a._lock:
            if self.stopped.is_set():
                info["stopped"] = True
        self._keep()
        return reply.get("tool_calls", [])

    def _call(self, calls: list[dict[str, Any]]) -> None:
        # runs the calls at once, each on a thread, and keeps their answers once all have answered,
        # or the turn is stopped: those yet to answer, answered so
        a, answered = self.agent, queue.SimpleQueue[int]()
        messages = []
        for call in calls:
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            with contextlib.suppress(ValueError):  # JSON text, as leat serve sends them
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            message = {"role": "tool", "tool_call_id": call["id"], "name": name, "content": ""}
            messages.append(message | {"info": {"arguments": arguments}})
        indexes = self._show(*messages)
        for message, index in zip(messages, indexes, strict=True):
            work = (message, index, answered)
            threading.Thread(target=self._answer, args=work, daemon=True).start()
        waiting = len(messages)
        while waiting and not self.stopped.is_set():
            with contextlib.suppress(queue.Empty):
                answered.get(timeout=0.1)
                waiting -= 1
        with a._lock:
            for message in messages:
                if not message["content"]:
                    message["content"] = "Stopped before it answered."
                    message["info"]["stopped"] = True
        self._keep()

    def _answer(self, message: dict[str, Any], index: int, answered: queue.SimpleQueue) -> None:
        # runs a call's tool, and puts its answer in its message, unless the turn stopped first
        a, arguments = self.agent, message["info"]["arguments"]
        try:
            if (tool := a.tools.get(message["name"])) is None:
                raise ValueError(f"there is no tool {message['name']!r}")
            if not isinstance(arguments, dict):
                raise ValueError(f"the arguments must be a JSON object, not {arguments!r}")
            result = tool.run(Context(self.id), **arguments)
        except Exception as e:  # for the model, which may try again
            result = Result(f"error: {e}", {"error": str(e)})
        with a._lock:
            if not message["content"]:
                message["content"] = result.content or "(nothing)"
                message["info"] |= result.info
                a._publish_message(self.id, index, message)
        answered.put(index)

    def _show(self, *messages: dict[str, Any]) -> list[int]:
        # shows messages to come in the conversation, for now; returns where they are
        with self.agent._lock:
            indexes = []
            for message in messages:
                indexes.append(self.kept + len(self.live))
                self.live.append(message)
                self.agent._publish_message(self.id, indexes[-1], message)
            return indexes

    def _keep(self) -> None:
        # keeps the messages shown, unless the conversation is gone
        a = self.agent
        with a._lock:
            if a._turns.get(self.id) is not self:
                raise _Deleted
            a.store.append(self.id, *self.live)
            for i, message in enumerate(self.live):
                a._publish_message(self.id, self.kept + i, message)
            self.kept += len(self.live)
            self.live = []

    def _end(self) -> None:
        a = self.agent
        with a._lock:
            if a._turns.get(self.id) is self:
                del a._turns[self.id]
                a._publish_summary(self.id)

    def _take_back(self, error: str) -> None:
        # removes the turn's messages, from `start`, and the conversation it began; tells the apps
        # why, and what the user wrote, to send again
        a = self.agent
        with a._lock:
            if a._turns.get(self.id) is not self:
                return
            del a._turns[self.id]
            self.live = []
            event = {"type": "error", "conversation": self.id, "error": error, "start": self.start}
            a.events.publish(event | {"content": self.content})
            if self.start == 1:  # its first turn
                a.store.delete(self.id)
                a.events.publish({"type": "deleted", "id": self.id})
            else:
                a.store.truncate(self.id, self.start)
                a._publish_summary(self.id)


class _Deleted(Exception):
    """The turn's conversation was deleted."""


def _system(memories: list[dict[str, Any]]) -> dict[str, Any]:
    # the system prompt of a conversation begun now, which knows these memories
    today = datetime.date.today()
    date = f"{today:%A}, {today.day} {today:%B %Y}"
    remembered = "\n".join(f"[{m['id']}] {m['text']}" for m in memories) or "Nothing yet."
    return {"role": "system", "content": SYSTEM.format(date=date, memories=remembered)}


def _title(content: str) -> str:
    # the first line of the first message, cut at a word
    line = content.strip().split("\n")[0]
    return line if len(line) <= TITLE else line[:TITLE].rsplit(" ", 1)[0] + "…"


def _api(message: dict[str, Any]) -> dict[str, Any]:
    # a message as the model reads it: without what only people see, nor an empty reasoning. The
    # reasoning is sent back, which templates such as Qwen3.5's show the steps of an agent's turn.
    return {k: v for k, v in message.items() if k != "info" and (v or k != "reasoning_content")}


def _speed(timings: dict[str, Any]) -> dict[str, Any]:
    # the reply's tokens and their rate after the first, which came of the prompt's last step; and
    # the prompt's tokens the engine had cached, and those it read
    n, ms = timings["predicted_n"], timings["predicted_ms"]
    speed = {"tokens": n, "cached": timings.get("cache_n"), "read": timings.get("prompt_n")}
    return speed | ({"rate": 1e3 * (n - 1) / ms} if n > 1 and ms else {})
