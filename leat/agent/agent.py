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

from leat.agent import context
from leat.agent.background import Background
from leat.agent.client import Client, Completion, EngineError
from leat.agent.store import Store
from leat.agent.tools import Context, Result, Tool, arguments, files, memory, numbered
from leat.agent.tools import tasks as scheduling
from leat.agent.workspace import Workspace

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
call search, then fetch the few pages most likely to answer, three or so, more only if they fall \
short. Cite what you use by the numbers the tools give their sources, as [1] or [2][3], after the \
words they support.

When the user tells you something about themselves worth knowing in later conversations, first \
call remember, then reply: who they are, the people in their life, their work and plans, how they \
like things done; each fact a memory of its own, written of "the user", as "The user's cat is \
called Pamuk." Not what they asked about, what a search finds again, or a task's details. When a \
memory changes, remember the new one in its place; when they ask you to forget something, forget \
it. Never say you noted something unless you called remember. To find what you talked about in \
earlier conversations that your memory below does not hold, call recall.

When the user wants something done later, once or again and again, as a reminder or a morning's \
briefing, call schedule.
{workspace}
What you remember of the user, each by its number:
{memories}"""
# of the system prompt, when the agent has a workspace
WORKSPACE = """
The user's files are in a workspace, where you read, write and edit them, and run Python among \
them in a sandbox without the network; the files they attach are named in their message. To make \
a document, first read the skill for its kind, then make it with run, and name its file in your \
answer, without a link: the app shows the user the files you make. The skills:
{skills}
"""
TITLE = 60  # characters of a conversation's title at most: its first message's start
ROUNDS = 25  # replies a turn takes at most; the last may call no tools, and answers, told so
LAST = "(You have made all the tool calls this message allows: answer now, from what you found.)"

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
    which calls the memory's tools and `tools`; and the user's files, in `workspace`, if any."""

    def __init__(
        self, store: Store, engine: Client, tools: list[Tool] | None = None,
        workspace: Workspace | None = None,
    ):  # fmt: skip
        self.store, self.engine, self.workspace, self.events = store, engine, workspace, Events()
        own = [*memory.tools(self), *scheduling.tools(self)]
        self.tools = {tool.name: tool for tool in [*own, *(tools or [])]}
        self._files: list[dict[str, Any]] | None = None  # the files the apps were last told of
        self.background: Background | None = None  # once started
        self._turns: dict[str, _Turn] = {}  # the running ones, by their conversation's id
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the agent's work in the background: naming conversations, and reviewing them
        for memories once idle, and running the tasks due, as background.Background does."""
        self.background = Background(self)
        self.background.start()

    def running(self, id: str) -> bool:
        """Whether a turn runs in a conversation."""
        with self._lock:
            return id in self._turns

    def rename(self, id: str, title: str) -> None:
        """Names a conversation, as the model did, telling the apps."""
        with self._lock:
            self.store.name(id, title)
            self._publish_summary(id)

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
            summarized = self.store.context(id).get("summarized")
            return self._summary(c) | {"messages": messages, "summarized": summarized}

    def send(
        self, id: str | None, content: str, think: bool = False, attached: list[str] | None = None,
        task: int | None = None, via: str | None = None,
    ) -> str:  # fmt: skip
        """Starts a turn of the user's message, in a new conversation without an id; returns the
        conversation's id. The model thinks before it replies if `think`, which takes longer, and
        reads of the files `attached`, in the workspace. A message of a scheduled task names it,
        and one sent by a messaging app, `via`, that. Raises NotFound if there is no such
        conversation or file, Busy if a turn runs in the conversation."""
        info: dict[str, Any] = {"think": think} | ({"task": task} if task else {})
        if via:
            info["via"] = via
        if attached:
            space = self.workspace
            if space is None or not all(space.path(name).is_file() for name in attached):
                raise NotFound(f"the workspace has not all of {', '.join(attached)}")
            info["files"] = attached
        message = {"role": "user", "content": content, "info": info}
        with self._lock:
            if id is None:
                system = _system(self.store.memories(), self.workspace is not None)
                id = self.store.create(_title(content), [system, message])["id"]
                start = 1
            else:
                if self.store.conversation(id) is None:
                    raise NotFound(f"there is no conversation {id}")
                if id in self._turns:
                    raise Busy("a reply is already running")
                start = len(self.store.messages(id))
                self.store.append(id, message)
            turn = self._turns[id] = _Turn(self, id, start, content, think, task)
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

    def remember(
        self, text: str, category: str = "about", replaces: int | None = None
    ) -> dict[str, Any]:
        """Remembers a fact of the user, in the conversations begun from now on, in place of the
        memory `replaces` if given. Raises ValueError if it may not be one, or if it does not fit
        the memory's room, NotFound if there is no memory to replace."""
        text = memory.checked(text, category)
        with self._lock:
            others = [m for m in self.store.memories() if m["id"] != replaces]
            if same := [m for m in others if m["text"].lower() == text.lower()]:
                raise ValueError(f"that is remembered already, as [{same[0]['id']}]")
            if (used := sum(len(m["text"]) for m in others)) + len(text) > memory.ROOM:
                raise ValueError(
                    f"the memory is full, {used} of its {memory.ROOM} characters: forget or "
                    "change the least useful memories first, or make one of two"
                )
            if replaces is None:
                m = self.store.add_memory(text, category)
            elif (changed := self.store.replace_memory(replaces, text, category)) is None:
                raise NotFound(f"there is no memory {replaces}")
            else:
                m = changed
            self.events.publish(self.memories_event())
        return m

    def forget(self, id: int) -> dict[str, Any]:
        """Forgets a memory, and returns it. Raises NotFound if there is no such memory."""
        with self._lock:
            if (m := self.store.delete_memory(id)) is None:
                raise NotFound(f"there is no memory {id}")
            self.events.publish(self.memories_event())
        return m

    def files_changed(self) -> None:
        """Tells the apps of the workspace's files, if they changed since they were last told."""
        with self._lock:
            if self.workspace is not None and (now := self.workspace.files()) != self._files:
                self._files = now
                self.events.publish({"type": "files", "files": now})

    def files_event(self) -> Event:
        return {"type": "files", "files": self.workspace.files() if self.workspace else []}

    def schedule(
        self, prompt: str, at: datetime.datetime, repeat: str, conversation: str | None = None
    ) -> dict[str, Any]:
        """Schedules a task, first at a time of the box's clock, then as often as `repeat` says,
        in a conversation. Raises ValueError if it may not be one, as at a time passed."""
        now = datetime.datetime.now()
        at = scheduling.first(at, repeat, now)
        prompt = scheduling.checked(prompt, at, repeat, now)
        with self._lock:
            if len(self.store.tasks()) >= scheduling.MOST:
                raise ValueError(f"{scheduling.MOST} tasks are scheduled, the most: cancel one")
            task = self.store.add_task(prompt, repeat, at.timestamp(), conversation)
            self.events.publish(self.tasks_event())
        if self.background is not None:
            self.background.wake()
        return task | {"schedule": scheduling.describe(task)}

    def unschedule(self, id: int) -> dict[str, Any]:
        """Cancels a task, and returns it. Raises NotFound if there is no such task."""
        with self._lock:
            if (task := self.store.delete_task(id)) is None:
                raise NotFound(f"there is no task {id}")
            self.events.publish(self.tasks_event())
        return task

    def tasks(self) -> list[dict[str, Any]]:
        """The tasks, the next due first, each with its times in words."""
        return [t | {"schedule": scheduling.describe(t)} for t in self.store.tasks()]

    def ran(self, id: int, due: float | None, conversation: str) -> None:
        """Sets when a task that ran runs next, in the conversation it ran in; or ends it."""
        with self._lock:
            self.store.advance(id, due, conversation)
            self.events.publish(self.tasks_event())

    def tasks_event(self) -> Event:
        return {"type": "tasks", "tasks": self.tasks()}

    def memories_event(self) -> Event:
        return {"type": "memories", "memories": self.store.memories()}

    def models(self) -> list[dict[str, Any]]:
        """The engine's models, as it lists them. Raises EngineError."""
        return self.engine.models()

    def limit(self) -> int | None:
        """The loaded model's context, in tokens, if the engine says. Raises EngineError."""
        loaded = [m for m in self.models() if m.get("status") == "loaded"]
        return loaded[0].get("max_context") if loaded else None

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

    def __init__(
        self, agent: Agent, id: str, start: int, content: str, think: bool, task: int | None
    ):  # fmt: skip
        self.agent, self.id, self.start, self.content, self.think = agent, id, start, content, think
        self.task = task  # the scheduled task the message is of, if any
        self.state = agent.store.context(id)  # the prompt's, as it was, to take the turn back to
        self.kept = start + 1  # the conversation's messages kept
        self.live: list[dict[str, Any]] = []  # those after, to keep: a reply, or its calls running
        self.stopped = threading.Event()
        self.completion: Completion | None = None
        self.limit: int | None = None  # the model's context, once the turn asks
        self.redone = False  # a reply the context cut off, after the prompt was made smaller
        self.sources: dict[str, int] = {}  # the conversation's, by address, numbered for citing

    def stop(self) -> None:
        self.stopped.set()
        if self.completion is not None:
            self.completion.close()

    def run(self) -> None:
        try:
            self.limit = self.agent.limit()
            self.sources = numbered(self.agent.store.messages(self.id))
            for n in range(ROUNDS):
                calls = self._reply(last=n == ROUNDS - 1)
                if not calls or self.stopped.is_set():
                    break
                self._call(calls)
                self.agent.files_changed()  # as a call may have changed them
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

    def _reply(self, last: bool = False) -> list[dict[str, Any]]:
        # streams a reply of the model's into the conversation and keeps it; returns the calls it
        # makes of the agent's tools, offered them but for the turn's `last` reply. The prompt is
        # made smaller first if it outgrows its share of the context, and again, the reply redone,
        # if the context cuts the reply off.
        a = self.agent
        messages, state = a.store.messages(self.id), a.store.context(self.id)
        declared = [] if last else [tool.declaration() for tool in a.tools.values()]
        extra = len(json.dumps(declared))
        if self.limit and context.estimate(messages, state, extra) > context.COMPACT * self.limit:
            state = self._compact(messages, state, extra)
        reply: dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": ""}
        info: dict[str, Any] = {}
        reply["info"] = info
        (index,) = self._show(reply)
        read = context.prompt(messages, state)
        body: dict[str, Any] = {"messages": read, **SAMPLING[self.think]}
        # as Qwen3's templates take it, and others ignore
        body["chat_template_kwargs"] = {"enable_thinking": self.think}
        if declared:
            body["tools"] = declared
        elif last and a.tools:
            read.append({"role": "user", "content": LAST})
        started, finish = time.monotonic(), None
        self.completion = a.engine.complete(body)
        if self.stopped.is_set():  # before the completion was there to close
            self.completion.close()
        for chunk in self.completion:
            choice = chunk["choices"][0] if chunk.get("choices") else {}
            delta, finish = choice.get("delta") or {}, choice.get("finish_reason") or finish
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
        if finish == "length" and self.limit and not self.redone:
            smaller = self._compact(messages, state, extra, force=True)
            if smaller is not state:  # made smaller: the reply is redone in its place
                self.redone = True
                with a._lock:
                    self.live.remove(reply)
                return self._reply(last)
        with a._lock:
            if self.stopped.is_set():
                info["stopped"] = True
            if finish == "length":
                info["cut"] = True
        if info.get("read") is not None:  # the engine's count, of which the next is estimated
            used = (info["cached"] or 0) + info["read"] + info["tokens"]
            a.store.set_context(self.id, state | {"used": used, "at": index + 1})
        self._keep()
        return reply.get("tool_calls", [])

    def _compact(
        self, messages: list[dict[str, Any]], state: context.State, extra: int, force: bool = False
    ) -> context.State:
        # makes the prompt smaller, as context.compact can, telling the apps where any summary
        # ends; returns the state, the same if it is no smaller
        assert self.limit is not None
        smaller = context.compact(messages, state, self.limit, extra, self._summarize, force)
        keys = ("cleared", "summarized")
        if [smaller.get(k) for k in keys] == [state.get(k) for k in keys]:
            return state
        a = self.agent
        with a._lock:
            a.store.set_context(self.id, smaller)
            if (summarized := smaller.get("summarized")) != state.get("summarized"):
                event = {"type": "compacted", "conversation": self.id, "summarized": summarized}
                a.events.publish(event)
        return smaller

    def _summarize(self, before: str | None, messages: list[dict[str, Any]], tokens: int) -> str:
        # a summary of messages in `tokens` at most, with that of those before them, by the model,
        # whose context the summary and its prompt must fit; of messages too many, the latest
        assert self.limit is not None
        text, so_far = context.transcript(messages), f"The summary so far:\n{before}\n\n"
        room = (self.limit - tokens) * context.CHARS
        room -= len(context.SUMMARIZE) + len(so_far if before else "") + 200
        user = (so_far + "What came after:\n\n" if before else "") + text[-room:]
        body = {
            "messages": [{"role": "system", "content": context.SUMMARIZE},
                         {"role": "user", "content": user}],
            "max_tokens": tokens, "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False},
        }  # fmt: skip
        return self.agent.engine.reply(body)["content"].strip() or (before or "")

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
        a = self.agent
        try:
            if (tool := a.tools.get(message["name"])) is None:
                raise ValueError(f"there is no tool {message['name']!r}")
            called = arguments(message["info"]["arguments"])
            result = tool.run(Context(self.id, self._cite), **called)
        except Exception as e:  # for the model, which may try again
            result = Result(f"error: {e}", {"error": str(e)})
        with a._lock:
            if not message["content"]:
                message["content"] = result.content or "(nothing)"
                message["info"] |= result.info
                a._publish_message(self.id, index, message)
        answered.put(index)

    def _cite(self, url: str, title: str) -> int:
        # a source's number in the conversation: its own, if it was read before, or the next
        with self.agent._lock:
            return self.sources.setdefault(url, max(self.sources.values(), default=0) + 1)

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
                if self.task:  # a scheduled task done, for the apps to say
                    done = {"type": "done", "conversation": self.id, "task": self.content}
                    a.events.publish(done)
        if a.background is not None:
            a.background.ended(self.id)

    def _take_back(self, error: str) -> None:
        # removes the turn's messages, from `start`, and the conversation it began, and puts the
        # prompt's state back as it was; tells the apps why, and what the user wrote, to send again
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
                summarized = a.store.context(self.id).get("summarized")
                a.store.truncate(self.id, self.start)
                a.store.set_context(self.id, self.state)
                if (before := self.state.get("summarized")) != summarized:
                    event = {"type": "compacted", "conversation": self.id, "summarized": before}
                    a.events.publish(event)
                a._publish_summary(self.id)


class _Deleted(Exception):
    """The turn's conversation was deleted."""


def _system(memories: list[dict[str, Any]], workspace: bool) -> dict[str, Any]:
    # the system prompt of a conversation begun now, which knows these memories, and the
    # workspace's tools if `workspace`
    today = datetime.date.today()
    date = f"{today:%A}, {today.day} {today:%B %Y}"
    remembered = memory.listing(memories)
    skills = "\n".join(f"- {path}: {about}" for path, about in files.skills())
    space = WORKSPACE.format(skills=skills) if workspace else ""
    content = SYSTEM.format(date=date, memories=remembered, workspace=space)
    return {"role": "system", "content": content}


def _title(content: str) -> str:
    # the first line of the first message, cut at a word
    line = content.strip().split("\n")[0]
    return line if len(line) <= TITLE else line[:TITLE].rsplit(" ", 1)[0] + "…"


def _speed(timings: dict[str, Any]) -> dict[str, Any]:
    # the reply's tokens and their rate after the first, which came of the prompt's last step; and
    # the prompt's tokens the engine had cached, and those it read
    n, ms = timings["predicted_n"], timings["predicted_ms"]
    speed = {"tokens": n, "cached": timings.get("cache_n"), "read": timings.get("prompt_n")}
    return speed | ({"rate": 1e3 * (n - 1) / ms} if n > 1 and ms else {})
