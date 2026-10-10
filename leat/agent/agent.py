"""The agent: conversations, the turns that run in them, and the events that tell every app watching
what changed.

A conversation may be held in a project, whose name and instructions its system prompt gives the
model, and whose files its tools work among; one in none works among its person's own.

A turn is a job of its own, on a thread of the box: the user's message, then the model's replies,
streamed into the conversation, each after the tools the last one called, until one calls none. The
app that sent the message watches it as any other does, and closing it stops nothing. A turn that
fails is taken back whole, the user's message too, to send again.

Every change is made, and its event published, under one lock: an app that reads a conversation
and watches the events after misses nothing, and an event it gets twice changes nothing more.
"""

import base64
import contextlib
import copy
import functools
import json
import mimetypes
import queue
import shutil
import threading
import time
from collections.abc import Collection, Iterable, Iterator
from typing import Any

from leat.agent import context
from leat.agent.background import Background
from leat.agent.client import Client, Completion, EngineError, whole
from leat.agent.index import Index
from leat.agent.store import Store
from leat.agent.tools import Context, Result, Tool, arguments, files, numbered
from leat.agent.workspace import Workspace

# the system prompt, fixed when a conversation starts so that every prompt after extends the last
SYSTEM = """\
You are Leat, an assistant that runs on a computer of the user's own, which keeps what they say \
and the files they share private.{named} Each of the user's messages begins with the date and time \
they sent it.

When a question needs facts you may not know, or that may have changed since you learned them, \
call search, then fetch the few pages most likely to answer, three or so, at once, each with the \
question you want it to answer, more only if they fall short. When the user asks you to research \
something, search it in several ways and read the pages that matter, ten or so, before you answer \
with a report: what you found, in sections, and what stays unsure. Cite what you use by the \
numbers the tools give their sources, as [1] or [2][3], after the words they support.
{workspace}"""
# of the system prompt, when the agent has a workspace
WORKSPACE = """
The user's files are in a workspace, where you read, write and edit them, and run Python among \
them in a sandbox without the network; the files they attach are named in their message, and the \
images among them shown, as those you read are, when you can see images. To make \
a document, first read the skill for its kind, then make it with run, and name its file in your \
answer, without a link: the app shows the user the files you make.{search}{ask} The skills:
{skills}
"""
# of the system prompt, when the agent searches the files
SEARCH = """ To answer from the files, call search_files \
for the passages that say it, in several ways if the first falls short, and read on in the files \
where the passages do; cite them by their numbers, as you cite pages."""
# and when it asks every file
ASK = """ To answer a question of every file, or of many, as \
each invoice's total, call ask_files once rather than reading them; then answer with its table, \
or the first rows of a long one, and say what stands out."""
# of the system prompt, of a conversation in a project
PROJECT = """
This conversation is in the project "{name}".{instructions}
"""
INSTRUCTIONS = " Its instructions, which hold for every conversation in it:\n\n{text}\n"
TITLE = 60  # characters of a conversation's title at most: its first message's start
ROUNDS = 25  # replies a turn takes at most; the last answers, told so, and may call no tools
LAST = "(You have made all the tool calls this message allows: answer now, from what you found.)"
# times a reply of nothing, neither text nor a call, is asked again at most: a model may end its
# reasoning without a word, as gpt-oss does of one reply in forty or so, which would end the turn
EMPTY = 2

Event = dict[str, Any]


class NotFound(Exception):
    """There is no such conversation, or project, of the person's."""


class Refused(Exception):
    """The person may not do that: delete a project of another's, as only the box's owner may."""


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
    """The conversations in `store`, the turns run by the model `engine` serves, which calls
    `tools`; and the files, in `workspace`, if any: each project's, and each person's own, read
    and kept by `index`, if one, as they change.

    Each conversation is a person's of the box's, `person` by its id, and an event that tells of
    one is to that person's apps alone: "to" says whose. Before the box
    has its first person, all is no one's, None's."""

    def __init__(
        self, store: Store, engine: Client, tools: list[Tool] | None = None,
        workspace: Workspace | None = None, index: Index | None = None,
    ):  # fmt: skip
        self.store, self.engine, self.workspace, self.events = store, engine, workspace, Events()
        self.index = index
        if index is not None:  # each file's state, told as it is read
            index.changed = self._indexed
        self.tools = {tool.name: tool for tool in tools or []}
        self._files: dict[str, list[dict[str, Any]]] = {}  # each space's, as the apps were told
        self._models: Event | None = None  # the engine's models, as the apps were last told
        self.background: Background | None = None  # once started
        self._turns: dict[str, _Turn] = {}  # the running ones, by their conversation's id
        self._lock = threading.Lock()
        self.settle()

    def start(self) -> None:
        """Starts the agent's work in the background: naming conversations, and telling the apps
        when the engine comes up or goes away, as background.Background does."""
        self.background = Background(self)
        self.background.start()
        if self.index is not None:
            self.index.start()

    def running(self, id: str) -> bool:
        """Whether a turn runs in a conversation."""
        with self._lock:
            return id in self._turns

    def rename(self, id: str, title: str) -> None:
        """Names a conversation, as the model did, telling the apps."""
        with self._lock:
            self.store.name(id, title)
            self._publish_summary(id)

    def conversations(self, person: int | None = None) -> list[dict[str, Any]]:
        """A person's conversations, without their messages, the latest updated first."""
        with self._lock:
            return [self._summary(c) for c in self.store.conversations(person)]

    def conversation(self, id: str, person: int | None = None) -> dict[str, Any] | None:
        """A person's conversation and its messages, with the one a turn is writing."""
        with self._lock:
            if (c := self._seen(id, person)) is None:
                return None
            messages = self.store.messages(id)
            if turn := self._turns.get(id):
                messages += copy.deepcopy(turn.live)
            summarized = self.store.context(id).get("summarized")
            return self._summary(c) | {"messages": messages, "summarized": summarized}

    def send(
        self, id: str | None, content: str, effort: str | None = None,
        attached: list[str] | None = None, person: int | None = None, project: str | None = None,
    ) -> str:  # fmt: skip
        """Starts a turn of a person's message, in a new conversation without an id, held in
        `project` if one; returns the conversation's id. The model reasons at `effort`, one of
        leat.chat's EFFORTS, which leat serve gives as near as the model can, or else at the
        model's own default; and reads of the files `attached`, in the workspace. Raises NotFound
        if there is no such conversation or project of the person's, or file, Busy if a turn runs
        in the conversation."""
        info: dict[str, Any] = {"at": time.time()} | ({"effort": effort} if effort else {})
        message = {"role": "user", "content": content, "info": info}
        with self._lock:
            held = self._project(project, person) if id is None and project else None
            if id is not None:
                project = self._own(id, person)["project"]
                if id in self._turns:
                    raise Busy("a reply is already running")
            if attached:
                space = self._space(project, person) if self.workspace else None
                if space is None or not all(space.path(name).is_file() for name in attached):
                    raise NotFound(f"the workspace has not all of {', '.join(attached)}")
                info["files"] = attached
            if id is None:
                who = (self.store.person(person) or {}) if person else {}
                system = _system(self.workspace is not None, who.get("name"), held, self.tools)
                title = _title(content)
                id = self.store.create(title, [system, message], person, project)["id"]
                start = 1
            else:
                start = len(self.store.messages(id))
                self.store.append(id, message)
            turn = _Turn(self, id, person, project, start, content, effort)
            self._turns[id] = turn
            self._publish_summary(id)
            self._publish_message(id, person, start, message)
        threading.Thread(target=turn.run, name=f"turn {id}", daemon=True).start()
        return id

    def stop(self, id: str, person: int | None = None) -> None:
        """Stops a person's conversation's turn, if one runs: its reply so far is kept."""
        with self._lock:
            self._own(id, person)
            turn = self._turns.get(id)
        if turn is not None:
            turn.stop()

    def delete(self, id: str, person: int | None = None) -> None:
        """Deletes a person's conversation, stopping its turn. Raises NotFound if it is not."""
        with self._lock:
            self._own(id, person)
            if (turn := self._turns.pop(id, None)) is not None:
                turn.stop()
            self.store.delete(id)
            self.events.publish({"type": "deleted", "id": id, "to": person})

    def projects(self, person: int | None = None) -> list[dict[str, Any]]:
        """The projects a person sees, the latest updated first."""
        return [_project(p) for p in self.store.projects(person)]

    def add_project(
        self, name: str, person: int | None = None, instructions: str = "", shared: bool = False
    ) -> dict[str, Any]:
        """A new project of a person's, which the apps of those who see it are told of."""
        with self._lock:
            project = self.store.add_project(name, person, instructions, shared)
            self._publish_projects()
        return _project(project)

    def change_project(self, id: str, person: int | None = None, **changes: Any) -> None:
        """Changes a project a person sees: its name, instructions, or whether it is shared, which
        its own person alone may change, as a project shared is seen by everyone. Raises
        NotFound if they see no such project, Refused if it is shared and not theirs."""
        with self._lock:
            project = self._project(id, person)
            if "shared" in changes and project["person"] != person:
                raise Refused("only whoever made a project may change whom it is shared with")
            self.store.change_project(id, **changes)
            self._publish_projects()
            if "shared" in changes:  # the conversations in it, and its files, seen or no longer
                self._publish_conversations()
                if self.workspace is not None:
                    self._publish_files(id, person)

    def delete_project(self, id: str, person: int | None = None) -> None:
        """Deletes a project a person sees, with every conversation in it, whoever's, stopping
        their turns: its own person's to do, or the box's owner's. Raises NotFound if they see no
        such project, Refused if it is not theirs to delete."""
        with self._lock:
            project, who = self._project(id, person), self.store.person(person or 0) or {}
            if project["person"] != person and not who.get("owner"):
                raise Refused("only whoever made a project, or this Leat's owner, may delete it")
            held = self.store.conversations_in(id)
            for c in held:
                if (turn := self._turns.pop(c["id"], None)) is not None:
                    turn.stop()
            self.store.delete_project(id)
            if self.workspace is not None:
                self._space(id, person).remove()
                self._files.pop(_folder(id, person), None)
                if self.index is not None:
                    self.index.forget(_folder(id, person))
            for c in held:
                self.events.publish({"type": "deleted", "id": c["id"], "to": c["person"]})
            self._publish_projects()

    def projects_changed(self) -> None:
        """Tells each person's apps of the projects they see, as they changed."""
        with self._lock:
            self._publish_projects()

    def space(self, project: str | None = None, person: int | None = None) -> Workspace:
        """The files of a project the person sees, or their own of no project. Raises NotFound if
        they see no such project, or the agent keeps no files."""
        with self._lock:
            if self.workspace is None:
                raise NotFound("this Leat keeps no files")
            if project is not None:
                self._project(project, person)
            return self._space(project, person)

    def files_changed(self, project: str | None = None, person: int | None = None) -> None:
        """Tells the apps of a space's files, a project's or a person's own, if they changed
        since they were last told, and has the index read them as they now are."""
        with self._lock:
            if self.workspace is not None:
                self._publish_files(project, person, changed=True)
        if self.index is not None:
            self.index.refresh(_folder(project, person))

    def files_event(self, project: str | None = None, person: int | None = None) -> Event:
        """A space's files, a project's or a person's own, as an event."""
        files = self._listing(project, person) if self.workspace else []
        return {"type": "files", "project": project, "files": files}

    def clear(self, person: int) -> None:
        """Deletes a person's own files, as their removal does."""
        if self.workspace is not None:
            self._space(None, person).remove()
            if self.index is not None:
                self.index.forget(_folder(None, person))

    def settle(self) -> None:
        """Moves the files that are no one's into the owner's space, once the box has one: those
        of the workspace before it had spaces, at its top, and those of a box that had no people,
        no one's own, at people/0."""
        if self.workspace is None:
            return
        root, no_one = self.workspace.root, self.workspace.root / "people" / "0"
        owner = next((p["id"] for p in self.store.people() if p["owner"]), None)
        strays = [path for path in root.iterdir() if path.name not in ("people", "projects")]
        if owner is not None and no_one.is_dir():
            strays += list(no_one.iterdir())
        if strays:
            home = self._space(None, owner)
            for path in strays:  # each by a name free there; a hidden one, of saved pages, once
                if not path.name.startswith("."):
                    path.rename(home.root / home.free(path.name))
                elif (home.root / path.name).exists():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.rename(home.root / path.name)
        if owner is not None and no_one.is_dir():
            no_one.rmdir()

    def models(self) -> list[dict[str, Any]]:
        """The engine's models, as it lists them. Raises EngineError."""
        return self.engine.models()

    def loaded(self) -> dict[str, Any]:
        """The loaded model, as the engine lists it, or {} if none is. Raises EngineError."""
        return next((m for m in self.models() if m.get("status") == "loaded"), {})

    def load(self, model: str) -> None:
        """Loads a model in the engine, telling every app as it starts and once it is done."""
        self.events.publish({"type": "loading", "model": model})
        try:
            self.engine.load(model)
        finally:
            self.models_changed(always=True)

    def models_changed(self, always: bool = False) -> None:
        """Tells the apps of the engine's models, if they changed since they were last told, as
        they do when the engine comes up or goes away; or `always`."""
        event = self.models_event()
        with self._lock:
            if always or event != self._models:
                self._models = event
                self.events.publish(event)
                if self.index is not None:  # whose scans a model that sees images may now read
                    self.index.rescan()

    def models_event(self) -> Event:
        """The engine's models, as an event, or why there are none."""
        try:
            return {"type": "models", "models": self.models()}
        except EngineError as e:
            return {"type": "models", "models": [], "error": str(e)}

    def _own(self, id: str, person: int | None) -> dict[str, Any]:
        # a person's conversation, or NotFound if it is not one
        if (c := self._seen(id, person)) is None:
            raise NotFound(f"there is no conversation {id}")
        return c

    def _seen(self, id: str, person: int | None) -> dict[str, Any] | None:
        # a person's conversation, in no project or one they see; or None
        if (c := self.store.conversation(id)) is None or c["person"] != person:
            return None
        if c["project"] is not None and not _sees(self.store.project(c["project"]), person):
            return None
        return c

    def _space(self, project: str | None, person: int | None) -> Workspace:
        # the files of a project, or of a person's own
        assert self.workspace is not None
        return self.workspace.space(_folder(project, person))

    def _listing(self, project: str | None, person: int | None) -> list[dict[str, Any]]:
        # a space's files, each with its state in the index if it is not read: "reading", or
        # "failed", with why
        files = self._space(project, person).files()
        if self.index is None:
            return files
        states = self.index.states(_folder(project, person))
        return [f | states.get(f["name"], {}) for f in files]

    def _indexed(self, folder: str) -> None:
        # tells the apps of a space's files, whose states changed as the index read them
        kind, id = folder.split("/", 1)
        with self._lock:
            if kind == "projects" and self.store.project(id) is not None:
                self._publish_files(id, None, changed=True)
            elif kind == "people":
                self._publish_files(None, int(id) or None, changed=True)

    def _publish_files(
        self, project: str | None, person: int | None, changed: bool = False
    ) -> None:
        # a space's files, to the apps of those who see it, if they `changed` since they were told
        files, folder = self._listing(project, person), _folder(project, person)
        if changed and self._files.get(folder) == files:
            return
        self._files[folder] = files
        event: Event = {"type": "files", "project": project, "files": files}
        if project is None:
            event["to"] = person
        elif not (held := self.store.project(project) or {}).get("shared"):
            event["to"] = held.get("person")
        self.events.publish(event)

    def _project(self, id: str, person: int | None) -> dict[str, Any]:
        # a project the person sees, or NotFound if they see none of that id
        if not _sees(project := self.store.project(id), person):
            raise NotFound(f"there is no project {id}")
        assert project is not None
        return project

    def _summary(self, c: dict[str, Any]) -> dict[str, Any]:
        running, keys = c["id"] in self._turns, ("id", "title", "updated", "project")
        return {k: c[k] for k in keys} | {"running": running}

    def _everyone(self) -> list[int | None]:
        # the box's people, by their ids; before it has any, no one's, None
        return [p["id"] for p in self.store.people()] or [None]

    def _publish_projects(self) -> None:
        # each person's projects, to their apps alone, as a project shared or no longer is seen
        # by others or no longer
        for person in self._everyone():
            event = {"type": "projects", "projects": self.projects(person), "to": person}
            self.events.publish(event)

    def _publish_conversations(self) -> None:
        for person in self._everyone():
            listed = [self._summary(c) for c in self.store.conversations(person)]
            self.events.publish({"type": "conversations", "conversations": listed, "to": person})

    def _publish_summary(self, id: str) -> None:
        if (c := self.store.conversation(id)) is not None:
            event = {"type": "conversation", "conversation": self._summary(c)}
            self.events.publish(event | {"to": c["person"]})

    def _publish_message(
        self, id: str, person: int | None, index: int, message: dict[str, Any]
    ) -> None:
        event = {"type": "message", "conversation": id, "index": index, "to": person}
        self.events.publish(event | {"message": copy.deepcopy(message)})


class _Turn:
    """A turn running in a person's conversation, of their message at `start`."""

    def __init__(
        self, agent: Agent, id: str, person: int | None, project: str | None, start: int,
        content: str, effort: str | None,
    ):  # fmt: skip
        self.agent, self.id, self.person, self.start = agent, id, person, start
        self.project = project
        # the files its tools work among: its project's, or its person's own
        self.workspace = agent._space(project, person) if agent.workspace else None
        self.content, self.effort = content, effort  # asked of the model, or its default if None
        self.tools = agent.tools  # the tools the model calls
        self.state = agent.store.context(id)  # the prompt's, as it was, to take the turn back to
        self.kept = start + 1  # the conversation's messages kept
        self.live: list[dict[str, Any]] = []  # those after, to keep: a reply, or its calls running
        self.stopped = threading.Event()
        self.completion: Completion | None = None
        self.limit: int | None = None  # the model's context, once the turn asks
        self.image = 0  # the tokens an image takes at most, of a model that sees them, once asked
        self.redone = False  # a reply the context cut off, after the prompt was made smaller
        self.empty = 0  # times the reply was asked again, as it said nothing
        self.sources: dict[str, int] = {}  # the conversation's, by address, numbered for citing

    def stop(self) -> None:
        self.stopped.set()
        if self.completion is not None:
            self.completion.close()

    def run(self) -> None:
        try:
            loaded = self.agent.loaded()
            self.limit, self.image = loaded.get("max_context"), loaded.get("image_tokens") or 0
            self.sources = numbered(self.agent.store.messages(self.id))
            for n in range(ROUNDS):
                calls = self._reply(last=n == ROUNDS - 1)
                if not calls or self.stopped.is_set():
                    break
                self._call(calls)
                self.agent.files_changed(self.project, self.person)  # as a call may have
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
        # makes of the agent's tools. The turn's `last` reply is told to answer, and may call
        # none, with tool_choice "none": the tools still declared, so that its prompt extends the
        # last, as gpt-oss, told to answer, would call them still. The prompt is made smaller
        # first if it outgrows its share of the context, and again, the reply redone, if the
        # context cuts the reply off.
        a = self.agent
        messages, state = a.store.messages(self.id), a.store.context(self.id)
        declared = [tool.declaration() for tool in self.tools.values()]
        extra = len(json.dumps(declared))
        if self.limit and self._estimate(messages, state, extra) > context.COMPACT * self.limit:
            state = self._compact(messages, state, extra)
        reply: dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": ""}
        info: dict[str, Any] = {}
        reply["info"] = info
        (index,) = self._show(reply)
        read = context.prompt(messages, state)
        # sampled as leat serve has the model, as its makers recommend it, or as set there
        body: dict[str, Any] = {"messages": self._seen(read)}
        if self.effort:
            body["reasoning_effort"] = self.effort
        # as Qwen3's templates take them, and others ignore: the replies of turns before rendered
        # as they were, their reasoning kept, so that the prompt of a message after a turn that
        # called tools extends the last, which the engine's cache holds, rather than changing it
        body["chat_template_kwargs"] = {"preserve_thinking": True}
        if declared:
            body["tools"] = declared
        if last and declared:
            body["messages"].append({"role": "user", "content": LAST})
            body["tool_choice"] = "none"
        started, finish = time.monotonic(), None
        chunks: Iterable[dict[str, Any]] = ()
        if not self.stopped.is_set():  # as it may be, while the prompt was made smaller
            chunks = self.completion = a.engine.complete(body, self.person)
            if self.stopped.is_set():  # before the completion was there to close
                self.completion.close()
        for chunk in chunks:
            choice = chunk["choices"][0] if chunk.get("choices") else {}
            delta, finish = choice.get("delta") or {}, choice.get("finish_reason") or finish
            with a._lock:
                info["model"] = chunk.get("model")
                for key in ("reasoning_content", "content"):
                    if text := delta.get(key):
                        info.setdefault("first", time.monotonic() - started)
                        event = {"type": "delta", "conversation": self.id, "index": index,
                                 "to": self.person}  # fmt: skip
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
        said = reply["content"].strip() or reply.get("tool_calls")
        if not said and finish == "stop" and self.empty < EMPTY and not self.stopped.is_set():
            self.empty += 1  # asked again, of the same prompt, which the engine's cache holds
            with a._lock:
                self.live.remove(reply)
            return self._reply(last)
        self.empty = 0
        with a._lock:
            if last:  # of a server that calls tools all the same, none answered
                reply.pop("tool_calls", None)
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

        def summarize(before: str | None, span: list[dict[str, Any]], tokens: int) -> str:
            return self._summarize(before, span, tokens, messages, state, extra)

        image = self.image or context.IMAGE
        try:
            smaller = context.compact(messages, state, self.limit, extra, summarize, force, image)
        except _Stopped:  # while summarizing: the turn ends as it is
            return state
        keys = ("cleared", "summarized")
        if [smaller.get(k) for k in keys] == [state.get(k) for k in keys]:
            return state
        a = self.agent
        with a._lock:
            a.store.set_context(self.id, smaller)
            if (summarized := smaller.get("summarized")) != state.get("summarized"):
                event = {"type": "compacted", "conversation": self.id, "summarized": summarized}
                a.events.publish(event | {"to": self.person})
        return smaller

    def _summarize(
        self, before: str | None, span: list[dict[str, Any]], tokens: int,
        messages: list[dict[str, Any]], state: context.State, extra: int,
    ) -> str:  # fmt: skip
        # a summary, in `tokens` at most, of the messages of `span`, with that of those before
        # them, by the model: asked at the end of the conversation's prompt, as the engine's cache
        # holds it, if it and the summary fit the context, as Claude Code asks its own; or of the
        # span alone, as a transcript, the latest of it that the context holds
        assert self.limit is not None
        asking = len(context.IN_PLACE) // context.CHARS
        if self._estimate(messages, state, extra) + asking + tokens < self.limit:
            note = {"role": "user", "content": context.IN_PLACE}
            body = {
                "messages": [*self._seen(context.prompt(messages, state)), note],
                "max_tokens": tokens,
                "temperature": 0.3, "tools": [t.declaration() for t in self.tools.values()],
                "reasoning_effort": "none", "chat_template_kwargs": {"preserve_thinking": True},
            }  # fmt: skip
            reply = self._whole(body)
            if (summary := reply["content"].strip()) and not reply.get("tool_calls"):
                return summary
        text, so_far = context.transcript(span), f"The summary so far:\n{before}\n\n"
        room = (self.limit - tokens) * context.CHARS
        room -= len(context.SUMMARIZE) + len(so_far if before else "") + 200
        latest = text[max(0, len(text) - room) :]
        user = (so_far + "What came after:\n\n" if before else "") + latest
        body = {
            "messages": [{"role": "system", "content": context.SUMMARIZE},
                         {"role": "user", "content": user}],
            "max_tokens": tokens, "temperature": 0.3, "reasoning_effort": "none",
        }  # fmt: skip
        return self._whole(body)["content"].strip() or (before or "")

    def _whole(self, body: dict[str, Any]) -> dict[str, Any]:
        # the model's reply to body, whole, which a stop cuts short: raises _Stopped then
        self.completion = self.agent.engine.complete(body, self.person)
        if self.stopped.is_set():  # before the completion was there to close
            self.completion.close()
        reply = whole(self.completion)
        if self.stopped.is_set():
            raise _Stopped
        return reply

    def _seen(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # the prompt's messages, each image a message shows before its text as a data: URL if
        # the model sees images; a tool's said to be unseen if it does not, or gone if the
        # workspace no longer has it, as the user's are named
        for m in messages:
            if not (names := m.pop("images", None)):
                continue
            urls = [url for name in names if (url := self._url(name))] if self.image else []
            if urls:
                parts = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
                m["content"] = [*parts, {"type": "text", "text": m.get("content") or ""}]
            elif m["role"] == "tool":
                m["content"] += (" It is no longer in the workspace." if self.image
                                 else " You cannot see it: the model takes no images.")  # fmt: skip
        return messages

    def _estimate(self, messages: list[dict[str, Any]], state: context.State, extra: int) -> int:
        return context.estimate(messages, state, extra, self.image or context.IMAGE)

    def _url(self, name: str) -> str | None:
        # a workspace's image as a data: URL, or None if it is gone
        if (space := self.workspace) is None:
            return None
        try:
            data = space.path(name).read_bytes()
        except (OSError, ValueError):
            return None
        kind = mimetypes.guess_type(name)[0] or "image/png"
        return f"data:{kind};base64,{base64.b64encode(data).decode()}"

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
            if (tool := self.tools.get(message["name"])) is None:
                raise ValueError(f"there is no tool {message['name']!r}")
            called = arguments(message["info"]["arguments"])
            progress = functools.partial(self._progress, message, index)
            context = Context(
                self.id, self._cite, self.person, self.workspace, progress, self.stopped
            )
            result = tool.run(context, **called)
        except Exception as e:  # for the model, which may try again
            result = Result(f"error: {e}", {"error": str(e)})
        with a._lock:
            if not message["content"]:
                message["content"] = result.content or "(nothing)"
                message["info"] |= result.info
                a._publish_message(self.id, self.person, index, message)
        answered.put(index)

    def _progress(self, message: dict[str, Any], index: int, info: dict[str, Any]) -> None:
        # shows how far a call is, in its message's info, while it runs
        with self.agent._lock:
            if not message["content"]:
                message["info"] |= info
                self.agent._publish_message(self.id, self.person, index, message)

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
                self.agent._publish_message(self.id, self.person, indexes[-1], message)
            return indexes

    def _keep(self) -> None:
        # keeps the messages shown, unless the conversation is gone
        a = self.agent
        with a._lock:
            if a._turns.get(self.id) is not self:
                raise _Deleted
            a.store.append(self.id, *self.live)
            for i, message in enumerate(self.live):
                a._publish_message(self.id, self.person, self.kept + i, message)
            self.kept += len(self.live)
            self.live = []

    def _end(self) -> None:
        a = self.agent
        with a._lock:
            if a._turns.get(self.id) is self:
                del a._turns[self.id]
                a._publish_summary(self.id)
        if a.background is not None:
            a.background.wake()

    def _take_back(self, error: str) -> None:
        # removes the turn's messages, from `start`, and the conversation it began, and puts the
        # prompt's state back as it was; tells the apps why, and what the user wrote, to send
        # again; and has the engine looked at now, which may have gone away, so that the apps are
        # told so, and again once it is back, rather than at the next look
        a = self.agent
        if a.background is not None:
            a.background.wake()
        with a._lock:
            if a._turns.get(self.id) is not self:
                return
            del a._turns[self.id]
            self.live = []
            event = {"type": "error", "conversation": self.id, "error": error, "start": self.start}
            a.events.publish(event | {"content": self.content, "to": self.person})
            if self.start == 1:  # its first turn
                a.store.delete(self.id)
                a.events.publish({"type": "deleted", "id": self.id, "to": self.person})
            else:
                summarized = a.store.context(self.id).get("summarized")
                a.store.truncate(self.id, self.start)
                a.store.set_context(self.id, self.state)
                if (before := self.state.get("summarized")) != summarized:
                    event = {"type": "compacted", "conversation": self.id, "summarized": before}
                    a.events.publish(event | {"to": self.person})
                a._publish_summary(self.id)


class _Deleted(Exception):
    """The turn's conversation was deleted."""


class _Stopped(Exception):
    """The turn was stopped while the model answered other than in its reply."""


def _system(
    workspace: bool, name: str | None = None, project: dict[str, Any] | None = None,
    tools: Collection[str] = (),
) -> dict[str, Any]:  # fmt: skip
    # the system prompt of a conversation begun now with the user, of a `name` if the box has
    # people, the workspace's tools if `workspace`, and of those named in `tools` search_files's
    # and ask_files's, and the project it is held in, if one
    skills = "\n".join(f"- {path}: {about}" for path, about in files.skills())
    search, ask = SEARCH * ("search_files" in tools), ASK * ("ask_files" in tools)
    space = WORKSPACE.format(skills=skills, search=search, ask=ask) if workspace else ""
    named = f" The user is {name}." if name else ""
    content = SYSTEM.format(named=named, workspace=space)
    if project is not None:
        text = project["instructions"].strip()
        instructions = INSTRUCTIONS.format(text=text) if text else ""
        content += PROJECT.format(name=project["name"], instructions=instructions)
    return {"role": "system", "content": content}


def _folder(project: str | None, person: int | None) -> str:
    # the folder of a space in the workspace: a project's, or a person's own, people/0 of no one's
    return f"projects/{project}" if project is not None else f"people/{person or 0}"


def _sees(project: dict[str, Any] | None, person: int | None) -> bool:
    # whether a person sees a project: their own, or one shared
    return project is not None and (bool(project["shared"]) or project["person"] == person)


def _project(p: dict[str, Any]) -> dict[str, Any]:
    # a project as the apps show it
    keys = ("id", "name", "instructions", "person", "updated")
    return {k: p[k] for k in keys} | {"shared": bool(p["shared"])}


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
