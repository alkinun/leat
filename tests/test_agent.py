import contextlib
import datetime
import http.client
import http.cookiejar
import json
import queue
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import ANY

import pytest

from leat.agent import agent as agent_module
from leat.agent import background, context
from leat.agent.agent import LAST, Agent, Busy, NotFound
from leat.agent.client import Client, EngineError
from leat.agent.context import message as _api
from leat.agent.server import Server
from leat.agent.store import Store
from leat.agent.tools import Result, Tool, memory, strings

HOLD = None  # in a scripted reply: wait there until the test releases it


class FakeEngine(ThreadingHTTPServer):
    """leat serve's API, streaming scripted replies in turn: each a list of deltas, or the message
    of an error to answer with."""

    daemon_threads = True

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []  # each completion's body
        self.replies: queue.SimpleQueue[list[dict[str, Any] | None] | str] = queue.SimpleQueue()
        self.released = threading.Event()
        self.loads: list[str] = []
        self.context: int | None = None  # the model's, said only if set
        self.status = "loaded"  # the model's
        self.vision = False  # whether it sees images
        super().__init__(("127.0.0.1", 0), _FakeHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _FakeHandler(BaseHTTPRequestHandler):
    server: FakeEngine

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        model = {"id": "fake", "status": self.server.status}
        if self.server.vision:
            model |= {"vision": True, "image_tokens": 300}
        if self.server.context:
            model["max_context"] = self.server.context
        self._json(200, {"object": "list", "data": [model]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/models/load":
            self.server.loads.append(body["model"])
            return self._json(200, {"id": body["model"], "status": "loaded"})
        self.server.requests.append(body)
        reply = self.server.replies.get(timeout=5)
        if isinstance(reply, str):
            return self._json(400, {"error": {"message": reply}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        finish = "stop"  # unless a delta says otherwise, as {"finish_reason": "length"}
        with contextlib.suppress(OSError):  # a client that stopped the reply
            for delta in reply:
                if delta is HOLD:
                    self.server.released.wait(timeout=5)
                elif "finish_reason" in delta:
                    finish = delta["finish_reason"]
                else:
                    self._chunk({"choices": [{"index": 0, "delta": delta}]})
            # the prompt's tokens, three characters each, none cached
            read = len(json.dumps(body["messages"])) // 3
            timings = {"predicted_n": 5, "predicted_ms": 100.0, "cache_n": 0, "prompt_n": read}
            choice = {"index": 0, "delta": {}, "finish_reason": finish}
            self._chunk({"choices": [choice], "timings": timings})
            self.wfile.write(b"data: [DONE]\n\n")

    def _chunk(self, chunk: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps({'model': 'fake'} | chunk)}\n\n".encode())

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def echo(context, text: str) -> Result:
    return Result(f"echo: {text}", {"echoed": text})


@pytest.fixture
def agent(engine, tmp_path) -> Agent:
    tool = Tool("echo", "Says the text again", strings(text="what to say"), echo)
    return Agent(Store(tmp_path / "leat.db"), Client(engine.url), [tool])


@pytest.fixture
def events(agent) -> Iterator[queue.SimpleQueue]:
    with agent.events.watch() as events:
        yield events


def until(events: queue.SimpleQueue, done: Callable[[dict], bool]) -> list[dict]:
    """The events up to the first that is `done`."""
    seen: list[dict] = []
    while not seen or not done(seen[-1]):
        seen.append(events.get(timeout=5))
    return seen


def stamped(agent: Agent, id: str, index: int) -> str:
    """The time a stored message of the user's begins with, as the model reads it."""
    return context.stamp(agent.store.messages(id)[index]["info"]["at"]) + "\n"


def ended(event: dict) -> bool:
    # the end of a turn: its conversation no longer running, or gone
    c = event.get("conversation")
    return event["type"] == "deleted" or event["type"] == "conversation" and not c["running"]


REPLY = [{"reasoning_content": "Hmm."}, {"content": "Hello"}, {"content": " there."}]


def call(name: str, arguments: Any, id: str = "call_1") -> dict:
    """A reply's call of a tool, as leat serve sends it, its arguments as JSON text."""
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"index": 0, "id": id, "type": "function", "function": {"name": name, "arguments": text}}


def test_turn(agent, engine, events):
    # a message starts a conversation, whose reply streams into it and stays
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi\nand more")
    seen = until(events, ended)
    assert [e["type"] for e in seen[:3]] == ["conversation", "message", "message"]
    assert seen[0]["conversation"] | {"updated": 0} == {
        "id": id, "title": "Hi", "updated": 0, "running": True, "character": None,
        "shared": False}  # fmt: skip
    user = {"role": "user", "content": "Hi\nand more",
            "info": {"think": False, "at": pytest.approx(time.time(), abs=5)}}  # fmt: skip
    assert seen[1]["message"] == user
    assert seen[2]["index"] == 2 and seen[2]["message"]["content"] == ""
    messages = agent.store.messages(id)
    system, user, reply = messages
    assert system["role"] == "system"
    assert "Each of the user's messages begins with the date and time" in system["content"]
    assert (reply["reasoning_content"], reply["content"]) == ("Hmm.", "Hello there.")
    info = reply["info"]
    assert info["model"] == "fake" and info["tokens"] == 5 and info["rate"] == pytest.approx(40.0)
    read = len(json.dumps(engine.requests[0]["messages"])) // 3
    assert info["first"] >= 0 and (info["cached"], info["read"]) == (0, read)
    assert seen[-2] == {"type": "message", "conversation": id, "index": 2, "message": reply,
                        "to": None}  # fmt: skip
    # the model read the system prompt and the message, not thinking, sampled as Qwen3.6
    # recommends then
    (request,) = engine.requests
    assert request["messages"] == [_api(m) for m in messages[:2]] and request["stream"] is True
    assert request["messages"][1]["content"] == stamped(agent, id, 1) + "Hi\nand more"
    assert request["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": True}
    assert request["temperature"] == 0.7 and request["presence_penalty"] == 1.5


def test_think(agent, engine, events):
    # a message may ask the model to think first, sampled as Qwen3.6 recommends for thinking
    engine.replies.put(REPLY)
    id = agent.send(None, "Prove it.", think=True)
    until(events, ended)
    assert engine.requests[-1]["chat_template_kwargs"] == {
        "enable_thinking": True, "preserve_thinking": True}  # fmt: skip
    assert engine.requests[-1]["temperature"] == 1.0
    assert agent.store.messages(id)[1]["info"]["think"] is True


def test_deltas(agent, engine, events):
    # each delta says where its text goes, so that an app that has some already, read with the
    # conversation, takes only what it lacks
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    deltas = [e for e in until(events, ended) if e["type"] == "delta"]
    reply = {"reasoning_content": "", "content": ""}
    for delta in deltas + deltas[1:]:  # some twice
        reply[delta["key"]] = reply[delta["key"]][: delta["at"]] + delta["text"]
    assert reply == {k: agent.store.messages(id)[2][k] for k in reply}


def test_next_turn(agent, engine, events):
    # the next turn sends the conversation so far, the reply's reasoning too but not its info
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    until(events, ended)
    engine.replies.put([{"content": "Fine."}])
    assert agent.send(id, "How are you?") == id
    until(events, ended)
    sent = engine.requests[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1] == {"role": "user", "content": stamped(agent, id, 1) + "Hi"}
    assert sent[2] == {"role": "assistant", "content": "Hello there.", "reasoning_content": "Hmm."}
    assert [m["content"] for m in agent.store.messages(id)][3:] == ["How are you?", "Fine."]
    # and the conversations, the latest updated first, are kept for the next run
    other = agent.send(None, "Another")
    engine.replies.put(REPLY)
    until(events, ended)
    again = Agent(agent.store, agent.engine)
    assert [c["id"] for c in again.conversations()] == [other, id]


def test_tools(agent, engine, events):
    # a reply's calls run, and the model replies again, reading their answers, until it calls none
    engine.replies.put([{"content": "Let me see."}, {"tool_calls": [call("echo", {"text": "hi"})]}])
    engine.replies.put([{"content": "It said hi."}])
    id = agent.send(None, "Echo hi")
    until(events, ended)
    _, _, step, answer, reply = agent.store.messages(id)
    function = {"name": "echo", "arguments": '{"text": "hi"}'}
    assert step["content"] == "Let me see."
    assert step["tool_calls"] == [{"id": "call_1", "type": "function", "function": function}]
    assert answer == {
        "role": "tool", "tool_call_id": "call_1", "name": "echo", "content": "echo: hi",
        "info": {"arguments": {"text": "hi"}, "echoed": "hi"},
    }  # fmt: skip
    assert reply["content"] == "It said hi."
    first, second = engine.requests
    names = ["remember", "forget", "recall", "schedule", "unschedule", "tasks", "add_to_list",
             "check_off", "lists", "echo"]  # fmt: skip
    assert [tool["function"]["name"] for tool in first["tools"]] == names
    assert second["messages"][-1] == _api(answer)


def test_bad_calls(agent, engine, events):
    # a call the tools cannot answer is answered with why, for the model to try again
    calls = [call("nothing", {}, "a"), call("echo", "{not JSON", "b"), call("echo", [1], "c"),
             call("echo", {"words": "hi"}, "d")]  # fmt: skip
    engine.replies.put([{"tool_calls": calls}])
    engine.replies.put([{"content": "Sorry."}])
    id = agent.send(None, "Hi")
    until(events, ended)
    answers = [m["content"] for m in agent.store.messages(id) if m["role"] == "tool"]
    assert answers[0] == "error: there is no tool 'nothing'"
    assert answers[1] == "error: the arguments must be a JSON object, not '{not JSON'"
    assert answers[2] == "error: the arguments must be a JSON object, not [1]"
    assert answers[3].startswith("error: ") and "'words'" in answers[3]


def test_calls_at_once(agent, engine, events):
    # a reply's calls run at once: each of these waits for the other
    both = threading.Barrier(2, timeout=5)
    meet = Tool("meet", "Waits for another", strings(), lambda c: Result(str(both.wait())))
    agent.tools["meet"] = meet
    engine.replies.put([{"tool_calls": [call("meet", {}, "a"), call("meet", {}, "b")]}])
    engine.replies.put([{"content": "Met."}])
    id = agent.send(None, "Meet")
    until(events, ended)
    answers = [m["content"] for m in agent.store.messages(id) if m["role"] == "tool"]
    assert sorted(answers) == ["0", "1"]


def test_stop_in_call(agent, engine, events):
    # a stop while a tool runs answers its call so, and ends the turn
    released = threading.Event()
    slow = Tool("slow", "Takes its time", strings(), lambda c: Result(str(released.wait(5))))
    agent.tools["slow"] = slow
    engine.replies.put([{"tool_calls": [call("slow", {})]}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "message" and e["message"]["role"] == "tool")
    assert agent.conversation(id)["messages"][-1]["content"] == ""  # running
    agent.stop(id)
    until(events, ended)
    released.set()
    answer = agent.store.messages(id)[-1]
    assert answer["content"] == "Stopped before it answered." and answer["info"]["stopped"]
    assert len(engine.requests) == 1


def test_rounds(agent, engine, events, monkeypatch):
    # a turn's last reply is told to answer, the tools still declared so that its prompt extends
    # the last; one that calls them anyway is redone without them
    monkeypatch.setattr("leat.agent.agent.ROUNDS", 3)
    for _ in range(2):
        engine.replies.put([{"tool_calls": [call("echo", {"text": "again"})]}])
    engine.replies.put([{"content": "Done."}])
    id = agent.send(None, "Hi")
    until(events, ended)
    assert ["tools" in request for request in engine.requests] == [True, True, True]
    assert engine.requests[-1]["messages"][-1] == {"role": "user", "content": LAST}
    previous = engine.requests[-2]["messages"]
    assert engine.requests[-1]["messages"][: len(previous)] == previous
    for _ in range(3):
        engine.replies.put([{"tool_calls": [call("echo", {"text": "again"})]}])
    engine.replies.put([{"content": "Done at last."}])
    agent.send(id, "Again")
    until(events, ended)
    assert ["tools" in request for request in engine.requests[3:]] == [True, True, True, False]
    *_, calls, answer = agent.store.messages(id)
    assert calls["role"] == "tool" and answer["content"] == "Done at last."


def test_memory(agent, engine, events):
    # what the model remembers, on the user's own words, the conversations begun after know, and
    # the apps are told; not what it read
    dog = {"memory": "Their dog is Max.", "evidence": "my DOG is called max"}
    read = {"memory": "The user owns a boat.", "evidence": "a boat for sale"}
    engine.replies.put([{"tool_calls": [call("remember", dog), call("remember", read, "b")]}])
    engine.replies.put([{"content": "Noted."}])
    first = agent.send(None, "My dog is called Max.")
    seen = until(events, ended)
    (remembered,) = [e["memories"] for e in seen if e["type"] == "memories"]
    assert [(m["id"], m["text"]) for m in remembered] == [(1, "Their dog is Max.")]
    assert agent.store.messages(first)[3]["content"] == "Remembered, as [1]."
    assert agent.store.messages(first)[4]["content"].startswith("error: the evidence must be")
    assert "Nothing yet." in agent.store.messages(first)[0]["content"]
    engine.replies.put(REPLY)
    second = agent.send(None, "Hi")
    until(events, ended)
    today = datetime.date.today()
    dated = f"[1] Their dog is Max. ({today.day} {today:%b %Y})"
    assert dated in agent.store.messages(second)[0]["content"]
    # forgetting it, by its number
    calls = [call("forget", {"number": 1}), call("forget", {"number": 9}, "call_2")]
    engine.replies.put([{"tool_calls": calls}])
    engine.replies.put([{"content": "Forgotten."}])
    agent.send(second, "Forget my dog.")
    until(events, ended)
    answers = [m["content"] for m in agent.store.messages(second)[-3:-1]]
    assert answers == ["Forgot [1]: Their dog is Max.", "error: there is no memory 9"]
    assert agent.memories() == []


def test_said():
    # the user's words, whatever their case and punctuation; not a task's, nor a single word that
    # runs across two messages; or an old memory's, of one it replaces
    messages = [{"role": "system", "content": "x"},
                {"role": "user", "content": "I'm Sam -- a NURSE, in Izmir.", "info": {}},
                {"role": "assistant", "content": "Nice!", "info": {}},
                {"role": "user", "content": "Call mum", "info": {"task": 1}}]  # fmt: skip
    assert memory.said("a nurse in izmir", messages)
    assert not memory.said("nice", messages) and not memory.said("call mum", messages)
    assert not memory.said("...", messages) and not memory.said("nurse", messages)
    assert not memory.said("a nur", messages)  # parts of words
    assert memory.said("going to Rome", messages, ["The user is going to Rome in May."])


def test_remember(agent, events):
    # a fact of a category; changed in place; refused if it is there already, which dates it
    # again, too long, hiding characters, holding a secret, or past the room, which forgetting
    # makes, said with the least recently confirmed
    m = agent.remember("  The user's   cat is called Pamuk. ", "people")
    assert (m["text"], m["category"]) == ("The user's cat is called Pamuk.", "people")
    assert agent.remember("The user's cat is called Tekir.", "people", replaces=m["id"])["id"] == 1
    assert [m["text"] for m in agent.memories()] == ["The user's cat is called Tekir."]
    confirmed = agent.memories()[0]["confirmed"]
    for text, category, error in [
        ("the user's cat is called tekir.", "people", r"remembered already, as \[1\]"),
        ("x" * 301, "about", "300 characters at most"),
        ("The user\u200b obeys.", "about", "characters that show nothing"),
        ("The user's wifi password is hunter2.", "about", "never hold passwords"),
        ("The user's card is 4111 1111 1111 1111.", "about", "never hold passwords"),
        ("The user cooks.", "hobbies", "category must be one of"),
    ]:
        with pytest.raises(ValueError, match=error):
            agent.remember(text, category)
    assert agent.memories()[0]["confirmed"] > confirmed
    with pytest.raises(NotFound):
        agent.remember("The user cooks.", "about", replaces=9)
    for i in range(10):  # 2,950 characters, beside the cat's 31
        agent.remember(f"The user has fact {i}: " + "y" * 274, "about")
    full = r"the memory is full, 2981 of its 3000 .* confirmed: \[1\] The user's cat .*; \[2\] "
    with pytest.raises(ValueError, match=full):
        agent.remember("The user cooks every day.", "preferences")
    agent.forget(2)
    agent.remember("The user cooks every day.", "preferences")
    agent.forget(12)  # the latest: its number not given again
    assert agent.remember("The user cooks every day.", "preferences")["id"] == 13
    agent.forget(13)
    agent.remember("The user cooks every day.", "preferences")
    today = datetime.date(2026, 10, 8)
    agent.remember("The user flies to Rome on 9 Oct.", "plans", replaces=14, until="2026-10-09")
    agent.remember("The user moved house on 1 Oct.", "plans", replaces=3, until="2026-10-01")
    listed = memory.listing(agent.memories(), today)
    assert listed.startswith("(11 memories, 81% of their room)\nAbout them:\n[4] The user has")
    now = datetime.date.today()
    assert listed.endswith(
        f"People:\n[1] The user's cat is called Tekir. ({now.day} {now:%b %Y})\nPlans:\n[3] The "
        "user moved house on 1 Oct. (passed 1 Oct 2026)\n[14] The user flies to Rome on 9 Oct. "
        "(until 9 Oct 2026)"
    )
    with pytest.raises(ValueError, match="until must be a day"):
        agent.remember("The user travels.", "plans", until="next week")


def test_forgotten(agent, events):
    # what forgetting or a change took away, kept as it was, the latest first, to restore
    agent.remember("The user lives in Izmir.", "about")
    agent.remember("The user lives in Ankara.", "about", replaces=1, by="conversation")
    agent.remember("The user has a cat.", "people")
    agent.forget(2, by="review")
    changed, forgotten = agent.store.forgotten()
    assert (changed["text"], changed["change"], changed["by"]) == (
        "The user has a cat.",
        "forgotten",
        "review",
    )
    assert (forgotten["text"], forgotten["change"]) == ("The user lives in Izmir.", "replaced")
    assert agent.restore(changed["id"])["text"] == "The user has a cat."
    assert agent.restore(forgotten["id"])["text"] == "The user lives in Izmir."
    assert [m["text"] for m in agent.memories()] == ["The user lives in Izmir.",
                                                     "The user has a cat."]  # fmt: skip
    assert agent.store.forgotten() == []
    with pytest.raises(NotFound):
        agent.restore(forgotten["id"])
    event = agent.memories_event()
    assert event["type"] == "memories" and event["forgotten"] == []
    # a household's memory one person made their own: its old text, everyone's to see, no other
    # may restore over theirs; they may, the household's again
    ada, bo = (agent.store.add_person(name)["id"] for name in ("Ada", "Bo"))
    shared = agent.remember("The family has a dog.", "household")
    agent.remember("I walk the dog.", "about", replaces=shared["id"], person=ada)
    (old,) = agent.store.forgotten(bo)
    with pytest.raises(NotFound, match="no longer yours"):
        agent.restore(old["id"], person=bo)
    assert agent.memories(ada)[-1]["text"] == "I walk the dog."
    restored = agent.restore(old["id"], person=ada)
    assert (restored["text"], restored["person"]) == ("The family has a dog.", None)


def test_household(agent, engine, events):
    # all that was no one's becomes the first person's, the owner's; then each person's
    # conversations, memories and tasks are their own, the household's memories everyone's, and
    # each event says whose it is
    engine.replies.put(REPLY)
    before = agent.send(None, "Hi")
    until(events, ended)
    agent.remember("The user lives in Izmir.", "about")
    owner, ada = agent.store.add_person("Alkın"), agent.store.add_person("Ada")
    assert (owner["owner"], ada["owner"]) == (1, 0)
    assert [c["id"] for c in agent.conversations(owner["id"])] == [before]
    assert agent.conversations(ada["id"]) == [] and agent.conversations() == []
    engine.replies.put(REPLY)
    with agent.events.watch() as told:
        mine = agent.send(None, "Hello", person=ada["id"])
        assert {e.get("to") for e in until(told, ended)} == {ada["id"]}
    assert "the home of the user, Ada:" in agent.store.messages(mine)[0]["content"]
    for other in (owner["id"], None):
        assert agent.conversation(mine, other) is None
        with pytest.raises(NotFound):
            agent.send(mine, "Hi", person=other)
        with pytest.raises(NotFound):
            agent.delete(mine, other)
        with pytest.raises(NotFound):
            agent.stop(mine, other)
    # memories: Ada's, the household's, which Ada and the owner both know, and not the owner's
    agent.remember("The user plays the violin.", "about", person=ada["id"])
    with agent.events.watch() as told:
        cat = agent.remember("The household's cat is called Pamuk.", "household", person=ada["id"])
        assert {told.get(timeout=1)["to"] for _ in range(2)} == {owner["id"], ada["id"]}
    assert cat["person"] is None
    known = lambda person: [m["text"] for m in agent.memories(person)]  # noqa: E731
    assert known(ada["id"]) == [
        "The user plays the violin.",
        "The household's cat is called Pamuk.",
    ]
    assert known(owner["id"]) == [
        "The user lives in Izmir.",
        "The household's cat is called Pamuk.",
    ]
    izmir = agent.memories(owner["id"])[0]["id"]
    with pytest.raises(NotFound):
        agent.forget(izmir, person=ada["id"])
    with pytest.raises(NotFound):
        agent.remember("The user lives in Rome.", "about", replaces=izmir, person=ada["id"])
    agent.forget(cat["id"], person=owner["id"])  # the household's: anyone's to forget
    (gone,) = agent.store.forgotten(ada["id"])
    assert agent.restore(gone["id"], ada["id"])["text"] == "The household's cat is called Pamuk."
    # tasks: each person's own
    soon = datetime.datetime.now() + datetime.timedelta(hours=1)
    task = agent.schedule("Practise", soon, "once", mine, ada["id"])
    assert agent.tasks(owner["id"]) == [] and [t["id"] for t in agent.tasks(ada["id"])] == [
        task["id"]
    ]
    with pytest.raises(NotFound):
        agent.unschedule(task["id"], owner["id"])
    # recall finds a person's own conversations alone
    found = agent.store.search("hello hi", person=owner["id"])
    assert {f["conversation"] for f in found} == {before}
    # a child's conversations begun after keep to a child's rules, a character's too
    assert agent.store.set_child(ada["id"], True) and not agent.store.set_child(owner["id"], True)
    tutor = agent.add_character("Ms Ada", "A tutor.")
    for character in (None, tutor["id"]):
        engine.replies.put(REPLY)
        id = agent.send(None, "Hi", person=ada["id"], character=character)
        until(events, ended)
        assert agent.store.messages(id)[0]["content"].endswith(agent_module.CHILD)


def test_character(agent, engine, events):
    # a conversation with one of the household's characters: their system prompt, knowing the
    # user's memories but without the memory's tools, nor reviewed; going on once they are removed
    agent.remember("The user is 9.", "about")
    with pytest.raises(ValueError, match="needs a name"):
        agent.add_character(" ", "Someone.")
    tutor = agent.add_character("Ms Ada", "A patient maths tutor, who asks before she tells.")
    assert agent.characters() == [tutor]
    with pytest.raises(NotFound):
        agent.send(None, "Hi", character=9)
    engine.replies.put([{"tool_calls": [call("remember", {"memory": "x", "evidence": "x"})]}])
    engine.replies.put([{"content": "What do you think 7 times 8 is?"}])
    id = agent.send(None, "What is 7 times 8?", character=tutor["id"])
    until(events, ended)
    system = agent.store.messages(id)[0]["content"]
    assert system.startswith("You are Ms Ada, a character that Leat plays")
    assert "A patient maths tutor" in system and "[1] The user is 9." in system
    names = [t["function"]["name"] for t in engine.requests[0]["tools"]]
    assert "remember" not in names and "forget" not in names and "recall" in names
    assert agent.store.messages(id)[3]["content"] == "error: there is no tool 'remember'"
    assert agent.conversations()[0]["character"] == tutor["id"]
    background.review(agent, id)  # no request: nothing of a roleplay is remembered
    assert len(engine.requests) == 2 and agent.store.reviewed(id) == 5
    agent.remove_character(tutor["id"])
    engine.replies.put([{"content": "Right!"}])
    agent.send(id, "56")
    until(events, ended)
    assert agent.conversations()[0]["character"] == tutor["id"] and agent.characters() == []
    assert "remember" not in [t["function"]["name"] for t in engine.requests[-1]["tools"]]
    for removed in (lambda: agent.remove_character(tutor["id"]),
                    lambda: agent.send(None, "Hi", character=tutor["id"])):  # fmt: skip
        with pytest.raises(NotFound):
            removed()


def test_lists(agent, events):
    # the household's lists: made as things are added, each once, whatever their case; checked
    # off by name or by the one that holds it; every app told
    from leat.agent.tools import lists

    added = lists.add(agent, "shopping list", ["Milk", "2 kg of apples", "milk", " "])
    assert added.content == "Added to Shopping: Milk, 2 kg of apples. It has 2 things."
    assert events.get(timeout=1)["lists"][0]["name"] == "Shopping"
    assert lists.add(agent, "Shopping", "bread, MILK").content.startswith(
        "Added to Shopping: bread."
    )
    done = lists.check_off(agent, "shopping", ["apples", "eggs"])
    assert done.content == "Checked off 2 kg of apples. Not on Shopping: eggs."
    assert lists.show(agent, "SHOPPING").content == "Shopping:\n- Milk\n- bread"
    assert lists.show(agent).content == "The lists: Shopping (2)."
    with pytest.raises(LookupError, match="there is no list 'chores'"):
        lists.check_off(agent, "chores", ["dishes"])


def test_recall_by_time(agent, engine, events):
    # no words: the latest conversations, each by its first message, but this one
    for content in ("Plan my week", "Fix my bike"):
        engine.replies.put([{"content": "Done."}])
        agent.send(None, content)
        until(events, ended)
    engine.replies.put([{"tool_calls": [call("recall", {"query": "", "days": 2})]}])
    engine.replies.put([{"content": "This and that."}])
    now = agent.send(None, "What did we talk about lately?")
    until(events, ended)
    answer = agent.store.messages(now)[3]
    day = datetime.date.today()
    assert answer["content"] == (f"“Fix my bike”, {day.day} {day:%B %Y}: the user began, Fix my "
                                 f"bike\n“Plan my week”, {day.day} {day:%B %Y}: the user began, "
                                 "Plan my week")  # fmt: skip


def test_name(agent, engine, events):
    # a conversation named after its first exchange, once
    engine.replies.put(REPLY)
    id = agent.send(None, "When should I plant tulip bulbs?")
    until(events, ended)
    engine.replies.put([{"content": "“Planting tulip bulbs.”\nMore"}])
    background.name(agent, id)
    asked = engine.requests[-1]["messages"]
    assert asked[0]["content"] == background.NAME
    said = "When should I plant tulip bulbs?\n\nAssistant: Hello there."
    assert asked[1]["content"] == f"User: {stamped(agent, id, 1)}{said}"
    assert agent.conversations()[0]["title"] == "Planting tulip bulbs"
    assert agent.store.unnamed() == []
    assert until(events, lambda e: e["type"] == "conversation")[-1]["conversation"]["title"] == (
        "Planting tulip bulbs")  # fmt: skip
    # one the model gives no name keeps its own, and is not named again
    engine.replies.put(REPLY)
    other = agent.send(None, "Hi")
    until(events, ended)
    engine.replies.put([{"content": "  "}])
    background.name(agent, other)
    assert agent.conversation(other)["title"] == "Hi" and agent.store.unnamed() == []


def test_review(agent, engine, events):
    # what is new in a conversation, reviewed for memories by the memory's calls alone, once
    agent.remember("The user asked about 2^2^2^2.", "about")
    engine.replies.put([{"content": "Nice to meet you, Sam."}])
    id = agent.send(None, "I'm Sam, a nurse.")
    until(events, ended)
    calls = [call("remember", {"memory": "The user's name is Sam.", "category": "about",
                               "evidence": "I'm Sam"}, "a"),
             call("remember", {"memory": "The user is a nurse.", "category": "work",
                               "evidence": "a nurse"}, "b"),
             call("forget", {"number": 1}, "c")]  # fmt: skip
    engine.replies.put([{"tool_calls": calls}])
    engine.replies.put([{"content": "Done."}])
    background.review(agent, id)
    first, second = engine.requests[-2:]
    today = datetime.date.today()
    assert first["messages"][0]["content"].endswith(
        f"About them:\n[1] The user asked about 2^2^2^2. ({today.day} {today:%b %Y})"
    )
    assert (
        first["messages"][1]["content"]
        == f"User: {stamped(agent, id, 1)}I'm Sam, a nurse.\n\nAssistant: Nice to meet you, Sam."
    )
    assert [t["function"]["name"] for t in first["tools"]] == ["remember", "forget"]
    assert [m["content"] for m in second["messages"][3:]] == ["Remembered, as [2].",
            "Remembered, as [3].", "Forgot [1]: The user asked about 2^2^2^2."]  # fmt: skip
    assert [(m["text"], m["category"]) for m in agent.memories()] == [
        ("The user's name is Sam.", "about"), ("The user is a nurse.", "work")]  # fmt: skip
    assert agent.store.reviewed(id) == 3 and agent.store.idle(time.time() + 1) == []
    background.review(agent, id)  # nothing new: no request
    assert len(engine.requests) == 3
    # of a long one, the latest, as much as half the model's context holds
    engine.replies.put([{"content": "OK."}])
    agent.send(id, "y" * 3000)
    until(events, ended)
    engine.context = 1000
    engine.replies.put([{"content": "Done."}])
    background.review(agent, id)
    assert engine.requests[-1]["messages"][1]["content"] == "y" * 1484 + "\n\nAssistant: OK."


def test_background_failures(agent, engine, events, monkeypatch):
    # a piece of the background's work that fails leaves the others done
    ids = []
    for text in ("One", "Two"):
        engine.replies.put([{"content": "Hi."}])
        ids.append(agent.send(None, text))
        until(events, ended)
        agent.store.name(ids[-1], text)
    reviewed = []

    def review(agent, id):
        if id == ids[0]:
            raise RuntimeError("a bug")
        reviewed.append(id)

    monkeypatch.setattr(background, "review", review)
    agent.background = background.Background(agent, idle=0)
    agent.background.start()
    deadline = time.time() + 5
    while not reviewed and time.time() < deadline:
        time.sleep(0.05)
    assert reviewed == [ids[1]]


def test_tidy(agent, engine, events):
    # a person's memory tidied: what says the same merged, by replaces alone, which adds nothing;
    # a quarter forgotten at most, or 3; the day noted, and all it did kept, to undo
    for i in range(5):
        agent.remember(f"The user has fact {i}.", "about")
    agent.remember("The user lives in Izmir.", "about")
    agent.remember("The user is based in Izmir.", "about")
    calls = [
        call("remember", {"memory": "The user lives in Izmir, Turkey.", "replaces": 6,
                          "evidence": "lives in Izmir"}, "a"),
        call("remember", {"memory": "The user is new.", "evidence": "x"}, "b"),
        *(call("forget", {"number": n}, f"f{n}") for n in (7, 1, 2, 3)),
    ]  # fmt: skip
    engine.replies.put([{"tool_calls": calls}])
    engine.replies.put([{"content": "Done."}])
    background.tidy(agent, None)
    said = [m["content"] for m in engine.requests[-1]["messages"] if m["role"] == "tool"]
    assert said[0] == "Changed [6]." and said[1].startswith("error: tidying changes memories")
    forgot = ["Forgot [7]: The user is based in Izmir.", "Forgot [1]: The user has fact 0.",
              "Forgot [2]: The user has fact 1."]  # fmt: skip
    assert said[2:5] == forgot
    assert said[5] == "error: a tidying forgets 3 memories at most"
    assert [m["text"] for m in agent.memories()] == [
        "The user has fact 2.", "The user has fact 3.", "The user has fact 4.",
        "The user lives in Izmir, Turkey."]  # fmt: skip
    assert {f["by"] for f in agent.store.forgotten()} == {"tidy"}
    assert agent.store.setting(background.TIDIED) == {"None": datetime.date.today().isoformat()}


def test_background(agent, engine, events):
    # once started, the agent names a conversation after its turn, and reviews it once idle; one
    # whose naming failed, as the engine was away, is named at the next look
    engine.replies.put(REPLY)
    unnamed = agent.send(None, "Hello")
    until(events, ended)
    agent.store.mark_reviewed(unnamed, 3)
    engine.replies.put([{"content": "Hello again"}])  # its name, at the first look
    agent.background = background.Background(agent, idle=0)
    agent.background.start()
    until(
        events,
        lambda e: e["type"] == "conversation" and e["conversation"]["title"] == "Hello again",
    )
    engine.replies.put(REPLY)
    engine.replies.put([{"content": "Greetings"}])  # its name
    engine.replies.put([{"content": "Done."}])  # its review
    id = agent.send(None, "Hi")
    until(
        events, lambda e: e["type"] == "conversation" and e["conversation"]["title"] == "Greetings"
    )
    deadline = time.time() + 5
    while agent.store.reviewed(id) < 3 and time.time() < deadline:
        time.sleep(0.05)
    assert agent.store.reviewed(id) == 3 and len(engine.requests) == 5


def test_recall(agent, engine, events):
    # recall finds what the user and the model said in other conversations, not the tools' answers
    engine.replies.put([{"tool_calls": [call("echo", {"text": "tulips"})]}])
    engine.replies.put([{"content": "Plant the tulips in October."}])
    garden = agent.send(None, "When should I plant bulbs?")
    until(events, ended)
    engine.replies.put([{"tool_calls": [call("recall", {"query": "tulips planting"})]}])
    engine.replies.put([{"content": "In October."}])
    now = agent.send(None, "What did you say about tulips?")
    until(events, ended)
    answer = agent.store.messages(now)[3]
    day = datetime.date.today()
    # each match with its turn's question and answer, in order: "planting" is "plant"
    assert answer["content"] == (f"“When should I plant bulbs?”, {day.day} {day:%B %Y}\n"
                                 "user: When should I plant bulbs?\n"
                                 "assistant: Plant the tulips in October.")  # fmt: skip
    found = [{"id": garden, "title": "When should I plant bulbs?"}]
    assert answer["info"]["conversations"] == found
    # not the conversation it is made in, nor a deleted one
    agent.delete(garden)
    assert agent.store.search("tulips", exclude=now) == [] and agent.store.search("") == []


def test_migration(tmp_path):
    # a state of the first version, conversations alone, gains memories and search, which finds
    # what was said before
    db = sqlite3.connect(tmp_path / "leat.db")
    db.executescript("""
        CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL NOT NULL,
                                    updated REAL NOT NULL);
        CREATE TABLE messages (conversation TEXT NOT NULL REFERENCES conversations (id)
                               ON DELETE CASCADE, position INTEGER NOT NULL, message TEXT NOT NULL,
                               PRIMARY KEY (conversation, position));
        INSERT INTO conversations VALUES ('0123456789ab', 'Bulbs', 0, 0);
        INSERT INTO messages VALUES ('0123456789ab', 0, '{"role": "system", "content": "tulips"}'),
                                    ('0123456789ab', 1, '{"role": "user", "content": "Tulips?"}');
    """)  # fmt: skip
    db.close()
    store = Store(tmp_path / "leat.db")
    assert [(f["role"], f["text"]) for f in store.search("tulip")] == [("user", "Tulips?")]
    assert store.memories() == [] and Store(tmp_path / "leat.db").conversations()[0]["id"]


def test_clearing(agent, engine, events):
    # past COMPACT of the context, a tool's long answer before the latest messages is cleared
    engine.context = 3000
    page = "x" * 6000  # some 2,000 tokens
    agent.tools["page"] = Tool("page", "A long page", strings(), lambda c: Result(page))
    engine.replies.put([{"tool_calls": [call("page", {})]}])
    engine.replies.put([{"content": "Read it."}])
    id = agent.send(None, "Read the page")
    until(events, ended)
    assert engine.requests[1]["messages"][-1]["content"] == page  # read whole while it is new
    engine.replies.put([{"content": "Sure."}])
    agent.send(id, "Thanks!")
    until(events, ended)
    cleared = [m for m in engine.requests[2]["messages"] if m["role"] == "tool"]
    assert cleared[0]["content"] == context.CLEARED
    assert agent.store.messages(id)[3]["content"] == page  # kept whole, for the app
    assert agent.store.context(id)["cleared"] == 4 and "summary" not in agent.store.context(id)


def test_summary(agent, engine, events):
    # what clearing cannot make small enough, the model summarizes, and the prompt goes on from it
    engine.context = 3000
    engine.replies.put([{"content": "Noted."}])
    id = agent.send(None, "a" * 3000)
    until(events, ended)
    engine.replies.put([{"content": "Goal: the user writes long."}])  # the summary
    engine.replies.put([{"content": "Noted again."}])
    with agent.events.watch() as seen:
        agent.send(id, "b" * 3000)
        compacted = until(seen, lambda e: e["type"] == "compacted")[-1]
        until(seen, ended)
    summarizing, reply = engine.requests[1:]
    assert summarizing["messages"][0]["content"] == context.SUMMARIZE
    assert f"User: {stamped(agent, id, 1)}" + "a" * 3000 in summarizing["messages"][1]["content"]
    system, last = reply["messages"]
    assert system["content"].endswith("go on from the summary:\n\nGoal: the user writes long.")
    assert last == {"role": "user", "content": stamped(agent, id, 3) + "b" * 3000}
    assert compacted["summarized"] == 3 == agent.conversation(id)["summarized"]


def test_summary_in_place(agent, engine, events):
    # a summary asked at the end of the conversation's prompt, as the engine's cache holds it,
    # when the prompt and the summary fit the context; its tools declared, that it extends the last
    engine.context = 4400
    engine.replies.put([{"content": "Noted."}])
    id = agent.send(None, "a" * 3600)
    until(events, ended)
    engine.replies.put([{"content": "Goal: the user writes long."}])  # the summary
    engine.replies.put([{"content": "Noted again."}])
    agent.send(id, "b" * 3600)
    until(events, ended)
    first, summarizing, reply = engine.requests
    asked = summarizing["messages"]
    assert asked[-1] == {"role": "user", "content": context.IN_PLACE}
    assert asked[:2] == first["messages"] and summarizing["tools"] == first["tools"]
    assert reply["messages"][0]["content"].endswith("Goal: the user writes long.")


def test_cut_off(agent, engine, events):
    # a reply the context cuts off is redone in its place, once the prompt is smaller
    engine.context = 3000
    agent.tools["page"] = Tool("page", "A long page", strings(), lambda c: Result("x" * 2400))
    engine.replies.put([{"tool_calls": [call("page", {})]}])
    engine.replies.put([{"content": "Read it."}])
    id = agent.send(None, "Read the page")
    until(events, ended)
    engine.replies.put([{"content": "Th"}, {"finish_reason": "length"}])
    engine.replies.put([{"content": "There."}])
    agent.send(id, "And?")
    until(events, ended)
    *_, reply = agent.store.messages(id)
    assert reply["content"] == "There." and "cut" not in reply["info"]
    assert engine.requests[-1]["messages"][3]["content"] == context.CLEARED
    # one cut off again after that is said to be cut off
    engine.replies.put([{"content": "Th"}, {"finish_reason": "length"}])
    engine.replies.put([{"content": "Goal: the page."}])  # the summary that makes it smaller
    engine.replies.put([{"content": "Th"}, {"finish_reason": "length"}])
    agent.send(id, "And then?")
    until(events, ended)
    assert agent.store.messages(id)[-1]["info"]["cut"] is True


def test_citations(agent, engine, events):
    # sources numbered across the conversation, from 1: one read again keeps its number
    def read(context, url):
        n = context.cite(url, "A page")
        return Result(f"[{n}] A page", {"url": url, "n": n})

    agent.tools["read"] = Tool("read", "Reads a page", strings(url="its address"), read)
    id = None
    for urls in (["a", "b"], ["b", "c"]):
        for i, url in enumerate(urls):
            engine.replies.put([{"tool_calls": [call("read", {"url": url}, f"{url}{i}")]}])
        engine.replies.put([{"content": "Read."}])
        id = agent.send(id, f"Read {' and '.join(urls)}")
        until(events, ended)
    assert [m["info"]["n"] for m in agent.store.messages(id) if m["role"] == "tool"] == [1, 2, 2, 3]


def test_schedule(agent, engine, events):
    # a task set in a conversation, by the model's call; past times refused, saying now's
    engine.replies.put([{"tool_calls": [call("schedule", {"task": "Remind the user to call Ada",
                                                          "at": "in 2 hours"})]}])  # fmt: skip
    engine.replies.put([{"content": "I will."}])
    id = agent.send(None, "Remind me in 2 hours to call Ada")
    until(events, ended)
    (task,) = agent.tasks()
    assert (task["prompt"], task["repeat"], task["conversation"]) == (
        "Remind the user to call Ada", "once", id)  # fmt: skip
    assert task["next"] == pytest.approx(time.time() + 7200, abs=5)
    assert agent.store.messages(id)[3]["content"] == f"Scheduled, as task [1]: {task['schedule']}."
    with pytest.raises(ValueError, match="that time has passed: it is"):
        agent.schedule("Too late", datetime.datetime.now(), "once", id)
    assert agent.unschedule(1)["prompt"] == "Remind the user to call Ada" and agent.tasks() == []
    with pytest.raises(NotFound):
        agent.unschedule(1)


def test_task_runs(agent, engine, events):
    # a task due is sent in its conversation, said to be one, and runs next a day after its time,
    # once however long it was missed; a task done for good goes
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    until(events, ended)
    yesterday = datetime.datetime.now().replace(second=0, microsecond=0) - datetime.timedelta(
        days=1, minutes=1)  # fmt: skip
    daily = agent.store.add_task("Give the weather", "daily", yesterday.timestamp(), id)
    engine.replies.put([{"content": "Sunny."}])
    background.run(agent, daily)
    done = until(events, lambda e: e["type"] == "done")[-1]
    assert done == {"type": "done", "conversation": id, "task": "Give the weather", "to": None}
    asked = engine.requests[-1]["messages"][-1]
    assert asked == {
        "role": "user",
        "content": stamped(agent, id, 3) + "(Your scheduled task [1] is due now: Give the weather)",
    }
    (task,) = agent.tasks()
    assert task["next"] == (yesterday + datetime.timedelta(days=2)).timestamp()
    # one whose conversation is gone runs in a new one; one that runs once is then gone
    once = agent.store.add_task("Say hello", "once", yesterday.timestamp(), "000000000000")
    engine.replies.put([{"content": "Hello."}])
    background.run(agent, once)
    other = until(events, lambda e: e["type"] == "done")[-1]["conversation"]
    assert other != id and [t["id"] for t in agent.tasks()] == [1]


def test_check(agent, engine, events):
    # a check tells only if its condition holds: one that finds nothing is withdrawn, as if it
    # never ran, and the apps are told nothing is done; one that finds it is kept, and told
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    until(events, ended)
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    check = agent.schedule("Look at tomorrow's weather in Izmir", tomorrow, "daily", id,
                           condition=" it will  rain ")  # fmt: skip
    assert check["condition"] == "it will rain"
    assert check["schedule"].endswith(", telling only if it will rain")
    engine.replies.put([{"content": "NOTHING."}])
    background.run(agent, agent.store.task(check["id"]))
    seen = until(events, ended)
    assert "withdrawn" in [e["type"] for e in seen] and "done" not in [e["type"] for e in seen]
    asked = engine.requests[-1]["messages"][-1]["content"]
    due = "(Your scheduled check [1] is due now: Look at tomorrow's weather in Izmir. Tell the "
    assert asked.endswith(due + "user only if it will rain; if not, reply NOTHING alone.)")
    assert len(agent.store.messages(id)) == 3  # as it was
    engine.replies.put([{"content": "Take an umbrella: rain is coming."}])
    background.run(agent, agent.store.task(check["id"]))
    assert until(events, lambda e: e["type"] == "done")[-1]["task"].startswith("Look at")
    assert agent.store.messages(id)[-1]["content"] == "Take an umbrella: rain is coming."


def test_task_waits(agent, engine, events, monkeypatch):
    # a task due in a conversation whose turn runs waits, without looking again and again, for the
    # turn's end
    engine.replies.put([{"content": "Hel"}, HOLD])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "delta")
    agent.store.name(id, "Greetings")  # not named by the next replies
    task = agent.store.add_task("Say hello", "once", time.time() - 1, id)
    looks = []
    due = agent.store.due
    monkeypatch.setattr(agent.store, "due", lambda now: looks.append(now) or due(now))
    agent.background = background.Background(agent, idle=60)
    agent.background.start()
    time.sleep(0.3)
    assert len(looks) == 1 and agent.store.due(time.time()) == [task]
    engine.replies.put([{"content": "Hello."}])
    engine.released.set()
    assert until(events, lambda e: e["type"] == "done")[-1]["task"] == "Say hello"
    assert agent.tasks() == []


def test_task_without_model(agent, engine, events):
    # a task due while the engine has no model, as the box starts, waits for one
    engine.status = "unloaded"
    task = agent.store.add_task("Say hello", "once", time.time() - 1, None)
    with pytest.raises(EngineError, match="no model is loaded"):
        background.run(agent, task)
    assert agent.store.due(time.time()) == [task] and agent.conversations() == []


def test_task_on_time(agent, engine, events):
    # a task set for sooner than the next look runs on time
    agent.background = background.Background(agent, idle=60)
    agent.background.start()
    time.sleep(0.1)  # waiting for the next look
    engine.replies.put([{"content": "Hello."}])
    agent.schedule("Say hello", datetime.datetime.now() + datetime.timedelta(seconds=0.3), "once")
    done = until(events, lambda e: e["type"] == "done")[-1]
    assert done["task"] == "Say hello"


def test_live_reply(agent, engine, events):
    # while a reply streams, the conversation shows it, and another message must wait for it
    engine.replies.put([{"content": "Hel"}, HOLD, {"content": "lo"}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "delta")
    c = agent.conversation(id)
    assert c["running"] and c["messages"][-1]["content"] == "Hel"
    with pytest.raises(Busy):
        agent.send(id, "Hello?")
    engine.released.set()
    until(events, ended)
    assert agent.conversation(id)["messages"][-1]["content"] == "Hello"
    with pytest.raises(NotFound):
        agent.send("0" * 12, "Hi")


def test_stop(agent, engine, events):
    # a stopped reply keeps what it had, and says it was stopped
    engine.replies.put([{"content": "Hel"}, HOLD, {"content": "lo"}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "delta")
    agent.stop(id)
    until(events, ended)
    reply = agent.store.messages(id)[-1]
    assert reply["content"] == "Hel" and reply["info"]["stopped"] is True


def test_stop_in_prefill(agent, engine, events):
    # a stop before the first token ends the reply too, as one waiting for a long prompt's prefill
    engine.replies.put([HOLD, {"content": "Hello"}])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "message" and e["message"]["role"] == "assistant")
    time.sleep(0.1)  # into the completion
    agent.stop(id)
    until(events, ended)
    assert agent.store.messages(id)[-1]["content"] == ""


def test_delete(agent, engine, events):
    # a conversation deleted while its reply streams is gone, the reply too
    engine.replies.put([{"content": "Hel"}, HOLD])
    id = agent.send(None, "Hi")
    until(events, lambda e: e["type"] == "delta")
    agent.delete(id)
    assert until(events, ended)[-1] == {"type": "deleted", "id": id, "to": None}
    assert agent.conversations() == [] and agent.store.messages(id) == []


def test_failure(agent, engine, events):
    # a turn that fails is taken back whole, saying why, with the message to send again; the
    # conversation it began too
    engine.replies.put("the prompt is too long")
    id = agent.send(None, "Hi")
    error, deleted = until(events, ended)[-2:]
    assert error == {"type": "error", "conversation": id, "error": "the prompt is too long",
                     "start": 1, "content": "Hi", "to": None}  # fmt: skip
    assert deleted == {"type": "deleted", "id": id, "to": None} and agent.conversations() == []
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi")
    until(events, ended)
    engine.replies.put("the prompt is too long")
    agent.send(id, "More")
    assert until(events, ended)[-2]["start"] == 3
    assert len(agent.store.messages(id)) == 3 and not agent.conversation(id)["running"]


def test_failure_after_summary(agent, engine, events):
    # a turn taken back leaves the prompt's state as it was before it, its summary gone too
    engine.context = 3000
    engine.replies.put([{"content": "Noted."}])
    id = agent.send(None, "a" * 3000)
    until(events, ended)
    before = agent.store.context(id)
    engine.replies.put([{"content": "Goal: the user writes long."}])  # the summary
    engine.replies.put("the engine broke")
    agent.send(id, "b" * 3000)
    until(events, ended)
    assert agent.store.context(id) == before and agent.conversation(id)["summarized"] is None


def test_unreachable(tmp_path, engine):
    # an engine that is not there fails the turn, saying so
    engine.shutdown()
    engine.server_close()
    agent = Agent(Store(tmp_path / "leat.db"), Client(engine.url))
    with agent.events.watch() as events:
        agent.send(None, "Hi")
        error = until(events, ended)[-2]
    assert error["type"] == "error" and "is not reachable" in error["error"]
    models = agent.models_event()
    assert models["models"] == [] and "is not reachable" in models["error"]


def test_models(agent, engine, events):
    assert agent.models_event()["models"] == [{"id": "fake", "status": "loaded"}]
    agent.load("other")
    assert engine.loads == ["other"]
    assert [e["type"] for e in until(events, lambda e: e["type"] == "models")] == [
        "loading", "models"]  # fmt: skip
    # the apps are told of the models as they change, as when the engine loads one or comes up
    agent.models_changed()
    assert events.empty()  # as they were told
    engine.status = "loading"
    agent.models_changed()
    assert events.get(timeout=1)["models"] == [{"id": "fake", "status": "loading"}]


def test_engine_stuck(tmp_path, monkeypatch):
    # an engine that takes the request but never answers is as good as away, once LIST has passed
    monkeypatch.setattr("leat.agent.client.LIST", 0.2)
    with socket.create_server(("127.0.0.1", 0)) as stuck:
        client = Client(f"http://127.0.0.1:{stuck.getsockname()[1]}")
        start = time.monotonic()
        with pytest.raises(EngineError, match="timed out"):
            client.models()
        assert time.monotonic() - start < 2


def test_hang_up_is_quiet(agent, capsys):
    # a page closed before its answer was written is no error to print; any other error is
    with Server(agent, port=0) as server:
        for error in (BrokenPipeError(), ConnectionResetError(), ValueError("a bug")):
            try:
                raise error
            except Exception:
                server.handle_error(None, ("127.0.0.1", 1))
    printed = capsys.readouterr().err
    assert "ValueError: a bug" in printed and "Broken" not in printed and "Reset" not in printed


@contextlib.contextmanager
def serving(agent: Agent, telegram: Any = None) -> Iterator[str]:
    """The agent's server, its household's owner set up, whose cookie urllib's requests send from
    then on, as a browser's do."""
    with Server(agent, port=0, telegram=telegram) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        processor = urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        urllib.request.install_opener(urllib.request.build_opener(processor))
        try:
            assert request(f"{url}/api/setup", "POST", {"name": "Alkın"})[0] == 200
            yield url
        finally:
            urllib.request.install_opener(None)  # type: ignore[arg-type]
            server.shutdown()


def cookie() -> str:
    """The Cookie header of the device serving() set up."""
    opener = urllib.request._opener  # type: ignore[attr-defined]
    jar = next(h.cookiejar for h in opener.handlers if hasattr(h, "cookiejar"))
    return "; ".join(f"{c.name}={c.value}" for c in jar)


@pytest.fixture
def server(agent) -> Iterator[str]:
    with serving(agent) as url:
        yield url


def request(url: str, method: str = "GET", body: Any = None, **headers: str) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data, headers, method=method)) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_app(server):
    for path in ("/", "/c/0123456789ab"):
        with urllib.request.urlopen(f"{server}{path}") as response:
            assert response.headers["Content-Type"] == "text/html; charset=utf-8"
            assert b"<title>leat</title>" in response.read()
    for path in ("/app.mjs", "/themes.mjs"):
        with urllib.request.urlopen(f"{server}{path}") as response:
            assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
    with urllib.request.urlopen(f"{server}/vendor/temml/Temml.woff2") as response:
        assert response.headers["Content-Type"] == "font/woff2"
    with urllib.request.urlopen(f"{server}/logo.svg") as response:
        assert response.headers["Content-Type"] == "image/svg+xml"
    for path in ("/server.py", "/vendor/temml/LICENSE", "/../store.py", "/api/nothing"):
        assert request(f"{server}{path}")[0] == 404


def test_api(server, engine, agent, events):
    engine.replies.put(REPLY)
    status, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    id = json.loads(body)["id"]
    assert status == 200 and agent.conversation(id, 1) is not None  # the owner's
    until(events, ended)
    # what is wrong with a request, said
    assert request(f"{server}/api/conversations", "POST", {"content": " "})[0] == 400
    assert request(f"{server}/api/conversations", "POST", [1])[0] == 400
    assert request(f"{server}/api/conversations", "POST", {"content": "Hi", "think": 1})[0] == 400
    assert request(f"{server}/api/conversations/0123456789ab/messages", "POST",
                   {"content": "Hi"})[0] == 404  # fmt: skip
    assert request(f"{server}/api/conversations/0123456789ab")[0] == 404
    status, body = request(f"{server}/api/conversations/{id}")
    assert status == 200 and json.loads(body)["messages"][1]["content"] == "Hi"
    assert request(f"{server}/api/conversations/{id}", "DELETE")[0] == 200
    assert agent.conversations(1) == []


def test_api_busy(server, engine, events):
    engine.replies.put([{"content": "Hel"}, HOLD])
    _, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    id = json.loads(body)["id"]
    until(events, lambda e: e["type"] == "delta")
    url = f"{server}/api/conversations/{id}"
    assert request(f"{url}/messages", "POST", {"content": "Hello?"})[0] == 409
    assert request(f"{url}/stop", "POST", {})[0] == 200
    until(events, ended)


def test_api_memories(server, agent):
    url = f"{server}/api/memories"
    status, body = request(url, "POST", {"text": " Lives in Izmir. "})
    assert status == 200 and json.loads(body)["text"] == "Lives in Izmir."
    assert [m["text"] for m in agent.memories(1)] == ["Lives in Izmir."]
    assert request(url, "POST", {"text": ""})[0] == 400
    assert request(f"{url}/{json.loads(body)['id']}", "DELETE")[0] == 200
    assert request(f"{url}/7", "DELETE")[0] == 404 and agent.memories(1) == []
    (forgotten,) = agent.store.forgotten(1)
    assert request(f"{url}/restore", "POST", {"id": forgotten["id"]})[0] == 200
    assert [m["text"] for m in agent.memories(1)] == ["Lives in Izmir."]
    assert request(url, "POST", {"text": "Likes tea.", "category": "drinks"})[0] == 400
    assert request(f"{server}/memory")[0] == 200  # the app's page of them


def test_api_characters(server, agent, engine, events):
    url = f"{server}/api/characters"
    status, body = request(url, "POST", {"name": "Ms Ada", "about": "A patient maths tutor."})
    tutor = json.loads(body)
    assert status == 200 and tutor["name"] == "Ms Ada"
    assert request(url, "POST", {"name": "Nobody"})[0] == 400
    engine.replies.put(REPLY)
    _, body = request(f"{server}/api/conversations", "POST",
                      {"content": "Hi", "character": tutor["id"]})  # fmt: skip
    until(events, ended)
    assert agent.conversations(1)[0]["character"] == tutor["id"]
    assert request(f"{url}/{tutor['id']}", "DELETE")[0] == 200
    assert request(f"{url}/{tutor['id']}", "DELETE")[0] == 404
    assert request(f"{server}/characters")[0] == 200  # the app's page of them


def test_api_lists(server, agent):
    from leat.agent.tools import lists

    assert request(f"{server}/api/lists", "POST", {"name": "Chores"})[0] == 200
    (chores,) = agent.store.lists()
    assert request(f"{server}/api/lists/{chores['id']}/items", "POST", {"text": "Dishes"})[0] == 200
    (item,) = lists.find(agent, "chores")["items"]
    assert request(f"{server}/api/lists/items/{item['id']}", "DELETE")[0] == 200
    assert request(f"{server}/api/lists/items/{item['id']}", "DELETE")[0] == 404
    assert request(f"{server}/api/lists/{chores['id']}", "DELETE")[0] == 200
    assert agent.store.lists() == [] and request(f"{server}/lists")[0] == 200


def test_api_tasks(server, agent, engine, events):
    in_an_hour = datetime.datetime.now() + datetime.timedelta(hours=1)
    task = agent.schedule("Remind the user to stretch", in_an_hour, "daily", person=1)
    assert request(f"{server}/api/tasks/{task['id']}", "DELETE")[0] == 200
    assert request(f"{server}/api/tasks/{task['id']}", "DELETE")[0] == 404
    assert request(f"{server}/tasks")[0] == 200  # the app's page of them
    # one the user schedules themselves, a check, and runs now, to try it, in a chat that is its
    url = f"{server}/api/tasks"
    body = {"prompt": "Look at tomorrow's weather", "at": "19:00", "repeat": "daily",
            "only_if": "it will rain"}  # fmt: skip
    status, made = request(url, "POST", body)
    check = json.loads(made)
    assert status == 200 and check["schedule"].endswith("telling only if it will rain")
    assert request(url, "POST", body | {"repeat": "yearly"})[0] == 400
    engine.replies.put([{"content": "It will rain."}])
    status, ran = request(f"{url}/{check['id']}/run", "POST", {})
    until(events, lambda e: e["type"] == "done")
    assert agent.store.task(check["id"])["conversation"] == json.loads(ran)["id"]
    assert agent.store.task(check["id"])["next"] == check["next"]  # its time as it was


def test_joining(server, agent, engine, events):
    # a device asks to join, showing a code the owner's device shows too; let in as a new person,
    # it sees their own alone, may not do what the owner alone may, and is unpaired by the owner
    jar = urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    phone = urllib.request.build_opener(jar)

    def ask(path: str, method: str = "GET", body: Any = None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        try:
            with phone.open(urllib.request.Request(f"{server}{path}", data, method=method)) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    assert ask("/api/me") == (401, {"error": {"message": ANY}, "empty": False})
    assert ask("/api/setup", "POST", {"name": "Eve"})[0] == 400  # the household has its owner
    status, asked = ask("/api/pairings", "POST", {"name": "Ada"})
    assert status == 200 and len(asked["code"]) == 6
    connection = http.client.HTTPConnection(server.replace("http://", ""))
    connection.request("GET", "/api/events", headers={"Cookie": cookie()})
    response = connection.getresponse()
    while not (line := response.readline()).startswith(b'data: {"type": "household"'):
        pass
    (request_,) = json.loads(line[6:])["requests"]
    assert (request_["name"], request_["code"]) == ("Ada", asked["code"])
    connection.close()
    assert ask(f"/api/pairings/{asked['id']}")[0] == 401  # asked after by a POST alone
    assert ask(f"/api/pairings/{asked['id']}", "POST")[0] == 202  # waiting
    status, ada = request(f"{server}/api/pairings/{asked['id']}/allow", "POST", {})
    assert status == 200 and json.loads(ada)["name"] == "Ada"
    assert ask(f"/api/pairings/{asked['id']}", "POST")[0] == 200  # its cookie, taken once
    assert ask(f"/api/pairings/{asked['id']}", "POST")[0] == 404
    assert ask("/api/me") == (200, {"person": 2, "name": "Ada", "owner": False, "device": 2})
    # Ada's own alone: her events tell of her conversations, not the owner's
    adas = "; ".join(f"{c.name}={c.value}" for c in jar.cookiejar)
    connection = http.client.HTTPConnection(server.replace("http://", ""))
    connection.request("GET", "/api/events", headers={"Cookie": adas})
    stream = connection.getresponse()
    engine.replies.put(REPLY)
    _, body = request(f"{server}/api/conversations", "POST", {"content": "Hi"})
    until(events, ended)
    assert ask(f"/api/conversations/{json.loads(body)['id']}")[0] == 404
    engine.replies.put(REPLY)
    _, mine = ask("/api/conversations", "POST", {"content": "Hello"})
    until(events, ended)
    told = []
    while not told or told[-1].get("type") != "conversation":
        if (line := stream.readline()).startswith(b"data: "):
            told.append(json.loads(line[6:]))
    assert told[0] == {"type": "conversations", "conversations": []}
    assert told[-1]["conversation"]["id"] == mine["id"]  # the owner's never told
    connection.close()
    for path, body in [("/api/models/load", {"model": "fake"}), ("/api/telegram", {"token": "x"}),
                       ("/api/pairings/0123456789abcdef/allow", {})]:  # fmt: skip
        assert ask(path, "POST", body)[0] in (403, 404)
    assert ask("/api/models/load", "POST", {"model": "fake"})[0] == 403
    assert ask("/api/people/1", "DELETE")[0] == 403
    assert ask("/api/people/2", "POST", {"child": True})[0] == 403
    assert request(f"{server}/api/people/2", "POST", {"child": True})[0] == 200
    assert request(f"{server}/api/people/1", "POST", {"child": True})[0] == 404  # the owner
    assert agent.store.person(2)["child"] == 1
    # unpaired by the owner, it must ask again; and removed, with all that is hers, and the
    # Telegram people who talked as her
    (device,) = [d["id"] for d in agent.store.devices() if d["person"] == 2]
    connection = http.client.HTTPConnection(server.replace("http://", ""), timeout=5)
    connection.request("GET", "/api/events", headers={"Cookie": adas})
    stream = connection.getresponse()
    assert request(f"{server}/api/devices/{device}", "DELETE")[0] == 200
    assert ask("/api/me")[0] == 401
    time.sleep(1.1)  # the stream open before ends, at the next event past a second
    agent.events.publish({"type": "files", "files": [], "to": "owner"})
    assert stream.read().endswith(b"\n\n")  # to its end, which the server closed
    connection.close()
    allowed = {"9": {"id": 9, "name": "Ada", "person": 2}, "7": {"id": 7, "name": "A", "person": 1}}
    agent.store.set_setting("telegram", {"allowed": allowed})
    assert request(f"{server}/api/people/2", "DELETE")[0] == 200
    assert agent.store.person(2) is None and agent.conversations(2) == []
    assert list(agent.store.setting("telegram")["allowed"]) == ["7"]
    assert request(f"{server}/api/people/1", "DELETE")[0] == 400  # the owner


def test_trust(server, engine):
    # a request naming the box by an address or a local name, and a write from its own page
    engine.replies.put(REPLY)
    url, own = f"{server}/api/conversations", server.replace("http://", "")
    assert request(f"{server}/", Host="evil.example")[0] == 403  # DNS rebinding
    assert request(f"{server}/", Host="leat.local:8000")[0] == 200
    assert request(f"{server}/", Host="[::1]:8000")[0] == 200
    hi = {"content": "Hi"}
    assert request(url, "POST", hi, Origin="http://evil.example")[0] == 403
    assert request(url, "POST", hi, Origin=f"http://{own}")[0] == 200


def test_events(server, agent, engine):
    # what there is, then what changes
    connection = http.client.HTTPConnection(server.replace("http://", ""))
    connection.request("GET", "/api/events", headers={"Cookie": cookie()})
    response = connection.getresponse()
    assert response.headers["Content-Type"] == "text/event-stream"

    def event() -> dict:
        while not (line := response.readline()).startswith(b"data: "):
            pass
        return json.loads(line[6:])

    assert event() == {"type": "conversations", "conversations": []}
    assert event() == {"type": "memories", "memories": [], "forgotten": []}
    assert event() == {"type": "files", "files": []}
    assert event() == {"type": "tasks", "tasks": []}
    assert event() == {"type": "characters", "characters": []}
    assert event() == {"type": "lists", "lists": []}
    assert event()["type"] == "household"  # the owner's
    assert event()["type"] == "models"
    engine.replies.put(REPLY)
    id = agent.send(None, "Hi", person=1)
    assert event()["conversation"]["id"] == id
    connection.close()
