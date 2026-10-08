"""The Telegram channel, against a fake of Telegram's Bot API and the fake engine."""

import contextlib
import json
import queue
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from leat.agent.agent import Agent, NotFound
from leat.agent.channels import telegram
from leat.agent.channels.telegram import Telegram, TelegramError, to_html
from leat.agent.client import Client
from leat.agent.store import Store
from leat.agent.tools import files
from leat.agent.workspace import Workspace
from tests.test_agent import HOLD, call, request, serving, until

TOKEN, OTHER = "123:ok", "789:ok"  # two bots' tokens
ME, STRANGER = (
    {"id": 7, "first_name": "Alkın", "username": "alkinun"},
    {"id": 9, "first_name": "Eve"},
)


class FakeBots(ThreadingHTTPServer):
    """Telegram's Bot API, for TOKEN's bot: the updates it is given to send, and the calls made."""

    daemon_threads = True

    def __init__(self) -> None:
        self.updates: queue.SimpleQueue[dict] = queue.SimpleQueue()
        self.calls: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self.count = 0
        self.polls = 0  # getUpdates answered
        self.offsets: list[tuple[str, int]] = []  # each getUpdates's token and offset
        super().__init__(("127.0.0.1", 0), _Bots)

    def update(self, sender: dict, **message: Any) -> None:
        self.count += 1
        chat = {"id": message.pop("chat_id", sender["id"]), "type": message.pop("chat", "private")}
        sent = {"message_id": self.count, "from": sender, "chat": chat} | message
        self.updates.put({"update_id": self.count, "message": sent})

    def drain(self) -> None:
        """Waits for the bot to have handled every update given: it asks for more once it has."""
        while not self.updates.empty():
            time.sleep(0.01)
        polls = self.polls
        while self.polls < polls + 2:
            time.sleep(0.01)

    def next(self, method: str) -> Any:
        """The next call of a method, those of others before it passed."""
        while True:
            name, body = self.calls.get(timeout=5)
            if name == method:
                return body


class _Bots(BaseHTTPRequestHandler):
    server: FakeBots

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:  # a file's download
        self._send(b"the file's bytes")

    def do_POST(self) -> None:
        _, bot, method = self.path.split("/")
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if bot.removeprefix("bot") not in (TOKEN, OTHER):
            return self._answer({"ok": False, "description": "Unauthorized"}, 401)
        body = json.loads(data) if self.headers["Content-Type"] == "application/json" else data
        if method != "getUpdates":
            self.server.calls.put((method, body))
        results = {"getMe": {"username": "leat_bot"}, "getFile": {"file_path": "docs/1.pdf"}}
        if method == "getUpdates":
            self.server.offsets.append((bot.removeprefix("bot"), body["offset"]))
            updates = []
            with contextlib.suppress(queue.Empty):
                updates.append(self.server.updates.get(timeout=0.2))
            self.server.polls += 1
            return self._answer({"ok": True, "result": updates})
        self._answer({"ok": True, "result": results.get(method, True)})

    def _answer(self, body: dict, status: int = 200) -> None:
        self._send(json.dumps(body).encode(), status)

    def _send(self, data: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def bots() -> Iterator[FakeBots]:
    bots = FakeBots()
    threading.Thread(target=bots.serve_forever, args=(0.01,), daemon=True).start()
    yield bots
    bots.shutdown()


@pytest.fixture
def bot(bots, engine, tmp_path) -> Telegram:
    workspace = Workspace(tmp_path / "workspace")
    agent = Agent(
        Store(tmp_path / "leat.db"), Client(engine.url), files.tools(workspace), workspace
    )
    bot = Telegram(agent, f"http://127.0.0.1:{bots.server_port}")
    bot.start()
    return bot


def test_connect(bot, bots):
    with pytest.raises(TelegramError, match="does not know that token"):
        bot.connect("456:bad")
    assert bot.connect(f" {TOKEN} ") == {"type": "telegram", "bot": "leat_bot", "allowed": [],
                                         "requests": []}  # fmt: skip
    assert [c["command"] for c in bots.next("setMyCommands")["commands"]] == ["new", "stop"]
    bot.disconnect()
    assert bot.state()["bot"] is None and "token" not in bot.agent.store.setting(telegram.KEY)
    # another bot's updates are read from its first, whatever the last's were numbered
    bot.connect(TOKEN)
    bots.update(STRANGER, text="Hi")
    bots.next("sendMessage")
    bots.drain()
    bot.connect(OTHER)
    bots.drain()
    assert bots.offsets[-1] == (OTHER, 0)


def test_people(bot, bots, engine):
    # a stranger is told once to ask, and waits; once allowed, they talk to the agent, whose reply
    # comes back as Telegram's HTML
    bot.connect(TOKEN)
    bots.update(STRANGER, text="Hi")
    assert bots.next("sendMessage")["text"] == telegram.REFUSED
    bots.update(STRANGER, text="Hello?")
    bots.update(STRANGER, text="Hi", chat="group")  # no group's
    bots.drain()
    assert bots.calls.empty()  # told once
    assert bot.state()["requests"] == [{"id": 9, "name": "Eve", "asked": pytest.approx(time.time(),
                                        abs=5)}]  # fmt: skip
    with pytest.raises(NotFound):
        bot.allow(8)
    bot.allow(9)
    assert bots.next("sendMessage")["text"].startswith("You're in!")
    engine.replies.put([{"content": "Hello **Eve** <3"}])
    bots.update(STRANGER, text="Hi")
    sent = bots.next("sendMessage")
    assert (sent["chat_id"], sent["text"], sent["parse_mode"]) == (
        9,
        "Hello <b>Eve</b> &lt;3",
        "HTML",
    )
    (conversation,) = bot.agent.conversations()
    assert bot.agent.store.messages(conversation["id"])[1]["info"]["via"] == "telegram"
    bot.refuse(9)
    assert bot.state()["allowed"] == []
    # let in as a person of the household's, whose conversation the chat's becomes, a new one
    bot.agent.store.add_person("Alkın")  # the owner, whose all before becomes
    ada = bot.agent.store.add_person("Ada")["id"]
    bots.update(STRANGER, text="Hi again")
    bots.next("sendMessage")
    bot.allow(9, ada)
    bots.next("sendMessage")
    engine.replies.put([{"content": "Hello Ada."}])
    bots.update(STRANGER, text="Hi")
    assert bots.next("sendMessage")["text"] == "Hello Ada."
    (adas,) = bot.agent.conversations(ada)
    assert adas["id"] != conversation["id"]


def test_deleted(bot, bots, engine):
    # a conversation deleted in the app while it answers a chat: the chat waits no more
    bot.connect(TOKEN)
    bot.agent.store.set_setting(telegram.KEY, bot.agent.store.setting(telegram.KEY) | {
        "allowed": {"7": {"id": 7, "name": "Alkın (@alkinun)"}}})  # fmt: skip
    engine.replies.put([{"content": "Hel"}, HOLD])
    with bot.agent.events.watch() as events:
        bots.update(ME, text="Hi")
        until(events, lambda e: e["type"] == "delta")
    bot.agent.delete(bot.agent.conversations()[0]["id"])
    deadline = time.time() + 5
    while bot.started and time.time() < deadline:
        time.sleep(0.01)
    assert bot.started == {}
    # a file too big for a bot to take is said to be
    bots.update(ME, document={"file_id": "f", "file_name": "big.iso", "file_size": 30 << 20})
    assert bots.next("sendMessage")["text"].startswith("That file is too big")


def test_chat(bot, bots, engine):
    # a chat is one conversation, till /new; a file sent is the workspace's, attached
    bot.connect(TOKEN)
    bot.agent.store.set_setting(telegram.KEY, bot.agent.store.setting(telegram.KEY) | {
        "allowed": {"7": {"id": 7, "name": "Alkın (@alkinun)"}}})  # fmt: skip
    for text in ("One", "Two"):
        engine.replies.put([{"content": f"Got {text}."}])
        bots.update(ME, text=text)
        assert bots.next("sendMessage")["text"] == f"Got {text}."
    assert len(bot.agent.conversations()) == 1
    bots.update(ME, text="/new@leat_bot\n")
    assert bots.next("sendMessage")["text"] == "A new conversation begins."
    engine.replies.put([{"content": "Read it."}])
    bots.update(ME, caption="Look", document={"file_id": "f", "file_name": "plan.pdf"})
    assert bots.next("sendMessage")["text"] == "Read it."
    latest = bot.agent.conversations()[0]["id"]
    assert len(bot.agent.conversations()) == 2
    assert bot.agent.store.messages(latest)[1]["info"]["files"] == ["plan.pdf"]
    assert bot.agent.workspace.path("plan.pdf").read_bytes() == b"the file's bytes"
    # the files a turn made follow its answer, by their names
    engine.replies.put(
        [{"tool_calls": [call("write", {"path": "notes/plan.txt", "content": "Plan"})]}]
    )
    engine.replies.put([{"content": "Made it."}])
    bots.update(ME, text="Write the plan")
    assert bots.next("sendMessage")["text"] == "Made it."
    sent = bots.next("sendDocument")
    assert b'filename="plan.txt"' in sent and b"\r\n\r\nPlan\r\n" in sent
    # a task of the chat's conversation is answered there
    engine.replies.put([{"content": "Stretch now!"}])
    task = bot.agent.store.add_task("Remind them to stretch", "once", time.time() - 1, latest)
    from leat.agent.background import run

    run(bot.agent, task)
    assert bots.next("sendMessage")["text"] == "Stretch now!"


def test_api(bot, bots):
    # the app's settings of the bot, through the agent's API
    with serving(bot.agent, bot) as server:
        url = f"{server}/api/telegram"
        status, body = request(url, "POST", {"token": "456:bad"})
        assert status == 400 and b"Telegram refused it: it does not know that token" in body
        assert request(url, "POST", {"token": " "})[0] == 400
        status, body = request(url, "POST", {"token": TOKEN})
        assert status == 200 and json.loads(body)["bot"] == "leat_bot"
        bots.update(STRANGER, text="Hi")
        bots.next("sendMessage")
        assert request(f"{url}/people", "POST", {"id": 8})[0] == 404
        assert request(f"{url}/people", "POST", {"id": 9})[0] == 200
        assert [(p["id"], p["person"]) for p in bot.state()["allowed"]] == [(9, 1)]  # the owner
        assert request(f"{url}/people/9", "DELETE")[0] == 200
        assert request(url, "DELETE")[0] == 200 and bot.state()["bot"] is None


def test_group(bot, bots, engine):
    # in a group, the bot answers the household's people alone, when they name it or answer it:
    # each a conversation of their own there, with the message they answer, knowing none of
    # theirs, and the web's tools alone; its answer an answer to their message
    bot.connect(TOKEN)
    bots.next("setMyCommands")
    settings = bot.agent.store.setting(telegram.KEY)
    allowed = {"7": {"id": 7, "name": "Alkın (@alkinun)", "person": None}}
    bot.agent.store.set_setting(telegram.KEY, settings | {"allowed": allowed})
    bot.agent.remember("The user lives in Izmir.", "about")
    family = -100
    bots.update(ME, chat="supergroup", chat_id=family, text="Lunch at 1?")  # not to the bot
    bots.update(STRANGER, chat="supergroup", chat_id=family, text="@leat_bot hi")  # a stranger
    bots.drain()
    assert bots.calls.empty() and bot.agent.conversations() == []
    engine.replies.put([{"content": "It is not true [1]."}])
    claim = {"message_id": 41, "from": STRANGER, "text": "The moon is made of cheese."}
    bots.update(ME, chat="supergroup", chat_id=family, text="@leat_bot is this true?",
                reply_to_message=claim)  # fmt: skip
    sent = bots.next("sendMessage")
    assert (sent["chat_id"], sent["text"]) == (family, "It is not true [1].")
    assert sent["reply_parameters"]["message_id"] == bots.count
    asked = engine.requests[-1]
    assert "asked in a group chat" in asked["messages"][0]["content"]
    assert "Izmir" not in asked["messages"][0]["content"]
    assert asked["messages"][1]["content"].endswith(
        "Eve wrote in the group:\n> The moon is made of cheese.\n\nis this true?"
    )
    assert {t["function"]["name"] for t in asked.get("tools", [])} <= {"search", "fetch", "weather"}
    (conversation,) = bot.agent.conversations()
    assert bot.agent.store.conversation(conversation["id"])["shared"] == 1


ONE, TWO = '<a href="https://one.org">[1]</a>', '<a href="https://two.org">[2]</a>'


@pytest.mark.parametrize(
    "markdown, expected",
    [
        ("# Title\n\nSome *very* **bold** and `a<b`.",
         "<b>Title</b>\n\nSome <i>very</i> <b>bold</b> and <code>a&lt;b</code>."),
        ("- one\n  * two", "• one\n  • two"),
        ("```python\nif a < b:\n    pass\n```", "<pre>if a &lt; b:\n    pass</pre>"),
        ("| a | b |\n|---|---|\n| 1 | 2 |", "<pre>| a | b |\n| 1 | 2 |</pre>"),
        ("See [the docs](https://x.org/a?b=1&c=2).",
         'See <a href="https://x.org/a?b=1&amp;c=2">the docs</a>.'),
        ("True [1][2], and [3].", f"True {ONE}{TWO}, and [3]."),
        ("> quoted", "<blockquote>quoted</blockquote>"),
        ("2 * 3 * 4", "2 * 3 * 4"),
    ],
)  # fmt: skip
def test_to_html(markdown, expected):
    assert to_html(markdown, {1: "https://one.org", 2: "https://two.org"}) == expected


def test_parts():
    paragraphs = ["a" * 1500, "b" * 1500, "c" * 5000]
    parts = telegram._parts("\n\n".join(paragraphs))
    assert [len(p) for p in parts] == [1500, 1500, 2048, 2048, 904]
