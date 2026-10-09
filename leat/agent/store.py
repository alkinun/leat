"""The agent's state, in one SQLite file: conversations and their messages, and memories.

A message is a dict in OpenAI's chat format, as the model reads it, and its "info": what only people
see of it, such as the model that wrote it and how fast. A conversation's messages are only ever
appended, but for a turn taken back whole, so that each step's prompt extends the last's. What the
user and the model said is indexed for full-text search, the tools' answers not.

A memory is a short sentence about the user, of a category, which every conversation begun after
it knows, dated when it was last confirmed, and a plan with its last day. What a change replaced or
forgetting removed is kept, to undo. A conversation is reviewed for memories once idle, to the
message it was reviewed to, and named by the model once.

A conversation's context is the state of what its prompt keeps of it, as leat.agent.context fits it
to the model's.

A task is a prompt the agent runs at its next time, in a conversation, first at its first, the
time asked for, or the first of its repeats after, whose wall clock and day of the month its repeats
keep; one done for good is deleted. Settings are values by name, of JSON.

The household is its people, the first its owner, and the devices paired to each, known by the hash
of a secret each holds. A conversation, a memory and a task are each a person's; a memory of the
household category is everyone's. Before the household has its first person, everything is no one's,
and becomes the owner's.
"""

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# each takes the state from the version of its index to the next: SQLite's user_version
_MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      created REAL NOT NULL,
      updated REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages (
      conversation TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
      position INTEGER NOT NULL,
      message TEXT NOT NULL,
      PRIMARY KEY (conversation, position)
    );
    """,
    """
    CREATE TABLE memories (
      id INTEGER PRIMARY KEY,
      text TEXT NOT NULL,
      created REAL NOT NULL
    );
    CREATE VIRTUAL TABLE search USING fts5 (
      content, conversation UNINDEXED, position UNINDEXED, role UNINDEXED,
      tokenize = 'porter unicode61'
    );
    INSERT INTO search
      SELECT message ->> '$.content', conversation, position, message ->> '$.role' FROM messages
      WHERE message ->> '$.role' IN ('user', 'assistant') AND message ->> '$.content' != '';
    """,
    """
    ALTER TABLE conversations ADD COLUMN context TEXT NOT NULL DEFAULT '{}';
    """,
    """
    ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'about';
    ALTER TABLE conversations ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE conversations ADD COLUMN named INTEGER NOT NULL DEFAULT 0;
    """,
    # a memory's number is never another's, as a conversation's prompt names the memories of when
    # it began: one forgotten there must not be one remembered since
    """
    CREATE TABLE numbered (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      text TEXT NOT NULL,
      created REAL NOT NULL,
      category TEXT NOT NULL DEFAULT 'about'
    );
    INSERT INTO numbered SELECT id, text, created, category FROM memories;
    DROP TABLE memories;
    ALTER TABLE numbered RENAME TO memories;
    """,
    """
    CREATE TABLE tasks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      prompt TEXT NOT NULL,
      repeat TEXT NOT NULL,
      first REAL NOT NULL,
      next REAL NOT NULL,
      conversation TEXT,
      created REAL NOT NULL
    );
    """,
    """
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """,
    # a memory's dates: when it was last made, changed or said again, and a plan's last day; and
    # every memory forgotten or changed, as it was, to undo
    """
    ALTER TABLE memories ADD COLUMN confirmed REAL;
    UPDATE memories SET confirmed = created;
    ALTER TABLE memories ADD COLUMN until TEXT;
    CREATE TABLE forgotten (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      memory INTEGER NOT NULL,
      text TEXT NOT NULL,
      category TEXT NOT NULL,
      until TEXT,
      created REAL NOT NULL,
      change TEXT NOT NULL,
      by TEXT NOT NULL,
      at REAL NOT NULL
    );
    """,
    # the household: its people, each device paired to one, and whose each conversation, memory
    # and task is; a memory of no one's is the household's
    """
    CREATE TABLE people (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      owner INTEGER NOT NULL DEFAULT 0,
      created REAL NOT NULL
    );
    CREATE TABLE devices (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      person INTEGER NOT NULL REFERENCES people (id) ON DELETE CASCADE,
      name TEXT NOT NULL,
      token TEXT NOT NULL UNIQUE,
      created REAL NOT NULL,
      seen REAL NOT NULL
    );
    ALTER TABLE conversations ADD COLUMN person INTEGER REFERENCES people (id);
    ALTER TABLE memories ADD COLUMN person INTEGER REFERENCES people (id);
    ALTER TABLE forgotten ADD COLUMN person INTEGER;
    ALTER TABLE tasks ADD COLUMN person INTEGER REFERENCES people (id);
    """,
    # the household's characters, each a conversation's to play; one removed is retired, its
    # conversations its still
    """
    CREATE TABLE characters (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      about TEXT NOT NULL,
      created REAL NOT NULL,
      removed REAL
    );
    ALTER TABLE conversations ADD COLUMN character INTEGER REFERENCES characters (id);
    """,
    # the household's lists, as its shopping, and their items
    """
    CREATE TABLE lists (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      created REAL NOT NULL
    );
    CREATE TABLE items (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      list INTEGER NOT NULL REFERENCES lists (id) ON DELETE CASCADE,
      text TEXT NOT NULL,
      created REAL NOT NULL
    );
    """,
    # a conversation in a group chat, whose words are not the user's alone
    """
    ALTER TABLE conversations ADD COLUMN shared INTEGER NOT NULL DEFAULT 0;
    """,
    # a person of the household who is a child, whose conversations keep to a child's rules
    """
    ALTER TABLE people ADD COLUMN child INTEGER NOT NULL DEFAULT 0;
    """,
    # a task that is a check, which tells the user only if its condition holds
    """
    ALTER TABLE tasks ADD COLUMN condition TEXT;
    """,
]
_SEARCHED = ("user", "assistant")  # the roles of the messages search finds
# a conversation's columns as the apps list it
_SUMMARY = "id, title, created, updated, person"
# the memories a person knows: their own, and the household's
_KNOWN = "(person IS ? OR category = 'household')"


class Store:
    """The state at `path`, made if it is not there; safe to use from any thread."""

    def __init__(self, path: Path | str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        (version,) = self._db.execute("PRAGMA user_version").fetchone()
        for i, migration in enumerate(_MIGRATIONS[version:], version + 1):
            with self._db:
                self._db.execute("BEGIN")
                for statement in migration.split(";"):
                    self._db.execute(statement)
                self._db.execute(f"PRAGMA user_version = {i}")

    def conversations(self, person: int | None = None) -> list[dict[str, Any]]:
        """A person's conversations, without their messages, the latest updated first."""
        sql = f"SELECT {_SUMMARY} FROM conversations WHERE person IS ? ORDER BY updated DESC"
        return [dict(row) for row in self._query(sql, person)]

    def conversation(self, id: str) -> dict[str, Any] | None:
        rows = self._query(f"SELECT {_SUMMARY} FROM conversations WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def create(
        self, title: str, messages: list[dict[str, Any]], person: int | None = None
    ) -> dict[str, Any]:
        """A new conversation of a person's, of these messages."""
        id, now = uuid.uuid4().hex[:12], time.time()
        with self._lock, self._db:
            self._db.execute("BEGIN")
            self._db.execute(
                "INSERT INTO conversations (id, title, created, updated, person)"
                " VALUES (?, ?, ?, ?, ?)", (id, title, now, now, person),
            )  # fmt: skip
            self._insert(id, 0, messages)
        return {"id": id, "title": title, "created": now, "updated": now, "person": person}

    def context(self, id: str) -> dict[str, Any]:
        rows = self._query("SELECT context FROM conversations WHERE id = ?", id)
        return json.loads(rows[0]["context"]) if rows else {}

    def set_context(self, id: str, context: dict[str, Any]) -> None:
        with self._lock:
            sql = "UPDATE conversations SET context = ? WHERE id = ?"
            self._db.execute(sql, (json.dumps(context, ensure_ascii=False), id))

    def messages(self, id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT message FROM messages WHERE conversation = ? ORDER BY position", id
        )
        return [json.loads(row["message"]) for row in rows]

    def append(self, id: str, *messages: dict[str, Any]) -> None:
        """Appends messages to a conversation, which they update."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            (n,) = self._db.execute(
                "SELECT count(*) FROM messages WHERE conversation = ?", (id,)
            ).fetchone()
            self._insert(id, n, list(messages))
            self._db.execute("UPDATE conversations SET updated = ? WHERE id = ?", (time.time(), id))

    def truncate(self, id: str, n: int) -> None:
        """Keeps a conversation's first n messages, taking back those after."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            for table in ("messages", "search"):
                sql = f"DELETE FROM {table} WHERE conversation = ? AND position >= ?"
                self._db.execute(sql, (id, n))

    def delete(self, id: str) -> None:
        with self._lock, self._db:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM conversations WHERE id = ?", (id,))
            self._db.execute("DELETE FROM search WHERE conversation = ?", (id,))

    def search(
        self, query: str, person: int | None = None, exclude: str | None = None, since: float = 0,
        limit: int = 8,
    ) -> list[dict[str, Any]]:  # fmt: skip
        """What a person and the model said in their conversations that matches any of the
        query's words, the best first, of conversations updated since a time: each its
        conversation's id, title and last update, its role and position, and the words around."""
        if not (words := re.findall(r"\w+", query)):
            return []
        rows = self._query(
            "SELECT s.conversation, c.title, c.updated, s.role, s.position,"
            " snippet(search, 0, '', '', '…', 32) AS text"
            " FROM search s JOIN conversations c ON c.id = s.conversation"
            " WHERE search MATCH ? AND c.person IS ? AND s.conversation IS NOT ?"
            " AND c.updated >= ? ORDER BY rank LIMIT ?",
            " OR ".join(f'"{word}"' for word in words), person, exclude, since, limit,
        )  # fmt: skip
        return [dict(row) for row in rows]

    def recent(
        self, since: float, person: int | None = None, exclude: str | None = None, limit: int = 10
    ) -> list[dict]:
        """A person's conversations updated since a time, the latest first: each's id, title and
        last update, and its first message of the user's."""
        rows = self._query(
            "SELECT c.id AS conversation, c.title, c.updated, m.message ->> '$.content' AS text"
            " FROM conversations c JOIN messages m ON m.conversation = c.id AND m.position = 1"
            " WHERE c.updated >= ? AND c.person IS ? AND c.id IS NOT ?"
            " ORDER BY c.updated DESC LIMIT ?",
            since, person, exclude, limit,
        )  # fmt: skip
        return [dict(row) for row in rows]

    def idle(self, before: float) -> list[str]:
        """The conversations last updated before a time with messages not yet reviewed."""
        rows = self._query(
            "SELECT c.id FROM conversations c WHERE c.updated < ? AND c.reviewed <"
            " (SELECT count(*) FROM messages m WHERE m.conversation = c.id)", before
        )  # fmt: skip
        return [row["id"] for row in rows]

    def reviewed(self, id: str) -> int:
        rows = self._query("SELECT reviewed FROM conversations WHERE id = ?", id)
        return rows[0]["reviewed"] if rows else 0

    def mark_reviewed(self, id: str, n: int) -> None:
        with self._lock:
            self._db.execute("UPDATE conversations SET reviewed = ? WHERE id = ?", (n, id))

    def unnamed(self) -> list[str]:
        """The conversations the model has not named, the latest updated first."""
        rows = self._query("SELECT id FROM conversations WHERE NOT named ORDER BY updated DESC")
        return [row["id"] for row in rows]

    def name(self, id: str, title: str) -> None:
        """Names a conversation, as the model did."""
        with self._lock:
            sql = "UPDATE conversations SET title = ?, named = 1 WHERE id = ?"
            self._db.execute(sql, (title, id))

    def memories(self, person: int | None = None) -> list[dict[str, Any]]:
        """The memories a person knows, their own and the household's, the oldest first."""
        sql = f"SELECT * FROM memories WHERE {_KNOWN} ORDER BY id"
        return [dict(row) for row in self._query(sql, person)]

    def memory(self, id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM memories WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def add_memory(
        self, text: str, category: str, until: str | None = None, person: int | None = None
    ) -> dict[str, Any]:
        """A new memory of a person's, or of no one's, the household's."""
        rows = self._query(
            "INSERT INTO memories (text, created, category, confirmed, until, person) VALUES"
            " (?, ?, ?, ?, ?, ?) RETURNING *", text, now := time.time(), category, now, until,
            person,
        )  # fmt: skip
        return dict(rows[0])

    def replace_memory(
        self, id: int, text: str, category: str, until: str | None, by: str,
        person: int | None = None,
    ) -> dict[str, Any] | None:  # fmt: skip
        """Replaces a memory's text, category and last day, and whose it is, keeping what it was;
        returns it as it is now, if it is there."""
        return self._change(
            id, "replaced", by, "UPDATE memories SET text = ?, category = ?, until = ?,"
            " confirmed = ?, person = ? WHERE id = ? RETURNING *", text, category, until,
            time.time(), person, id,
        )  # fmt: skip

    def confirm_memory(self, id: int) -> None:
        """Dates a memory said again now."""
        self._query("UPDATE memories SET confirmed = ? WHERE id = ?", time.time(), id)

    def delete_memory(self, id: int, by: str) -> dict[str, Any] | None:
        """Forgets a memory, keeping what it was; returns it, if it was there."""
        sql = "DELETE FROM memories WHERE id = ? RETURNING *"
        return self._change(id, "forgotten", by, sql, id)

    def forgotten(self, person: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """The memories a person knew that were forgotten or changed, as they were, the latest
        first."""
        sql = f"SELECT * FROM forgotten WHERE {_KNOWN} ORDER BY id DESC LIMIT ?"
        return [dict(row) for row in self._query(sql, person, limit)]

    def forgetting(self, id: int) -> dict[str, Any] | None:
        """A memory forgotten or changed, as it was, by its number among the forgotten."""
        rows = self._query("SELECT * FROM forgotten WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def restore(self, id: int) -> dict[str, Any] | None:
        """Puts back a memory as it was before it was forgotten or changed, by its number among
        the forgotten; returns it, if it can be: not a change of a memory forgotten since."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            if (old := self._row("SELECT * FROM forgotten WHERE id = ?", id)) is None:
                return None
            if old["change"] == "forgotten":
                sql = ("INSERT OR IGNORE INTO memories (id, text, category, until, created, person,"
                       " confirmed) VALUES (?, ?, ?, ?, ?, ?, ?)")  # fmt: skip
                keys = ("memory", "text", "category", "until", "created", "person")
                values: tuple[Any, ...] = (*(old[k] for k in keys), time.time())
            else:
                sql = ("UPDATE memories SET text = ?, category = ?, until = ?, person = ?,"
                       " confirmed = ? WHERE id = ?")  # fmt: skip
                values = (old["text"], old["category"], old["until"], old["person"], time.time(),
                          old["memory"])  # fmt: skip
            if not self._db.execute(sql, values).rowcount:
                return None
            self._db.execute("DELETE FROM forgotten WHERE id = ?", (id,))
            return self._row("SELECT * FROM memories WHERE id = ?", old["memory"])

    def setting(self, key: str) -> Any:
        """A setting's value, or None if it has none."""
        rows = self._query("SELECT value FROM settings WHERE key = ?", key)
        return json.loads(rows[0]["value"]) if rows else None

    def set_setting(self, key: str, value: Any) -> None:
        """Sets a setting's value, or removes it, of None."""
        with self._lock:
            if value is None:
                self._db.execute("DELETE FROM settings WHERE key = ?", (key,))
            else:
                sql = "INSERT OR REPLACE INTO settings VALUES (?, ?)"
                self._db.execute(sql, (key, json.dumps(value, ensure_ascii=False)))

    def tasks(self, person: int | None = None, everyone: bool = False) -> list[dict[str, Any]]:
        """A person's tasks, or `everyone`'s, the next due first."""
        if everyone:
            return [dict(row) for row in self._query("SELECT * FROM tasks ORDER BY next")]
        sql = "SELECT * FROM tasks WHERE person IS ? ORDER BY next"
        return [dict(row) for row in self._query(sql, person)]

    def task(self, id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM tasks WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def add_task(
        self, prompt: str, repeat: str, first: float, conversation: str | None,
        person: int | None = None, condition: str | None = None, next: float | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """A task of a person's, a check if it has a condition, next at its first time or at
        `next`, one of its repeats after."""
        rows = self._query(
            "INSERT INTO tasks (prompt, repeat, first, next, conversation, created, person,"
            " condition) VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING *",
            prompt, repeat, first, first if next is None else next, conversation, time.time(),
            person, condition,
        )  # fmt: skip
        return dict(rows[0])

    def due(self, now: float) -> list[dict[str, Any]]:
        """The tasks due by a time, the earliest first."""
        rows = self._query("SELECT * FROM tasks WHERE next <= ? ORDER BY next", now)
        return [dict(row) for row in rows]

    def advance(self, id: int, due: float | None, conversation: str) -> None:
        """Sets when a task runs next, and where, or deletes it if it is done for good."""
        with self._lock:
            if due is None:
                self._db.execute("DELETE FROM tasks WHERE id = ?", (id,))
            else:
                sql = "UPDATE tasks SET next = ?, conversation = ? WHERE id = ?"
                self._db.execute(sql, (due, conversation, id))

    def delete_task(self, id: int) -> dict[str, Any] | None:
        rows = self._query("DELETE FROM tasks WHERE id = ? RETURNING *", id)
        return dict(rows[0]) if rows else None

    def people(self) -> list[dict[str, Any]]:
        """The household's people, the owner first."""
        return [dict(row) for row in self._query("SELECT * FROM people ORDER BY id")]

    def person(self, id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM people WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def add_person(self, name: str) -> dict[str, Any]:
        """A new person of the household's; the first its owner, whose all that was no one's
        becomes, but the household's memories."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            first = self._db.execute("SELECT count(*) FROM people").fetchone()[0] == 0
            person = self._row(
                "INSERT INTO people (name, owner, created) VALUES (?, ?, ?) RETURNING *",
                name, int(first), time.time(),
            )  # fmt: skip
            assert person is not None
            if first:
                for table in ("conversations", "tasks"):
                    sql = f"UPDATE {table} SET person = ? WHERE person IS NULL"
                    self._db.execute(sql, (person["id"],))
                for table in ("memories", "forgotten"):
                    sql = f"UPDATE {table} SET person = ? WHERE person IS NULL"
                    self._db.execute(f"{sql} AND category != 'household'", (person["id"],))
        return person

    def rename_person(self, id: int, name: str) -> None:
        self._query("UPDATE people SET name = ? WHERE id = ?", name, id)

    def remove_person(self, id: int) -> None:
        """Removes a person who is not the owner, and all that is theirs: their conversations,
        memories, tasks and devices."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            for (conversation,) in self._db.execute(
                "SELECT id FROM conversations WHERE person = ?", (id,)
            ).fetchall():
                self._db.execute("DELETE FROM search WHERE conversation = ?", (conversation,))
            for table in ("conversations", "memories", "forgotten", "tasks", "devices"):
                self._db.execute(f"DELETE FROM {table} WHERE person = ?", (id,))
            self._db.execute("DELETE FROM people WHERE id = ? AND NOT owner", (id,))

    def devices(self) -> list[dict[str, Any]]:
        """The devices paired, without their tokens, the latest seen first."""
        sql = "SELECT id, person, name, created, seen FROM devices ORDER BY seen DESC"
        return [dict(row) for row in self._query(sql)]

    def device(self, token: str) -> dict[str, Any] | None:
        """The device of a token's hash, without it, if it is paired."""
        sql = "SELECT id, person, name, created, seen FROM devices WHERE token = ?"
        rows = self._query(sql, token)
        return dict(rows[0]) if rows else None

    def add_device(self, person: int, name: str, token: str) -> dict[str, Any]:
        """Pairs a device to a person, by its token's hash."""
        rows = self._query(
            "INSERT INTO devices (person, name, token, created, seen) VALUES (?, ?, ?, ?, ?)"
            " RETURNING id, person, name, created, seen",
            person, name, token, now := time.time(), now,
        )  # fmt: skip
        return dict(rows[0])

    def seen(self, id: int) -> None:
        self._query("UPDATE devices SET seen = ? WHERE id = ?", time.time(), id)

    def remove_device(self, id: int) -> bool:
        return bool(self._query("DELETE FROM devices WHERE id = ? RETURNING id", id))

    def _change(self, id: int, change: str, by: str, sql: str, *parameters: Any) -> dict | None:
        # changes a memory by sql, keeping what it was in forgotten, `by` whom: the conversation,
        # the review, or the user in the app; returns it as sql does, if it is there
        with self._lock, self._db:
            self._db.execute("BEGIN")
            if (old := self._row("SELECT * FROM memories WHERE id = ?", id)) is None:
                return None
            self._db.execute(
                "INSERT INTO forgotten (memory, text, category, until, created, person, change,"
                " by, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (id, old["text"], old["category"], old["until"], old["created"], old["person"],
                 change, by, time.time()),
            )  # fmt: skip
            return self._row(sql, *parameters)

    def _row(self, sql: str, *parameters: Any) -> dict[str, Any] | None:
        # the first row of sql, as a dict, in a transaction the caller holds the lock of
        cursor = self._db.execute(sql, parameters)
        cursor.row_factory = sqlite3.Row
        return dict(row) if (row := cursor.fetchone()) is not None else None

    def _insert(self, id: str, start: int, messages: list[dict[str, Any]]) -> None:
        rows = [(id, start + i, json.dumps(m, ensure_ascii=False)) for i, m in enumerate(messages)]
        self._db.executemany("INSERT INTO messages VALUES (?, ?, ?)", rows)
        searched = [
            (m["content"], id, start + i, m["role"])
            for i, m in enumerate(messages)
            if m["role"] in _SEARCHED and isinstance(m.get("content"), str) and m["content"]
        ]
        self._db.executemany("INSERT INTO search VALUES (?, ?, ?, ?)", searched)

    def _query(self, sql: str, *parameters: Any) -> list[sqlite3.Row]:
        with self._lock:
            cursor = self._db.execute(sql, parameters)
            cursor.row_factory = sqlite3.Row
            return cursor.fetchall()
