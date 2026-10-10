"""The agent's state, in one SQLite file: conversations and their messages, and the accounts.

A message is a dict in OpenAI's chat format, as the model reads it, and its "info": what only people
see of it, such as the model that wrote it and how fast. A conversation's messages are only ever
appended, but for a turn taken back whole, so that each step's prompt extends the last's. A
conversation is named by the model once.

A conversation's context is the state of what its prompt keeps of it, as leat.agent.context fits it
to the model's.

The accounts are the box's people, the first its owner, and the devices paired to each, known by the
hash of a secret each holds. A conversation is a person's. Before the box has its first person,
every conversation is no one's, and becomes the owner's.

A project is a client's or a matter's: its files, instructions for the model, and the conversations
held in it. It is a person's own, or shared with everyone on the box; a conversation in it is still
its person's alone, and seen only while they see the project.
"""

import json
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
    # the household's features gone: memories, tasks, characters, lists, the search of what was said
    # the settings, a conversation's character and group chat and how far it was reviewed, and
    # whether a person is a child. The conversations are made anew without those columns, which
    # SQLite drops from no table that refers to another; their messages kept, as foreign keys are
    # not enforced while the state migrates
    """
    CREATE TABLE kept (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      created REAL NOT NULL,
      updated REAL NOT NULL,
      context TEXT NOT NULL DEFAULT '{}',
      named INTEGER NOT NULL DEFAULT 0,
      person INTEGER REFERENCES people (id)
    );
    INSERT INTO kept SELECT id, title, created, updated, context, named, person FROM conversations;
    DROP TABLE conversations;
    ALTER TABLE kept RENAME TO conversations;
    DROP TABLE memories;
    DROP TABLE forgotten;
    DROP TABLE tasks;
    DROP TABLE characters;
    DROP TABLE items;
    DROP TABLE lists;
    DROP TABLE search;
    DROP TABLE settings;
    ALTER TABLE people DROP COLUMN child;
    """,
    # the projects, each a person's own or shared, and the project a conversation is held in
    """
    CREATE TABLE projects (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      instructions TEXT NOT NULL DEFAULT '',
      person INTEGER REFERENCES people (id),
      shared INTEGER NOT NULL DEFAULT 0,
      created REAL NOT NULL,
      updated REAL NOT NULL
    );
    ALTER TABLE conversations ADD COLUMN project TEXT REFERENCES projects (id);
    """,
]
# a conversation's columns as the apps list it
_SUMMARY = "id, title, created, updated, person, project"
# the projects a person sees: their own, and those shared
_SEEN = "(shared OR person IS ?)"


class Store:
    """The state at `path`, made if it is not there; safe to use from any thread."""

    def __init__(self, path: Path | str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._db.execute("PRAGMA journal_mode = WAL")
        (version,) = self._db.execute("PRAGMA user_version").fetchone()
        for i, migration in enumerate(_MIGRATIONS[version:], version + 1):
            with self._db:
                self._db.execute("BEGIN")
                for statement in migration.split(";"):
                    self._db.execute(statement)
                if self._db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise sqlite3.IntegrityError(f"the state's migration {i} broke a reference")
                self._db.execute(f"PRAGMA user_version = {i}")
        # enforced once migrated, as a table made anew is dropped before its copy takes its name,
        # which would delete what refers to it
        self._db.execute("PRAGMA foreign_keys = ON")

    def conversations(self, person: int | None = None) -> list[dict[str, Any]]:
        """A person's conversations, without their messages, the latest updated first: those of
        no project, and of the projects they see."""
        sql = (
            f"SELECT {_SUMMARY} FROM conversations WHERE person IS ? AND (project IS NULL OR"
            f" project IN (SELECT id FROM projects WHERE {_SEEN})) ORDER BY updated DESC"
        )
        return [dict(row) for row in self._query(sql, person, person)]

    def conversations_in(self, project: str) -> list[dict[str, Any]]:
        """The conversations held in a project, whoever's."""
        sql = f"SELECT {_SUMMARY} FROM conversations WHERE project = ?"
        return [dict(row) for row in self._query(sql, project)]

    def conversation(self, id: str) -> dict[str, Any] | None:
        rows = self._query(f"SELECT {_SUMMARY} FROM conversations WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def create(
        self, title: str, messages: list[dict[str, Any]], person: int | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """A new conversation of a person's, of these messages, in a project if one."""
        id, now = uuid.uuid4().hex[:12], time.time()
        with self._lock, self._db:
            self._db.execute("BEGIN")
            self._db.execute(
                "INSERT INTO conversations (id, title, created, updated, person, project)"
                " VALUES (?, ?, ?, ?, ?, ?)", (id, title, now, now, person, project),
            )  # fmt: skip
            self._insert(id, 0, messages)
        return {"id": id, "title": title, "created": now, "updated": now, "person": person,
                "project": project}  # fmt: skip

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
            sql = "DELETE FROM messages WHERE conversation = ? AND position >= ?"
            self._db.execute(sql, (id, n))

    def delete(self, id: str) -> None:
        with self._lock, self._db:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM conversations WHERE id = ?", (id,))

    def unnamed(self) -> list[str]:
        """The conversations the model has not named, the latest updated first."""
        rows = self._query("SELECT id FROM conversations WHERE NOT named ORDER BY updated DESC")
        return [row["id"] for row in rows]

    def name(self, id: str, title: str) -> None:
        """Names a conversation, as the model did."""
        with self._lock:
            sql = "UPDATE conversations SET title = ?, named = 1 WHERE id = ?"
            self._db.execute(sql, (title, id))

    def projects(self, person: int | None = None) -> list[dict[str, Any]]:
        """The projects a person sees, the latest updated first."""
        sql = f"SELECT * FROM projects WHERE {_SEEN} ORDER BY updated DESC"
        return [dict(row) for row in self._query(sql, person)]

    def project(self, id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM projects WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def add_project(
        self, name: str, person: int | None = None, instructions: str = "", shared: bool = False
    ) -> dict[str, Any]:
        """A new project of a person's."""
        rows = self._query(
            "INSERT INTO projects (id, name, instructions, person, shared, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *",
            uuid.uuid4().hex[:12], name, instructions, person, int(shared), now := time.time(), now,
        )  # fmt: skip
        return dict(rows[0])

    def change_project(self, id: str, **changes: Any) -> dict[str, Any] | None:
        """Changes a project's name, instructions or whether it is shared, as `changes` give them;
        returns it as it now is, or None if there is none."""
        assert set(changes) <= {"name", "instructions", "shared"}
        sets = "".join(f"{key} = ?, " for key in changes)
        sql = f"UPDATE projects SET {sets}updated = ? WHERE id = ? RETURNING *"
        rows = self._query(sql, *changes.values(), time.time(), id)
        return dict(rows[0]) if rows else None

    def delete_project(self, id: str) -> None:
        """Deletes a project, with every conversation in it, whoever's."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM conversations WHERE project = ?", (id,))
            self._db.execute("DELETE FROM projects WHERE id = ?", (id,))

    def people(self) -> list[dict[str, Any]]:
        """The box's people, the owner first."""
        return [dict(row) for row in self._query("SELECT * FROM people ORDER BY id")]

    def person(self, id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM people WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def add_person(self, name: str) -> dict[str, Any]:
        """A new person of the box's; the first its owner, whose every conversation and project
        that was no one's becomes."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            first = self._db.execute("SELECT count(*) FROM people").fetchone()[0] == 0
            person = self._row(
                "INSERT INTO people (name, owner, created) VALUES (?, ?, ?) RETURNING *",
                name, int(first), time.time(),
            )  # fmt: skip
            assert person is not None
            if first:
                for table in ("conversations", "projects"):
                    sql = f"UPDATE {table} SET person = ? WHERE person IS NULL"
                    self._db.execute(sql, (person["id"],))
        return person

    def remove_person(self, id: int) -> None:
        """Removes a person who is not the owner, and all that is theirs: their conversations,
        devices and own projects, with the conversations in those. The projects they shared
        become the owner's."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            own = "SELECT id FROM projects WHERE person = ? AND NOT shared"
            self._db.execute(f"DELETE FROM conversations WHERE project IN ({own})", (id,))
            self._db.execute("DELETE FROM projects WHERE person = ? AND NOT shared", (id,))
            owner = "SELECT id FROM people WHERE owner"
            self._db.execute(f"UPDATE projects SET person = ({owner}) WHERE person = ?", (id,))
            for table in ("conversations", "devices"):
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

    def _row(self, sql: str, *parameters: Any) -> dict[str, Any] | None:
        # the first row of sql, as a dict, in a transaction the caller holds the lock of
        cursor = self._db.execute(sql, parameters)
        cursor.row_factory = sqlite3.Row
        return dict(row) if (row := cursor.fetchone()) is not None else None

    def _insert(self, id: str, start: int, messages: list[dict[str, Any]]) -> None:
        rows = [(id, start + i, json.dumps(m, ensure_ascii=False)) for i, m in enumerate(messages)]
        self._db.executemany("INSERT INTO messages VALUES (?, ?, ?)", rows)

    def _query(self, sql: str, *parameters: Any) -> list[sqlite3.Row]:
        with self._lock:
            cursor = self._db.execute(sql, parameters)
            cursor.row_factory = sqlite3.Row
            return cursor.fetchall()
