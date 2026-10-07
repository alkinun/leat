"""The agent's state, in one SQLite file: conversations and their messages, and memories.

A message is a dict in OpenAI's chat format, as the model reads it, and its "info": what only people
see of it, such as the model that wrote it and how fast. A conversation's messages are only ever
appended, but for a turn taken back whole, so that each step's prompt extends the last's. What the
user and the model said is indexed for full-text search, the tools' answers not.

A memory is a short sentence about the user, which every conversation begun after it knows.

A conversation's context is the state of what its prompt keeps of it, as leat.agent.context fits it
to the model's.
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
]
_SEARCHED = ("user", "assistant")  # the roles of the messages search finds
_SUMMARY = "id, title, created, updated"  # a conversation's columns, but its context


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

    def conversations(self) -> list[dict[str, Any]]:
        """Every conversation, without its messages, the latest updated first."""
        rows = self._query(f"SELECT {_SUMMARY} FROM conversations ORDER BY updated DESC")
        return [dict(row) for row in rows]

    def conversation(self, id: str) -> dict[str, Any] | None:
        rows = self._query(f"SELECT {_SUMMARY} FROM conversations WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def create(self, title: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """A new conversation of these messages."""
        id, now = uuid.uuid4().hex[:12], time.time()
        with self._lock, self._db:
            self._db.execute("BEGIN")
            sql = "INSERT INTO conversations (id, title, created, updated) VALUES (?, ?, ?, ?)"
            self._db.execute(sql, (id, title, now, now))
            self._insert(id, 0, messages)
        return {"id": id, "title": title, "created": now, "updated": now}

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

    def search(self, query: str, exclude: str | None = None, limit: int = 8) -> list[dict]:
        """What the user and the model said that matches any of the query's words, the best first,
        each its conversation's id, title and last update, its role, and the words around."""
        if not (words := re.findall(r"\w+", query)):
            return []
        rows = self._query(
            "SELECT s.conversation, c.title, c.updated, s.role,"
            " snippet(search, 0, '', '', '…', 24) AS text"
            " FROM search s JOIN conversations c ON c.id = s.conversation"
            " WHERE search MATCH ? AND s.conversation IS NOT ? ORDER BY rank LIMIT ?",
            " OR ".join(f'"{word}"' for word in words),
            exclude,
            limit,
        )
        return [dict(row) for row in rows]

    def memories(self) -> list[dict[str, Any]]:
        """The memories, the oldest first."""
        return [dict(row) for row in self._query("SELECT * FROM memories ORDER BY id")]

    def add_memory(self, text: str) -> dict[str, Any]:
        now, sql = time.time(), "INSERT INTO memories (text, created) VALUES (?, ?)"
        with self._lock:
            cursor = self._db.execute(sql, (text, now))
        return {"id": cursor.lastrowid, "text": text, "created": now}

    def delete_memory(self, id: int) -> dict[str, Any] | None:
        """Deletes a memory, and returns it, if it is there."""
        rows = self._query("DELETE FROM memories WHERE id = ? RETURNING *", id)
        return dict(rows[0]) if rows else None

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
