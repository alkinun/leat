"""The agent's state, in one SQLite file: conversations and their messages.

A message is a dict in OpenAI's chat format, as the model reads it, and its "info": what only people
see of it, such as the model that wrote it and how fast. A conversation's messages are only ever
appended, but for a turn taken back whole, so that each step's prompt extends the last's.
"""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
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
"""


class Store:
    """The state at `path`, made if it is not there; safe to use from any thread."""

    def __init__(self, path: Path | str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)

    def conversations(self) -> list[dict[str, Any]]:
        """Every conversation, without its messages, the latest updated first."""
        rows = self._query("SELECT * FROM conversations ORDER BY updated DESC")
        return [dict(row) for row in rows]

    def conversation(self, id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM conversations WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def create(self, title: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """A new conversation of these messages."""
        id, now = uuid.uuid4().hex[:12], time.time()
        with self._lock, self._db:
            self._db.execute("BEGIN")
            self._db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (id, title, now, now))
            self._insert(id, 0, messages)
        return {"id": id, "title": title, "created": now, "updated": now}

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
        with self._lock:
            sql = "DELETE FROM messages WHERE conversation = ? AND position >= ?"
            self._db.execute(sql, (id, n))

    def delete(self, id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM conversations WHERE id = ?", (id,))

    def _insert(self, id: str, start: int, messages: list[dict[str, Any]]) -> None:
        rows = [(id, start + i, json.dumps(m, ensure_ascii=False)) for i, m in enumerate(messages)]
        self._db.executemany("INSERT INTO messages VALUES (?, ?, ?)", rows)

    def _query(self, sql: str, *parameters: Any) -> list[sqlite3.Row]:
        with self._lock:
            cursor = self._db.execute(sql, parameters)
            cursor.row_factory = sqlite3.Row
            return cursor.fetchall()
