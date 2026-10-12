"""The agent's state, in one SQLite file: conversations and their messages, and the accounts.

A message is a dict in OpenAI's chat format, as the model reads it, and its "info": what only people
see of it, such as the model that wrote it and how fast. A conversation's messages are only ever
appended, but for a turn taken back whole, so that each step's prompt extends the last's. A
conversation is named by the model once.

A conversation's context is the state of what its prompt keeps of it, as leat.agent.context fits it
to the model's.

The accounts are the box's people, its admins among them, the first always, and the devices paired
to each, known by the hash of a secret each holds. The box has an admin always, once it has a
person. A conversation is a person's. Before the box has its first person, every conversation is
no one's, and becomes theirs.

A project is a client's or a matter's: its files, instructions for the model, and the conversations
held in it. It is the person's who made it, and seen by them alone, or by the people they chose, its
members, too, or shared with everyone on the box; a conversation in it is still its person's alone,
and seen only while they see the project.

A workflow is a request saved to make again, by its name: a project's, seen by those who see the
project, or a person's own. One may run by itself on a schedule, as leat.agent.schedule says, as
the person who set it, counted from when it was set or last ran.

The settings are the box's, each a value of JSON by its key, as whether it is kept from the
internet.

The activity is what was done on the box, by whom, and where, as its admins look back on it: a
chat begun, a file uploaded or read for one, a project shared; never what anyone asked, nor what
was answered. It is kept a year.
"""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# the state, as a new one is made: of version BASE, SQLite's user_version, as the 25 migrations that
# made it up to the box's first were folded into it
SCHEMA = """
-- the conversations, each a person's, or no one's before the box has its first person, in a
-- project or not; its context, what its prompt keeps of it; whether the model named it
CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  context TEXT NOT NULL DEFAULT '{}',
  named INTEGER NOT NULL DEFAULT 0,
  person INTEGER REFERENCES people (id),
  project TEXT REFERENCES projects (id)
);
-- their messages, each a dict as JSON, in order
CREATE TABLE messages (
  conversation TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  message TEXT NOT NULL,
  PRIMARY KEY (conversation, position)
);
-- the box's people, its admins among them
CREATE TABLE people (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  admin INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL
);
-- each device paired to a person, by its secret's hash: an app's, of a browser's cookie, or a
-- network drive's, of a password its computer keeps, which reaches the person's files alone
CREATE TABLE devices (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person INTEGER NOT NULL REFERENCES people (id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  token TEXT NOT NULL UNIQUE,
  created REAL NOT NULL,
  seen REAL NOT NULL,
  kind TEXT NOT NULL DEFAULT 'app'
);
-- the projects, each a person's own, shared with everyone or with its members; and the folder its
-- files are synced with, if one, as leat.agent.sync has it, when they last were, and why they
-- were not, if they were not
CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  instructions TEXT NOT NULL DEFAULT '',
  person INTEGER REFERENCES people (id),
  shared INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  source TEXT,
  synced REAL,
  unsynced TEXT
);
-- the people a project is shared with, besides whoever made it, if it is not with everyone
CREATE TABLE members (
  project TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
  person INTEGER NOT NULL REFERENCES people (id) ON DELETE CASCADE,
  PRIMARY KEY (project, person)
);
-- the workflows, each a request saved by its name, a project's or a person's own; its schedule,
-- if it runs by itself, the time it counts from, and who set it, whose its runs are
CREATE TABLE workflows (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  prompt TEXT NOT NULL,
  project TEXT REFERENCES projects (id) ON DELETE CASCADE,
  person INTEGER REFERENCES people (id) ON DELETE CASCADE,
  created REAL NOT NULL,
  schedule TEXT,
  since REAL,
  by INTEGER REFERENCES people (id) ON DELETE SET NULL
);
-- the box's settings, each a value of JSON by its key
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- what was done on the box: when, by whom, or by Leat for whom, what, in which project by its
-- name, and of what
CREATE TABLE activity (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at REAL NOT NULL,
  person INTEGER,
  leat INTEGER NOT NULL DEFAULT 0,
  action TEXT NOT NULL,
  place TEXT,
  detail TEXT
);
CREATE INDEX activity_at ON activity (at);
"""
BASE = 25
# each takes the state from the version BASE and its index to the next
_MIGRATIONS: list[str] = []
KEPT = 365 * 86400  # seconds the activity is kept
# a conversation's columns as the apps list it
_SUMMARY = "id, title, created, updated, person, project"
# the projects a person, :me, sees: their own, those they are a member of, and those shared
_SEEN = "(shared OR person IS :me OR id IN (SELECT project FROM members WHERE person IS :me))"
# a project's columns, with its members' ids
_PROJECT = "*, (SELECT json_group_array(person) FROM members WHERE project = projects.id) members"
# the projects of whoever made them alone, shared with no one
_OWN = "NOT shared AND id NOT IN (SELECT project FROM members)"


class Store:
    """The state at `path`, made if it is not there; safe to use from any thread."""

    def __init__(self, path: Path | str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._db.execute("PRAGMA journal_mode = WAL")
        (version,) = self._db.execute("PRAGMA user_version").fetchone()
        if version == 0 and self._db.execute("SELECT 1 FROM sqlite_schema").fetchone() is None:
            self._db.executescript(f"BEGIN; {SCHEMA}; PRAGMA user_version = {BASE}; COMMIT;")
            version = BASE
        if version < BASE:
            raise ValueError(f"{path} is the state of a Leat older than any this one reads")
        for i, migration in enumerate(_MIGRATIONS[version - BASE :], version + 1):
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
            f"SELECT {_SUMMARY} FROM conversations WHERE person IS :me AND (project IS NULL OR"
            f" project IN (SELECT id FROM projects WHERE {_SEEN})) ORDER BY updated DESC"
        )
        return [dict(row) for row in self._query(sql, me=person)]

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
        sql = f"SELECT {_PROJECT} FROM projects WHERE {_SEEN} ORDER BY updated DESC"
        return [_project(row) for row in self._query(sql, me=person)]

    def synced_projects(self) -> list[dict[str, Any]]:
        """The projects whose files are synced with a folder, whoever's."""
        sql = f"SELECT {_PROJECT} FROM projects WHERE source IS NOT NULL"
        return [_project(row) for row in self._query(sql)]

    def project(self, id: str) -> dict[str, Any] | None:
        """A project, with its members' ids, `members`, the first first."""
        rows = self._query(f"SELECT {_PROJECT} FROM projects WHERE id = ?", id)
        return _project(rows[0]) if rows else None

    def add_project(
        self, name: str, person: int | None = None, instructions: str = "", shared: bool = False
    ) -> dict[str, Any]:
        """A new project of a person's."""
        self._query(
            "INSERT INTO projects (id, name, instructions, person, shared, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            id := uuid.uuid4().hex[:12], name, instructions, person, int(shared),
            now := time.time(), now,
        )  # fmt: skip
        project = self.project(id)
        assert project is not None
        return project

    def change_project(self, id: str, **changes: Any) -> None:
        """Changes a project's name, instructions, whether it is shared with everyone, or its
        members, by their ids, as `changes` give them, which make it the latest changed; or the
        folder it is kept a copy of, and when it was last synced and why not, which do not."""
        assert set(changes) <= {"name", "instructions", "shared", "members", "source", "synced",
                                "unsynced"}  # fmt: skip
        members = changes.pop("members", None)
        with self._lock, self._db:
            self._db.execute("BEGIN")
            if members is not None or set(changes) - {"source", "synced", "unsynced"}:
                changes["updated"] = time.time()
            sets = ", ".join(f"{key} = ?" for key in changes) or "id = id"
            sql = f"UPDATE projects SET {sets} WHERE id = ? RETURNING person"
            if (made := self._row(sql, *changes.values(), id)) is None:
                return
            if members is not None:
                self._db.execute("DELETE FROM members WHERE project = ?", (id,))
                rows = [(id, m) for m in set(members) - {made["person"]}]
                self._db.executemany("INSERT INTO members VALUES (?, ?)", rows)

    def delete_project(self, id: str) -> None:
        """Deletes a project, with every conversation in it, whoever's."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            for table in ("conversations", "workflows"):
                self._db.execute(f"DELETE FROM {table} WHERE project = ?", (id,))
            self._db.execute("DELETE FROM projects WHERE id = ?", (id,))

    def workflows(self, person: int | None = None) -> list[dict[str, Any]]:
        """The workflows a person sees, their own and those of the projects they see, by name."""
        sql = (
            "SELECT * FROM workflows WHERE (project IS NULL AND person IS :me) OR project IN"
            f" (SELECT id FROM projects WHERE {_SEEN}) ORDER BY name COLLATE NOCASE"
        )
        return [dict(row) for row in self._query(sql, me=person)]

    def workflow(self, id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM workflows WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def add_workflow(
        self, name: str, prompt: str, person: int | None = None, project: str | None = None
    ) -> dict[str, Any]:
        """A new workflow: a project's, if one, or else a person's own."""
        rows = self._query(
            "INSERT INTO workflows (name, prompt, project, person, created) VALUES (?, ?, ?, ?, ?)"
            " RETURNING *", name, prompt, project, None if project else person, time.time(),
        )  # fmt: skip
        return dict(rows[0])

    def delete_workflow(self, id: int) -> None:
        self._query("DELETE FROM workflows WHERE id = ?", id)

    def schedule_workflow(self, id: int, schedule: str | None, by: int | None) -> None:
        """Has a workflow run by itself on a schedule, as a person, counted from now; or not, of
        None."""
        self._query(
            "UPDATE workflows SET schedule = ?, since = ?, by = ? WHERE id = ?",
            schedule, time.time() if schedule else None, by if schedule else None, id,
        )  # fmt: skip

    def ran_workflow(self, id: int, at: float) -> None:
        """Counts a workflow's schedule from a time it ran."""
        self._query("UPDATE workflows SET since = ? WHERE id = ?", at, id)

    def scheduled(self) -> list[dict[str, Any]]:
        """The workflows that run by themselves."""
        rows = self._query("SELECT * FROM workflows WHERE schedule IS NOT NULL")
        return [dict(row) for row in rows]

    def use(self, since: float) -> list[dict[str, Any]]:
        """Each person's use since a time: the conversations they began, and the messages they
        sent, counted alone, not read; the first person first."""
        sql = """
          SELECT p.id, p.name,
            (SELECT count(*) FROM conversations c WHERE c.person = p.id AND c.created >= ?1) chats,
            (SELECT count(*) FROM messages m JOIN conversations c ON c.id = m.conversation
             WHERE c.person = p.id AND m.message ->> '$.role' = 'user'
             AND m.message ->> '$.info.at' >= ?1) messages
          FROM people p ORDER BY p.id"""
        return [dict(row) for row in self._query(sql, since)]

    def note(
        self, person: int | None, action: str, place: str | None = None,
        detail: str | None = None, leat: bool = False,
    ) -> None:  # fmt: skip
        """Notes something done, by a person, or by Leat for one, or of none, in a project, by its
        name, if in one, and of what; forgetting what was done a year ago."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            now, sql = time.time(), "INSERT INTO activity VALUES (NULL, ?, ?, ?, ?, ?, ?)"
            self._db.execute(sql, (now, person, int(leat), action, place, detail))
            self._db.execute("DELETE FROM activity WHERE at < ?", (now - KEPT,))

    def activity(self, before: int | None = None, n: int = 100) -> list[dict[str, Any]]:
        """What was done, the latest first, n at most, before the one of the id `before`: each with
        the name of whoever did it, if they are still the box's."""
        sql = (
            "SELECT a.id, a.at, a.leat, a.action, a.place, a.detail, p.name AS person"
            " FROM activity a"
            " LEFT JOIN people p ON p.id = a.person WHERE ?1 IS NULL OR a.id < ?1"
            " ORDER BY a.id DESC LIMIT ?2"
        )
        return [dict(row) for row in self._query(sql, before, n)]

    def setting(self, key: str) -> Any:
        """A setting's value, or None if it has none."""
        rows = self._query("SELECT value FROM settings WHERE key = ?", key)
        return json.loads(rows[0]["value"]) if rows else None

    def set_setting(self, key: str, value: Any) -> None:
        """Sets a setting, or of None takes it away."""
        if value is None:
            self._query("DELETE FROM settings WHERE key = ?", key)
        else:
            sql = "INSERT OR REPLACE INTO settings VALUES (?, ?)"
            self._query(sql, key, json.dumps(value, ensure_ascii=False))

    def people(self) -> list[dict[str, Any]]:
        """The box's people, the first first."""
        return [dict(row) for row in self._query("SELECT * FROM people ORDER BY id")]

    def person(self, id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM people WHERE id = ?", id)
        return dict(rows[0]) if rows else None

    def add_person(self, name: str, admin: bool = False) -> dict[str, Any]:
        """A new person of the box's, an admin or not; the first an admin always, whose every
        conversation and project that was no one's becomes."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            first = self._db.execute("SELECT count(*) FROM people").fetchone()[0] == 0
            person = self._row(
                "INSERT INTO people (name, admin, created) VALUES (?, ?, ?) RETURNING *",
                name, int(first or admin), time.time(),
            )  # fmt: skip
            assert person is not None
            if first:
                for table in ("conversations", "projects", "workflows"):
                    sql = f"UPDATE {table} SET person = ? WHERE person IS NULL"
                    self._db.execute(sql, (person["id"],))
        return person

    def rename(self, id: int, name: str) -> bool:
        """Renames a person; whether there is one of the id."""
        return bool(self._query("UPDATE people SET name = ? WHERE id = ? RETURNING id", name, id))

    def set_admin(self, id: int, admin: bool) -> None:
        """Makes a person an admin, or no longer one. Raises LookupError if there is no such
        person, ValueError of the last admin, as the box has one always."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            if self._row("SELECT id FROM people WHERE id = ?", id) is None:
                raise LookupError(f"there is no person {id}")
            self._db.execute("UPDATE people SET admin = ? WHERE id = ?", (int(admin), id))
            if not self._db.execute("SELECT count(*) FROM people WHERE admin").fetchone()[0]:
                raise ValueError("this Leat needs an admin: make someone else one first")

    def remove_person(self, id: int, heir: int) -> None:
        """Removes a person, and all that is theirs: their conversations, devices and own projects,
        shared with no one, with the conversations in those. A project they shared with chosen
        people becomes its first member's, so that no one else sees it; one shared with everyone,
        their heir's. Raises ValueError of the last admin, as the box has one always."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            own = f"SELECT id FROM projects WHERE person = ? AND {_OWN}"
            for table in ("conversations", "workflows"):
                self._db.execute(f"DELETE FROM {table} WHERE project IN ({own})", (id,))
            self._db.execute(f"DELETE FROM projects WHERE person = ? AND {_OWN}", (id,))
            first = "(SELECT min(person) FROM members WHERE project = projects.id)"
            sql = f"UPDATE projects SET person = {first} WHERE person = ? AND NOT shared"
            self._db.execute(sql, (id,))
            made = "(SELECT person FROM projects WHERE id = project)"
            self._db.execute(f"DELETE FROM members WHERE person = {made}")
            self._db.execute("UPDATE projects SET person = ? WHERE person = ?", (heir, id))
            for table in ("conversations", "devices", "workflows"):
                self._db.execute(f"DELETE FROM {table} WHERE person = ?", (id,))
            self._db.execute("DELETE FROM people WHERE id = ?", (id,))
            if not self._db.execute("SELECT count(*) FROM people WHERE admin").fetchone()[0]:
                raise ValueError("this Leat needs an admin: make someone else one first")

    def devices(self) -> list[dict[str, Any]]:
        """The devices paired, without their tokens, the latest seen first."""
        sql = "SELECT id, person, name, kind, created, seen FROM devices ORDER BY seen DESC"
        return [dict(row) for row in self._query(sql)]

    def device(self, token: str) -> dict[str, Any] | None:
        """The device of a token's hash, without it, if it is paired."""
        sql = "SELECT id, person, name, kind, created, seen FROM devices WHERE token = ?"
        rows = self._query(sql, token)
        return dict(rows[0]) if rows else None

    def add_device(self, person: int, name: str, token: str, kind: str = "app") -> dict[str, Any]:
        """Pairs a device to a person, by its token's hash: an app's, or a network drive's."""
        rows = self._query(
            "INSERT INTO devices (person, name, token, kind, created, seen) VALUES"
            " (?, ?, ?, ?, ?, ?) RETURNING id, person, name, kind, created, seen",
            person, name, token, kind, now := time.time(), now,
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

    def _query(self, sql: str, *parameters: Any, **named: Any) -> list[sqlite3.Row]:
        # the rows of sql, of its parameters, by their places or their names
        with self._lock:
            cursor = self._db.execute(sql, named or parameters)
            cursor.row_factory = sqlite3.Row
            return cursor.fetchall()


def _project(row: sqlite3.Row) -> dict[str, Any]:
    # a project as a dict, its members' ids a list, the first first
    return dict(row) | {"members": sorted(json.loads(row["members"]))}
