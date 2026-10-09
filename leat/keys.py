"""API keys: each of a person or an app, by a name, which leat serve asks of every request once it
has a file of them, and counts each one's use by.

The file keeps each key's SHA-256 alone, so that a copy of it lets no one in, and is readable by its
owner alone. A key is shown once, as it is made. The server reads the file again whenever it
changes, so that a key added or removed counts from the next request on, without a restart.
"""

import datetime
import hashlib
import json
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any

PREFIX = "leat-"  # of every key, so that one pasted where it should not be is known for one
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")  # a key's name, as a metric's label takes it


class Keys:
    """The keys of the file at `path`, which need not be there yet."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seen: tuple[int, int] | None = None  # the file's time and size as last read
        self._names: dict[str, str] = {}  # each key's name, by its hash

    def add(self, name: str) -> str:
        """Makes a key of a name, and returns it: the only time it is shown. Raises ValueError if
        the name is no name, or one a key has already."""
        if not NAME.fullmatch(name):
            raise ValueError(
                f"a key's name is 1 to 64 letters, digits, dots, dashes or underscores: {name!r}"
            )
        with self._lock:
            keys = self._read()
            if name in keys:
                raise ValueError(f"there is a key named {name} already: remove it first")
            key = PREFIX + secrets.token_urlsafe(32)
            today = datetime.date.today().isoformat()
            self._write(keys | {name: {"sha256": _hash(key), "created": today}})
        return key

    def remove(self, name: str) -> None:
        """Removes a key, which lets no one in from the next request on. Raises LookupError if
        there is no key of that name."""
        with self._lock:
            keys = self._read()
            if name not in keys:
                raise LookupError(f"there is no key named {name}")
            del keys[name]
            self._write(keys)

    def listed(self) -> list[dict[str, str]]:
        """The keys' names and the days they were made, without the keys."""
        with self._lock:
            return [{"name": n, "created": k.get("created", "")} for n, k in self._read().items()]

    def name(self, key: str | None) -> str | None:
        """The name of a key, or None if it is none of the file's, as it is now."""
        if not key:
            return None
        with self._lock:
            try:
                stat = self.path.stat()
                seen: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                seen = None
            if seen != self._seen:
                self._names = {k["sha256"]: n for n, k in self._read().items()}
                self._seen = seen
            return self._names.get(_hash(key))

    def _read(self) -> dict[str, Any]:
        # the file's keys by name, none if it is not there. Raises ValueError if it is no file
        # of keys.
        try:
            keys = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        if not isinstance(keys, dict) or not all(
            isinstance(k, dict) and isinstance(k.get("sha256"), str) for k in keys.values()
        ):
            raise ValueError(f"{self.path} is no file of leat's keys")
        return keys

    def _write(self, keys: dict[str, Any]) -> None:
        # the file whole, or as it was, made readable by its owner alone before a key is in it
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = self.path.with_name(self.path.name + ".new")
        fd = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)  # one left over by a write that failed keeps what it had
        with os.fdopen(fd, "w") as f:
            json.dump(keys, f, indent=2)
            f.write("\n")
        os.replace(new, self.path)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()
