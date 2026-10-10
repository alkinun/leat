"""Backups: the agent's state, its people, conversations, projects and workflows, and every file,
copied each night to a folder the owner chooses, as a disk or a network drive mounted on the box.

Each backup is a snapshot of its own, a folder named by when it was made, which holds leat.db and
the workspace as they were: the state copied by SQLite's backup, which the agent's writes meanwhile
do not tear, and each file copied, or linked to the last snapshot's where it is unchanged and the
folder's file system can, so that a night's backup takes the room only of what changed. A snapshot
is written under a hidden name, and takes its own once whole, so that one cut short by a failure or
the power is never taken for whole. KEPT are kept, the latest. The index is not kept: it is made
anew from the files.

To restore one, stop leat agent and copy its leat.db and workspace into the agent's --data.
"""

import datetime
import os
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leat.agent.agent import Agent

KEPT = 14  # snapshots kept, the latest
HOUR = 3  # the hour of the night after which each day's backup is made
RETRY = 3600  # seconds after a failed backup before the next try
PREFIX = "leat-"  # of a snapshot's name, before when it was made


class Backups:
    """The backups of an agent's state and files, to the folder the owner chose, kept in its
    settings with how the last went; each told to the owner's apps."""

    def __init__(self, agent: "Agent"):
        self.agent, self.store = agent, agent.store
        self._running = threading.Lock()  # held while a backup is made

    def state(self) -> dict[str, Any]:
        """The backups as the owner's apps show them: the folder, the last backup, how many are
        kept, and whether one is being made, or why the last failed."""
        folder = self.store.setting("backup.folder")
        kept = len(_snapshots(Path(folder))) if folder and Path(folder).is_dir() else 0
        return {
            "type": "backups", "folder": folder, "last": self.store.setting("backup.last"),
            "error": self.store.setting("backup.error"), "kept": kept,
            "running": self._running.locked(),
        }  # fmt: skip

    def choose(self, folder: str | None) -> None:
        """Backs up to a folder from now on, or to none. Raises ValueError of one that is not an
        absolute path to a folder the box can write to."""
        if folder is not None:
            path = Path(folder)
            if not path.is_absolute():
                raise ValueError("the folder must be named by its whole path, as /media/backup")
            if not path.is_dir():
                raise ValueError(f"{folder} is not a folder on this computer")
            if not os.access(path, os.W_OK):
                raise ValueError(f"Leat cannot write to {folder}")
        self.store.set_setting("backup.folder", folder)
        self.store.set_setting("backup.error", None)
        self._publish()

    def due(self, now: datetime.datetime | None = None) -> bool:
        """Whether a backup is due: none was made today after HOUR, there is a folder, and none
        failed in the last RETRY seconds."""
        now = now or datetime.datetime.now()
        if not self.store.setting("backup.folder") or now.hour < HOUR:
            return False
        failed = self.store.setting("backup.failed")
        if failed and time.time() - failed < RETRY:
            return False
        last = self.store.setting("backup.last")
        made = datetime.datetime.fromtimestamp(last["at"]) if last else None
        return made is None or made < now.replace(hour=HOUR, minute=0, second=0, microsecond=0)

    def run(self) -> None:
        """Makes a backup now, if none is being made, telling the owner's apps as it begins and
        how it went."""
        if not self._running.acquire(blocking=False):
            return
        try:
            self._publish()
            folder = Path(self.store.setting("backup.folder") or "")
            try:
                made = back_up(self.agent, folder)
            except OSError as e:
                self.store.set_setting("backup.error", _said(e, folder))
                self.store.set_setting("backup.failed", time.time())
            else:
                self.store.set_setting("backup.last", made)
                for key in ("backup.error", "backup.failed"):
                    self.store.set_setting(key, None)
        finally:
            self._running.release()
            self._publish()

    def _publish(self) -> None:
        self.agent.events.publish(self.state() | {"to": "owner"})


def back_up(agent: "Agent", folder: Path, now: datetime.datetime | None = None) -> dict[str, Any]:
    """Makes a snapshot of an agent's state and files in a folder, and keeps the KEPT latest;
    returns when it was made, its name, and how many files and bytes it holds. Raises OSError if
    the folder is not there, or cannot be written to."""
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not there")
    now = now or datetime.datetime.now()
    name = f"{PREFIX}{now:%Y-%m-%d-%H%M%S}"
    partial = folder / f".{name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    (partial / "workspace").mkdir(parents=True)
    try:
        agent.store.backup(partial / "leat.db")
        root = agent.workspace.root if agent.workspace else None
        files, size = _copy(root, partial / "workspace", _latest(folder))
        partial.rename(folder / name)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    for old in _snapshots(folder)[:-KEPT]:
        shutil.rmtree(old, ignore_errors=True)
    return {"at": now.timestamp(), "name": name, "files": files, "bytes": size}


def _copy(source: Path | None, target: Path, last: Path | None) -> tuple[int, int]:
    # copies a workspace's files to a snapshot's, each linked to the last snapshot's where it is
    # as it was, as its size and time of change say, and the file system can; links out of the
    # workspace and hidden files, as saved pages, are not copied. Returns the files and bytes
    files = size = 0
    if source is None:
        return files, size
    for path in sorted(source.rglob("*")):
        name = path.relative_to(source)
        if path.is_symlink() or not path.is_file() or any(p.startswith(".") for p in name.parts):
            continue
        stat, copy = path.stat(), target / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        before = last / "workspace" / name if last else None
        if before and before.is_file() and _same(before.stat(), stat):
            try:
                os.link(before, copy)
            except OSError:  # a file system without links, as exFAT
                shutil.copy2(path, copy)
        else:
            shutil.copy2(path, copy)
        files, size = files + 1, size + stat.st_size
    return files, size


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_size == b.st_size and a.st_mtime == b.st_mtime


def _snapshots(folder: Path) -> list[Path]:
    # a folder's snapshots, the oldest first
    return sorted(p for p in folder.glob(f"{PREFIX}*") if p.is_dir())


def _latest(folder: Path) -> Path | None:
    snapshots = _snapshots(folder)
    return snapshots[-1] if snapshots else None


def _said(error: OSError, folder: Path) -> str:
    # why a backup failed, as the owner reads it
    if isinstance(error, FileNotFoundError) and not folder.is_dir():
        return f"{folder} is not there: is its disk connected, or its drive mounted?"
    if getattr(error, "errno", None) == 28:  # ENOSPC
        return f"{folder} is full"
    return f"it failed: {error.strerror or error}"
