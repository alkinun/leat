"""Backups: snapshots of the state and the files, unchanged files linked to the last snapshot's,
the latest KEPT kept, one cut short never taken for whole; when one is due, how each went as the
owner is told, and the owner's alone to set up."""

import datetime
import json
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from leat.agent import backup
from leat.agent.agent import Agent
from leat.agent.client import Client
from leat.agent.store import Store
from leat.agent.tools import files
from leat.agent.workspace import Workspace
from tests.test_agent import request, serving


@pytest.fixture
def agent(tmp_path) -> Agent:
    return Agent(Store(tmp_path / "leat.db"), Client("http://127.0.0.1:9"), [],
                 Workspace(tmp_path / "workspace"))  # fmt: skip


def night(day: int, hour: int = 3) -> datetime.datetime:
    return datetime.datetime(2026, 10, day, hour, 0, 5)


def test_snapshots(agent, tmp_path):
    # the state and the files, whole; the files unchanged since linked to the last snapshot's;
    # hidden files and links out not kept; the latest KEPT kept
    folder = tmp_path / "usb"
    folder.mkdir()
    agent.add_project("Yılmaz Ltd")
    files.write(agent.space(), "notes.txt", "notes")
    files.write(agent.space(), "Invoices/1.txt", "one")
    files.write(agent.space(), ".web/page.md", "a page")
    (agent.space().root / "out").symlink_to(tmp_path / "leat.db")
    made = backup.back_up(agent, folder, night(10))
    assert made == {"at": night(10).timestamp(), "name": "leat-2026-10-10-030005", "files": 2,
                    "bytes": 8}  # fmt: skip
    first = folder / made["name"]
    kept = sqlite3.connect(first / "leat.db").execute("SELECT name FROM projects").fetchall()
    assert kept == [("Yılmaz Ltd",)]
    assert sorted(p.relative_to(first / "workspace").as_posix() for p in first.rglob("*.txt")) == [
        "people/0/Invoices/1.txt", "people/0/notes.txt"]  # fmt: skip
    files.write(agent.space(), "notes.txt", "notes, changed")
    second = folder / backup.back_up(agent, folder, night(11))["name"]
    one, again = (s / "workspace/people/0/Invoices/1.txt" for s in (first, second))
    assert one.stat().st_ino == again.stat().st_ino  # linked, as it is unchanged
    assert (second / "workspace/people/0/notes.txt").read_text() == "notes, changed"
    assert (first / "workspace/people/0/notes.txt").read_text() == "notes"
    for day in range(12, 12 + backup.KEPT):
        backup.back_up(agent, folder, night(day))
    names = sorted(p.name for p in folder.iterdir())
    assert len(names) == backup.KEPT and names[0] == "leat-2026-10-12-030005"


def test_cut_short(agent, tmp_path, monkeypatch):
    # a backup that fails leaves nothing that might be taken for a snapshot
    folder = tmp_path / "usb"
    folder.mkdir()
    files.write(agent.space(), "a.txt", "a")

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(backup.shutil, "copy2", full)
    with pytest.raises(OSError):
        backup.back_up(agent, folder, night(10))
    assert list(folder.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        backup.back_up(agent, tmp_path / "gone", night(10))


def test_due(agent, tmp_path, monkeypatch):
    # once a night, after HOUR, to a folder chosen; a failure tried again after RETRY, and said
    folder = tmp_path / "usb"
    folder.mkdir()
    backups = agent.backups
    assert not backups.due(night(10))
    with pytest.raises(ValueError, match="whole path"):
        backups.choose("usb")
    with pytest.raises(ValueError, match="not a folder"):
        backups.choose(str(tmp_path / "nothing"))
    backups.choose(str(folder))
    assert not backups.due(night(10, hour=2)) and backups.due(night(10))
    with agent.events.watch() as told:
        backups.run()
        running, done = told.get(), told.get()
    assert running["running"] and not done["running"] and done["to"] == "owner"
    assert done["last"]["files"] == 0 and done["kept"] == 1 and done["error"] is None
    now = datetime.datetime.fromtimestamp(done["last"]["at"])
    assert not backups.due(now.replace(hour=23))
    assert backups.due(now.replace(hour=4) + datetime.timedelta(days=1))
    shutil.rmtree(folder)  # its disk taken away
    backups.run()
    assert "is not there: is its disk connected" in backups.state()["error"]
    assert not backups.due(now.replace(hour=4) + datetime.timedelta(days=1))  # not till RETRY
    agent.store.set_setting("backup.failed", time.time() - backup.RETRY - 1)
    assert backups.due(now.replace(hour=4) + datetime.timedelta(days=1))


def test_api(agent, tmp_path):
    # the owner's alone: the folder chosen, and a backup made now; told as it goes
    folder = tmp_path / "usb"
    folder.mkdir()
    with serving(agent) as url:
        assert request(f"{url}/api/backups/now", "POST", {})[0] == 400  # no folder yet
        assert request(f"{url}/api/backups", "POST", {"folder": 1})[0] == 400
        assert request(f"{url}/api/backups", "POST", {"folder": str(folder)})[0] == 200
        assert request(f"{url}/api/backups/now", "POST", {})[0] == 200
        for _ in range(50):
            if agent.backups.state()["kept"] and not agent.backups.state()["running"]:
                break
            time.sleep(0.05)
        assert agent.backups.state()["kept"] == 1
        assert request(f"{url}/api/backups", "POST", {"folder": None})[0] == 200
        assert agent.backups.state()["folder"] is None
    assert json.loads(json.dumps(agent.backups.state()))["type"] == "backups"
    assert Path(folder).is_dir()
