"""Syncing a project's files with a folder of the box's, as a network drive's: a one-way copy, of
what is new or changed, without what is gone, hidden or a link out; refused of a folder too big,
or not a folder of a name; the owner's alone to set, through the API."""

import json
import os
import time
from collections.abc import Iterator

import pytest

from leat.agent import sync
from leat.agent.agent import Agent
from leat.agent.client import Client
from leat.agent.store import Store
from leat.agent.sync import mirror
from leat.agent.workspace import Workspace
from tests.test_agent import request, serving


@pytest.fixture
def agent(tmp_path) -> Agent:
    return Agent(Store(tmp_path / "leat.db"), Client("http://127.0.0.1:9"), [],
                 Workspace(tmp_path / "workspace"))  # fmt: skip


@pytest.fixture
def server(agent) -> Iterator[str]:
    with serving(agent) as url:
        yield url


def test_mirror(tmp_path, monkeypatch):
    # the folder's files copied, those unchanged since not, those gone deleted, with their
    # folders left empty; hidden files and links out of it not copied
    source, target = tmp_path / "nas" / "Yılmaz", tmp_path / "copy"
    (source / "2026").mkdir(parents=True)
    (source / "lease.txt").write_text("rent")
    (source / "2026" / "march.txt").write_text("invoice")
    (source / ".DS_Store").write_text("junk")
    (source / "out").symlink_to(tmp_path / "secret.txt")
    assert mirror(source, target) == (2, 0)
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*")) == [
        "2026", "2026/march.txt", "lease.txt"]  # fmt: skip
    assert mirror(source, target) == (0, 0)  # unchanged
    (source / "lease.txt").write_text("rent, raised")
    os.utime(source / "lease.txt", (time.time() + 5, time.time() + 5))
    (source / "2026" / "march.txt").unlink()
    assert mirror(source, target) == (1, 1)
    assert (target / "lease.txt").read_text() == "rent, raised" and not (target / "2026").exists()
    monkeypatch.setattr(sync, "FILES", 0)
    with pytest.raises(ValueError, match="more than 0 files"):
        mirror(source, target)
    with pytest.raises(FileNotFoundError, match="is it mounted|drive mounted"):
        mirror(tmp_path / "gone", target)


def test_sync(server, agent, tmp_path):
    # a project's files synced with a folder, the first time at once, in a folder of its name;
    # refused of a folder not named by its whole path, of none, or of a whole disk; a sync that
    # fails says why; the owner's alone, through the API
    source = tmp_path / "nas" / "Yılmaz Arşiv"
    source.mkdir(parents=True)
    (source / "lease.txt").write_text("rent")
    project = agent.add_project("Yılmaz Ltd", 1)["id"]
    url = f"{server}/api/projects/{project}/sync"
    for wrong in ("nas", str(tmp_path / "gone"), "/", 1):
        assert request(url, "POST", {"source": wrong})[0] in (400, 404)
    assert request(url, "POST", {"source": str(source)})[0] == 200
    for _ in range(100):
        if (agent.store.project(project) or {}).get("synced"):
            break
        time.sleep(0.02)
    space = agent.space(project, 1)
    assert [f["name"] for f in space.files()] == ["Yılmaz Arşiv/lease.txt"]
    held = agent.projects(1)[0]
    assert held["source"] == str(source) and held["unsynced"] is None
    assert agent.syncs_due() == []
    (source / "lease.txt").unlink()
    source.rmdir()
    agent.synced(project)
    assert "is not there" in agent.projects(1)[0]["unsynced"]
    assert [f["name"] for f in space.files()] == ["Yılmaz Arşiv/lease.txt"]  # kept meanwhile
    actions = [a["action"] for a in json.loads(request(f"{server}/api/activity")[1])["activity"]]
    assert actions[:2] == ["synced the files", "chose to sync the files with"]
    assert request(url, "POST", {"source": None})[0] == 200
    assert agent.projects(1)[0]["source"] is None and agent.syncs_due() == []
