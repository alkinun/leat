"""Backups: the office's data, encrypted, each night, to a USB disk or a network share of the
office's, from which an admin restores it, onto this box or a new one, as one broken is replaced.

    python -m leat.box.backup             # a request of leat agent's, on the socket systemd took
    python -m leat.box.backup --run       # a backup now, as leat-backup.timer starts each night
    python -m leat.box.backup --restore   # a restore asked for, apart, as leat agent is stopped

The backups are restic's: a repository in the folder "Leat backup" of the target, encrypted by a
recovery key the box makes as it is set up, which its admin is shown once and writes down, as a new
box restores of it alone. The box keeps it, and the share's password, on its data's partition, which
its TPM unlocks and a factory reset erases. Kept: a backup of each of the last 7 days, of the last 5
weeks and of the last 12 months.

Backed up: leat agent's data, its conversations and people, the files, their earlier versions and
their index, its databases copied first by SQLite's backup, as they are written to meanwhile. Not
the box's identity, nor its settings: a new box has its own.

Requests, each a JSON object, as leat.box.system's, answered {}, or what was asked, or
{"error": why}:

    {"backup": "status"}               its target, its last backup, or why it failed, and the USB
                                       disks, for an admin to choose one
    {"backup": {"disk": UUID}}         backs up to the USB disk of that file system, each night;
    {"backup": {"share": "//nas/share", "user": …, "password": …}}    or to that network share:
                                       answered {"key": …}, the recovery key of a new repository;
                                       {"found": true} of one there, whose key is to be given, as
                                       "key", to back up to it, and restore of it
    {"backup": "now"}                  a backup now
    {"backup": "off"}                  no more backups; those made are left
    {"backup": "snapshots"}            the backups, to restore one: newest first
    {"restore": {"snapshot": ID, "device": DEVICE}}   restores one: leat agent stopped, its data
                                       put in its place, and the device asking, of DEVICE, kept
"""

import contextlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from leat.box.system import LONGEST, Refused, write

SOCKET = "/run/leat/backup.sock"  # its requests', which leat agent asks on
WAIT = 150  # seconds leat agent waits for an answer, as a share's mount and restic's look take
DATA = Path("/var/lib/leat")  # leat agent's
# the target and its key, the root's alone; the last backup's status and restore's, anyone's, as
# leat agent's watch reads it
STATE = Path("/var/lib/leat-backup")
STATUS = STATE / "status.json"
# where the target is mounted, while a backup, restore or request runs, at a point of each's own
MOUNTED = Path("/run/leat-backup")
IDENTITY = Path("/etc/leat/box.env")
FOLDER = "Leat backup"  # the repository's, in the target
HOST = "leat-box"  # every snapshot's, of whichever box, as a new box backs up to an old one's
DATABASES = ["leat.db", "index.db"]
COPIED = ".backup"  # the databases' copies, of their backup, in the data's folder while it runs
# what of the data's folder is no backup's: the databases, of their copies; runs' scratch; and the
# sandbox's libraries, of the image's
EXCLUDED = ["*.db", "*.db-wal", "*.db-shm", ".workspace-runs", "sandbox"]
KEEP = ["--keep-daily", "7", "--keep-weekly", "5", "--keep-monthly", "12"]
# a recovery key's letters, none mistaken for another, as a setup code's; 6 groups of 4: 120 bits
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
GROUPS, GROUP = 6, 4
FILE_SYSTEMS = {"vfat": "vfat", "exfat": "exfat", "ntfs": "ntfs3", "ext4": "ext4", "btrfs": "btrfs",
                "xfs": "xfs"}  # fmt: skip
SHARE = re.compile(r"//[A-Za-z0-9.-]+/[^/\x00]+(/[^\x00]*)?")
NOT_THERE, WRONG_KEY = 10, 12  # restic's exit codes of a repository not there, and a wrong key


def handle(request: Any) -> dict[str, Any]:
    """The answer to a request, as the helper gives it."""
    try:
        if not isinstance(request, dict):
            raise Refused("a request is a JSON object")
        if "restore" in request:
            return restore_later(request["restore"])
        what = request.get("backup")
        if what == "status":
            return status()
        if what == "now":
            _start("leat-backup.service")
            return {}
        if what == "off":
            for name in ("target.json", "key", STATUS.name):
                (STATE / name).unlink(missing_ok=True)
            return {}
        if what == "snapshots":
            return {"snapshots": snapshots()}
        if isinstance(what, dict):
            return set_target(what)
        raise Refused(f"there is no such request: {json.dumps(request)[:100]}")
    except (Refused, OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}


def status() -> dict[str, Any]:
    """Where the box backs up to, without the share's password; its last backup, and its last try's
    error, if it failed; whether one runs now; and the USB disks there are."""
    target = _target()
    return {"target": target and _shown(target), "last": _read(STATUS.name),
            "restore": _read("restore.json"),
            "running": _active("leat-backup.service") or _active("leat-restore.service"),
            "disks": disks()}  # fmt: skip


def disks() -> list[dict[str, Any]]:
    """The file systems of the USB disks plugged into the box, of those the box mounts: each its
    UUID, label, type, disk's model, size and room left, in bytes."""
    columns = "NAME,UUID,LABEL,FSTYPE,SIZE,FSAVAIL,TRAN,MODEL"
    told = subprocess.run(["lsblk", "--json", "--bytes", "--output", columns], capture_output=True,
                          text=True, timeout=10, check=True).stdout  # fmt: skip
    found = []
    for disk in json.loads(told).get("blockdevices", []):
        if disk.get("tran") != "usb":
            continue
        for part in [disk, *disk.get("children", [])]:
            if part.get("fstype") in FILE_SYSTEMS and part.get("uuid"):
                found.append({"uuid": part["uuid"], "label": part.get("label") or "",
                              "fstype": part["fstype"], "model": (disk.get("model") or "").strip(),
                              "size": part["size"], "free": part.get("fsavail")})  # fmt: skip
    return found


def set_target(target: dict[str, Any]) -> dict[str, Any]:
    """Backs up to `target` from now on, each night: of a repository there, of its key, or of one
    made, backed up to at once, whose recovery key is answered; or answers that one is there, whose
    key is to be given. Of one there, nothing is backed up before tonight, as a new box restores
    its backups first."""
    checked = _checked(target)
    key = target.get("key")
    if key is not None and not isinstance(key, str):
        raise Refused("a recovery key is text")
    with mounted(checked) as root:
        repository = root / FOLDER
        if key is not None:
            key = normal(key)
            _restic(repository, key, "cat", "config")  # Refused of a wrong one
            made: dict[str, Any] = {}
        elif _there(repository):
            return {"found": True}
        else:
            key = new_key()
            _restic(repository, key, "init")
            made = {"key": key}
    STATE.mkdir(mode=0o711, exist_ok=True)
    STATUS.unlink(missing_ok=True)  # of a target before
    write(STATE / "key", key, 0o600)
    write(STATE / "target.json", json.dumps(checked), 0o600)
    if "key" in made:
        _start("leat-backup.service")
    return made


def run() -> None:
    """Backs up now, to the box's target, and forgets backups past keeping; its status written,
    as the next status reads it."""
    target, key = _target(), _key()
    if target is None or key is None:
        return
    started = time.time()
    try:
        copied = _copy_databases()
        try:
            with mounted(target) as root:
                repository = root / FOLDER
                excluded = [a for e in EXCLUDED for a in ("--exclude", str(DATA / e))]
                done = _restic(repository, key, "backup", "--json", "--host", HOST, "--tag",
                               f"box={_name()}", *excluded, str(DATA))  # fmt: skip
                told = [json.loads(line) for line in done.splitlines()]
                summary: dict[str, Any] = next(
                    (m for m in told if m.get("message_type") == "summary"), {}
                )
                _restic(repository, key, "forget", "--host", HOST, *KEEP, "--prune")
        finally:
            shutil.rmtree(copied, ignore_errors=True)
    except (Refused, OSError, subprocess.SubprocessError, sqlite3.Error) as e:
        _status({"time": started, "error": str(e)})
        raise SystemExit(f"no backup: {e}") from None
    _status({"time": started, "ok": time.time(), "size": summary.get("total_bytes_processed"),
             "added": summary.get("data_added")})  # fmt: skip


def snapshots() -> list[dict[str, Any]]:
    """The backups there are, newest first: each its ID, time, of ISO 8601, size, and the name of
    the box it was made of."""
    target, key = _target(), _key()
    if target is None or key is None:
        raise Refused("the box backs up nowhere")
    with mounted(target) as root:
        told = json.loads(_restic(root / FOLDER, key, "snapshots", "--json") or "[]")
    return [{"id": s["id"], "time": s["time"], "size": (s.get("summary") or {}).get(
                 "total_bytes_processed"), "box": _tagged(s, "box")}
            for s in sorted(told, key=lambda s: s["time"], reverse=True)]  # fmt: skip


def restore_later(asked: Any) -> dict[str, Any]:
    """Restores a backup, apart, as leat agent, which asked, is stopped meanwhile: answered at
    once."""
    if not isinstance(asked, dict) or not isinstance(asked.get("snapshot"), str):
        raise Refused('a restore is of {"snapshot": ID, "device": …}')
    if not re.fullmatch(r"[0-9a-f]{8,64}|latest", asked["snapshot"]):
        raise Refused("that is no backup's ID")
    if _target() is None:
        raise Refused("the box backs up nowhere")
    STATE.mkdir(mode=0o711, exist_ok=True)
    write(STATE / "asked.json", json.dumps(asked), 0o600)
    _start("leat-restore.service")
    return {}


def restore() -> None:
    """Restores the backup asked for, of asked.json: made apart, leat agent stopped only as its
    data's folder is swapped for it, then started again; the device asking, which this box's state
    had, kept beside it, as restored.json, for leat agent to pair, as the backup may not have it.
    The data before deleted once swapped; kept, as "before", if it could not be put back."""
    asked = json.loads((STATE / "asked.json").read_text())
    work, before = STATE / "restoring", STATE / "before"
    shutil.rmtree(work, ignore_errors=True)
    started = time.time()
    try:
        target, key = _target(), _key()
        if target is None or key is None:
            raise Refused("the box backs up nowhere")
        with mounted(target) as root:
            repository = root / FOLDER
            told = json.loads(_restic(repository, key, "snapshots", "--json", asked["snapshot"])
                              or "[]")  # fmt: skip
            if not told:
                raise Refused("there is no such backup")
            _restic(repository, key, "restore", told[-1]["id"], "--target", str(work))
        restored = work / DATA.relative_to("/")
        for name in DATABASES:
            with contextlib.suppress(FileNotFoundError):
                (restored / COPIED / name).replace(restored / name)
        if not (restored / "leat.db").exists():
            raise Refused("that backup holds no conversations: it is none of Leat's")
        shutil.rmtree(restored / COPIED, ignore_errors=True)
        kept = {"box": _tagged(told[-1], "box"), "device": asked.get("device")}
        write(restored / "restored.json", json.dumps(kept), 0o600)
        subprocess.run(["chown", "-R", "leat:leat", str(restored)], check=True, timeout=600)
        subprocess.run(["systemctl", "stop", "leat-agent.service"], check=True, timeout=120)
        try:
            DATA.replace(before)  # refused of a "before" there still, of a restore that failed
            try:
                restored.replace(DATA)
            except OSError:
                before.replace(DATA)
                raise
        finally:  # its sandbox's link made again, and leat agent started, whatever was swapped
            subprocess.run(["systemd-tmpfiles", "--create"], timeout=60, check=False)
            subprocess.run(["systemctl", "start", "leat-agent.service"], timeout=120, check=False)
    except (Refused, OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
        _status({"time": started, "error": f"not restored: {e}"}, "restore")
        raise SystemExit(f"not restored: {e}") from None
    finally:
        (STATE / "asked.json").unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(before, ignore_errors=True)
    _status({"time": started, "ok": time.time(), "snapshot": told[-1]["id"]}, "restore")


@contextlib.contextmanager
def mounted(target: dict[str, Any]) -> Iterator[Path]:
    """The target mounted, while the block runs: its root's path."""
    point, credentials = MOUNTED / f"target-{os.getpid()}", MOUNTED / f"credentials-{os.getpid()}"
    point.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        if "disk" in target:
            device = _device(target["disk"])
            options = "nosuid,nodev,noexec"
            command = ["mount", "-t", FILE_SYSTEMS[device["fstype"]], "-o", options, device["path"]]
        else:
            write(credentials, f"username={target['user']}\npassword={target['password']}\n",
                  0o600)  # fmt: skip
            options = (f"credentials={credentials},nosuid,nodev,noexec,uid=0,gid=0,"
                       "file_mode=0600,dir_mode=0700")  # fmt: skip
            command = ["mount", "-t", "cifs", "-o", options, target["share"]]
        try:
            done = subprocess.run([*command, str(point)], capture_output=True, text=True,
                                  timeout=60)  # fmt: skip
        finally:
            credentials.unlink(missing_ok=True)
        if done.returncode:
            raise Refused(f"the backup's target is out of reach: {done.stderr.strip()}")
        try:
            yield point
        finally:
            subprocess.run(["umount", str(point)], capture_output=True, timeout=120, check=False)
    finally:
        with contextlib.suppress(OSError):
            point.rmdir()


def new_key() -> str:
    """A recovery key: 6 groups of 4 letters, of 120 bits."""
    return "-".join("".join(secrets.choice(LETTERS) for _ in range(GROUP)) for _ in range(GROUPS))


def normal(key: str) -> str:
    """A recovery key as it was made, of one typed, its case, spaces and dashes anyhow. Raises
    Refused of one that is none."""
    letters = "".join(key.upper().split()).replace("-", "")
    if len(letters) != GROUPS * GROUP or any(c not in LETTERS for c in letters):
        raise Refused(f"a recovery key is {GROUPS} groups of {GROUP} letters and digits")
    return "-".join(letters[i : i + GROUP] for i in range(0, len(letters), GROUP))


def _checked(target: dict[str, Any]) -> dict[str, Any]:
    # a target as the box keeps it: a USB disk's file system, of its UUID, or a share, of SMB, its
    # address of either slashes. Raises Refused of one that is neither
    if isinstance(target.get("disk"), str):
        device = _device(target["disk"])
        return {"disk": device["uuid"], "label": device["label"] or device["model"]}
    share, user, password = target.get("share"), target.get("user"), target.get("password")
    if not (isinstance(share, str) and isinstance(user, str) and isinstance(password, str)):
        raise Refused('a target is {"disk": UUID} or {"share": …, "user": …, "password": …}')
    share = share.strip().replace("\\", "/").removeprefix("smb:")
    if not SHARE.fullmatch(share):
        raise Refused(f"{share} is no network share: it is \\\\server\\share, as Windows writes it")
    if any(c in user + password for c in "\n\x00") or len(user) > 256 or len(password) > 256:
        raise Refused("that user or password is none a share takes")
    return {"share": share, "user": user, "password": password}


def _device(uuid: str) -> dict[str, Any]:
    # the USB disk's file system of that UUID, its path and type. Raises Refused if none is there
    for disk in disks():
        if disk["uuid"] == uuid:
            return {**disk, "path": f"/dev/disk/by-uuid/{uuid}"}
    raise Refused("that USB disk is not plugged into the box")


def _restic(repository: Path, key: str, *args: str) -> str:
    # restic's output of a command on the repository. Raises Refused of a wrong key, or of one
    # there is none of, and of any other error, saying it
    environment = {**os.environ, "RESTIC_REPOSITORY": str(repository), "RESTIC_PASSWORD": key,
                   "RESTIC_CACHE_DIR": "/var/cache/leat-backup"}  # fmt: skip
    done = subprocess.run(["restic", "--retry-lock", "1m", *args], capture_output=True, text=True,
                          env=environment, check=False)  # fmt: skip
    if done.returncode == WRONG_KEY:
        raise Refused("that recovery key is not this backup's")
    if done.returncode == NOT_THERE:
        raise Refused("there is no backup there")
    if done.returncode:
        raise Refused(f"restic failed: {done.stderr.strip()[-500:]}")
    return done.stdout


def _there(repository: Path) -> bool:
    # whether a repository is there, of any key: its config, which restic init writes first
    return (repository / "config").exists()


def _copy_databases() -> Path:
    # the databases copied into the data's folder, of SQLite's backup, consistent as they are
    # written to meanwhile; returns their folder
    copied = DATA / COPIED
    shutil.rmtree(copied, ignore_errors=True)
    copied.mkdir(mode=0o700)
    for name in DATABASES:
        if (DATA / name).exists():
            source = sqlite3.connect(f"file:{DATA / name}?mode=ro", uri=True)
            target = sqlite3.connect(copied / name)
            try:
                source.backup(target)
            finally:
                source.close()
                target.close()
    return copied


def _status(status: dict[str, Any], of: str = "status") -> None:
    # the last backup's status, or restore's, of "restore", anyone's to read
    STATE.mkdir(mode=0o711, exist_ok=True)
    write(STATE / f"{of}.json", json.dumps(status), 0o644)


def _read(name: str) -> dict[str, Any] | None:
    # a status kept, if there is one
    with contextlib.suppress(FileNotFoundError, ValueError):
        return json.loads((STATE / name).read_text())
    return None


def _target() -> dict[str, Any] | None:
    return _read("target.json")


def _key() -> str | None:
    with contextlib.suppress(FileNotFoundError):
        return (STATE / "key").read_text().strip()
    return None


def _shown(target: dict[str, Any]) -> dict[str, Any]:
    # a target as an admin is shown it: never the share's password
    return {k: v for k, v in target.items() if k != "password"}


def _name() -> str:
    # the box's name, of its identity, which its snapshots are tagged with
    with contextlib.suppress(OSError):
        for line in IDENTITY.read_text().splitlines():
            if line.startswith("LEAT_NAME="):
                return line.split("=", 1)[1]
    return ""


def _tagged(snapshot: dict[str, Any], tag: str) -> str | None:
    # a snapshot's tag of that name's value
    return next((t.split("=", 1)[1] for t in snapshot.get("tags") or []
                 if t.startswith(f"{tag}=")), None)  # fmt: skip


def _start(unit: str) -> None:
    subprocess.run(["systemctl", "--no-block", "start", unit], check=True, capture_output=True,
                   timeout=10)  # fmt: skip


def _active(unit: str) -> bool:
    told = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True,
                          timeout=10, check=False).stdout.strip()  # fmt: skip
    return told in ("active", "activating")


def main() -> None:
    # a request on stdin, the socket systemd accepted, and its answer on stdout; or, of --run, a
    # backup now, and of --restore, the restore asked for
    if sys.argv[1:] == ["--run"]:
        return run()
    if sys.argv[1:] == ["--restore"]:
        return restore()
    raw = sys.stdin.buffer.read(LONGEST + 1)
    try:
        request = json.loads(raw) if len(raw) <= LONGEST else None
    except ValueError:
        request = None
    sys.stdout.write(json.dumps(handle(request)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
