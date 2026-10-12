"""The workspace: the user's files, which they upload and the agent makes, and Python run among them
in a sandbox.

The workspace is the box's, and holds spaces, each a workspace of its own: a person's own files, at
people/<id>, and each project's, at projects/<id>. A conversation's tools work in its space alone,
whose folder alone its sandbox sees.

The sandbox is bubblewrap's: the code sees /usr, the sandbox's environment of libraries, which it
cannot change, and the workspace, at /workspace; no network, no other file of the box's, and a
memory and a time it may not pass. Files of the kinds whose reading parses them, as PDFs, are read
there too, so that a file made to attack its parser attacks the sandbox. Python runs isolated, so
that a file of the workspace's, as docx.py, cannot take a library's place, and writes no bytecode.

Nothing Leat changes is lost: a file changed or deleted, by a tool or by its user, is kept first as
an earlier version, in the history beside the workspace, which no sandbox sees, to be restored for
KEPT seconds, HISTORY versions of a file at most. The code a run runs sees the workspace through an
overlay, its changes written apart, and made in the workspace once it ends, each file it changes
or deletes kept first, so that no code the model writes changes a file unkept; where the kernel
mounts no overlay, as one too old, the workspace is the code's to change, and runs keep nothing.
A space's history goes with it, as a project deleted takes its files' earlier versions too.
"""

import contextlib
import difflib
import functools
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

TIMEOUT = 120  # seconds a run may take
MEMORY = 4 << 30  # bytes of memory a run may take
OUTPUT = 20000  # bytes of a run's output kept, its start and its end
LONGEST = 50_000_000  # bytes of a document's text kept, which is read whole
NAME = 255  # bytes of a file's name at most, as Linux's file systems take
UNPACKED = 1 << 30  # bytes a zip unpacked may hold at most
MEMBERS = 5000  # files a zip unpacked may hold at most
HISTORY = 20  # earlier versions of a file kept at most
KEPT = 30 * 86400  # seconds an earlier version is kept
# unpacks the zip at sys.argv[1] into the folder sys.argv[2], refusing one of more than
# sys.argv[3] bytes or sys.argv[4] files; its names of folders kept but for any that would lead
# out, and taken as UTF-8 where a zip says not, as many do of them; macOS's leavings and hidden
# files left out, and a folder that holds all the rest, as a folder compressed has, left out too.
# It prints how many files it unpacked
_UNPACK = """
import shutil, sys, zipfile
from pathlib import Path
with zipfile.ZipFile(sys.argv[1]) as z:
    members = [m for m in z.infolist() if not m.is_dir()]
    if len(members) > int(sys.argv[4]):
        sys.exit(f"it holds {len(members)} files, more than {sys.argv[4]}")
    if sum(m.file_size for m in members) > int(sys.argv[3]):
        sys.exit(f"it holds more than {int(sys.argv[3]) >> 20} MB")
    named = []
    for m in members:
        name = m.filename
        if not m.flag_bits & 0x800:
            try:
                name = name.encode("cp437").decode("utf-8")
            except UnicodeError:
                pass
        parts = [p for p in name.replace("\\\\", "/").split("/") if p not in ("", ".", "..")]
        if parts and parts[0] != "__MACOSX" and not any(p.startswith(".") for p in parts):
            named.append((m, parts))
    if len({parts[0] for _, parts in named}) == 1 and all(len(parts) > 1 for _, parts in named):
        named = [(m, parts[1:]) for m, parts in named]
    for m, parts in named:
        target = Path(sys.argv[2], *parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(m) as source, open(target, "wb") as out:
            shutil.copyfileobj(source, out)
print(len(named))
"""
# the sandbox's view of the box: the system, read-only, and fonts' settings, for charts
_SYSTEM = [
    "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
    "--symlink", "usr/bin", "/bin", "--ro-bind-try", "/etc/fonts", "/etc/fonts",
    "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
]  # fmt: skip


@dataclass(frozen=True)
class Ran:
    """What came of a run: its exit status, None if it ran out of time, and what it printed."""

    status: int | None
    output: str

    @property
    def why(self) -> str:
        """Why it failed, as it said last."""
        return (self.output.strip().splitlines() or ["it stopped"])[-1]


class Workspace:
    """The files at `root`, and Python run among them by the interpreter of `environment`, a
    virtual environment of the libraries the sandbox offers, or the system's without one. Their
    earlier versions are kept at `history`, and runs write at `scratch`, each beside the root by
    default, where its sandbox sees neither."""

    def __init__(
        self, root: Path, environment: Path | None = None, history: Path | None = None,
        scratch: Path | None = None,
    ):  # fmt: skip
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.environment = environment.resolve() if environment else None
        self.history = (history or self.root.parent / f".{self.root.name}-history").resolve()
        self.scratch = (scratch or self.root.parent / f".{self.root.name}-runs").resolve()

    def space(self, name: str) -> "Workspace":
        """The workspace of a folder of this one's, as one of its own, whose code runs with the
        same libraries, and whose history is its own: a person's files, or a project's."""
        return Workspace(self.path(name), self.environment, self.history / name, self.scratch)

    def remove(self) -> None:
        """Deletes the workspace's folder, with every file in it, and their earlier versions."""
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.history, ignore_errors=True)

    def keep(self, name: str, move: bool = False) -> None:
        """Keeps a file as it is now as an earlier version, or each of a folder's, moved there if
        `move`, as a file deleted is; then forgets the file's oldest past HISTORY or KEPT. Hidden
        files are not kept, as a page saved to read on, which is no one's work, nor empty ones, as a
        network drive puts before a file's content."""
        path = self.path(name)
        if path.is_dir():
            for inside in sorted(path.rglob("*")):
                if inside.is_file():
                    self.keep(inside.relative_to(self.root).as_posix(), move)
            return
        relative = path.relative_to(self.root).as_posix()
        if not path.is_file() or hidden(relative) or not path.stat().st_size:  # nothing to keep
            return
        versions = self.history / relative
        versions.mkdir(parents=True, exist_ok=True)
        kept = versions / str(time.time_ns())
        if move:
            shutil.move(path, kept)
        else:
            shutil.copy2(path, kept)
        self._forget(versions)

    def versions(self) -> list[dict[str, Any]]:
        """The earlier versions of the files, the latest first: each's file's name, its version,
        when it was kept, its size, and whether the file is deleted, as no file has its name now."""
        found = []
        for kept in self.history.rglob("*") if self.history.is_dir() else []:
            if not kept.name.isdigit() or kept.is_dir():
                continue
            name = kept.parent.relative_to(self.history).as_posix()
            found.append({
                "name": name, "version": kept.name, "kept": int(kept.name) / 1e9,
                "size": kept.lstat().st_size, "deleted": not (self.root / name).is_file(),
            })  # fmt: skip
        return sorted(found, key=lambda v: v["kept"], reverse=True)

    def restore(self, name: str, version: str) -> None:
        """Makes a file as an earlier version of it was, the file as it is now kept first.
        Raises FileNotFoundError if there is no such version."""
        relative = self.path(name).relative_to(self.root).as_posix()
        kept = self.history / relative / version
        if not version.isdigit() or not kept.is_file():
            raise FileNotFoundError(f"there is no earlier version {version} of {name}")
        path = self.root / relative
        if path.is_dir():
            self.delete(relative)
        self.keep(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kept, path)

    def forget(self) -> None:
        """Forgets every earlier version kept longer than KEPT, as the box does each day."""
        if self.history.is_dir():
            for versions in {kept.parent for kept in self.history.rglob("*") if kept.is_file()}:
                self._forget(versions)

    def _forget(self, versions: Path) -> None:
        # forgets a file's versions past the HISTORY latest, or kept longer than KEPT, and the
        # folders of its name left empty
        oldest = time.time_ns() - int(KEPT * 1e9)
        kept = [v for v in versions.iterdir() if v.name.isdigit()]
        kept.sort(key=lambda v: int(v.name))
        for i, version in enumerate(kept):
            if i < len(kept) - HISTORY or int(version.name) < oldest:
                version.unlink(missing_ok=True)
        while versions != self.history and versions.is_dir() and not any(versions.iterdir()):
            versions.rmdir()
            versions = versions.parent

    def path(self, name: str) -> Path:
        """The path of a file in the workspace, named relative to it. Raises ValueError for a
        name outside it, such as an absolute one, or one up from it."""
        if any(len(part.encode()) > NAME for part in Path(name).parts):
            raise ValueError(f"a file's name is {NAME} bytes at most")
        path = (self.root / name.lstrip("/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"{name} is outside the workspace")
        return path

    def inside(self, name: str) -> str:
        """A file's path as the sandbox's code sees it, at /workspace."""
        return f"/workspace/{self.path(name).relative_to(self.root).as_posix()}"

    def files(self) -> list[dict[str, Any]]:
        """Every file, by its name in the workspace, the latest changed first; hidden ones not, nor
        links out of it, as the sandbox's code may make to any path."""
        found = []
        for path in self.root.rglob("*"):
            name = path.relative_to(self.root).as_posix()
            if hidden(name):
                continue
            if not path.is_file() or not path.resolve().is_relative_to(self.root):
                continue
            try:  # one deleted meanwhile, as by code a call runs at once with this one, is gone
                stat = path.stat()
            except FileNotFoundError:
                continue
            found.append({"name": name, "size": stat.st_size, "modified": stat.st_mtime})
        return sorted(found, key=lambda f: f["modified"], reverse=True)

    def missing(self, name: str) -> FileNotFoundError:
        """The error of a file there is not, which names the files of names nearest its, as a
        model that wrote one wrongly, or in another language, may take the one it meant."""
        names = [f["name"] for f in self.files()]
        near = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        said = f"; the nearest: {', '.join(near)}" if near else ""
        return FileNotFoundError(f"there is no file {name}{said}")

    def free(self, name: str, folders: bool = False) -> str:
        """A name for a new file at the workspace's top, as `name` but for what makes it a path,
        numbered as "notes (2).txt" if a file has it; or with `folders` in the folders `name`
        gives, but for any that would lead out of the workspace or are hidden, free in the
        last."""
        parts = name.replace("\\", "/").split("/")
        within = [_short(p.lstrip(". ")) for p in parts[:-1] if p.lstrip(". ")] if folders else []
        base = Path(parts[-1].lstrip(". ")).name or "file"
        stem, suffix = Path(base).stem, Path(base).suffix
        if len(base.encode()) > NAME - 8:  # its start and its kind, as long as a name may be with
            # room for a number, " (2)"
            suffix = suffix if len(suffix.encode()) < 16 else ""
            stem = _short(stem, NAME - 8 - len(suffix.encode()))
            base = stem + suffix
        n, candidate = 1, base
        while self.root.joinpath(*within, candidate).exists():
            n += 1
            candidate = f"{stem} ({n}){suffix}"
        return "/".join([*within, candidate])

    def new(self, name: str) -> str:
        """A name for a new file, as free() gives it of `folders`, its folders made."""
        made = self.free(name, folders=True)
        self.path(made).parent.mkdir(parents=True, exist_ok=True)
        return made

    def delete(self, name: str) -> None:
        """Deletes a file, or a folder with every file in it, each kept as an earlier version.
        Raises FileNotFoundError if there is none."""
        path = self.path(name)
        if path.is_dir() and path != self.root:
            self.keep(name, move=True)
            shutil.rmtree(path)  # and what was not kept, hidden files and folders left empty
        elif path.is_file():
            self.keep(name, move=True)
            path.unlink(missing_ok=True)  # one not kept, as a hidden one
        else:
            raise self.missing(name)

    def unpack(self, name: str) -> tuple[str, int]:
        """Unpacks a zip into a folder of its name, free at its place, in the sandbox, and deletes
        it; returns the folder's name and how many files it holds. Raises ValueError if it cannot
        be unpacked, or holds more than UNPACKED bytes or MEMBERS files, the zip kept."""
        file = self.path(name)
        stem = file.relative_to(self.root).as_posix().removesuffix(file.suffix)
        folder = self.free(stem, folders=True)
        ran = self.run(_UNPACK, self.inside(name), f"/workspace/{folder}", str(UNPACKED),
                       str(MEMBERS))  # fmt: skip
        if ran.status != 0:
            shutil.rmtree(self.path(folder), ignore_errors=True)
            raise ValueError(f"{file.name} could not be unpacked: {ran.why}")
        file.unlink()
        return folder, int(ran.output.strip().splitlines()[-1])

    def given(self, code: str, given: Any, *names: str, timeout: float = TIMEOUT) -> Any:
        """Runs code in the sandbox, of `given`, as JSON in a hidden file of the workspace's, its
        path sys.argv[1], and of the files named, their paths the arguments after; returns what
        it printed last, as JSON. Raises ValueError if it fails."""
        inside = f".given-{uuid.uuid4().hex[:8]}.json"
        self.path(inside).write_text(json.dumps(given, ensure_ascii=False), encoding="utf-8")
        try:
            paths = [self.inside(n) for n in names]
            ran = self.run(code, f"/workspace/{inside}", *paths, timeout=timeout, kept=None)
        finally:
            self.path(inside).unlink(missing_ok=True)
        lines = ran.output.strip().splitlines() or ["it printed nothing"]
        if ran.status != 0:
            raise ValueError(f"it failed: {lines[-1]}")
        return json.loads(lines[-1])

    def run(
        self, code: str, *args: str, timeout: float = TIMEOUT, kept: int | None = OUTPUT
    ) -> Ran:
        """Runs Python code in the sandbox, in the workspace, with `args` as its sys.argv[1:]; of
        its output, `kept` bytes at most, its start and its end, or of None all of it to
        LONGEST."""
        if shutil.which("bwrap") is None:
            raise RuntimeError("the sandbox needs bubblewrap, which is not installed")
        python, mounts = "/usr/bin/python3", []
        if self.environment is not None:
            python = str(self.environment / "bin" / "python")
            mounts = ["--ro-bind", str(self.environment), str(self.environment)]
        upper = None
        if _overlays(self.scratch):
            self.scratch.mkdir(parents=True, exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix="run-", dir=self.scratch))
            upper, work = run / "upper", run / "work"
            upper.mkdir()
            work.mkdir()
            seen = ["--overlay-src", str(self.root), "--overlay", str(upper), str(work)]
        else:
            seen = ["--bind", str(self.root)]
        command = [
            "prlimit", f"--as={MEMORY}", "--",
            "bwrap", *_SYSTEM, *mounts, *seen, "/workspace",
            "--chdir", "/workspace", "--unshare-all", "--die-with-parent", "--new-session",
            "--clearenv", "--setenv", "PATH", f"{Path(python).parent}:/usr/bin",
            "--setenv", "HOME", "/tmp", "--setenv", "MPLBACKEND", "Agg",
            python, "-I", "-B", "-c", code, *args,
        ]  # fmt: skip
        env, stopped = {"PATH": os.environ.get("PATH", "/usr/bin")}, threading.Event()
        out = subprocess.PIPE
        try:
            with subprocess.Popen(
                command, stdout=out, stderr=subprocess.STDOUT, env=env
            ) as process:

                def stop() -> None:  # out of time
                    stopped.set()
                    process.kill()

                timer = threading.Timer(timeout, stop)
                timer.start()
                assert process.stdout is not None
                output = _kept(process.stdout, kept)
                status = process.wait()
                timer.cancel()
            if upper is not None:
                self._apply(upper)
        finally:
            if upper is not None:
                _remove_run(upper.parent)
        return Ran(None if stopped.is_set() else status, output)

    def _apply(self, upper: Path) -> None:
        # makes in the workspace the changes a run wrote in its overlay's upper folder, each file
        # changed or deleted kept first: a file or a link written there takes the place of any
        # of its name, a whiteout, a character device, deletes one, and an opaque folder, one
        # deleted and made anew, replaces the folder of its name
        for folder, folders, names in os.walk(upper):
            at = Path(folder).relative_to(upper)
            if at.parts and _opaque(Path(folder)):
                self._clear(at.as_posix())
            for name in folders:
                target = self.root / at / name
                if target.is_symlink() or target.is_file():
                    self._clear((at / name).as_posix())
                target.mkdir(parents=True, exist_ok=True)
            for name in names:
                source, target = Path(folder) / name, self.root / at / name
                self._clear((at / name).as_posix())
                if not stat.S_ISCHR(source.lstat().st_mode):  # a whiteout deletes alone
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, target)

    def _clear(self, name: str) -> None:
        # deletes what has a name, a link not followed, and nothing kept of it, a file or a
        # folder kept first, so that a run's change takes its place
        path = self.root / name
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            self.delete(name)


@functools.cache
def _overlays(scratch: Path) -> bool:
    # whether the sandbox mounts an overlay, its writes at `scratch`: as an unprivileged user,
    # which needs Linux 5.11 at least, and bubblewrap 0.11
    if shutil.which("bwrap") is None:
        return False
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        probe = Path(tempfile.mkdtemp(prefix="probe-", dir=scratch))
    except OSError:
        return False
    try:
        for part in ("lower", "upper", "work"):
            (probe / part).mkdir()
        command = [
            "bwrap", *_SYSTEM, "--overlay-src", str(probe / "lower"), "--overlay",
            str(probe / "upper"), str(probe / "work"), "/workspace", "--unshare-all",
            "--die-with-parent", "/usr/bin/true",
        ]  # fmt: skip
        return subprocess.run(command, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        _remove_run(probe)


def _opaque(folder: Path) -> bool:
    # whether an overlay's upper folder is opaque: deleted and made anew, hiding what it held
    for name in ("user.overlay.opaque", "trusted.overlay.opaque"):
        try:
            if os.getxattr(folder, name, follow_symlinks=False) == b"y":
                return True
        except OSError:
            continue
    return False


def _remove_run(folder: Path) -> None:
    # deletes a folder of a run's, its overlay's work folder too, which overlayfs leaves unreadable
    for inside in folder.rglob("*") if folder.is_dir() else []:
        if inside.is_dir() and not inside.is_symlink():
            with contextlib.suppress(OSError):
                inside.chmod(0o700)
    shutil.rmtree(folder, ignore_errors=True)


def hidden(name: str) -> bool:
    """Whether a file of the workspace, by its name there, is one no one sees, nor keeps: one of a
    hidden folder's, or its own, as a page saved to read on, or the owner file Word makes beside a
    document while it is open, ~$, which is no one's work."""
    return any(part.startswith((".", "~$")) for part in name.split("/"))


def _short(name: str, n: int = NAME) -> str:
    # a name cut to n bytes at most, between its characters
    return name.encode()[:n].decode(errors="ignore")


def _kept(stream: IO[bytes], n: int | None) -> str:
    # a process's output, read as it comes, which never waits on a full pipe: its start and end
    # if it is longer than n bytes, or of None its start to LONGEST bytes, so that a loop of
    # prints never fills leat's memory
    keep = LONGEST if n is None else n
    head, tail, total = bytearray(), bytearray(), 0
    for chunk in iter(lambda: stream.read(1 << 16), b""):
        total += len(chunk)
        head += chunk[: keep - len(head)]
        if n is not None:
            tail = (tail + chunk)[-keep:]
    if total <= keep:
        return head.decode(errors="replace")
    if n is None:
        return f"{head.decode(errors='replace')}\n… ({total - keep} bytes left out)"
    half = n // 2
    start, end = head[:half].decode(errors="replace"), tail[-half:].decode(errors="replace")
    return f"{start}\n… ({total - 2 * half} bytes left out) …\n{end}"
