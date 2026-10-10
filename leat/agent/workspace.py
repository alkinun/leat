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
"""

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

TIMEOUT = 120  # seconds a run may take
MEMORY = 4 << 30  # bytes of memory a run may take
OUTPUT = 20000  # bytes of a run's output kept, its start and its end
LONGEST = 50_000_000  # bytes of a document's text kept, which is read whole
NAME = 255  # bytes of a file's name at most, as Linux's file systems take
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


class Workspace:
    """The files at `root`, and Python run among them by the interpreter of `environment`, a
    virtual environment of the libraries the sandbox offers, or the system's without one."""

    def __init__(self, root: Path, environment: Path | None = None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.environment = environment.resolve() if environment else None

    def space(self, name: str) -> "Workspace":
        """The workspace of a folder of this one's, as one of its own, whose code runs with the
        same libraries: a person's files, or a project's."""
        return Workspace(self.path(name), self.environment)

    def remove(self) -> None:
        """Deletes the workspace's folder, with every file in it."""
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name: str) -> Path:
        """The path of a file in the workspace, named relative to it. Raises ValueError for a
        name outside it, such as an absolute one, or one up from it."""
        if any(len(part.encode()) > NAME for part in Path(name).parts):
            raise ValueError(f"a file's name is {NAME} bytes at most")
        path = (self.root / name.lstrip("/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"{name} is outside the workspace")
        return path

    def files(self) -> list[dict[str, Any]]:
        """Every file, by its name in the workspace, the latest changed first; hidden ones not, nor
        links out of it, as the sandbox's code may make to any path."""
        found = []
        for path in self.root.rglob("*"):
            name = path.relative_to(self.root).as_posix()
            if any(part.startswith(".") for part in name.split("/")):
                continue
            if not path.is_file() or not path.resolve().is_relative_to(self.root):
                continue
            try:  # one deleted meanwhile, as by code a call runs at once with this one, is gone
                stat = path.stat()
            except FileNotFoundError:
                continue
            found.append({"name": name, "size": stat.st_size, "modified": stat.st_mtime})
        return sorted(found, key=lambda f: f["modified"], reverse=True)

    def free(self, name: str) -> str:
        """A name for a new file at the workspace's top, as `name` but for what makes it a path,
        numbered as "notes (2).txt" if a file has it."""
        base = Path(name.replace("\\", "/").split("/")[-1].lstrip(". ")).name or "file"
        stem, suffix = Path(base).stem, Path(base).suffix
        if len(base.encode()) > NAME - 8:  # its start and its kind, as long as a name may be with
            # room for a number, " (2)"
            suffix = suffix if len(suffix.encode()) < 16 else ""
            stem = stem.encode()[: NAME - 8 - len(suffix.encode())].decode(errors="ignore")
            base = stem + suffix
        n, candidate = 1, base
        while (self.root / candidate).exists():
            n += 1
            candidate = f"{stem} ({n}){suffix}"
        return candidate

    def delete(self, name: str) -> None:
        path = self.path(name)
        if not path.is_file():
            raise FileNotFoundError(f"there is no file {name}")
        path.unlink()

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
        command = [
            "prlimit", f"--as={MEMORY}", "--",
            "bwrap", *_SYSTEM, *mounts, "--bind", str(self.root), "/workspace",
            "--chdir", "/workspace", "--unshare-all", "--die-with-parent", "--new-session",
            "--clearenv", "--setenv", "PATH", f"{Path(python).parent}:/usr/bin",
            "--setenv", "HOME", "/tmp", "--setenv", "MPLBACKEND", "Agg",
            python, "-I", "-B", "-c", code, *args,
        ]  # fmt: skip
        env, stopped = {"PATH": os.environ.get("PATH", "/usr/bin")}, threading.Event()
        out = subprocess.PIPE
        with subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env) as process:

            def stop() -> None:  # out of time
                stopped.set()
                process.kill()

            timer = threading.Timer(timeout, stop)
            timer.start()
            assert process.stdout is not None
            output = _kept(process.stdout, kept)
            status = process.wait()
            timer.cancel()
        return Ran(None if stopped.is_set() else status, output)


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
