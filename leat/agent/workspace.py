"""The workspace: the user's files, which they upload and the agent makes, and Python run among them
in a sandbox.

The sandbox is bubblewrap's: the code sees /usr, the sandbox's environment of libraries, which it
cannot change, and the workspace, at /workspace; no network, no other file of the box's, and a
memory and a time it may not pass. Files of the kinds whose reading parses them, as PDFs, are read
there too, so that a file made to attack its parser attacks the sandbox. Python runs isolated, so
that a file of the workspace's, as docx.py, cannot take a library's place, and writes no bytecode.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIMEOUT = 120  # seconds a run may take
MEMORY = 4 << 30  # bytes of memory a run may take
OUTPUT = 20000  # characters of a run's output kept, its start and its end
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

    def path(self, name: str) -> Path:
        """The path of a file in the workspace, named relative to it. Raises ValueError for a
        name outside it, such as an absolute one, or one up from it."""
        path = (self.root / name.lstrip("/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"{name} is outside the workspace")
        return path

    def files(self) -> list[dict[str, Any]]:
        """Every file, by its name in the workspace, the latest changed first; hidden ones not."""
        found = []
        for path in self.root.rglob("*"):
            name = path.relative_to(self.root).as_posix()
            if path.is_file() and not any(part.startswith(".") for part in name.split("/")):
                stat = path.stat()
                found.append({"name": name, "size": stat.st_size, "modified": stat.st_mtime})
        return sorted(found, key=lambda f: f["modified"], reverse=True)

    def free(self, name: str) -> str:
        """A name for a new file at the workspace's top, as `name` but for what makes it a path,
        numbered as "notes (2).txt" if a file has it."""
        base = Path(name.replace("\\", "/").split("/")[-1].lstrip(". ")).name or "file"
        stem, suffix = Path(base).stem, Path(base).suffix
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
        its output, `kept` characters at most, its start and its end, or all of it of None."""
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
        try:
            done = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                env={"PATH": os.environ.get("PATH", "/usr/bin")},
            )  # fmt: skip
        except subprocess.TimeoutExpired as e:
            return Ran(None, _kept((e.output or b"").decode(errors="replace"), kept))
        return Ran(done.returncode, _kept(done.stdout.decode(errors="replace"), kept))


def _kept(output: str, n: int | None) -> str:
    # an output's start and end, if it is longer than n characters
    if n is None or len(output) <= n:
        return output
    half = n // 2
    return f"{output[:half]}\n… ({len(output) - n} characters left out) …\n{output[-half:]}"
