"""Syncing a project with a folder of the box's, as an office's network drive mounted there: a copy
of its files kept in the project's files, in a folder of its name, one way, the folder's as they
are, every SYNC seconds and once it is chosen.

A file new or changed in the folder, as its size and time of change tell, is copied, and one gone
from it deleted from the copy; links out of the folder and hidden files are not copied. A copy
rather than the folder itself keeps the index and the sandbox as they are for every file, and the
project's files there while the drive is away. A folder of more than FILES files is refused, as
one chosen by mistake, a whole drive's.
"""

import os
import shutil
from pathlib import Path

SYNC = 300  # seconds between a project's syncs
FILES = 50000  # files of a folder synced at most


def mirror(source: Path, target: Path) -> tuple[int, int]:
    """Makes `target` a copy of `source`'s files; returns how many files it copied and deleted.
    Raises OSError if the folder is not there, or cannot be read, ValueError if it holds more than
    FILES files."""
    if not source.is_dir():
        raise FileNotFoundError(f"{source} is not there: is its drive mounted?")
    wanted = {}
    for root, folders, names in os.walk(source):
        folders[:] = [f for f in folders if not f.startswith(".")]
        for name in names:
            path = Path(root, name)
            if name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            wanted[path.relative_to(source)] = path
            if len(wanted) > FILES:
                raise ValueError(f"{source} holds more than {FILES} files: choose a folder in it")
    copied = deleted = 0
    for relative, path in wanted.items():
        copy, stat = target / relative, path.stat()
        if copy.is_file() and _same(copy.stat(), stat):
            continue
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        copied += 1
    for copy in sorted(target.rglob("*"), reverse=True) if target.is_dir() else []:
        if copy.is_file() and copy.relative_to(target) not in wanted:
            copy.unlink()
            deleted += 1
        elif copy.is_dir() and not any(copy.iterdir()):
            copy.rmdir()
    return copied, deleted


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)
