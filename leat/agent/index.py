"""The index: what each space's files say, read once and kept, to read again at once and to search.

A space's files are read in the background as they come and change: each one's text, as
leat.agent.documents reads it, kept until the file changes, as its size and the time it changed
tell; and its passages, each some lines of one of its places, which SQLite's FTS5 finds by the
words they share with a query, the likeliest first. The index is a copy of what the files say,
kept beside the agent's state, and one of another version is made anew from them.

A query's words find those written as they are and those that begin as they do, as Turkish's
suffixes and English's endings change a word: one of more than STEM letters finds the words that
begin with its first letters but its last three, STEM at least, so that "sözleşmesi" finds
"sözleşmenin", and "termination" "terminate". Dotted and dotless i are one letter, and a letter
and its accented forms are one.

With a transcriber, an image's text is read too, and a PDF's scanned pages', by the model that sees
images; while none does, a PDF's scans are left unread, and read once one does, at rescan().

With an embedder, each passage is embedded too, after its file is read, and a search finds them by
what they mean as well as by their words: the passages likeliest by each, those by meaning MEANT
similar at least and NEAR the most similar, ranked as their reciprocal ranks add up, as reciprocal
rank fusion ranks them, so that a question finds a passage in other words or another language, and
an invoice's number still finds its own.
"""

import collections
import mimetypes
import queue
import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from leat.agent import documents
from leat.agent.client import EngineError
from leat.agent.context import picture
from leat.agent.embeddings import Embedder, similarity
from leat.agent.ocr import Transcriber
from leat.agent.workspace import Workspace

VERSION = 3  # of the index's tables: one of another is made anew
PASSAGE = 1200  # characters of a passage at most
FOUND = 8  # passages a search finds at most
EACH = 3  # of one file at most
STEM = 5  # letters of a word's beginning that a query's word finds at least
# the words of every sentence, English's and Turkish's, as fold() has them, which a search leaves
# out
_COMMON = """
a an the is are was were be been am of to in on at for and or but with by from as it its this that
these those what which who whom how why when where much many do does did can could should would
will shall may might must about into than then there their they them your you my me i we our us he
she his her has have had not no if so any all some
ve veya ile bu şu o bir için de da mi mu mü ne nasil kaç ki gibi daha en çok ama fakat ya hangi
neden niye nerede kim olan olarak var yok her kadar
"""
COMMON = frozenset(_COMMON.split())
CANDIDATES = 50  # passages found by their words, and by what they mean, before they are ranked
# the similarity, a cosine, of a passage found by what it means, at least, and below the most
# similar's at most: Qwen3-Embedding 0.6B's cosines of a firm's files and questions of them were
# 0.46 to 0.60 of the files that answered and 0.36 to 0.56 of those that did not, but within 0.05
# of the first only of those that came close to answering
MEANT, NEAR = 0.45, 0.05
FUSION = 60  # of reciprocal rank fusion: a rank's score is 1 / (FUSION + its rank)
EMBEDDED = 32  # passages embedded at once
_TABLES = """
CREATE TABLE documents (
  folder TEXT NOT NULL,
  name TEXT NOT NULL,
  size INTEGER NOT NULL,
  modified REAL NOT NULL,
  text TEXT,
  error TEXT,
  scans INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (folder, name)
);
CREATE VIRTUAL TABLE passages USING fts5 (
  words, text UNINDEXED, folder UNINDEXED, name UNINDEXED, place UNINDEXED, start UNINDEXED,
  tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE vectors (passage INTEGER PRIMARY KEY, folder TEXT NOT NULL, vector BLOB NOT NULL);
CREATE INDEX vectors_folder ON vectors (folder);
"""
# deletes the vectors of a file's passages, of its folder and name, before they are deleted
_UNEMBED = (
    "DELETE FROM vectors WHERE passage IN (SELECT rowid FROM passages WHERE folder = ?1"
    " AND name = ?2)"
)


class Index:
    """The index at `path` of the spaces of `workspace`, each by its folder there, as
    "projects/<id>", whose images and scans `transcriber`, if any, reads, and whose passages
    `embedder`, if any, embeds. `changed` is called with a space's folder once its files' states
    change, as one begins to be read and once it is, on the index's thread."""

    def __init__(
        self, path: Path | str, workspace: Workspace, transcriber: Transcriber | None = None,
        embedder: Embedder | None = None,
    ):  # fmt: skip
        self.workspace, self.transcriber, self.embedder = workspace, transcriber, embedder
        self.changed: Callable[[str], None] = lambda folder: None
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode = WAL")
        if self._db.execute("PRAGMA user_version").fetchone()[0] != VERSION:
            for table in ("documents", "passages", "vectors"):
                self._db.execute(f"DROP TABLE IF EXISTS {table}")
            self._db.executescript(_TABLES)
            self._db.execute(f"PRAGMA user_version = {VERSION}")
        self._lock = threading.Lock()
        self._waiting: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._reading: dict[str, set[str]] = {}  # each space's files still to read, by folder

    def start(self) -> None:
        """Reads every space's new and changed files, then each space's as it is refreshed, on a
        thread of its own."""
        self.rescan()
        threading.Thread(target=self._work, name="index", daemon=True).start()

    def rescan(self) -> None:
        """Has every space's files read again that are new or changed, or were left unread as no
        model saw images, as one may now."""
        for checked in (self.transcriber, self.embedder):
            if checked is not None:
                checked.recheck()
        for kind in ("people", "projects"):
            for folder in sorted((self.workspace.root / kind).glob("*")):
                self.refresh(f"{kind}/{folder.name}")

    def refresh(self, folder: str) -> None:
        """Has a space's files read again, those new or changed since, and those gone forgotten."""
        self._waiting.put(folder)

    def text(self, space: Workspace, name: str) -> str:
        """A file's text, as kept if the file is as it was, or else read and kept. Raises
        FileNotFoundError if there is no such file, ValueError if it cannot be read as text."""
        file = space.path(name)
        name, folder = file.relative_to(space.root).as_posix(), self._folder(space)
        stat = file.stat()
        with self._lock:
            row = self._db.execute(
                "SELECT text FROM documents WHERE folder = ? AND name = ? AND size = ?"
                " AND modified = ? AND text IS NOT NULL",
                (folder, name, stat.st_size, stat.st_mtime),
            ).fetchone()  # fmt: skip
        if row is not None:
            return row[0]
        text, scans = self._read(space, name)
        self._keep(folder, name, stat.st_size, stat.st_mtime, text, scans=scans)
        return text

    def search(
        self, space: Workspace, query: str, n: int = FOUND, file: str | None = None
    ) -> list[dict[str, Any]]:
        """The passages of a space's files, or of one `file`, likeliest to say what a query asks,
        by their words and, with an embedder, by what they mean, n at most, EACH of a file of all:
        each's file, place, its start in the file's text, and its text."""
        folder, scores = self._folder(space), collections.defaultdict[int, float](float)
        for ranked in (self._worded(folder, query, file), self._meant(folder, query, file)):
            for rank, passage in enumerate(ranked):
                scores[passage] += 1 / (FUSION + rank + 1)
        best = sorted(scores, key=scores.__getitem__, reverse=True)
        with self._lock:
            rows = {row[0]: row[1:] for row in self._db.execute(
                f"SELECT rowid, name, place, start, text FROM passages WHERE rowid IN"
                f" ({', '.join('?' * len(best))})", best)}  # fmt: skip
        found: list[dict[str, Any]] = []
        for name, place, start, text in (rows[passage] for passage in best if passage in rows):
            if (file or sum(f["name"] == name for f in found) < EACH) and len(found) < n:
                found.append({"name": name, "place": place, "start": start, "text": text})
        return found

    def _worded(self, folder: str, query: str, file: str | None) -> list[int]:
        # the passages that share the query's words, the likeliest first, by their rows
        if not (match := _match(query)):
            return []
        with self._lock:
            return [row for (row,) in self._db.execute(
                "SELECT rowid FROM passages WHERE passages MATCH ?1 AND folder = ?2"
                " AND (?3 IS NULL OR name = ?3) ORDER BY rank LIMIT ?4",
                (match, folder, file, CANDIDATES))]  # fmt: skip

    def _meant(self, folder: str, query: str, file: str | None) -> list[int]:
        # the passages that mean what the query asks, MEANT similar at least, the likeliest
        # first, by their rows; none without an embedder, or while it fails
        if self.embedder is None or not self.embedder.available():
            return []
        try:
            asked = self.embedder.question(query)
        except EngineError:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT v.passage, v.vector FROM vectors v JOIN passages p ON p.rowid = v.passage"
                " WHERE v.folder = ?1 AND (?2 IS NULL OR p.name = ?2)", (folder, file),
            ).fetchall()  # fmt: skip
        scored = sorted(((similarity(asked, vector), row) for row, vector in rows), reverse=True)
        least = max(MEANT, scored[0][0] - NEAR) if scored else MEANT
        return [row for score, row in scored[:CANDIDATES] if score >= least]

    def states(self, folder: str) -> dict[str, dict[str, str]]:
        """The states of a space's files that are not wholly read, by their names: "reading";
        "failed", with why; or "scanned", of a PDF whose scanned pages wait for a model that sees
        images."""
        with self._lock:
            rows = self._db.execute(
                "SELECT name, error FROM documents WHERE folder = ?"
                " AND (error IS NOT NULL OR scans > 0)", (folder,),
            ).fetchall()  # fmt: skip
            reading = set(self._reading.get(folder, ()))
        states = {name: {"state": "failed", "error": error} if error else {"state": "scanned"}
                  for name, error in rows}  # fmt: skip
        return states | {name: {"state": "reading"} for name in reading}

    def forget(self, folder: str) -> None:
        """Forgets a space's files, as a project deleted takes them."""
        with self._lock, self._db:
            self._db.execute("BEGIN")
            for table in ("documents", "passages", "vectors"):
                self._db.execute(f"DELETE FROM {table} WHERE folder = ?", (folder,))

    def _work(self) -> None:
        while True:
            folder = self._waiting.get()
            try:
                self.update(folder)
            except Exception as e:  # a bug's, which must not stop the reading of the others
                print(f"leat agent: the index failed to read {folder}: {e!r}", flush=True)

    def update(self, folder: str) -> None:
        """Reads a space's files that are new or changed, now, forgetting those gone, each told of
        as it begins to be read and once it is."""
        if not (self.workspace.root / folder).is_dir():
            return self.forget(folder)
        space = self.workspace.space(folder)
        on_disk = {f["name"]: f for f in space.files()}
        with self._lock:
            sql = "SELECT name, size, modified, scans FROM documents WHERE folder = ?"
            known = {row[0]: row[1:] for row in self._db.execute(sql, (folder,))}
        sees = self.transcriber is not None and self.transcriber.available()
        gone = [name for name in known if name not in on_disk]
        stale = [
            name for name, f in on_disk.items()
            if (known.get(name, ())[:2] != (f["size"], f["modified"])
                or sees and known[name][2] > 0)  # its scans unread, as they can now be
            and (documents.readable(space, name) or sees and picture(name))
        ]  # fmt: skip
        with self._lock, self._db:
            self._db.execute("BEGIN")
            for name in gone:
                self._db.execute(_UNEMBED, (folder, name))
                for table in ("documents", "passages"):
                    sql = f"DELETE FROM {table} WHERE folder = ? AND name = ?"
                    self._db.execute(sql, (folder, name))
            self._reading[folder] = set(stale)
        if gone or stale:
            self.changed(folder)
        for name in stale:
            try:
                file = space.path(name).stat()
                text, scans = self._read(space, name)
                self._keep(folder, name, file.st_size, file.st_mtime, text, scans=scans)
            except FileNotFoundError:  # gone meanwhile
                pass
            except (ValueError, RuntimeError) as e:  # as the sandbox not being there
                f = on_disk[name]
                self._keep(folder, name, f["size"], f["modified"], None, str(e))
            with self._lock:
                self._reading[folder].discard(name)
            self.changed(folder)
        self._embed(folder)

    def _embed(self, folder: str) -> None:
        # embeds a space's passages not yet embedded, EMBEDDED at a time, with the embedder if it
        # embeds: those it fails to, at the next update
        if self.embedder is None or not self.embedder.available():
            return
        while True:
            with self._lock:
                rows = self._db.execute(
                    "SELECT p.rowid, p.text FROM passages p LEFT JOIN vectors v ON"
                    " v.passage = p.rowid WHERE p.folder = ? AND v.passage IS NULL LIMIT ?",
                    (folder, EMBEDDED),
                ).fetchall()  # fmt: skip
            if not rows:
                return
            try:
                vectors = self.embedder.passages([text for _, text in rows])
            except EngineError:
                return
            with self._lock, self._db:  # those of passages still there, as a file may change
                self._db.execute("BEGIN")
                self._db.executemany(
                    "INSERT OR REPLACE INTO vectors SELECT ?1, ?2, ?3"
                    " WHERE EXISTS (SELECT 1 FROM passages WHERE rowid = ?1)",
                    [(row, folder, v) for (row, _), v in zip(rows, vectors, strict=True)],
                )

    def _read(self, space: Workspace, name: str) -> tuple[str, int]:
        # a file's text, an image's as the transcriber reads it, and a PDF's scanned pages' too;
        # and the scans left unread, as no model sees images. Raises ValueError if it cannot be
        # read, FileNotFoundError if it is gone
        sees = self.transcriber is not None and self.transcriber.available()
        if picture(name):
            if not sees:
                raise ValueError(f"{name} is an image, which only a model that sees images reads")
            assert self.transcriber is not None
            data, kind = space.path(name).read_bytes(), mimetypes.guess_type(name)[0]
            try:
                return self.transcriber(data, kind or "image/png"), 0
            except EngineError as e:
                raise ValueError(f"{name} could not be read: {e}") from e
        text = documents.text(space, name)
        if not (scans := documents.scans(name, text)) or not sees:
            return text, len(scans)
        assert self.transcriber is not None
        try:
            pages = documents.render(space, name, scans)
            read = {n: self.transcriber(png, "image/png") for n, png in pages.items()}
        except (ValueError, EngineError):  # read as it is, its scans left for later
            return text, len(scans)
        return documents.transcribed(text, read), 0

    def _keep(
        self, folder: str, name: str, size: int, modified: float, text: str | None,
        error: str | None = None, scans: int = 0,
    ) -> None:  # fmt: skip
        # keeps a file's text and its passages, and how many of its scans are unread; or why it
        # could not be read
        found = passages(name, text) if text is not None else []
        with self._lock, self._db:
            self._db.execute("BEGIN")
            self._db.execute(_UNEMBED, (folder, name))
            sql = "DELETE FROM passages WHERE folder = ? AND name = ?"
            self._db.execute(sql, (folder, name))
            self._db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (folder, name, size, modified, text, error, scans),
            )
            self._db.executemany(
                "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
                [(fold(said), said, folder, name, place, start) for start, place, said in found],
            )

    def _folder(self, space: Workspace) -> str:
        return space.root.relative_to(self.workspace.root).as_posix()


def passages(name: str, text: str) -> list[tuple[int, str, str]]:
    """A file's text in passages, each some of its lines, PASSAGE characters at most, a line
    longer cut between words, within one of its places: each's start in the text, its place, as
    "page 3", or "" before any, and its text."""
    marks = [(0, ""), *documents.places(name, text)]
    found = []
    for (start, place), (end, _) in zip(marks, [*marks[1:], (len(text), "")], strict=True):
        first = last = None  # the passage's span, as its lines are added
        for line in _lines(text, start, end):
            if first is not None and line[1] - first > PASSAGE:
                found.append((first, place, text[first:last]))
                first = None
            first, last = line[0] if first is None else first, line[1]
        if first is not None:
            found.append((first, place, text[first:last]))
    return found


def fold(text: str) -> str:
    """Text as the index matches it: in lowercase, with dotted and dotless i one letter, as
    Turkish's İ and ı, and English's I and i, are."""
    return text.lower().replace("ı", "i").replace("̇", "")


def _lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    # the spans of the lines of text[start:end] that say something, each PASSAGE characters at
    # most, a line longer cut between its words
    spans = []
    for line in re.finditer(r"[^\n]*\S[^\n]*", text[start:end]):
        at, stop = start + line.start(), start + line.end()
        while stop - at > PASSAGE:
            cut = text.rfind(" ", at + 1, at + PASSAGE)
            cut = cut if cut > at else at + PASSAGE
            spans.append((at, cut))
            at = cut + 1 if text[cut] == " " else cut
        spans.append((at, stop))
    return spans


def _match(query: str) -> str:
    # an FTS5 query of a search's words: any of them, as written or begun, but for those of a
    # letter alone, and the words of every sentence, English's and Turkish's, unless it has no
    # others
    words = list(dict.fromkeys(re.findall(r"\w+", fold(query))))
    words = [w for w in words if w not in COMMON] or words
    terms = []
    for word in words:
        if len(word) > STEM:
            terms.append(f'"{word[: max(STEM, len(word) - 3)]}"*')
        elif len(word) > 3:
            terms.append(f'"{word}"*')
        elif len(word) > 1:
            terms.append(f'"{word}"')
    return " OR ".join(terms)
