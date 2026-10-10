"""Embeddings of the files' passages and of questions, as the embedding model leat serve has beside
its model gives them, if it has one, as Qwen3-Embedding: the index finds passages by what they mean,
as well as by their words, so that a question finds a passage that says it in other words, or in
another language.

A question is embedded with an instruction of what it is for, a passage as it is, as the model was
taught. An embedding keeps its first DIMS dimensions, made of unit length again, as Qwen3-Embedding
was taught to be cut short: the cosine of two is their dot product, which pure Python reckons over
a project's passages at once.
"""

import array
import math
import threading
import time

from leat.agent.client import Client, EngineError

DIMS = 512  # of an embedding, kept
CHECK = 60  # seconds the engine's word on whether it embeds holds
# what a question is embedded for, as Qwen3-Embedding is asked
TASK = "Given a question, retrieve passages of the user's files that answer it"


class Embedder:
    """Embeds texts by the embedding model `engine` has beside its model, if it has one."""

    def __init__(self, engine: Client):
        self.engine = engine
        self._checked, self._has = 0.0, False
        self._lock = threading.Lock()

    def available(self) -> bool:
        """Whether the engine embeds, as it said CHECK seconds ago at most."""
        with self._lock:
            if time.monotonic() - self._checked > CHECK:
                try:
                    self.engine.embed(["."])
                    self._has = True
                except EngineError:
                    self._has = False
                self._checked = time.monotonic()
            return self._has

    def recheck(self) -> None:
        """Has the engine asked again, at the next available()."""
        with self._lock:
            self._checked = 0.0

    def passages(self, texts: list[str]) -> list[bytes]:
        """Passages' embeddings, each as the index keeps it. Raises EngineError."""
        return [_kept(v) for v in self.engine.embed(texts)]

    def question(self, text: str) -> bytes:
        """A question's embedding, as the index keeps one. Raises EngineError."""
        return _kept(self.engine.embed([f"Instruct: {TASK}\nQuery: {text}"])[0])


def similarity(a: bytes, b: bytes) -> float:
    """The cosine of two embeddings as the index keeps them."""
    return math.sumprod(array.array("f", a), array.array("f", b))


def _kept(vector: list[float]) -> bytes:
    # an embedding cut to DIMS dimensions, of unit length, as float32's bytes
    cut = vector[:DIMS]
    length = math.sqrt(math.sumprod(cut, cut)) or 1.0
    return array.array("f", (x / length for x in cut)).tobytes()
