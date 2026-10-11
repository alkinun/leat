"""Embeddings of the files' passages and of questions, as the embedding model leat serve has beside
its model gives them, if it has one, as Qwen3-Embedding or EmbeddingGemma 2: the index finds
passages by what they mean, as well as by their words, so that a question finds a passage that says
it in other words, or in another language.

A question and a passage are each embedded as the model was taught, its form, which its name tells,
and so are the similarities of the passages that answer a question. An embedding keeps its first
DIMS dimensions, made of unit length again, as both were taught to be cut short: the cosine of two
is their dot product, which pure Python reckons over a project's passages at once.
"""

import array
import math
import threading
import time
from dataclasses import dataclass

from leat.agent.client import Client, EngineError

DIMS = 512  # of an embedding, kept
CHECK = 60  # seconds the engine's word on whether it embeds, and by which model, holds


@dataclass(frozen=True)
class Form:
    """How an embedding model was taught to embed a question and a passage, each a template of its
    text, and the similarity, a cosine, of a passage that answers a question: `meant` at least,
    and below the most similar's by `near` at most."""

    question: str
    passage: str
    meant: float
    near: float


# Qwen3-Embedding's: a question with an instruction of what it is for, a passage as it is. Its
# 0.6B's cosines of a firm's files and questions of them were 0.46 to 0.60 of the files that
# answered and 0.36 to 0.56 of those that did not, but within 0.05 of the first only of those that
# came close to answering
QWEN3 = Form(
    "Instruct: Given a question, retrieve passages of the user's files that answer it\nQuery: {}",
    "{}", 0.45, 0.05,
)  # fmt: skip
# EmbeddingGemma 2's, of its prompts for search, whose cosines run higher: of twelve questions of a
# firm's files in other words, in English and German, 0.69 to 0.83 of the passages that answered,
# eleven of them 0.70 at least, and 0.49 to 0.77 of those that did not, those that answered below
# the most similar by 0.08 at most; of three questions the files do not answer, 0.61 to 0.70 at most
GEMMA2 = Form("task: search result | query: {}", "title: none | text: {}", 0.70, 0.08)


def form(model: str) -> Form:
    """An embedding model's form, by its name, as leat serve names its GGUF: EmbeddingGemma's, or
    else Qwen3-Embedding's."""
    return GEMMA2 if "embeddinggemma" in model.lower() else QWEN3


class Embedder:
    """Embeds texts by the embedding model `engine` has beside its model, if it has one: `model`,
    its name, as the engine said it last, or None while it embeds nothing."""

    def __init__(self, engine: Client):
        self.engine = engine
        self.model: str | None = None
        self._checked = 0.0
        self._lock = threading.Lock()

    @property
    def form(self) -> Form:
        """How the model embeds."""
        return form(self.model or "")

    def available(self) -> bool:
        """Whether the engine embeds, as it said CHECK seconds ago at most."""
        with self._lock:
            if time.monotonic() - self._checked > CHECK:
                try:
                    self.model = self.engine.embed(["."])[0]
                except EngineError:
                    self.model = None
                self._checked = time.monotonic()
            return self.model is not None

    def recheck(self) -> None:
        """Has the engine asked again, at the next available()."""
        with self._lock:
            self._checked = 0.0

    def passages(self, texts: list[str]) -> list[bytes]:
        """Passages' embeddings, each as the index keeps it. Raises EngineError."""
        _, vectors = self.engine.embed([self.form.passage.format(t) for t in texts])
        return [_kept(v) for v in vectors]

    def question(self, text: str) -> bytes:
        """A question's embedding, as the index keeps one. Raises EngineError."""
        return _kept(self.engine.embed([self.form.question.format(text)])[1][0])


def similarity(a: bytes, b: bytes) -> float:
    """The cosine of two embeddings as the index keeps them."""
    return math.sumprod(array.array("f", a), array.array("f", b))


def _kept(vector: list[float]) -> bytes:
    # an embedding cut to DIMS dimensions, of unit length, as float32's bytes
    cut = vector[:DIMS]
    length = math.sqrt(math.sumprod(cut, cut)) or 1.0
    return array.array("f", (x / length for x in cut)).tobytes()
