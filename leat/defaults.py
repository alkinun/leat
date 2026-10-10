"""The sampling a model runs with where a request leaves it out: as its makers recommend it, apart
as it reasons or not, under an operator's own, by model, of a file of JSON.

What a model's makers recommend comes of its GGUF's general.sampling keys, which llama.cpp's
converter writes of its generation_config.json, and, of the families whose files say none of it or
not all, of their generation_config.json and model cards: Qwen's sets apart as the model thinks or
not, and its presence_penalty, which no GGUF key holds. What neither says is OpenAI's default.
"""

import json
import math
import re
import struct
import threading
from pathlib import Path
from typing import Any

# the options, as a request names them, and what each must be
OPTIONS: dict[str, tuple[type, float, float]] = {
    "temperature": (float, 0, 2), "top_k": (int, 0, 1 << 20), "top_p": (float, 0, 1),
    "min_p": (float, 0, 1), "presence_penalty": (float, -2, 2),
}  # fmt: skip
# OpenAI's, of what no one recommends otherwise
OPENAI = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "presence_penalty": 0.0}
# the GGUF keys llama.cpp's converter writes, by the option each is
_KEYS = {"general.sampling.temp": "temperature", "general.sampling.top_k": "top_k",
         "general.sampling.top_p": "top_p", "general.sampling.min_p": "min_p"}  # fmt: skip

Options = dict[str, float]
# Qwen3.6's cards, "Best Practices", of 27B and 35B A3B alike: thinking for general tasks, and
# instruct
_QWEN36: tuple[Options, Options] = (
    {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
)
# each family's, as its makers recommend it: by general.architecture and, where several makers'
# models share one, a word of general.name; the set as the model reasons, and as it does not, if
# its makers recommend another
_FAMILIES: list[tuple[str, str, Options, Options | None]] = [
    # OpenAI's README: "We recommend sampling with temperature=1.0 and top_p=1.0"
    ("gpt-oss", "", {"temperature": 1.0, "top_p": 1.0}, None),
    ("qwen35", "", *_QWEN36),
    ("qwen35moe", "", *_QWEN36),
    # Qwen3's cards: thinking mode, and non-thinking
    ("qwen3", "", {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
     {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}),
    ("qwen3moe", "", {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
     {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}),
    # the generation_config.json of each
    ("qwen2", "", {"temperature": 0.7, "top_p": 0.8, "top_k": 20}, None),
    ("gemma3", "", {"top_p": 0.95, "top_k": 64}, None),
    ("gemma4", "", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}, None),
    ("llama", "llama", {"temperature": 0.6, "top_p": 0.9}, None),
    ("llama", "mistralsmall", {"temperature": 0.15}, None),
]  # fmt: skip


def recommended(metadata: dict[str, Any]) -> tuple[Options, Options]:
    """A model's sampling as its makers recommend it, every option of OPTIONS: as it reasons, and
    as it does not. Its GGUF's keys say the first, as its generation_config.json does, and the
    second too where its family's makers recommend no other."""
    arch = metadata.get("general.architecture", "")
    named = re.sub(r"[^a-z0-9]", "", f"{metadata.get('general.name', '')}".lower())
    family = next(((r, p) for a, word, r, p in _FAMILIES if a == arch and word in named), None)
    reasoning, plain = family or ({}, None)
    written = {option: _f32(metadata[key]) for key, option in _KEYS.items() if key in metadata}
    reasons = OPENAI | reasoning | written
    return reasons, OPENAI | plain if plain is not None else reasons


def _f32(value: Any) -> Any:
    # a float32 of a GGUF's as the shortest decimal of it, 0.95 rather than 0.949999988079071: the
    # same float32, which sampling takes, as /v1/models says it
    if not isinstance(value, float) or not math.isfinite(value):
        return value
    exact = struct.pack("<f", value)
    shortest = (float(f"{value:.{digits}g}") for digits in range(1, 18))
    return next(t for t in shortest if struct.pack("<f", t) == exact)


def checked(options: Any, where: str) -> Options:
    """Options as a file of them gives them, each of OPTIONS and of its kind. Raises ValueError,
    saying `where`, if they are not."""
    if not isinstance(options, dict):
        raise ValueError(f"{where} must be an object of sampling options")
    for name, value in options.items():
        kind, low, high = OPTIONS.get(name, (None, 0, 0))
        if kind is None:
            raise ValueError(f"{where} has {name!r}, none of {', '.join(OPTIONS)}")
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        if not ok or (kind is int and value != int(value)) or not low <= value <= high:
            what = "an integer" if kind is int else "a number"
            raise ValueError(f"{where}'s {name} must be {what} from {low} to {high}")
    return {k: int(v) if OPTIONS[k][0] is int else float(v) for k, v in options.items()}


class Overrides:
    """An operator's sampling, of a file of JSON: by a model's id, or "*" for every model, the
    options that win over its makers', as {"gpt-oss-20b": {"temperature": 0.8}}. The file is read
    again as it changes, so that a change counts from the next request on."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seen: tuple[int, int] | None = None
        self._models: dict[str, Options] = {}

    def of(self, model: str) -> Options:
        """The options the file sets of a model, those of "*" under its own. Raises ValueError if
        the file is no file of them."""
        with self._lock:
            stat = self.path.stat()
            if (seen := (stat.st_mtime_ns, stat.st_size)) != self._seen:
                models = json.loads(self.path.read_text() or "{}")
                if not isinstance(models, dict):
                    raise ValueError(f"{self.path} must be an object of models' sampling")
                self._models = {m: checked(o, f"{self.path}'s {m}") for m, o in models.items()}
                self._seen = seen
            return self._models.get("*", {}) | self._models.get(model, {})
