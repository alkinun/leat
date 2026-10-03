"""Generation: chunked prefill and token-by-token decode, each replayed from one compiled graph."""

import itertools
from collections.abc import Iterator
from pathlib import Path

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.gguf import GGUF
from leat.model import Config, Transformer
from leat.sampler import sample
from leat.tokenizer import Tokenizer


class Engine:
    """A loaded model, ready to generate.

    The KV cache holds `slots` sequences, and a prompt is prefilled only past the longest prefix
    any of them shares with it. A prompt that extends a slot's tokens continues in that slot, so
    multi-turn chat costs only the latest turn. Any other prompt takes an empty slot, or else the
    least recently used one, and first copies that prefix in: a system prompt that several
    conversations share, say.
    """

    def __init__(
        self, path: str | Path, max_context: int = 4096, prefill_chunk: int = 512, slots: int = 1
    ):
        if slots < 1:
            raise ValueError(f"slots must be at least 1, got {slots}")
        self.gguf = gguf = GGUF.open(path)
        self.tokenizer = Tokenizer(gguf.metadata)
        self.config = Config.from_gguf(gguf.metadata)
        self.model = Transformer(self.config, gguf.load(), max_context, slots)
        self.max_context, self.prefill_chunk, self.slots = max_context, prefill_chunk, slots
        self._pos = UOp.variable("start_pos", 0, max_context - 1)
        self._len = UOp.variable("chunk_len", 1, prefill_chunk)
        self._slot = UOp.variable("slot", 0, slots - 1)
        self._source = UOp.variable("source", 0, slots - 1)
        self._prefix = UOp.variable("prefix", 1, max_context - 1)
        # TinyJit runs a function once as is, then captures it on the second call: capture on the
        # first, a fresh process's slow call for each graph. The device's random state must exist
        # by then, or the graph would create it anew every call.
        Tensor.rand(1).realize()  # on the default device, which holds the model
        self._prefill, self._decode = TinyJit(self._step), TinyJit(self._step)
        self._copy = TinyJit(self.model.copy)
        self._prefill.cnt = self._decode.cnt = self._copy.cnt = 1
        self._cached: list[list[int]] = [[] for _ in range(slots)]  # tokens each slot holds
        self._used = [0] * slots  # when each slot last started a generation
        self._clock = itertools.count(1)
        self._generating = False

    def generate(
        self, prompt: list[int], max_tokens: int, temperature: float = 0.0, ignore_eog: bool = False
    ) -> Iterator[int]:
        """Yields up to `max_tokens` ids; stops early at end of generation or the context limit.

        One generation runs at a time: exhaust or close one before starting the next.
        """
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
        if not 0 < len(prompt) < self.max_context:
            raise ValueError(
                f"prompt must have 1 to {self.max_context - 1} tokens, got {len(prompt)}"
            )
        if self._generating:
            # each graph returns its output in a buffer that its next call overwrites
            raise RuntimeError("a generation is unfinished: exhaust or close it first")
        slot = self._claim(prompt)
        cached = self._cached[slot]  # the prompt's prefix the slot holds, kept up to date
        self._generating = True
        try:
            pos, slot_var = len(cached), self._slot.bind(slot)
            temp = Tensor([temperature], dtype=dtypes.float32)
            padded = Tensor([prompt + [0] * (self.max_context - len(prompt))], dtype=dtypes.int32)
            while pos < len(prompt):
                n = min(self.prefill_chunk, len(prompt) - pos)
                start, length = self._pos.bind(pos), self._len.bind(n)
                chunk = padded[:, start : start + length]
                token = self._prefill(chunk, slot_var, start, temp)
                pos += n
            cached += prompt[len(cached) :]
            for remaining in reversed(range(max_tokens)):
                yield (t := int(token.item()))
                eog = t in self.tokenizer.eog_ids and not ignore_eog
                if not remaining or eog or pos >= self.max_context:
                    return
                token = self._decode(token, slot_var, self._pos.bind(pos), temp)
                cached.append(t)
                pos += 1
        finally:
            self._generating = False

    def reset(self) -> None:
        """Forgets every cached prefix, so the next prompt is prefilled from scratch."""
        self._cached = [[] for _ in range(self.slots)]

    def _claim(self, prompt: list[int]) -> int:
        # a slot for the prompt, made to hold the longest prefix of it that any slot holds
        slots = range(self.slots)
        shared = [_shared(prompt, cached) for cached in self._cached]
        extended = [s for s in slots if shared[s] == len(self._cached[s])]  # empty slots too
        if extended:
            slot = max(extended, key=lambda s: shared[s])
        else:
            slot = min(slots, key=lambda s: self._used[s])
        source = max(slots, key=lambda s: shared[s])
        if (prefix := shared[source]) > shared[slot]:
            self._cached[slot] = []  # while the copy overwrites it
            self._copy(self._source.bind(source), self._slot.bind(slot), self._prefix.bind(prefix))
        self._cached[slot] = prompt[:prefix]  # what stays valid if prefill is interrupted
        self._used[slot] = next(self._clock)
        return slot

    def _step(self, tokens: Tensor, slot: UOp, start_pos: UOp, temperature: Tensor) -> Tensor:
        hidden = self.model(tokens, start_pos, slot)
        return sample(self.model.logits(hidden[:, -1, :]), temperature).realize()


def _shared(prompt: list[int], cached: list[int]) -> int:
    # how many leading tokens a slot holds of the prompt, all but its last at most: generation
    # starts from the logits of running that one
    pairs = zip(prompt[:-1], cached, strict=False)
    return sum(1 for _ in itertools.takewhile(lambda p: p[0] == p[1], pairs))
