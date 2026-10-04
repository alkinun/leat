"""Generation: chunked prefill and token-by-token decode, replayed from compiled graphs."""

import array
import itertools
import random
from collections.abc import Generator
from pathlib import Path

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.gguf import GGUF
from leat.model import Config, Transformer
from leat.sampler import sample
from leat.tokenizer import Tokenizer

# prompt tokens up to which a chunk takes a graph bound to that many, whose kernels size their
# work for so few: the matrix kernels take tiles of 16 tokens
FEW = 16


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
        self._few = UOp.variable("few_len", 1, min(FEW, prefill_chunk))
        self._slot = UOp.variable("slot", 0, slots - 1)
        self._source = UOp.variable("source", 0, slots - 1)
        # TinyJit runs a function once as is, then captures it on the second call: capture on the
        # first, a fresh process's slow call for each graph
        self._chunk, self._few_chunk, self._decode = (TinyJit(self._step) for _ in range(3))
        self._copy = TinyJit(self.model.copy)
        for jit in (self._chunk, self._few_chunk, self._decode, self._copy):
            jit.cnt = 1
        self._cached: list[list[int]] = [[] for _ in range(slots)]  # tokens each slot holds
        self._used = [0] * slots  # when each slot last started a generation
        self._clock = itertools.count(1)
        self._generating = False

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        ignore_eog: bool = False,
    ) -> Generator[int, None, None]:
        """Yields up to `max_tokens` ids; stops early at end of generation or the context limit.

        Sampling is greedy at temperature 0; above, a `seed` makes it repeatable. One generation
        runs at a time: exhaust or close one before starting the next.
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
        seed = random.getrandbits(32) if seed is None else seed % 2**32
        self._generating = True
        try:
            pos, slot_var = len(cached), self._slot.bind(slot)
            temp = Tensor([temperature], dtype=dtypes.float32)
            seeds = Tensor([seed], dtype=dtypes.uint32)
            while pos < len(prompt):
                chunk = prompt[pos : pos + self.prefill_chunk]
                token = self._prefill(chunk, pos, slot_var, temp, seeds)
                pos += len(chunk)
            cached += prompt[len(cached) :]
            for remaining in reversed(range(max_tokens)):
                yield (t := int(token.item()))
                eog = t in self.tokenizer.eog_ids and not ignore_eog
                if not remaining or eog or pos >= self.max_context:
                    return
                token = self._decode(token, slot_var, self._pos.bind(pos), temp, seeds)
                cached.append(t)
                pos += 1
        finally:
            self._generating = False

    def warm_up(self) -> None:
        """Compiles the graphs generation replays, which takes seconds in a fresh process, so that
        the first prompt runs at full speed. Leaves no prefix cached."""
        # a prefill of more than FEW tokens and a decode step, then a prefill of few
        for prompt, n in (([0] * min(FEW + 1, self.max_context - 1), 2), ([0, 0], 1)):
            self.reset()
            for _ in self.generate(prompt, n, ignore_eog=True):
                pass
        if self.slots > 1:  # copying a cached prefix to another slot has a graph too
            self._copy(self._source.bind(0), self._slot.bind(1))
        self.reset()

    def cached_prefix(self, prompt: list[int]) -> int:
        """How many leading tokens of `prompt` the cache holds: generate() prefills the rest."""
        return max(_shared(prompt, cached) for cached in self._cached)

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
            self._copy(self._source.bind(source), self._slot.bind(slot))
        self._cached[slot] = prompt[:prefix]  # what stays valid if prefill is interrupted
        self._used[slot] = next(self._clock)
        return slot

    def _prefill(
        self, chunk: list[int], pos: int, slot: UOp, temperature: Tensor, seed: Tensor
    ) -> Tensor:
        # runs prompt tokens from position pos, and samples the next: a single one as a decode
        # step, up to FEW in the graph bound to that many, more in the one bound to prefill_chunk
        start, sampling, n = self._pos.bind(pos), (temperature, seed), len(chunk)
        if n == 1:
            return self._decode(_ids(chunk, 1), slot, start, *sampling)
        graph, length = (self._few_chunk, self._few) if n <= FEW else (self._chunk, self._len)
        tokens = _ids(chunk, int(length.vmax)).shrink(((0, 1), (0, length.bind(n))))
        return graph(tokens, slot, start, *sampling)

    def _step(
        self, tokens: Tensor, slot: UOp, start_pos: UOp, temperature: Tensor, seed: Tensor
    ) -> Tensor:
        hidden = self.model(tokens, start_pos, slot)
        logits = self.model.logits(hidden[:, -1, :])
        return sample(logits, temperature, seed, start_pos + tokens.shape[1]).realize()


def _ids(tokens: list[int], size: int) -> Tensor:
    # (1, size) token ids, padded: from bytes, as tinygrad converts a list value by value
    padded = array.array("i", tokens + [0] * (size - len(tokens))).tobytes()
    return Tensor(padded, dtype=dtypes.int32).reshape(1, size)


def _shared(prompt: list[int], cached: list[int]) -> int:
    # how many leading tokens a slot holds of the prompt, all but its last at most: generation
    # starts from the logits of running that one
    pairs = zip(prompt[:-1], cached, strict=False)
    return sum(1 for _ in itertools.takewhile(lambda p: p[0] == p[1], pairs))
