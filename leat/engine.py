"""Generation: chunked prefill and batched decode, replayed from compiled graphs."""

import array
import itertools
import random
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.gguf import GGUF
from leat.model import Config, Span, Transformer
from leat.sampler import sample
from leat.tokenizer import Tokenizer

# prompt tokens up to which a chunk takes a graph bound to that many, whose kernels size their
# work for so few: the matrix kernels take tiles of 16 tokens
FEW_TOKENS = 16


@dataclass(eq=False)
class Sequence:
    """A generation in progress: its prompt, how it samples, and the tokens it has generated."""

    prompt: list[int]
    max_tokens: int
    temperature: float
    seed: int
    ignore_eog: bool
    slot: int
    tokens: list[int] = field(default_factory=list)  # generated so far, the last not yet run
    done: bool = False  # generated all it will, or cancelled
    # its temperature and seed for the graphs, as (1,) tensors
    sampling: tuple[Tensor, Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # uploaded along with the first chunk's tokens
        temperature = Tensor([self.temperature], dtype=dtypes.float32)
        self.sampling = temperature, Tensor([self.seed], dtype=dtypes.uint32)


@dataclass
class _Batch:
    # a decode step's sequences, and what the next step reuses if it runs the same ones
    sequences: list[Sequence]
    tokens: Tensor  # the step's output, the sequences' next tokens: (1, n)
    sampling: tuple[Tensor, Tensor]  # their temperatures and seeds


class Engine:
    """A loaded model, ready to generate.

    Several sequences generate at once, one per slot of the KV cache: each step prefills a chunk
    of at most one prompt and then decodes a token of every other sequence in one batch, which
    reads each weight once for all of them. On the reference ops a sequence generates as it would
    alone; the kernels for several tokens round differently from those for one, so a batched
    sequence may take another token where two are close.

    A prompt is prefilled only past the longest prefix any slot shares with it. A prompt that
    extends a slot's tokens continues in that slot, so multi-turn chat costs only the latest turn.
    Any other prompt takes a free slot, empty or else the least recently used, and first copies
    that prefix in: a system prompt that several conversations share, say.
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
        self._len = UOp.variable("chunk_len", 1, prefill_chunk)
        self._few = UOp.variable("few_len", 1, min(FEW_TOKENS, prefill_chunk))
        # the slot and position of each sequence of a decode step, the first a chunk's too
        self._slot_vars = [UOp.variable(f"slot{i}", 0, slots - 1) for i in range(slots)]
        self._pos_vars = [UOp.variable(f"pos{i}", 0, max_context - 1) for i in range(slots)]
        self._source = UOp.variable("source", 0, slots - 1)
        # a graph for each number of sequences a decode step may run, rather than padding fewer:
        # a sequence's attention reads all its keys and values, a repeated one's too
        self._chunk, self._few_chunk = _graph(self._step), _graph(self._step)
        self._decode = {n: _graph(self._step) for n in range(1, slots + 1)}
        self._copy = _graph(self.model.copy)
        self._batch: _Batch | None = None  # the last decode step's
        self._cached: list[list[int]] = [[] for _ in range(slots)]  # tokens each slot holds
        self._used = [0] * slots  # when each slot last started a generation
        self._clock = itertools.count(1)
        self.active: list[Sequence] = []  # in the order they started

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        ignore_eog: bool = False,
    ) -> Generator[int, None, None]:
        """Yields up to `max_tokens` ids; stops early at end of generation or the context limit.

        Sampling is greedy at temperature 0; above, a `seed` makes it repeatable. The sequence
        generates alone: no other may be active, and none may start until this one is exhausted
        or closed. start() and step() generate several at once.
        """
        if self.active:
            raise RuntimeError("another generation is unfinished: exhaust or close it first")
        sequence = self.start(prompt, max_tokens, temperature, seed, ignore_eog)
        try:
            while not sequence.done:
                for _, token in self.step():
                    yield token
        finally:
            self.cancel(sequence)

    def start(
        self,
        prompt: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        ignore_eog: bool = False,
    ) -> Sequence:
        """Starts a generation in a free slot, as generate() would; step() advances it."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
        if not 0 < len(prompt) < self.max_context:
            raise ValueError(
                f"prompt must have 1 to {self.max_context - 1} tokens, got {len(prompt)}"
            )
        if len(self.active) == self.slots:
            raise RuntimeError(f"all {self.slots} slots are generating")
        seed = random.getrandbits(32) if seed is None else seed % 2**32
        sequence = Sequence(prompt, max_tokens, temperature, seed, ignore_eog, self._claim(prompt))
        self.active.append(sequence)
        return sequence

    def step(self) -> list[tuple[Sequence, int]]:
        """Advances the active sequences: a chunk of the first one's prompt still to prefill, then
        a token of each one past its prompt. Returns the tokens generated, a sequence's first at
        the end of its prompt; sequences that are done leave `active`."""
        decoding = [s for s in self.active if s.tokens]
        out = []
        prefilling = next((s for s in self.active if not s.tokens), None)
        if prefilling and (token := self._prefill(prefilling)) is not None:
            out.append((prefilling, token))
        if decoding:
            tokens = self._decode_step(decoding)
            out += zip(decoding, tokens, strict=True)
        for sequence, token in out:
            sequence.tokens.append(token)
            self._check(sequence)
        return out

    def cancel(self, sequence: Sequence) -> None:
        """Ends a sequence early, freeing its slot; its cache keeps what it ran, for prefixes."""
        sequence.done = True
        if sequence in self.active:
            self.active.remove(sequence)

    def warm_up(self) -> None:
        """Compiles the graphs generation replays, which takes seconds in a fresh process, so that
        the first prompt runs at full speed. Leaves no prefix cached."""
        # a prefill of more than FEW_TOKENS tokens and a decode step, then a prefill of few
        for prompt, n in (([0] * min(FEW_TOKENS + 1, self.max_context - 1), 2), ([0, 0], 1)):
            self.reset()
            for _ in self.generate(prompt, n, ignore_eog=True):
                pass
        for n in range(2, self.slots + 1):  # decode steps of several, each in a slot of its own
            sampling = Tensor([0.0] * n), Tensor([0] * n, dtype=dtypes.uint32)
            rows = [v.bind(i) for i in range(n) for v in (self._slot_vars[i], self._pos_vars[i])]
            self._decode[n](_ids([0] * n, n), *sampling, *rows)
        self._batch = None
        if self.slots > 1:  # copying a cached prefix to another slot has a graph too
            self._copy(self._source.bind(0), self._slot_vars[0].bind(1))
        self.reset()

    def cached_prefix(self, prompt: list[int]) -> int:
        """How many leading tokens of `prompt` the cache holds: generation prefills the rest."""
        return max(_shared(prompt, cached) for cached in self._cached)

    def reset(self) -> None:
        """Forgets every cached prefix, so the next prompt is prefilled from scratch, and ends
        every active sequence."""
        for sequence in list(self.active):
            self.cancel(sequence)
        self._cached = [[] for _ in range(self.slots)]

    def _claim(self, prompt: list[int]) -> int:
        # a free slot for the prompt, made to hold the longest prefix of it that any slot holds
        busy = {s.slot for s in self.active}
        free = [s for s in range(self.slots) if s not in busy]
        shared = [_shared(prompt, cached) for cached in self._cached]
        extended = [s for s in free if shared[s] == len(self._cached[s])]  # empty slots too
        if extended:
            slot = max(extended, key=lambda s: shared[s])
        else:
            slot = min(free, key=lambda s: self._used[s])
        source = max(range(self.slots), key=lambda s: shared[s])
        if (prefix := shared[source]) > shared[slot]:
            self._cached[slot] = []  # while the copy overwrites it
            self._copy(self._source.bind(source), self._slot_vars[0].bind(slot))
        self._cached[slot] = prompt[:prefix]
        self._used[slot] = next(self._clock)
        return slot

    def _prefill(self, sequence: Sequence) -> int | None:
        # runs the next chunk of the sequence's prompt: a single token as a decode step, up to
        # FEW_TOKENS in the graph bound to that many, more in the one bound to prefill_chunk.
        # Returns the token sampled after the prompt's last chunk.
        cached = self._cached[sequence.slot]
        pos, chunk = len(cached), sequence.prompt[len(cached) : len(cached) + self.prefill_chunk]
        row = self._slot_vars[0].bind(sequence.slot), self._pos_vars[0].bind(pos)
        if (n := len(chunk)) == 1:
            graph, tokens = self._decode[1], _ids(chunk, 1)
            self._batch = None  # whose output this graph overwrites
        else:
            few = n <= FEW_TOKENS
            graph, length = (self._few_chunk, self._few) if few else (self._chunk, self._len)
            tokens = _ids(chunk, int(length.vmax)).shrink(((0, 1), (0, length.bind(n))))
        token = graph(tokens, *sequence.sampling, *row)
        cached += chunk
        return int(token.item()) if len(cached) == len(sequence.prompt) else None

    def _decode_step(self, sequences: list[Sequence]) -> list[int]:
        # runs each sequence's last token. The same sequences as the last step's take that step's
        # output as their tokens, and its temperatures and seeds: nothing to upload.
        n, last = len(sequences), self._batch
        if last is not None and last.sequences == sequences:
            tokens, sampling = last.tokens, last.sampling
        else:
            tokens = _ids([s.tokens[-1] for s in sequences], n)
            temperature = Tensor([s.temperature for s in sequences], dtype=dtypes.float32)
            sampling = temperature, Tensor([s.seed for s in sequences], dtype=dtypes.uint32)
        bound = []
        for i, s in enumerate(sequences):
            held = len(self._cached[s.slot])
            bound += [self._slot_vars[i].bind(s.slot), self._pos_vars[i].bind(held)]
        out = self._decode[n](tokens, *sampling, *bound)
        self._batch = _Batch(list(sequences), out.reshape(1, n), sampling)
        for s in sequences:
            self._cached[s.slot].append(s.tokens[-1])
        return out.numpy().ravel().tolist()

    def _check(self, sequence: Sequence) -> None:
        # ends a sequence at end of generation, at max_tokens, or with the cache full
        token, held = sequence.tokens[-1], len(self._cached[sequence.slot])
        eog = token in self.tokenizer.eog_ids and not sequence.ignore_eog
        if eog or len(sequence.tokens) == sequence.max_tokens or held >= self.max_context:
            self.cancel(sequence)

    def _step(self, tokens: Tensor, temperature: Tensor, seed: Tensor, *rows: UOp) -> Tensor:
        # rows: the slot and start position of each span, in turn. A single span takes every
        # token, as a chunk of prompt; several take one each, as a decode step does.
        pairs = list(zip(rows[::2], rows[1::2], strict=True))
        if len(pairs) == 1:
            (slot, start), length = pairs[0], tokens.shape[1]
            hidden = self.model.run(tokens, [Span(slot, start, length)])
            logits = self.model.logits(hidden[:, -1, :])
            return sample(logits, temperature, seed, start + length).realize()
        hidden = self.model.run(tokens, [Span(slot, start) for slot, start in pairs])
        logits = self.model.logits(hidden).reshape(len(pairs), -1)
        positions = Tensor.stack(*(Tensor(start + 1) for _, start in pairs))
        return sample(logits, temperature, seed, positions).realize()


def _graph[T](fxn: Callable[..., T]) -> Callable[..., T]:
    # TinyJit runs a function once as is, then captures it on the second call: this captures on
    # the first, a fresh process's slow call for each graph
    jit = TinyJit(fxn)
    jit.cnt = 1
    return jit


def _ids(tokens: list[int], size: int) -> Tensor:
    # (1, size) token ids, padded: from bytes, as tinygrad converts a list value by value
    padded = array.array("i", tokens + [0] * (size - len(tokens))).tobytes()
    return Tensor(padded, dtype=dtypes.int32).reshape(1, size)


def _shared(prompt: list[int], cached: list[int]) -> int:
    # how many leading tokens a slot holds of the prompt, all but its last at most: generation
    # starts from the logits of running that one
    pairs = zip(prompt[:-1], cached, strict=False)
    return sum(1 for _ in itertools.takewhile(lambda p: p[0] == p[1], pairs))
