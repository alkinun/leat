"""Generation: chunked prefill and batched decode, replayed from compiled graphs."""

import array
import functools
import itertools
import random
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.draft import Drafter
from leat.draft import load as load_drafter
from leat.gguf import GGUF
from leat.kernels import MATVEC_TOKENS, VARIABLES
from leat.model import Config, Transformer
from leat.ops import Span
from leat.sampler import GREEDY, Sampling, sample
from leat.tokenizer import Tokenizer
from leat.vision import Image, Vision, blank

# prompt tokens up to which a chunk takes a graph bound to that many, whose kernels size their
# work for so few: the matrix kernels take tiles of 16 tokens
FEW_TOKENS = 16
# prompt tokens a step prefills at most while other sequences decode, which wait for it: for Llama
# 3.1 8B on the 3090, a chunk of 256 takes 57 ms against 111 for 512, and prefills 2.3% slower
SHARED_CHUNK = 256
# tokens before a prompt's end where a model with recurrent state keeps it: a later prompt that
# shares the prompt up to there goes on from it, as a chat's next turn, which renders the last
# turn's opening anew, or an agent's next step. The rest takes the graph for few tokens.
KEEP_BACK = FEW_TOKENS
# sequences a decode step runs at most in one graph, more in turn: as many tokens as the
# matrix-vector kernels take, which on AMD are the only kernels for several
BATCH = MATVEC_TOKENS
# tokens a drafter guesses ahead of a sequence, which the target checks at once
DRAFT_TOKENS = 3
# tokens a speculative step runs at most, its sequences' last ones and their drafts: as many as
# the matrix-vector kernels take, as the kernels for more cost more than the guesses save, so that
# several sequences draft fewer each. For Qwen3.6 35B A3B on the 3090, 3 sequences drafting 3
# each decoded at 0.88 times the speed of plain steps, 4 at 0.77, and drafting 1 each, 1.11 and
# 1.01 times.
SPECULATIVE_TOKENS = MATVEC_TOKENS


@dataclass(eq=False)
class Sequence:
    """A generation in progress: its prompt and the images it holds, by key, how it samples, and
    the tokens it has generated."""

    prompt: list[int]
    max_tokens: int
    sampling: Sampling
    seed: int
    ignore_eog: bool
    slot: int
    images: dict[int, Image] = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)  # generated so far, the last not yet run
    done: bool = False  # generated all it will, or cancelled
    # its sampling options and seed for the graphs, as (1, 5) and (1,) tensors
    options: tuple[Tensor, Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # uploaded along with the first chunk's tokens
        values = Tensor([self.sampling.values()], dtype=dtypes.float32)
        self.options = values, Tensor([self.seed], dtype=dtypes.uint32)


@dataclass
class _Batch:
    # a decode batch's sequences, and what the next with the same ones reuses
    sequences: list[Sequence]
    tokens: Tensor  # the step's output, the sequences' next tokens: (1, n)
    options: tuple[Tensor, Tensor]  # their sampling options and seeds


class Engine:
    """A loaded model, ready to generate.

    Several sequences generate at once, one per slot of the KV cache: each step prefills a chunk of
    at most one prompt, of SHARED_CHUNK tokens at most while others decode, and then decodes a token
    of every other sequence in one batch, which reads each weight once for all of them. On the
    reference ops a sequence generates as it would alone; the kernels for several tokens round
    differently from those for one, so a batched sequence may take another token where two are
    close.

    With a drafter, sequences decoding without presence_penalty decode speculatively, as many at
    once as the drafter's `sequences`: it guesses each one's next tokens, DRAFT_TOKENS at most and
    fewer for more sequences, SPECULATIVE_TOKENS in all with their last ones, and the target runs
    them in one step, sampling after each as it would have, and keeps each sequence's up to the
    first it would not have generated, and the token it generated there. As a draw depends only
    on the seed and the position, each sequence generates the tokens it would without, faster
    where the guesses hold.

    A prompt is prefilled only past the longest prefix any slot shares with it. A prompt that
    extends a slot's tokens continues in that slot, so multi-turn chat costs only the latest turn.
    Any other prompt takes a free slot, empty or else the least recently used, and first copies
    that prefix in: a system prompt that several conversations share, say. A model with recurrent
    state, as Qwen3.5's Gated DeltaNet, holds it for all the tokens a slot ran, and so shares a
    slot's tokens only when it shares all of them, or else all those before the state it kept
    KEEP_BACK tokens before its last prompt's end.

    With a vision encoder, prompts hold images: each at as many positions as it has embeddings,
    which run in a chunk of their own, the image's whole, and see each other, as Gemma's do. The
    image is encoded when its chunk runs, and not again where the cache holds it.
    """

    def __init__(
        self, path: str | Path, max_context: int = 4096, prefill_chunk: int = 512, slots: int = 1,
        draft: str | Path | None = None, vision: str | Path | None = None,
    ):  # fmt: skip
        if slots < 1:
            raise ValueError(f"slots must be at least 1, got {slots}")
        # decode steps of powers of 2 sequences, and of `slots` or BATCH: others are padded with
        # rows in a spare slot, at its first position, which write where no sequence reads and
        # attend over a single key
        most = min(slots, BATCH)
        self._batches = sorted({1 << i for i in range(most.bit_length()) if 1 << i < most} | {most})
        # those that take padding, more than one past the batch before
        pairs = zip(self._batches[1:], self._batches[:-1], strict=True)
        self._padded = {n for n, before in pairs if n > before + 1}
        self._spare = slots  # the padding's slot, past the others, if any batch takes padding
        self.gguf = gguf = GGUF.open(path)
        self.tokenizer = Tokenizer(gguf.metadata)
        self.config = Config.from_gguf(gguf.metadata)
        cache_slots = slots + bool(self._padded)
        weights = gguf.load(names=filter(self.config.uses, gguf.tensors))
        # with a drafter, a recurrent model keeps its states after each token a speculative step
        # runs, to go back to the last each sequence keeps
        saved = SPECULATIVE_TOKENS if draft is not None else 0
        self.model = Transformer(self.config, weights, max_context, cache_slots, saved)
        self.drafter: Drafter | None = None if draft is None else load_drafter(draft, self.model)
        self.vision: Vision | None = None
        if vision is not None:
            self.vision = Vision(GGUF.open(vision), self.tokenizer, self.config.dim)
            self._image_len = UOp.variable("image_len", 1, self.vision.tokens)
            self._encode, self._image_chunk = graph(self.vision.encode), graph(self._step)
        # speculative steps of 1 sequence or more, as many as the drafter takes, each drafting a
        # token at least
        drafting = 0 if self.drafter is None else min(slots, self.drafter.sequences)
        self._speculated = [n for n in range(1, drafting + 1) if _drafts(n)]
        # the tokens each slot's sequence has generated, for presence_penalty
        vocab = int(self.model.output.shape[0])
        self._seen = Tensor.zeros(cache_slots, vocab, dtype=dtypes.bool).contiguous().realize()
        self.max_context, self.prefill_chunk, self.slots = max_context, prefill_chunk, slots
        self._len = UOp.variable("chunk_len", 1, prefill_chunk)
        self._few = UOp.variable("few_len", 1, min(FEW_TOKENS, prefill_chunk))
        # the slot and position of each row of a decode step, the first a chunk's too
        rows = range(most)
        self._slot_vars = [UOp.variable(f"slot{i}", 0, cache_slots - 1) for i in rows]
        self._pos_vars = [UOp.variable(f"pos{i}", 0, max_context - 1) for i in rows]
        self._source = UOp.variable("source", 0, slots - 1)
        # the sequences of a padded decode step, whose padding a mixture of experts skips: a
        # padding row would read experts of its own
        self._live = UOp.variable("live", 1, most) if self.config.experts else None
        self._chunk, self._few_chunk = graph(self._step), graph(self._step)
        self._decode = {n: graph(functools.partial(self._step, decode=True)) for n in self._batches}
        self._copy = graph(self._copy_slot)
        if self.drafter is not None:
            # each slot's normed hidden state that gave its last token, from which drafts go on;
            # the rows of a speculative step's tokens, of which one of each sequence's becomes it
            dim = self.config.dim
            self._hidden = Tensor.zeros(cache_slots, 1, dim).contiguous().realize()
            self._rows = Tensor.zeros(saved, 1, dim).contiguous().realize()
            self._speculate = {n: graph(self._speculative_step) for n in self._speculated}
            self._settle = {n: graph(self._settled) for n in self._speculated}
            # each sequence's tokens a step keeps, but one; its position, past a prompt's first
            # token, which the drafter's attention needs to know
            each = range(max(self._speculated))
            self._kept_vars = [UOp.variable(f"kept{i}", 0, DRAFT_TOKENS) for i in each]
            self._draft_vars = [UOp.variable(f"draft_pos{i}", 1, max_context - 1) for i in each]
        self._recurrent = any(self.config.recurrent)
        self._keep, self._restore = graph(self.model.keep), graph(self.model.restore)
        self._kept: list[list[int]] = [[] for _ in range(slots)]  # tokens before each kept state
        self._last: dict[int, _Batch] = {}  # the last decode step's batches, by graph
        self._cached: list[list[int]] = [[] for _ in range(slots)]  # tokens each slot holds
        self._used = [0] * slots  # when each slot last started a generation
        self._clock = itertools.count(1)
        self.active: list[Sequence] = []  # in the order they started

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        sampling: Sampling = GREEDY,
        seed: int | None = None,
        ignore_eog: bool = False,
        images: Iterable[Image] = (),
    ) -> Generator[int, None, None]:
        """Yields up to `max_tokens` ids; stops early at end of generation or the context limit.

        `sampling` is greedy by default; a `seed` makes a draw repeatable. The sequence
        generates alone: no other may be active, and none may start until this one is exhausted
        or closed. start() and step() generate several at once.
        """
        if self.active:
            raise RuntimeError("another generation is unfinished: exhaust or close it first")
        sequence = self.start(prompt, max_tokens, sampling, seed, ignore_eog, images)
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
        sampling: Sampling = GREEDY,
        seed: int | None = None,
        ignore_eog: bool = False,
        images: Iterable[Image] = (),
    ) -> Sequence:
        """Starts a generation in a free slot, as generate() would; step() advances it. The
        prompt shows each of `images` with its tokens."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
        if not 0 < len(prompt) < self.max_context:
            raise ValueError(
                f"prompt must have 1 to {self.max_context - 1} tokens, got {len(prompt)}"
            )
        shown = {image.key: image for image in images}
        for start, end in _images(prompt):
            if (image := shown.get(prompt[start])) is None or end - start != image.size:
                raise ValueError(f"the prompt's image at {start} is none of those given")
        if len(self.active) == self.slots:
            raise RuntimeError(f"all {self.slots} slots are generating")
        seed = random.getrandbits(32) if seed is None else seed % 2**32
        slot = self._claim(prompt)
        sequence = Sequence(prompt, max_tokens, sampling, seed, ignore_eog, slot, shown)
        self.active.append(sequence)
        return sequence

    def step(self) -> list[tuple[Sequence, int]]:
        """Advances the active sequences: a chunk of the first one's prompt still to prefill, then
        a token of each one past its prompt. Returns the tokens generated, a sequence's first at
        the end of its prompt; sequences that are done leave `active`."""
        decoding = [s for s in self.active if s.tokens]
        speculating = bool(decoding) and self._speculates(decoding)
        out = []
        prefilling = next((s for s in self.active if not s.tokens), None)
        size = min(self.prefill_chunk, SHARED_CHUNK) if decoding else self.prefill_chunk
        if prefilling and (token := self._prefill(prefilling, size)) is not None:
            out.append((prefilling, token))
        if decoding and not speculating:
            tokens = self._decode_step(decoding)
            out += zip(decoding, tokens, strict=True)
        for sequence, token in out:
            sequence.tokens.append(token)
            self._check(sequence)
        if speculating:
            out += self._speculative(decoding)
        return out

    def cancel(self, sequence: Sequence) -> None:
        """Ends a sequence early, freeing its slot; its cache keeps what it ran, for prefixes."""
        sequence.done = True
        if sequence in self.active:
            self.active.remove(sequence)

    def warm_up(self) -> None:
        """Compiles the graphs generation replays, which takes seconds in a fresh process, so that
        the first prompt runs at full speed. Leaves no prefix cached."""
        # a prefill of more than FEW_TOKENS tokens and a decode step, then a prefill of few; the
        # first longer by the KEEP_BACK tokens after the state a recurrent model keeps, and with a
        # drafter by a speculative step
        longer = FEW_TOKENS + 1 + (KEEP_BACK if self._recurrent else 0)
        first = 2 + (self.drafter is not None)
        for prompt, n in (([0] * min(longer, self.max_context - 1), first), ([0, 0], 1)):
            self.reset()
            for _ in self.generate(prompt, n, ignore_eog=True):
                pass
        # decode steps of several, each at a slot's first position, and of one where speculative
        # steps take its place
        for n in self._batches[0 if self.drafter else 1 :]:
            options = Tensor([[0.0] * 5] * n), Tensor([0] * n, dtype=dtypes.uint32)
            rows = [
                x for i in range(n) for x in (self._slot_vars[i].bind(i), self._pos_vars[i].bind(0))
            ]
            live = self._live.bind(n) if self._live is not None and n in self._padded else None
            self._decode[n](_ids([0] * n, n), *options, *rows, live=live)
        # speculative steps of several, each from a slot's second position
        for n in self._speculated[1:]:
            sampling = [
                t for _ in range(n) for t in (Tensor([[0.0] * 5]), Tensor([0], dtype=dtypes.uint32))
            ]
            slots = [self._slot_vars[i].bind(i) for i in range(n)]
            starts = [self._draft_vars[i].bind(1) for i in range(n)]
            self._speculate[n](_ids([0] * n, n), *sampling, *slots, *starts)
            self._settle[n](*(x for i in range(n) for x in (self._kept_vars[i].bind(0), slots[i])))
        self._last = {}
        if self.slots > 1:  # copying a cached prefix to another slot has a graph too
            self._copy(self._source.bind(0), self._slot_vars[0].bind(1))
        if self._recurrent:  # and keeping recurrent state, and going back to it
            self._keep(self._slot_vars[0].bind(0))
            self._restore(self._slot_vars[0].bind(0))
        if self.vision is not None:  # and encoding an image, and its chunk
            image = self.image(blank())
            if len(image.tokens) < self.max_context:
                self.reset()
                for _ in self.generate(image.tokens, 1, ignore_eog=True, images=[image]):
                    pass
        self.reset()

    def image(self, data: bytes) -> Image:
        """An image of its file's bytes, for a prompt: its tokens show it there, and start()
        takes it."""
        if self.vision is None:
            raise ValueError("the model has no vision encoder")
        return self.vision.image(data)

    def cached_prefix(self, prompt: list[int]) -> int:
        """How many leading tokens of `prompt` the cache holds: generation prefills the rest. Of
        a model with recurrent state, those of a slot's or of a free slot's kept state."""
        busy = {s.slot for s in self.active}
        kept = [len(k) for s, k in enumerate(self._kept) if s not in busy and _resumes(prompt, k)]
        return max(self._shared(prompt) + kept)

    def reset(self) -> None:
        """Forgets every cached prefix, so the next prompt is prefilled from scratch, and ends
        every active sequence."""
        for sequence in list(self.active):
            self.cancel(sequence)
        self._cached = [[] for _ in range(self.slots)]
        self._kept = [[] for _ in range(self.slots)]

    def _claim(self, prompt: list[int]) -> int:
        # a free slot for the prompt, made to hold the longest prefix of it that any slot holds
        busy = {s.slot for s in self.active}
        free = [s for s in range(self.slots) if s not in busy]
        if self._recurrent:
            self._resume(prompt, free)
        shared = self._shared(prompt)
        extended = [s for s in free if shared[s] == len(self._cached[s])]  # empty slots too
        if extended:
            slot = max(extended, key=lambda s: shared[s])
        else:
            slot = min(free, key=lambda s: self._used[s])
        source = max(range(self.slots), key=lambda s: shared[s])
        if (prefix := shared[source]) > shared[slot]:
            self._cached[slot], self._kept[slot] = [], []  # while the copy overwrites it
            self._copy(self._source.bind(source), self._slot_vars[0].bind(slot))
        self._cached[slot] = prompt[:prefix]
        if prefix < len(self._kept[slot]):  # the slot no longer holds the tokens before it
            self._kept[slot] = []
        self._used[slot] = next(self._clock)
        return slot

    def _resume(self, prompt: list[int], free: list[int]) -> None:
        # takes a free slot back to its kept recurrent state, where the prompt shares all the
        # tokens before it, and more than all of any slot's
        best = max(self._shared(prompt))
        kept = [s for s in free if best < len(self._kept[s]) and _resumes(prompt, self._kept[s])]
        if kept:
            slot = max(kept, key=lambda s: len(self._kept[s]))
            self._restore(self._slot_vars[0].bind(slot))
            self._cached[slot] = list(self._kept[slot])

    def _shared(self, prompt: list[int]) -> list[int]:
        # how many leading tokens of the prompt each slot holds that generation may start from: of
        # a slot with recurrent state, all its tokens or none
        shared = [_shared(prompt, cached) for cached in self._cached]
        if self._recurrent:
            shared = [n if n == len(c) else 0 for n, c in zip(shared, self._cached, strict=True)]
        return shared

    def _prefill(self, sequence: Sequence, size: int) -> int | None:
        # runs the next chunk of the sequence's prompt: an image whole, or else up to `size`
        # tokens before the next, up to FEW_TOKENS in the graph bound to that many, more in the
        # one bound to prefill_chunk. Returns the token sampled after the prompt's last chunk.
        cached, mark = self._cached[sequence.slot], len(sequence.prompt) - KEEP_BACK
        pos = len(cached)
        row = self._slot_vars[0].bind(sequence.slot), self._pos_vars[0].bind(pos)
        if (key := sequence.prompt[pos]) < 0:
            image = sequence.images[key]
            chunk = sequence.prompt[pos : pos + image.size]
            token = self._prefill_image(image, sequence.options, row)
        else:
            if self._recurrent and pos < mark:  # a chunk ends where the state is kept
                size = min(size, mark - pos)
            chunk = sequence.prompt[pos : pos + size]
            chunk = chunk[: next((i for i, t in enumerate(chunk) if t < 0), len(chunk))]
            few = (n := len(chunk)) <= FEW_TOKENS
            graph, length = (self._few_chunk, self._few) if few else (self._chunk, self._len)
            tokens = _ids(chunk, int(length.vmax)).shrink(((0, 1), (0, length.bind(n))))
            token = graph(tokens, *sequence.options, *row)
        cached += chunk
        if self._recurrent and len(cached) == mark:
            self._keep(self._slot_vars[0].bind(sequence.slot))
            self._kept[sequence.slot] = list(cached)
        return int(token.item()) if len(cached) == len(sequence.prompt) else None

    def _prefill_image(
        self, image: Image, options: tuple[Tensor, Tensor], row: tuple[UOp, UOp]
    ) -> Tensor:
        # runs an image's embeddings in the image chunk's graph, its positions' tokens the one
        # that fills an image, which a drafter takes in
        assert self.vision is not None
        n, dim = self._image_len.bind(image.size), self.config.dim
        embeddings = self._encode(image.pixels, image.positions).reshape(1, -1, dim)
        tokens = _ids([self.vision.fill] * image.size, self.vision.tokens)
        x, tokens = embeddings.shrink(((0, 1), (0, n), (0, dim))), tokens.shrink(((0, 1), (0, n)))
        return self._image_chunk(tokens, *options, *row, image=x)

    def _decode_step(self, sequences: list[Sequence]) -> list[int]:
        # runs each sequence's last token, in batches of BATCH at most. A batch of the same
        # sequences as the last step's batch in its graph takes that batch's output as its
        # tokens, and its sampling options and seeds, uploading nothing, unless another batch of
        # this step takes the graph too, overwriting the output
        batches = [sequences[i : i + BATCH] for i in range(0, len(sequences), BATCH)]
        sizes = [next(n for n in self._batches if n >= len(b)) for b in batches]
        last, self._last, out = self._last, {}, []
        for batch, n in zip(batches, sizes, strict=True):
            reuse = last.get(n) if sizes.count(n) == 1 else None
            out += self._decode_batch(
                batch, n, reuse if reuse and reuse.sequences == batch else None
            )
        return out

    def _decode_batch(self, sequences: list[Sequence], n: int, last: _Batch | None) -> list[int]:
        # runs each sequence's last token in the graph of n, padded with rows in the spare slot
        k = len(sequences)
        if last is not None:
            tokens, options = last.tokens, last.options
        else:
            pad = [0] * (n - k)
            tokens = _ids([s.tokens[-1] for s in sequences] + pad, n)
            values = [s.sampling.values() for s in sequences] + [[0.0] * 5] * (n - k)
            seed = Tensor([s.seed for s in sequences] + pad, dtype=dtypes.uint32)
            options = Tensor(values, dtype=dtypes.float32), seed
        rows = [(s.slot, len(self._cached[s.slot])) for s in sequences]
        bound = []
        for i, (slot, pos) in enumerate(rows + [(self._spare, 0)] * (n - k)):
            bound += [self._slot_vars[i].bind(slot), self._pos_vars[i].bind(pos)]
        live = self._live.bind(k) if self._live is not None and n in self._padded else None
        out = self._decode[n](tokens, *options, *bound, live=live)
        self._last[n] = _Batch(list(sequences), out.reshape(1, n), options)
        for s in sequences:
            self._cached[s.slot].append(s.tokens[-1])
        return out.numpy().ravel()[:k].tolist()

    def _speculates(self, sequences: list[Sequence]) -> bool:
        # whether sequences decoding decode speculatively: with a drafter, few enough to draft a
        # token each, and each without presence_penalty, which tokens drafted would change, with
        # room for its drafts in the cache, and with more than one token to go
        drafts = _drafts(len(sequences))

        def each(sequence: Sequence) -> bool:
            room = len(self._cached[sequence.slot]) + drafts < self.max_context
            more = sequence.max_tokens - len(sequence.tokens) > 1
            return sequence.sampling.presence_penalty == 0 and room and more

        few = len(sequences) in self._speculated
        return self.drafter is not None and few and all(map(each, sequences))

    def _speculative(self, sequences: list[Sequence]) -> list[tuple[Sequence, int]]:
        # a speculative step of the sequences: of each, the drafts it keeps, those the target
        # generated too, each until the first it would not have, and the token the target
        # generated after them, each appended and checked as step() does
        n, drafts = len(sequences), _drafts(len(sequences))
        slots = [self._slot_vars[i].bind(s.slot) for i, s in enumerate(sequences)]
        positions = [
            self._draft_vars[i].bind(len(self._cached[s.slot])) for i, s in enumerate(sequences)
        ]
        options = [t for s in sequences for t in s.options]
        tokens = _ids([s.tokens[-1] for s in sequences], n)
        out = self._speculate[n](tokens, *options, *slots, *positions).numpy().tolist()
        generated: list[tuple[Sequence, int]] = []
        settled = []
        for i, (sequence, row) in enumerate(zip(sequences, out, strict=True)):
            drafted, sampled = row[:drafts], row[drafts:]
            pairs = enumerate(zip(drafted, sampled[:drafts], strict=True))
            kept = next((j for j, (d, t) in pairs if d != t), drafts)
            ran, before = [sequence.tokens[-1], *drafted[:kept]], len(sequence.tokens)
            for token, after in zip(ran, sampled[: kept + 1], strict=True):
                self._cached[sequence.slot].append(token)
                sequence.tokens.append(after)
                generated.append((sequence, after))
                self._check(sequence)
                if sequence.done:
                    break
            last = len(sequence.tokens) - before - 1  # the row of the last token generated
            settled += [self._kept_vars[i].bind(last), slots[i]]
        # the row that gave each sequence's last token is its slot's hidden state, for the next
        # drafts, and the recurrent states after it its states
        self._settle[n](*settled)
        self._last = {}  # the decode steps' outputs are no longer the sequences' last tokens
        return generated

    def _speculative_step(self, tokens: Tensor, *args: Tensor | UOp) -> Tensor:
        # for n sequences' last tokens (1, n): each one's sampling options and seed, then the
        # slots, then the positions of the tokens. The drafter's tokens after each, `drafts` of
        # them, then the target's run of each token and its drafts, keeping its hidden rows, and
        # its draws after each: each sequence's drafts and draws, (n, 2 * drafts + 1)
        assert self.drafter is not None
        n = int(tokens.shape[1])
        drafts = _drafts(n)
        ran = drafts + 1
        options = cast(list[Tensor], list(args[: 2 * n]))
        slots = cast(list[int | UOp], list(args[2 * n : 3 * n]))
        starts = cast(list[int | UOp], list(args[3 * n :]))
        last = Tensor.cat(*(self._hidden[slot : slot + 1] for slot in slots)).reshape(1, n, -1)
        drafted = self.drafter.draft(tokens, last, slots, starts, drafts)
        run = tokens.reshape(n, 1).cat(drafted, dim=1).reshape(1, n * ran)
        spans = [Span(slot, start, ran) for slot, start in zip(slots, starts, strict=True)]
        hidden = self.model.run(run, spans, save=True)
        rows = self._rows[: n * ran]
        rows.assign(hidden.reshape(rows.shape)).realize()
        each = rows.reshape(n, ran, -1)
        before = last.reshape(n, 1, -1).cat(each[:, :drafts], dim=1)
        self.drafter.follow(run, before.reshape(1, n * ran, -1), spans)
        logits = self.model.logits(rows.reshape(n * ran, -1))
        values = Tensor.cat(*(t.expand(ran, int(t.shape[1])) for t in options[0::2]))
        seeds = Tensor.cat(*(t.expand(ran) for t in options[1::2]))
        positions = Tensor.stack(*(Tensor(start + 1 + i) for start in starts for i in range(ran)))
        drawn = sample(logits, values, seeds, positions).cast(dtypes.int32).reshape(n, ran)
        return drafted.cast(dtypes.int32).cat(drawn, dim=1).realize()

    def _settled(self, *rows: UOp) -> None:
        # for each sequence of a speculative step, its row's last token kept and its slot: the
        # slot's hidden state and recurrent states those after it
        pairs = list(zip(rows[::2], rows[1::2], strict=True))
        ran = _drafts(len(pairs)) + 1
        for i, (kept, slot) in enumerate(pairs):
            row = i * ran + kept
            self._hidden[slot : slot + 1].assign(self._rows[row : row + 1]).realize()
            self.model.rewind(slot, row)

    def _copy_slot(self, source: UOp, slot: UOp) -> None:
        # slot `source`'s cache and states, and what the drafter holds of it, to slot `slot`
        self.model.copy(source, slot)
        if self.drafter is not None:
            self.drafter.copy(source, slot)

    def _check(self, sequence: Sequence) -> None:
        # ends a sequence at end of generation, at max_tokens, or with the cache full
        token, held = sequence.tokens[-1], len(self._cached[sequence.slot])
        eog = token in self.tokenizer.eog_ids and not sequence.ignore_eog
        if eog or len(sequence.tokens) == sequence.max_tokens or held >= self.max_context:
            self.cancel(sequence)

    def _step(
        self, tokens: Tensor, options: Tensor, seed: Tensor, *rows: UOp, live: UOp | None = None,
        decode: bool = False, image: Tensor | None = None,
    ) -> Tensor:  # fmt: skip
        # rows: the slot and start position of each span, in turn. A single span takes every
        # token, as a chunk of prompt or a decode step of one, or an image's embeddings in place
        # of its tokens; several take one each, as a decode step does, the first `live` of them
        # its sequences' if given, the rest padding.
        pairs = list(zip(rows[::2], rows[1::2], strict=True))
        slots = [slot for slot, _ in pairs]
        seen = self._generated(tokens, slots, decode)
        if len(pairs) == 1:
            (slot, start), length = pairs[0], tokens.shape[1]
            spans = [Span(slot, start, length, causal=image is None)]
            if image is None:
                hidden = self.model.run(tokens, spans)
            else:  # the embeddings as they are, not scaled as the tokens' are
                hidden = self.model.forward(image, spans)
            hidden = self._followed(tokens, hidden, spans)
            logits = self.model.logits(hidden[:, -1, :])
            return sample(logits, options, seed, start + length, seen).realize()
        spans = [Span(slot, start) for slot, start in pairs]
        hidden = self._followed(tokens, self.model.run(tokens, spans, live), spans)
        logits = self.model.logits(hidden).reshape(len(pairs), -1)
        positions = Tensor.stack(*(Tensor(start + 1) for _, start in pairs))
        return sample(logits, options, seed, positions, seen).realize()

    def _followed(self, tokens: Tensor, hidden: Tensor, spans: list[Span]) -> Tensor:
        # hidden (1, T, dim), the target's of a run of the spans' tokens; with a drafter, which
        # takes in the run, given each token's hidden state before it, the last of each span as
        # its slot's, which gave its last token
        if self.drafter is None:
            return hidden
        hidden = hidden.contiguous().realize()
        if len(spans) == 1:  # the slot's before the first, none at a sequence's start
            span = spans[0]
            first = Tensor(span.start > 0).where(self._hidden[span.slot : span.slot + 1], 0.0)
            before, lasts = first.cat(hidden, dim=1)[:, : hidden.shape[1]], [hidden[:, -1:]]
        else:
            before = Tensor.cat(*(self._hidden[s.slot : s.slot + 1] for s in spans), dim=1)
            lasts = [hidden[:, i : i + 1] for i in range(len(spans))]
        self.drafter.follow(tokens, before, spans)
        Tensor.realize(*(self._hidden[s.slot : s.slot + 1].assign(last)
                         for s, last in zip(spans, lasts, strict=True)))  # fmt: skip
        return hidden

    def _generated(self, tokens: Tensor, slots: list[UOp], decode: bool) -> Tensor:
        # the tokens each row's sequence has generated, for presence_penalty: a decode step runs
        # the last, which it adds to its slot's, a chunk of prompt none, and empties its slot's.
        # Adding a token twice changes nothing, so the step needs not wait for these writes.
        if not decode:
            self._seen[slots[0] : slots[0] + 1].assign(self._seen[:1].zeros_like()).realize()
            return self._seen[:1].zeros_like()
        hot = Tensor.arange(self._seen.shape[1]).reshape(1, -1) == tokens.reshape(-1, 1)
        seen = Tensor.cat(*(self._seen[slot : slot + 1] for slot in slots)) | hot
        Tensor.realize(
            *(self._seen[s : s + 1].assign(seen[i : i + 1]) for i, s in enumerate(slots))
        )
        return seen


class graph[T]:
    """A TinyJit of fxn that captures its graph on the first call: TinyJit runs a function once
    as is, then captures it on the second, which a fresh process makes the slow one. Each call
    also passes the variables the kernels bind inside, which fxn does not see: a graph replays
    with the values its arguments carry alone."""

    def __init__(self, fxn: Callable[..., T]):
        def inner(*args: Any, **kwargs: Any) -> T:
            return fxn(*args, **{k: v for k, v in kwargs.items() if k not in VARIABLES})

        self.jit = TinyJit(inner)
        self.jit.cnt = 1

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        return self.jit(*args, **kwargs, **VARIABLES)

    @property
    def captured(self) -> Any:
        return self.jit.captured

    @property
    def cnt(self) -> int:  # the calls so far, the first counting as two
        return self.jit.cnt


def _drafts(sequences: int) -> int:
    # the tokens each of a speculative step's sequences drafts: 0 for too many to draft any
    return max(min(DRAFT_TOKENS, SPECULATIVE_TOKENS // sequences - 1), 0)


def _ids(tokens: list[int], size: int) -> Tensor:
    # (1, size) token ids, padded: from bytes, as tinygrad converts a list value by value
    padded = array.array("i", tokens + [0] * (size - len(tokens))).tobytes()
    return Tensor(padded, dtype=dtypes.int32).reshape(1, size)


def _resumes(prompt: list[int], kept: list[int]) -> bool:
    # whether a prompt may go on from a state kept after `kept`: it shares all of them
    return len(kept) == _shared(prompt, kept)


def _shared(prompt: list[int], cached: list[int]) -> int:
    # how many leading tokens a slot holds of the prompt, all but its last at most, as generation
    # starts from the logits of running that one, and none of an image, which runs whole
    pairs = zip(prompt[:-1], cached, strict=False)
    n = sum(1 for _ in itertools.takewhile(lambda p: p[0] == p[1], pairs))
    while 0 < n < len(prompt) and prompt[n] < 0 and prompt[n - 1] == prompt[n]:
        n -= 1
    return n


def _images(prompt: list[int]) -> Generator[tuple[int, int], None, None]:
    # where each image of a prompt starts and ends: runs of a negative key
    start = 0
    for i, token in enumerate(prompt):
        if token < 0 and (i == 0 or prompt[i - 1] != token):
            start = i
        if token < 0 and (i + 1 == len(prompt) or prompt[i + 1] != token):
            yield start, i + 1
