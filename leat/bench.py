"""Measurements: speed as llama-bench reports it, and quality as perplexity or KL divergence.

Quality follows llama-perplexity: chunks start with BOS where the tokenizer adds one, and only
their second half is scored.
"""

import math
import random
import statistics
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.chat import ChatTemplate
from leat.engine import BATCH, Engine, graph
from leat.sampler import GREEDY, Sampling


@dataclass(frozen=True)
class Speed:
    prefill: float  # tokens/s over a prompt, including sampling the first token
    decode: float  # tokens/s in all generating after one-token prompts, `sequences` at once
    weight_gbs: float  # weight bytes streamed per second while decoding, once per batch
    sequences: int = 1


@dataclass(frozen=True)
class Quality:
    perplexity: float
    # against a reference: KL(reference || leat) per position, as llama-perplexity defines it,
    # and the fraction of positions where both pick the same most likely token
    kl_mean: float | None = None
    kl_p99: float | None = None
    kl_max: float | None = None
    top1: float | None = None


def speed(
    engine: Engine, prompt_tokens: int = 512, gen_tokens: int = 128, reps: int = 3,
    sequences: int = 1,
) -> Speed:  # fmt: skip
    # as llama-bench: all prompt runs, then all generation runs, each after two warm-up runs, the
    # first of which captures the graph. Generation runs `sequences` at once, timed while all
    # are past their prompts, as llama-batched-bench's.
    rng = random.Random(0)
    prompt = [rng.randrange(engine.config.vocab_size) for _ in range(max(prompt_tokens, sequences))]
    prefill, decode = [], []
    for _ in range(reps + 2):
        engine.reset()
        start = time.perf_counter()
        next(engine.generate(prompt[:prompt_tokens], 1))
        prefill.append(prompt_tokens / (time.perf_counter() - start))
    for _ in range(reps + 2):
        engine.reset()
        started = [
            engine.start(prompt[i : i + 1], gen_tokens, ignore_eog=True) for i in range(sequences)
        ]
        while not all(s.tokens for s in started):
            engine.step()
        start, made = time.perf_counter(), 0
        while len(engine.active) == sequences:
            made += len(engine.step())
        decode.append(made / (time.perf_counter() - start))
    engine.reset()
    # the bytes a token reads: the embedding only where it is also the output layer, and of a
    # mixture of experts, only the share each token uses
    c = engine.config
    tensors = {n: t for n, t in engine.gguf.tensors.items() if c.uses(n)}
    tied = "output.weight" not in tensors
    share = {n: c.experts_used / c.experts if "_exps." in n else 1.0 for n in tensors}
    streamed = sum(
        t.nbytes * share[n] for n, t in tensors.items() if n != "token_embd.weight" or tied
    )
    tg = statistics.median(decode[2:])
    reads = -(-sequences // BATCH) / sequences  # weight reads per token: one per batch a step
    return Speed(statistics.median(prefill[2:]), tg, streamed * tg * reads / 1e9, sequences)


# requests of the kinds a home assistant gets, for decode speed on replies whose tokens a drafter
# guesses as it would in use, where it guesses random prompts' continuations seldom
CHAT_PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with tests.",
    "Explain how a refrigerator works to a ten-year-old.",
    "Give me a weekly meal plan for a family of four on a budget.",
    "What are the main differences between TCP and UDP? Answer with a table.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Summarize the causes of the French Revolution in bullet points.",
    "How do I set up a Raspberry Pi as a home media server? Step by step.",
    "Translate into Spanish and French: 'Please remember to water the plants on Tuesday.'",
]


CHAT_CONTEXT = 256  # tokens a chat prompt takes at most, in its template


@dataclass(frozen=True)
class ChatSpeed:
    decode: float  # tokens/s in all, of `sequences` replies to CHAT_PROMPTS at once
    per_step: float  # tokens a reply takes per step it decodes in: more than 1 speculatively
    sequences: int = 1


def chat_speed(
    engine: Engine, gen_tokens: int = 256, reps: int = 2, sequences: int = 1,
    sampling: Sampling = GREEDY,
) -> ChatSpeed:  # fmt: skip
    """Decode speed over replies to CHAT_PROMPTS in the model's chat template, `sequences` at
    once, each to a prompt of its own, sampled as `sampling` says, timed while all are past their
    prompts: the best of `reps` runs, after one that compiles the graphs. Prompts keep their last
    tokens where the context would not hold them and the reply."""
    template = ChatTemplate(engine.gguf.metadata, engine.tokenizer)
    room = max(engine.max_context - gen_tokens, 1)
    prompts = [
        template.encode([{"role": "user", "content": text}])[-room:] for text in CHAT_PROMPTS
    ]
    best: tuple[float, float] = (0.0, 0.0)
    for rep in range(reps + 1):
        engine.reset()
        started = [
            engine.start(
                prompts[i % len(prompts)],
                gen_tokens if rep else 8,
                sampling,
                seed=i,
                ignore_eog=True,
            )
            for i in range(sequences)
        ]
        while not all(s.tokens for s in started):
            engine.step()
        start, made, steps = time.perf_counter(), 0, 0
        while engine.active:
            decoding = len(engine.active)
            made += len(engine.step())
            steps += decoding
        if rep:
            best = max(best, (made / (time.perf_counter() - start), made / steps))
    engine.reset()
    return ChatSpeed(*best, sequences)


def perplexity(
    engine: Engine, text: str, ctx: int = 512, chunks: int | None = None, decode: bool = False
) -> Quality:
    _check(engine, ctx, chunks)
    tokens = engine.tokenizer.encode(text)
    n = min(len(tokens) // ctx, len(tokens) if chunks is None else chunks)
    if n < 1:
        raise ValueError(f"the text has {len(tokens)} tokens, fewer than one chunk of {ctx}")
    nll, score = 0.0, _scorer(engine, ctx, decode)
    for i in range(n):
        chunk = tokens[i * ctx : (i + 1) * ctx]
        nll += _nll(score(chunk), chunk)
    return Quality(math.exp(nll / (n * (ctx - 1 - ctx // 2))))


def base_chunk(base: Path) -> int:
    """The chunk size of logits saved by llama-perplexity, which max_context must hold."""
    with open(base, "rb") as f:
        return _header(f, base)[0]


def kl_divergence(
    engine: Engine, base: Path, chunks: int | None = None, decode: bool = False
) -> Quality:
    """Compares against logits saved by `llama-perplexity --kl-divergence-base`."""
    with open(base, "rb") as f:
        ctx, vocab, n_chunks = _header(f, base)
        _check(engine, ctx, chunks)
        if vocab != engine.config.vocab_size:
            raise ValueError(f"{base} has a vocab of {vocab}, the model {engine.config.vocab_size}")
        tokens = list(memoryview(f.read(4 * ctx * n_chunks)).cast("i"))
        first, row = ctx // 2, 2 * ((vocab + 1) // 2) + 4  # uint16s per position
        nll, same, kls, score = 0.0, 0, list[float](), _scorer(engine, ctx, decode)
        for i in range(n_chunks if chunks is None else min(n_chunks, chunks)):
            chunk = tokens[i * ctx : (i + 1) * ctx]
            lp = score(chunk)
            stored = Tensor(f.read(2 * row * (ctx - 1 - first))).bitcast(dtypes.uint16)
            stored = stored.reshape(-1, row)
            scale, low = stored[:, :4].bitcast(dtypes.float32).chunk(2, dim=1)  # per position
            ref = stored[:, 4 : 4 + vocab].float() * scale + low
            # llama.cpp skips reference log-probs at or below -16, where its storage clamps
            kl = (ref > -16).where(ref.exp() * (ref - lp), 0).sum(-1)
            kls += cast(list[float], kl.tolist())
            same += int((ref.argmax(-1) == lp.argmax(-1)).sum().item())
            nll += _nll(lp, chunk)
    kls.sort()
    n = len(kls)
    return Quality(math.exp(nll / n), sum(kls) / n, _percentile(kls, 0.99), kls[-1], same / n)


def _check(engine: Engine, ctx: int, chunks: int | None) -> None:
    # chunks of at least one position scored, that the engine's context holds, and at least one
    if ctx < 3:
        raise ValueError(f"chunks of {ctx} tokens score none: they take 3 or more")
    if ctx > engine.max_context:
        raise ValueError(f"chunks of {ctx} tokens are over max_context {engine.max_context}")
    if chunks is not None and chunks < 1:
        raise ValueError(f"chunks must be at least 1, got {chunks}")


def _percentile(ordered: list[float], fraction: float) -> float:
    # between the two values about it, as llama-perplexity's
    at = fraction * (len(ordered) - 1)
    i, part = int(at), at - int(at)
    return (1 - part) * ordered[i] + part * ordered[min(i + 1, len(ordered) - 1)]


def _header(f: BinaryIO, base: Path) -> tuple[int, int, int]:
    # a llama-perplexity logits file's chunk size, vocab size and number of chunks
    magic, ctx, vocab, n_chunks = struct.unpack("<8s3i", f.read(20))
    if magic != b"_logits_":
        raise ValueError(f"{base} is not a llama-perplexity logits file")
    return ctx, vocab, n_chunks


def _scorer(engine: Engine, ctx: int, decode: bool) -> Callable[[list[int]], Tensor]:
    # a chunk's log-probabilities at positions ctx/2 .. ctx-2, each predicting the next token,
    # from graphs captured for the first chunk and replayed for the rest, whose log-probabilities
    # each overwrite the last's. With decode, those positions run one token at a time, through
    # the kernels generation uses.
    first, model, tok = ctx // 2, engine.model, engine.tokenizer

    def tokens(chunk: list[int]) -> list[int]:
        return [tok.bos_id] + chunk[1:] if tok.add_bos and tok.bos_id is not None else chunk

    if not decode:  # in a graph, which plans its buffers: a plain call holds every layer's at once
        score = graph(lambda t: model.logits(model(t, 0)[:, first : ctx - 1])[0].log_softmax(-1))
        return lambda chunk: score(Tensor([tokens(chunk)], dtype=dtypes.int32))
    rows = Tensor.zeros(ctx - 1 - first, engine.config.vocab_size).contiguous().realize()

    def step(token: Tensor, pos: UOp) -> None:
        rows[pos - first : pos - first + 1].assign(
            model.logits(model(token, pos))[0].log_softmax(-1)
        )
        rows.realize()

    jit, pos = TinyJit(step), UOp.variable("start_pos", 0, engine.max_context - 1)

    def run(chunk: list[int]) -> Tensor:
        ids = tokens(chunk)
        model(Tensor([ids[:first]], dtype=dtypes.int32), 0).realize()
        for i in range(first, ctx - 1):
            jit(Tensor([[ids[i]]], dtype=dtypes.int32), pos.bind(i))
        return rows

    return run


def _nll(logprobs: Tensor, chunk: list[int]) -> float:
    targets = Tensor(chunk[len(chunk) // 2 + 1 :], dtype=dtypes.int32).unsqueeze(-1)
    return -float(logprobs.gather(-1, targets).sum().item())
