"""Measurements: speed as llama-bench reports it, and quality as perplexity or KL divergence.

Quality follows llama-perplexity: chunks start with BOS where the tokenizer adds one, and only
their second half is scored.
"""

import math
import random
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.engine import BATCH, Engine, graph


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


def perplexity(
    engine: Engine, text: str, ctx: int = 512, chunks: int | None = None, decode: bool = False
) -> Quality:
    _check(ctx, chunks)
    tokens = engine.tokenizer.encode(text)
    n = min(len(tokens) // ctx, len(tokens) if chunks is None else chunks)
    if n < 1:
        raise ValueError(f"the text has {len(tokens)} tokens, fewer than one chunk of {ctx}")
    nll = 0.0
    for i in range(n):
        chunk = tokens[i * ctx : (i + 1) * ctx]
        nll += _nll(_logprobs(engine, chunk, decode), chunk)
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
        _check(ctx, chunks)
        if vocab != engine.config.vocab_size:
            raise ValueError(f"{base} has a vocab of {vocab}, the model {engine.config.vocab_size}")
        if ctx > engine.max_context:
            raise ValueError(
                f"{base} has chunks of {ctx} tokens, over max_context {engine.max_context}"
            )
        tokens = list(memoryview(f.read(4 * ctx * n_chunks)).cast("i"))
        first, row = ctx // 2, 2 * ((vocab + 1) // 2) + 4  # uint16s per position
        nll, same, kls = 0.0, 0, list[float]()
        for i in range(n_chunks if chunks is None else min(n_chunks, chunks)):
            chunk = tokens[i * ctx : (i + 1) * ctx]
            lp = _logprobs(engine, chunk, decode)
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


def _check(ctx: int, chunks: int | None) -> None:
    # chunks of at least one position scored, and at least one of them
    if ctx < 3:
        raise ValueError(f"chunks of {ctx} tokens score none: they take 3 or more")
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


def _logprobs(engine: Engine, chunk: list[int], decode: bool) -> Tensor:
    # log-probabilities at positions ctx/2 .. ctx-2, each predicting the next token. With decode,
    # those positions run one token at a time, through the kernels generation uses.
    ctx, first, model = len(chunk), len(chunk) // 2, engine.model
    tok = engine.tokenizer
    tokens = [tok.bos_id] + chunk[1:] if tok.add_bos and tok.bos_id is not None else chunk
    if not decode:  # in a graph, which plans its buffers: a plain call holds every layer's at once
        score = graph(lambda t: model.logits(model(t, 0)[:, first : ctx - 1])[0].log_softmax(-1))
        return score(Tensor([tokens], dtype=dtypes.int32))
    model(Tensor([tokens[:first]], dtype=dtypes.int32), 0).realize()
    rows = Tensor.zeros(ctx - 1 - first, engine.config.vocab_size).contiguous().realize()

    def step(token: Tensor, pos: UOp) -> None:
        rows[pos - first : pos - first + 1].assign(
            model.logits(model(token, pos))[0].log_softmax(-1)
        )
        rows.realize()

    jit, pos = TinyJit(step), UOp.variable("start_pos", 0, engine.max_context - 1)
    for i in range(first, ctx - 1):
        jit(Tensor([[tokens[i]]], dtype=dtypes.int32), pos.bind(i))
    return rows


def _nll(logprobs: Tensor, chunk: list[int]) -> float:
    targets = Tensor(chunk[len(chunk) // 2 + 1 :], dtype=dtypes.int32).unsqueeze(-1)
    return -float(logprobs.gather(-1, targets).sum().item())
