"""Measurements: speed as llama-bench reports it, and quality as perplexity or KL divergence.

Quality follows llama-perplexity: chunks start with BOS and only their second half is scored.
"""

import math
import random
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.engine import Engine


@dataclass(frozen=True)
class Speed:
    prefill: float  # tokens/s over a prompt, including sampling the first token
    decode: float  # tokens/s generating after a one-token prompt
    weight_gbs: float  # weight bytes streamed per second while decoding


@dataclass(frozen=True)
class Quality:
    perplexity: float
    # against a reference: KL(reference || leat) per position, as llama-perplexity defines it,
    # and the fraction of positions where both pick the same most likely token
    kl_mean: float | None = None
    kl_p99: float | None = None
    kl_max: float | None = None
    top1: float | None = None


def speed(engine: Engine, prompt_tokens: int = 512, gen_tokens: int = 128, reps: int = 3) -> Speed:
    rng = random.Random(0)
    prompt = [rng.randrange(engine.config.vocab_size) for _ in range(prompt_tokens)]
    prefill, decode = [], []
    for _ in range(reps + 2):  # the first two runs execute eagerly and capture the graphs
        engine.reset()
        start = time.perf_counter()
        next(engine.generate(prompt, 1))
        prefill.append(prompt_tokens / (time.perf_counter() - start))
        engine.reset()
        tokens = engine.generate(prompt[:1], gen_tokens, ignore_eog=True)
        next(tokens)
        start = time.perf_counter()
        decode.append(sum(1 for _ in tokens) / (time.perf_counter() - start))
    tensors = engine.gguf.tensors
    tied = "output.weight" not in tensors  # then the embedding is read whole, as the output layer
    streamed = sum(t.nbytes for n, t in tensors.items() if n != "token_embd.weight" or tied)
    tg = statistics.median(decode[2:])
    return Speed(statistics.median(prefill[2:]), tg, streamed * tg / 1e9)


def perplexity(
    engine: Engine, text: str, ctx: int = 512, chunks: int | None = None, decode: bool = False
) -> Quality:
    tokens = engine.tokenizer.encode(text)
    n = min(len(tokens) // ctx, chunks or len(tokens))
    if n < 1:
        raise ValueError(f"the text has {len(tokens)} tokens, fewer than one chunk of {ctx}")
    nll = 0.0
    for i in range(n):
        chunk = tokens[i * ctx : (i + 1) * ctx]
        nll += _nll(_logprobs(engine, chunk, decode), chunk)
    return Quality(math.exp(nll / (n * (ctx - 1 - ctx // 2))))


def kl_divergence(
    engine: Engine, base: Path, chunks: int | None = None, decode: bool = False
) -> Quality:
    """Compares against logits saved by `llama-perplexity --kl-divergence-base`."""
    with open(base, "rb") as f:
        magic, ctx, vocab, n_chunks = struct.unpack("<8s3i", f.read(20))
        if magic != b"_logits_":
            raise ValueError(f"{base} is not a llama-perplexity logits file")
        if vocab != engine.config.vocab_size:
            raise ValueError(f"{base} has a vocab of {vocab}, the model {engine.config.vocab_size}")
        if ctx > engine.max_context:
            raise ValueError(
                f"{base} has chunks of {ctx} tokens, over max_context {engine.max_context}"
            )
        tokens = list(memoryview(f.read(4 * ctx * n_chunks)).cast("i"))
        first, row = ctx // 2, 2 * ((vocab + 1) // 2) + 4  # uint16s per position
        nll, same, kls = 0.0, 0, list[float]()
        for i in range(min(n_chunks, chunks or n_chunks)):
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
    return Quality(math.exp(nll / n), sum(kls) / n, kls[int(0.99 * (n - 1))], kls[-1], same / n)


def _logprobs(engine: Engine, chunk: list[int], decode: bool) -> Tensor:
    # log-probabilities at positions ctx/2 .. ctx-2, each predicting the next token. With decode,
    # those positions run one token at a time, through the kernels generation uses.
    ctx, first, model = len(chunk), len(chunk) // 2, engine.model
    tokens = chunk if engine.tokenizer.bos_id is None else [engine.tokenizer.bos_id] + chunk[1:]
    if not decode:
        hidden = model(Tensor([tokens], dtype=dtypes.int32), 0)[:, first : ctx - 1]
        return model.logits(hidden)[0].log_softmax(-1)
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
