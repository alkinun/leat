"""Model math in plain tinygrad ops, which run on any device.

Ops dispatch to hand-written kernels where one applies; LEAT_KERNELS=ref turns them off.
"""

import itertools
import math
import os
from dataclasses import dataclass

from tinygrad import Tensor, UOp, dtypes

from leat import kernels
from leat.quant import NATIVE, QTensor


def linear(x: Tensor, w: QTensor, residual: Tensor | None = None) -> Tensor:
    # residual + x @ w.T, with the addition inside the matrix kernel where there is one
    return linears(x, w, residual=residual)[0]


def linears(
    x: Tensor,
    *ws: QTensor,
    norm: tuple[Tensor, float] | None = None,
    residual: Tensor | None = None,
) -> list[Tensor]:
    # x @ w.T for each w, after rms_norm(x, *norm) if given and plus a residual with one w;
    # kernels share one quantization of x
    if _fast() and all(kernels.supports_matvec(x, w) for w in ws):
        return kernels.matvecs(x, *ws, norm=norm, residual=residual)
    if _fast() and all(kernels.supports_matmul(x, w) for w in ws):
        return kernels.matmuls(x, *ws, norm=norm, residual=residual)
    if norm is not None:
        x = rms_norm(x, *norm)
    outs = [x @ w.dequant(x.dtype).T for w in ws]
    return outs if residual is None else [residual + out for out in outs]


def feed_forward(
    x: Tensor, gate: QTensor, up: QTensor, down: QTensor, norm: tuple[Tensor, float],
    kind: str = "silu", residual: bool = True,
) -> Tensor:  # fmt: skip
    # x + glu(kind, n @ gate.T, n @ up.T) @ down.T for n = rms_norm(x, *norm); without x if not
    # residual. Kernels take gate and up together where they share a type and shape; a few
    # tokens take the matrix-vector kernels, though the matrix kernels would also accept them.
    paired = _fast() and gate.type == up.type and gate.shape == up.shape
    if paired and kernels.supports_matvec(x, gate):
        hidden = kernels.swiglu(x, gate, up, norm, kind)
    elif paired and all(kernels.supports_matmul(x, w) for w in (gate, up, down)):
        return kernels.feed_forward(x, gate, up, down, norm, kind, residual)
    else:
        hidden = glu(kind, *linears(x, gate, up, norm=norm))
    return linear(hidden, down, residual=x if residual else None)


def glu(kind: str, g: Tensor, u: Tensor) -> Tensor:
    # how an MLP's gate and up combine: act(g) * u for act SiLU or GELU, tanh's approximation, or
    # gpt-oss's clamped SwiGLU, "oai", as ggml's swiglu_oai
    if kind == "oai":
        g, u = g.minimum(7.0), u.clip(-7.0, 7.0)
        return g * (g * 1.702).sigmoid() * (u + 1)
    return (g.gelu() if kind == "gelu" else g.silu()) * u


def add_normed(
    x: Tensor, parts: list[tuple[Tensor, Tensor]], weight: Tensor | None, eps: float,
    scale: Tensor | None = None,
) -> Tensor:  # fmt: skip
    # (x + the sum of rms_norm(part, its weight) over parts, normed again with `weight` if given)
    # times `scale` if given: how Gemma 4's blocks add their outputs to the residual
    if _fast() and kernels.supports_add_normed(x):
        return kernels.add_normed(x, parts, weight, eps, scale)
    total = sum((rms_norm(part, w, eps) for part, w in parts[1:]), rms_norm(*parts[0], eps))
    out = x + (total if weight is None else rms_norm(total, weight, eps))
    return out if scale is None else out * scale


def router(x: Tensor, norm: tuple[Tensor, float], w: QTensor, bias: Tensor | None = None) -> Tensor:
    # rms_norm(x, *norm) @ w.T, plus the bias if given: the scores a mixture of experts' router
    # gives each expert, or other projections of F32 weights as sensitive to rounding
    if _fast() and kernels.supports_scores(x, w):
        scores = kernels.scores(x, norm, w)
    else:
        scores = linear(rms_norm(x, *norm), w)
    return scores if bias is None else scores + bias


def mixture(
    x: Tensor, scores: Tensor, gate: QTensor, up: QTensor | None, down: QTensor, used: int,
    norm: tuple[Tensor, float], kind: str = "silu", scales: Tensor | None = None,
    residual: bool = True, biases: tuple[Tensor, Tensor, Tensor] | None = None,
    live: int | UOp | None = None,
) -> Tensor:  # fmt: skip
    # x + a mixture of experts for n = rms_norm(x, *norm), as feed_forward, given the router's
    # scores (B, T, experts): each token takes the MLPs of the `used` experts it scores highest,
    # weighted by the softmax of their scores and by each expert's scale if given. Experts are
    # stacked matrices (experts, rows, cols), with biases (experts, rows) of gate, up and down if
    # given; where up is None, gate stacks both, the gate's rows first in each. Only the chosen
    # experts are read. Tokens past the first `live`, if given, pad a batch: the kernels give
    # them no experts, rather than reading experts of their own, and the reference ops theirs.
    if _fast() and kernels.supports_mixture(x, gate, up, down):
        args = (used, norm, kind, scales, residual, biases, live)
        return kernels.mixture(x, scores, gate, up, down, *args)
    top, experts = scores.topk(used)
    weights = top.softmax(-1) if scales is None else top.softmax(-1) * scales[experts]
    B, T, dim = x.shape
    if math.prod(x.max_shape[:-1]) * used > gate.shape[0]:  # more pairs than experts
        chosen = (experts.unsqueeze(-1) == Tensor.arange(gate.shape[0])).float()
        each = (chosen * weights.unsqueeze(-1)).sum(-2)  # (B, T, experts): 0 but for the chosen
        mixed = _every_expert(rms_norm(x, *norm), each, gate, up, down, kind, biases)
        return x + mixed if residual else mixed
    ids = experts.flatten()
    n = rms_norm(x, *norm).unsqueeze(2).expand(B, T, used, dim).reshape(-1, 1, dim)
    g = n @ _take(gate, ids).dequant().transpose(1, 2)
    g, u = g.chunk(2, dim=-1) if up is None else (g, n @ _take(up, ids).dequant().transpose(1, 2))
    if biases is not None:
        g, u = g + biases[0][ids].unsqueeze(1), u + biases[1][ids].unsqueeze(1)
    out = glu(kind, g, u) @ _take(down, ids).dequant().transpose(1, 2)
    if biases is not None:
        out = out + biases[2][ids].unsqueeze(1)
    mixed = (out.reshape(B, T, used, dim) * weights.reshape(B, T, used, 1)).sum(2)
    return x + mixed if residual else mixed


def _every_expert(
    n: Tensor, each: Tensor, gate: QTensor, up: QTensor | None, down: QTensor, kind: str,
    biases: tuple[Tensor, Tensor, Tensor] | None,
) -> Tensor:  # fmt: skip
    # the sum over experts of each token's weight `each` (B, T, experts) of the expert times its
    # MLP of n: every expert runs every token, as many tokens choose most experts, rather than a
    # copy of the chosen ones' matrices for each pair
    total = None
    for e in range(gate.shape[0]):
        g = n @ _take(gate, Tensor([e])).dequant()[0].T
        g, u = g.chunk(2, dim=-1) if up is None else (g, n @ _take(up, Tensor([e])).dequant()[0].T)
        if biases is not None:
            g, u = g + biases[0][e], u + biases[1][e]
        out = glu(kind, g, u) @ _take(down, Tensor([e])).dequant()[0].T
        if biases is not None:
            out = out + biases[2][e]
        out = out * each[..., e : e + 1]
        total = out if total is None else total + out
    assert total is not None
    return total


def _take(w: QTensor, index: Tensor) -> QTensor:
    # the matrices w[index] of a stack of them, still in storage
    rest = w.data.shape[1:]
    data = w.data.reshape(w.shape[0], -1, *rest)[index]
    return QTensor(data.reshape(-1, *rest), w.type, (-1, *w.shape[1:]))


def _fast() -> bool:
    return os.environ.get("LEAT_KERNELS") != "ref"


def halved(x: Tensor) -> bool:
    # whether plain ops on x take matrices in f16, summed in f32, on the matrix cores, as
    # llama.cpp's cuBLAS does: on a GPU of them, but with LEAT_KERNELS=ref
    return _fast() and kernels.on_matrix_cores(x)


def embedding(tokens: Tensor, w: QTensor) -> Tensor:
    # Gathers whole rows of storage, then decodes only those; tinygrad lowers the gather to a load.
    # A bound number of tokens gathers as many as there may be: tinygrad leaves a copy along a
    # symbolic axis to one thread per block.
    vocab, dim = w.shape
    shape, padded = tokens.shape, tokens.max_shape
    rows = w.data.reshape(vocab, -1)[tokens.pad_to(padded).flatten()]
    if w.type in NATIVE:
        out = rows.reshape(*padded, dim).float()
    else:
        out = QTensor(rows.reshape(-1, w.data.shape[1]), w.type, (*padded, dim)).dequant()
    return out.shrink_to((*shape, dim))


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    x = x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()
    return x if weight is None else x * weight


def rotary(x: Tensor, cos: Tensor, sin: Tensor, halves: bool) -> Tensor:
    # rotates the first R dimensions: adjacent pairs, or with halves dimension i with i + R/2; the
    # others stay. x: (B, H, T, D); cos, sin: (T, R/2)
    rotated = 2 * cos.shape[-1]
    x, rest = x[..., :rotated], x[..., rotated:]
    if halves:
        x0, x1 = x.chunk(2, dim=-1)
        out = (x0 * cos - x1 * sin).cat(x0 * sin + x1 * cos, dim=-1)
    else:
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        out = Tensor.stack(x0 * cos - x1 * sin, x0 * sin + x1 * cos, dim=-1).flatten(-2)
    return out if rest.shape[-1] == 0 else out.cat(rest, dim=-1)


@dataclass(frozen=True)
class Span:
    """`length` consecutive tokens of one sequence, from position `start` of cache slot `slot`;
    each sees those before it, or if not `causal` all the span's, as an image's do."""

    slot: int | UOp
    start: int | UOp
    length: int | UOp = 1
    causal: bool = True


def rotate(
    q: Tensor, k: Tensor, v: Tensor, cache: Tensor, spans: list[Span],
    rope: tuple[tuple[Tensor, Tensor], int] | None, halves: bool,
    biases: tuple[Tensor, Tensor, Tensor] | None, norms: tuple[Tensor, Tensor] | None,
    v_norm: bool, eps: float, own: bool = False,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    # q (1, T, H, D), k and v (1, T, KV_H, D), the spans' tokens in turn: plus their biases (H * D
    # or KV_H * D), if given; each head of q and k normed with its weight, if given, and of v
    # without, if v_norm; the first R dimensions of q and k rotated by RoPE's tables (positions,
    # R/2), or each slot's (slots, positions, R/2), at their positions, or if `own` by the tokens'
    # own angles (T, R/2) of a single span, for rope ((cos, sin), R), if given; and k and v stored
    # there in their slots of the cache. Returns q (1, H, T, D) and the cache.
    args = (rope, halves, biases, norms, v_norm, eps)
    if own:
        return _rotate(q, k, v, cache, spans[0], *args, own=True)
    if _fast() and kernels.supports_rotate(q, cache) and (rows := _rows(spans)) is not None:
        slots, positions = rows
        return kernels.rotate(q, k, v, cache, slots, positions, *args)
    if len(spans) == 1:
        return _rotate(q, k, v, cache, spans[0], *args)
    outs, at = [], 0
    for span in spans:
        n = int(span.length)
        part, cache = _rotate(q[:, at : at + n], k[:, at : at + n], v[:, at : at + n], cache,
                              span, *args)  # fmt: skip
        outs.append(part)
        at += n
    return outs[0].cat(*outs[1:], dim=2), cache


def _rotate(
    q: Tensor, k: Tensor, v: Tensor, cache: Tensor, span: Span,
    rope: tuple[tuple[Tensor, Tensor], int] | None, halves: bool,
    biases: tuple[Tensor, Tensor, Tensor] | None, norms: tuple[Tensor, Tensor] | None,
    v_norm: bool, eps: float, own: bool = False,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    # rotate() for one span
    T, slot, start_pos = q.shape[1], span.slot, span.start
    if biases is not None:
        q, k, v = (t + b.reshape(t.shape[2:]) for t, b in zip((q, k, v), biases, strict=True))
    if norms is not None:
        q, k = rms_norm(q, norms[0], eps), rms_norm(k, norms[1], eps)
    if v_norm:
        v = rms_norm(v, None, eps)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    if rope is not None:
        tables = rope[0] if own or rope[0][0].ndim == 2 else (t[slot] for t in rope[0])
        cos, sin = tables if own else (table[start_pos : start_pos + T] for table in tables)
        q, k = (rotary(t, cos, sin, halves) for t in (q, k))
    new = Tensor.stack(k, v.transpose(1, 2)).cast(cache.dtype)
    cache[:, slot : slot + 1, :, start_pos : start_pos + T].assign(new)
    return q, cache


Rows = list[int | UOp] | int | UOp  # one per token, or one for a single span's tokens


def _rows(spans: list[Span]) -> tuple[Rows, Rows] | None:
    # the slot and position of each token of several spans, as the kernels take them, or the
    # slot and start of a single span; None for several spans of lengths not known in advance
    if len(spans) == 1:
        return spans[0].slot, spans[0].start
    if all(isinstance(s.length, int) for s in spans):
        tokens = [(s.slot, s.start + i) for s in spans for i in range(int(s.length))]
        return [slot for slot, _ in tokens], [pos for _, pos in tokens]
    return None


def _tokens(spans: list[Span]) -> bool:
    # whether every span is a single token
    return all(isinstance(s.length, int) and s.length == 1 for s in spans)


def attention(
    q: Tensor, cache: Tensor, spans: list[Span], scale: float, window: int = 0,
    sinks: Tensor | None = None,
) -> Tensor:  # fmt: skip
    # q: (1, H, T, D), the spans' tokens in turn; cache: (2, slots, KV_H, positions, D). Each
    # token attends over its span's slot, causally or over all the span's tokens as the span
    # has it, and over only the last `window` positions before it if given, with scores
    # q.k * scale; with a sink per head, if given, a score that takes its
    # share of the softmax and adds no value, as gpt-oss's. Returns (1, T, H * D), the layout the
    # output projection reads.
    # a token per row: of a single span of one, or of several spans, each token its own row
    rows = _rows(spans) if len(spans) > 1 or _tokens(spans) else None
    if _fast() and rows is not None and kernels.supports_attention(q, cache):
        slots, positions = (r if isinstance(r, list) else [r] for r in rows)
        lengths = [pos + 1 for pos in positions]
        # spans of several tokens: the rows of each one's last
        several = any(s.length != 1 for s in spans)
        ends = [n - 1 for n in itertools.accumulate(int(s.length) for s in spans)]
        return kernels.attention(
            q, cache, slots, lengths, scale, window, sinks, ends if several else None
        )
    if len(spans) == 1:
        span = spans[0]
        if _fast() and kernels.supports_flash_attention(q, cache):
            return kernels.flash_attention(
                q, cache, span.slot, span.start, scale, window, sinks, span.causal
            )
        return _attention(q, cache, span, scale, window, sinks)
    outs, at = [], 0
    for span in spans:
        n = int(span.length)
        outs.append(_attention(q[:, :, at : at + n], cache, span, scale, window, sinks))
        at += n
    return outs[0].cat(*outs[1:], dim=1)


def _attention(
    q: Tensor, cache: Tensor, span: Span, scale: float, window: int, sinks: Tensor | None
) -> Tensor:
    # attention() for one span, in plain ops
    B, H, T, D = q.shape
    slot, start_pos = span.slot, span.start
    k, v = (cache[i, slot : slot + 1, :, : start_pos + T].cast(q.dtype) for i in (0, 1))
    mask = None
    causal = span.causal and not (isinstance(T, int) and T == 1)
    if window or causal:
        full = Tensor.full((1, 1, T, k.shape[2]), float("-inf"), dtype=q.dtype)
        # later positions, where the span sees only those before each token, and positions
        # `window` or more back
        masks = ([full.triu(start_pos + 1)] if causal else []) + (
            [full.tril(start_pos - window)] if window else [])  # fmt: skip
        mask = sum(masks[1:], masks[0])
    if sinks is None:
        out = (q * (scale * math.sqrt(D))).scaled_dot_product_attention(k, v, mask, enable_gqa=True)
        return out.transpose(1, 2).reshape(B, T, H * D)
    k, v = (z.repeat_interleave(int(H) // int(z.shape[1]), dim=1) for z in (k, v))
    scores = q @ k.transpose(-1, -2) * scale + (0 if mask is None else mask)
    scores = scores.cat(sinks.reshape(1, H, 1, 1).expand(B, H, T, 1), dim=-1)
    out = scores.softmax(-1)[..., :-1] @ v
    return out.transpose(1, 2).reshape(B, T, H * D)


def delta_net(
    mixed: Tensor, z: Tensor, gates: Tensor, conv: Tensor,
    decay: tuple[Tensor, Tensor], norm: tuple[Tensor, float], states: tuple[Tensor, Tensor],
    spans: list[Span], saved: tuple[Tensor, Tensor] | None = None,
) -> Tensor:  # fmt: skip
    # Gated DeltaNet, Qwen3.5's linear attention, over the spans' tokens in turn, each from and
    # into its slot's states, as llama.cpp's. mixed (1, T, channels) holds each token's queries,
    # keys and values, which pass a causal convolution of conv's weights (channels, width) over
    # the token and the width - 1 inputs before it, which the conv state (slots, width - 1,
    # channels) holds, and then SiLU. Queries and keys, L2-normed and queries scaled, have fewer
    # heads than values: value head h takes their head h mod their heads. Each value head's state
    # (slots, heads, key dims, value dims) decays by exp(a * softplus(alpha + bias)) for decay
    # (a, bias), then moves the values it holds for the token's key a sigmoid(beta) share toward
    # the token's values, gates (1, T, 2 * heads) holding its alphas, then betas; the output,
    # the values the state holds for the token's query, is
    # normed with norm and gated by SiLU of z (1, T, heads * value dims). A span from position 0
    # starts its sequence, from zero states. The states after each token go to `saved`, if
    # given, (at least T, width - 1, channels) and (at least T, heads, key dims, value dims), the
    # t-th token's to row t, from which a sequence may go back to any of them. Returns (1, T,
    # heads * value dims).
    args = (mixed, z, gates, conv, decay, norm, states)
    lengths = {s.length for s in spans}
    even = len(lengths) == 1 and all(isinstance(n, int) for n in lengths)
    if _fast() and kernels.supports_delta_net(mixed, states[1]) and (len(spans) == 1 or even):
        tokens = mixed.shape[1] if len(spans) == 1 else spans[0].length
        slots, starts = [s.slot for s in spans], [s.start for s in spans]
        return kernels.delta_net(*args, slots, starts, tokens, saved)
    if len(spans) == 1:
        return _delta_net(mixed, z, gates, conv, decay, norm, states, spans[0], saved)
    outs, at = [], 0
    for span in spans:
        n = int(span.length)
        m, g, a = (t[:, at : at + n] for t in (mixed, z, gates))
        kept = None if saved is None else (saved[0][at : at + n], saved[1][at : at + n])
        outs.append(_delta_net(m, g, a, conv, decay, norm, states, span, kept))
        at += n
    return outs[0].cat(*outs[1:], dim=1)


def _delta_net(
    mixed: Tensor, z: Tensor, gates: Tensor, conv: Tensor,
    decay: tuple[Tensor, Tensor], norm: tuple[Tensor, float], states: tuple[Tensor, Tensor],
    span: Span, saved: tuple[Tensor, Tensor] | None = None,
) -> Tensor:  # fmt: skip
    # delta_net() for one span, a token at a time. A bound number of tokens runs as many as there
    # may be, those past the span's with no decay or update, which leave the states as they are.
    T, slot = mixed.shape[1], span.slot
    n = T if isinstance(T, int) else int(T.vmax)
    heads, k_dim, v_dim = (int(d) for d in states[1].shape[1:])
    width, eps = int(conv.shape[1]), norm[1]
    mixed, z, gates = (t.pad_to(t.max_shape) for t in (mixed, z, gates))
    alpha, beta = gates.chunk(2, dim=-1)
    conv_state, state = (Tensor(span.start > 0).where(t[slot : slot + 1], 0.0).contiguous()
                         for t in states)  # fmt: skip
    # read first: realizing both writes below, tinygrad may write the conv state before the
    # recurrence reads its inputs from it
    Tensor.realize(conv_state, state)
    inputs = conv_state.cat(mixed, dim=1)  # the convolution's: those before the span's, then its
    mixed = inputs[:, :n] * conv[:, 0]
    for j in range(1, width):
        mixed = mixed + inputs[:, j : j + n] * conv[:, j]
    mixed = mixed.silu()
    k_heads = (int(mixed.shape[-1]) - heads * v_dim) // (2 * k_dim)
    q, k, v = mixed.split([k_heads * k_dim, k_heads * k_dim, heads * v_dim], dim=-1)
    q, k = (_l2_norm(t.reshape(1, n, k_heads, k_dim), eps).repeat(1, 1, heads // k_heads, 1)
            for t in (q, k))  # fmt: skip
    q, v = q / math.sqrt(k_dim), v.reshape(1, n, heads, v_dim)
    real = (Tensor.arange(n) < Tensor(T)).reshape(1, n, 1)  # tokens, not padding
    decays = real.where(decay[0] * _softplus(alpha + decay[1]), 0.0).exp()
    shares = real.where(beta.sigmoid(), 0.0)
    out, after = [], []
    for t in range(n):
        state = state * decays[:, t].reshape(1, heads, 1, 1)
        key = k[:, t].unsqueeze(-1)
        delta = (v[:, t] - (state * key).sum(2)) * shares[:, t].unsqueeze(-1)
        state = state + key * delta.unsqueeze(2)
        out.append((state * q[:, t].unsqueeze(-1)).sum(2))
        after.append(state)
    # realized now: no later op reads them, so the step's outputs would not
    writes = [states[0][slot : slot + 1].assign(inputs[:, T : T + width - 1]),
              states[1][slot : slot + 1].assign(state)]  # fmt: skip
    if saved is not None:  # after token t, the inputs t + 1 .. t + width - 1 and the state
        windows = [inputs[:, t + 1 : t + width] for t in range(n)]
        writes += [
            saved[0][:n].assign(Tensor.cat(*windows)),
            saved[1][:n].assign(Tensor.cat(*after)),
        ]
    Tensor.realize(*writes)
    gated = rms_norm(Tensor.stack(*out, dim=1), *norm) * z.reshape(1, n, heads, v_dim).silu()
    return gated.reshape(1, n, heads * v_dim).shrink_to((1, T, heads * v_dim))


def _l2_norm(x: Tensor, eps: float) -> Tensor:
    # x / sqrt(|x|^2 + eps), as llama.cpp's Gated DeltaNet and transformers' have it
    return x * (x.square().sum(-1, keepdim=True) + eps).rsqrt()


def _softplus(x: Tensor) -> Tensor:
    # log(1 + exp(x)), without overflow
    return x.relu() + (1 + (-x.abs()).exp()).log()


def argmax(x: Tensor) -> Tensor:
    # index of each row's largest value, the first on ties: (B, V) -> (B, 1) int32
    if _fast() and kernels.supports_argmax(x):
        return kernels.argmax(x)
    return x.argmax(-1, keepdim=True).cast(dtypes.int32)


def cutoff(scores: Tensor, top_k: Tensor, top_p: Tensor, min_p: Tensor) -> Tensor:
    # the score below which each row of scores (B, V), log-probabilities but for a constant, drops
    # its tokens, each option (B, 1): -inf where no option is set. top_k keeps the k likeliest
    # tokens, at 0 all; top_p then the likeliest of those whose probabilities, renormalized, sum
    # to top_p, one at least; min_p those at least min_p times as likely as the likeliest. The
    # kernels cut within
    # a hundredth below where this would, and keep no token 20 nats below the likeliest.
    if _fast() and kernels.supports_cutoff(scores):
        top, by_k, by_p = kernels.cutoff(scores, top_k, top_p)
    else:
        top, by_k, by_p = _cutoff(scores, top_k, top_p)
    cut = (top_p < 1).where(by_p, by_k).maximum(top + min_p.log())
    return ((top_k > 0) | (top_p < 1) | (min_p > 0)).where(cut, -math.inf)


def _cutoff(scores: Tensor, top_k: Tensor, top_p: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # the top score, the k-th, and the last of the top k that top_p keeps: each token while the
    # likelier ones' share of the top k's probability falls short of top_p
    ranked, n = scores.sort(-1, descending=True)[0], int(scores.shape[-1])
    rank = Tensor.arange(1, n + 1).reshape(1, n)
    k = (top_k > 0).where(top_k, n)
    weights = (rank <= k).where((ranked - ranked[:, :1]).exp(), 0.0)
    likelier, kept = weights.cumsum(-1) - weights, weights.sum(-1, keepdim=True)
    by_k = (rank == k.minimum(n)).where(ranked, 0.0).sum(-1, keepdim=True)
    keep = (rank == 1) | (likelier < top_p * kept)  # the likeliest even at top_p 0
    by_p = ((rank <= k) & keep).where(ranked, math.inf).min(-1, keepdim=True)
    return ranked[:, :1], by_k, by_p
