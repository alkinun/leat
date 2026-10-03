"""Model math in plain tinygrad ops, the reference every fast kernel is tested against.

Ops dispatch to hand-written kernels where one applies; LEAT_KERNELS=ref turns them off.
"""

import math
import os

from tinygrad import Tensor, UOp, dtypes

from leat import nv
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
    if _fast() and all(nv.supports_matvec(x, w) for w in ws):
        return nv.matvecs(x, *ws, norm=norm, residual=residual)
    if _fast() and all(nv.supports_matmul(x, w) for w in ws):
        return nv.matmuls(x, *ws, norm=norm, residual=residual)
    if norm is not None:
        x = rms_norm(x, *norm)
    outs = [x @ w.dequant(x.dtype).T for w in ws]
    return outs if residual is None else [residual + out for out in outs]


def feed_forward(
    x: Tensor, gate: QTensor, up: QTensor, down: QTensor, norm: tuple[Tensor, float],
    gelu: bool = False, residual: bool = True,
) -> Tensor:  # fmt: skip
    # x + (act(n @ gate.T) * (n @ up.T)) @ down.T for n = rms_norm(x, *norm) and act SiLU, or GELU
    # if gelu; without x if not residual. Kernels take gate and up together where they share a
    # type and shape; one token takes the matrix-vector kernels, though the matrix kernels would
    # also accept it.
    paired = _fast() and gate.type == up.type and gate.shape == up.shape
    if paired and nv.supports_matvec(x, gate):
        hidden = nv.swiglu(x, gate, up, norm, gelu)
    elif (
        paired and not gelu and residual and all(nv.supports_matmul(x, w) for w in (gate, up, down))
    ):
        return nv.feed_forward(x, gate, up, down, norm)
    else:
        g, u = linears(x, gate, up, norm=norm)
        hidden = (g.gelu() if gelu else g.silu()) * u
    return linear(hidden, down, residual=x if residual else None)


def router(x: Tensor, norm: tuple[Tensor, float], w: QTensor) -> Tensor:
    # the scores a mixture of experts' router gives each expert: rms_norm(x, *norm) @ w.T
    if _fast() and nv.supports_scores(x, w):
        return nv.scores(x, norm, w)
    return linear(rms_norm(x, *norm), w)


def mixture(
    x: Tensor, scores: Tensor, gate: QTensor, up: QTensor | None, down: QTensor, used: int,
    norm: tuple[Tensor, float], gelu: bool = False, scales: Tensor | None = None,
    residual: bool = True,
) -> Tensor:  # fmt: skip
    # x + a mixture of experts for n = rms_norm(x, *norm), as feed_forward, given the router's
    # scores (B, T, experts): each token takes the MLPs of the `used` experts it scores highest,
    # weighted by the softmax of their scores and by each expert's scale if given. Experts are
    # stacked matrices (experts, rows, cols); where up is None, gate stacks both, the gate's rows
    # first in each. Only the chosen experts are read.
    if _fast() and nv.supports_mixture(x, gate, up, down):
        return nv.mixture(x, scores, gate, up, down, used, norm, gelu, scales, residual)
    top, experts = scores.topk(used)
    weights = top.softmax(-1) if scales is None else top.softmax(-1) * scales[experts]
    B, T, dim = x.shape
    ids = experts.flatten()
    n = rms_norm(x, *norm).unsqueeze(2).expand(B, T, used, dim).reshape(-1, 1, dim)
    g = n @ _take(gate, ids).dequant().transpose(1, 2)
    g, u = g.chunk(2, dim=-1) if up is None else (g, n @ _take(up, ids).dequant().transpose(1, 2))
    out = ((g.gelu() if gelu else g.silu()) * u) @ _take(down, ids).dequant().transpose(1, 2)
    mixed = (out.reshape(B, T, used, dim) * weights.reshape(B, T, used, 1)).sum(2)
    return x + mixed if residual else mixed


def _take(w: QTensor, index: Tensor) -> QTensor:
    # the matrices w[index] of a stack of them, still in storage
    rest = w.data.shape[1:]
    data = w.data.reshape(w.shape[0], -1, *rest)[index]
    return QTensor(data.reshape(-1, *rest), w.type, (-1, *w.shape[1:]))


def _fast() -> bool:
    return os.environ.get("LEAT_KERNELS") != "ref"


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


def rope(x: Tensor, cos: Tensor, sin: Tensor, halves: bool = False) -> Tensor:
    # rotates adjacent pairs of dimensions, or with halves dimension i with i + D/2.
    # x: (B, H, T, D); cos, sin: (T, D/2)
    if halves:
        x0, x1 = x.chunk(2, dim=-1)
        return (x0 * cos - x1 * sin).cat(x0 * sin + x1 * cos, dim=-1)
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    return Tensor.stack(x0 * cos - x1 * sin, x0 * sin + x1 * cos, dim=-1).flatten(-2)


def attention(
    q: Tensor, cache: Tensor, slot: int | UOp, start_pos: int | UOp, scale: float, window: int = 0
) -> Tensor:
    # q: (1, H, T, D) at positions start_pos.. ; cache: (2, slots, KV_H, positions, D), causal
    # over the slot's positions, and over only the last `window` of them if given, with scores
    # q.k * scale. Returns (1, T, H * D), the layout the output projection reads.
    B, H, T, D = q.shape
    if _fast() and nv.supports_attention(q, cache):
        # one token: the heads already follow each other; a transpose here would cost a copy
        return nv.attention(q, cache, slot, start_pos + T, scale, window).reshape(B, T, H * D)
    if _fast() and nv.supports_flash_attention(q, cache):
        return nv.flash_attention(q, cache, slot, start_pos, scale, window)
    k, v = (cache[i, slot : slot + 1, :, : start_pos + T].cast(q.dtype) for i in (0, 1))
    mask = None
    if window or not (isinstance(T, int) and T == 1):
        full = Tensor.full((1, 1, T, k.shape[2]), float("-inf"), dtype=q.dtype)
        mask = full.triu(start_pos + 1)  # later positions
        if window:  # and positions `window` or more back
            mask = mask + full.tril(start_pos - window)
    out = (q * (scale * math.sqrt(D))).scaled_dot_product_attention(k, v, mask, enable_gqa=True)
    return out.transpose(1, 2).reshape(B, T, H * D)


def argmax(x: Tensor) -> Tensor:
    # index of each row's largest value, the first on ties: (B, V) -> (B, 1) int32
    if _fast() and nv.supports_argmax(x):
        return nv.argmax(x)
    return x.argmax(-1, keepdim=True).cast(dtypes.int32)
