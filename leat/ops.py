"""Model math in plain tinygrad ops, the reference every fast kernel is tested against.

Ops dispatch to hand-written kernels where one applies; LEAT_KERNELS=ref turns them off.
"""

import os

from tinygrad import Tensor, UOp, dtypes

from leat import nv
from leat.quant import NATIVE, QTensor


def linear(x: Tensor, w: QTensor, residual: Tensor | None = None) -> Tensor:
    # residual + x @ w.T, with the addition inside the matrix kernel where there is one
    if _fast() and nv.supports(x, w):
        return nv.linears(x, w, residual=residual)[0]
    out = x @ w.dequant(x.dtype).T
    return out if residual is None else residual + out


def _fast() -> bool:
    return os.environ.get("LEAT_KERNELS") != "ref"


def linears(x: Tensor, *ws: QTensor, norm: tuple[Tensor, float] | None = None) -> list[Tensor]:
    # x @ w.T for each w, after rms_norm(x, *norm) if given; kernels share one quantization of x
    if _fast() and all(nv.supports(x, w) for w in ws):
        return nv.linears(x, *ws, norm=norm)
    if norm is not None:
        x = rms_norm(x, *norm)
    return [x @ w.dequant(x.dtype).T for w in ws]


def swiglu(
    x: Tensor, gate: QTensor, up: QTensor, norm: tuple[Tensor, float] | None = None
) -> Tensor:
    # silu(x @ gate.T) * (x @ up.T), after rms_norm(x, *norm) if given
    same = gate.type == up.type and gate.shape == up.shape
    if _fast() and same and nv.supports(x, gate) and nv.supports(x, up):
        return nv.swiglu(x, gate, up, norm)
    g, u = linears(x, gate, up, norm=norm)
    return g.silu() * u


def feed_forward(
    x: Tensor, gate: QTensor, up: QTensor, down: QTensor, norm: tuple[Tensor, float]
) -> Tensor:
    # x + swiglu(x, gate, up, norm) @ down.T: llama's MLP with its residual
    fits = gate.type == up.type and gate.shape == up.shape
    if _fast() and fits and all(nv.supports(x, w) for w in (gate, up, down)):
        return nv.feed_forward(x, gate, up, down, norm)
    return linear(swiglu(x, gate, up, norm), down, residual=x)


def embedding(tokens: Tensor, w: QTensor) -> Tensor:
    # Gathers whole rows of storage, then decodes only those; tinygrad lowers the gather to a load.
    # A bound number of tokens gathers as many as there may be: tinygrad leaves a copy along a
    # symbolic axis to one thread per block.
    vocab, dim = w.shape
    shape, tokens = tokens.shape, tokens.pad_to(tokens.max_shape)
    rows = w.data.reshape(vocab, -1)[tokens.flatten()]
    if w.type in NATIVE:
        out = rows.reshape(*tokens.shape, dim).float()
    else:
        out = QTensor(rows.reshape(-1, w.data.shape[1]), w.type, (*tokens.shape, dim)).dequant()
    return out.shrink_to((*shape, dim))


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * weight


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # rotates adjacent pairs, ggml's "normal" mode, which is how GGUF lays out llama's q and k.
    # x: (B, H, T, D); cos, sin: (T, D/2)
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    return Tensor.stack(x0 * cos - x1 * sin, x0 * sin + x1 * cos, dim=-1).flatten(-2)


def attention(q: Tensor, cache: Tensor, start_pos: int | UOp) -> Tensor:
    # q: (B, H, T, D) at positions start_pos.. ; cache: (2, B, KV_H, max_context, D), causal.
    # Returns (B, T, H * D), the layout the output projection reads.
    B, H, T, D = q.shape
    if _fast() and nv.supports_attention(q, cache):
        # one token: the heads already follow each other; a transpose here would cost a copy
        return nv.attention(q, cache, start_pos + T).reshape(B, T, H * D)
    if _fast() and nv.supports_flash_attention(q, cache):
        return nv.flash_attention(q, cache, start_pos)
    k, v = (
        cache[0, :, :, : start_pos + T].cast(q.dtype),
        cache[1, :, :, : start_pos + T].cast(q.dtype),
    )
    mask = None
    if not (isinstance(T, int) and T == 1):
        mask = Tensor.full((1, 1, T, k.shape[2]), float("-inf"), dtype=q.dtype).triu(start_pos + 1)
    out = q.scaled_dot_product_attention(k, v, mask, enable_gqa=True)
    return out.transpose(1, 2).reshape(B, T, H * D)


def argmax(x: Tensor) -> Tensor:
    # index of each row's largest value, the first on ties: (B, V) -> (B, 1) int32
    if _fast() and nv.supports_argmax(x):
        return nv.argmax(x)
    return x.argmax(-1, keepdim=True).cast(dtypes.int32)
