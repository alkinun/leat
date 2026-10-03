"""Model math in plain tinygrad ops, the reference every fast kernel is tested against.

Ops dispatch to hand-written kernels where one applies; LEAT_KERNELS=ref turns them off.
"""

import os

from tinygrad import Tensor, UOp

from leat import nv
from leat.quant import NATIVE, QTensor


def linear(x: Tensor, w: QTensor) -> Tensor:
    return linears(x, w)[0]


def _fast() -> bool:
    return os.environ.get("LEAT_KERNELS") != "ref"


def linears(x: Tensor, *ws: QTensor) -> list[Tensor]:
    # x @ w.T for each w; kernels share one quantization of the input
    if _fast() and all(nv.supports(x, w) for w in ws):
        return nv.linears(x, *ws)
    return [x @ w.dequant(x.dtype).T for w in ws]


def embedding(tokens: Tensor, w: QTensor) -> Tensor:
    # gathers whole rows of storage, then decodes only those; tinygrad lowers the gather to a load
    vocab, dim = w.shape
    rows = w.data.reshape(vocab, -1)[tokens.flatten()]
    if w.type in NATIVE:
        return rows.reshape(*tokens.shape, dim).float()
    return QTensor(rows.reshape(-1, w.data.shape[1]), w.type, (*tokens.shape, dim)).dequant()


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * weight


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # rotates adjacent pairs, ggml's "normal" mode, which is how GGUF lays out llama's q and k.
    # x: (B, H, T, D); cos, sin: (T, D/2)
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    return Tensor.stack(x0 * cos - x1 * sin, x0 * sin + x1 * cos, dim=-1).flatten(-2)


def attention(q: Tensor, cache: Tensor, start_pos: int | UOp) -> Tensor:
    # q: (B, H, T, D) at positions start_pos.. ; cache: (2, B, KV_H, max_context, D), causal
    T = q.shape[2]
    if _fast() and nv.supports_attention(q, cache):
        return nv.attention(q, cache, start_pos + T)
    k, v = (
        cache[0, :, :, : start_pos + T].cast(q.dtype),
        cache[1, :, :, : start_pos + T].cast(q.dtype),
    )
    mask = None
    if not (isinstance(T, int) and T == 1):
        mask = Tensor.full((1, 1, T, k.shape[2]), float("-inf"), dtype=q.dtype).triu(start_pos + 1)
    return q.scaled_dot_product_attention(k, v, mask, enable_gqa=True)
