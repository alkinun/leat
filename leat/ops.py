"""Model math in plain tinygrad ops, the reference every fast kernel is tested against.

Ops dispatch to hand-written kernels where one applies; LEAT_KERNELS=ref turns them off.
"""

import os

from tinygrad import Tensor, UOp

from leat import nv
from leat.quant import NATIVE, QTensor


def linear(x: Tensor, w: QTensor) -> Tensor:
    if os.environ.get("LEAT_KERNELS") != "ref" and nv.supports(x, w):
        return nv.linear(x, w)
    return x @ w.dequant(x.dtype).T


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


def attention(q: Tensor, k: Tensor, v: Tensor, start_pos: int | UOp) -> Tensor:
    # q: (B, H, T, D); k, v: (B, KV_H, start_pos + T, D), causal from the end
    T, S = q.shape[2], k.shape[2]
    mask = None
    if not (isinstance(T, int) and T == 1):
        mask = Tensor.full((1, 1, T, S), float("-inf"), dtype=q.dtype).triu(start_pos + 1)
    return q.scaled_dot_product_attention(k.cast(q.dtype), v.cast(q.dtype), mask, enable_gqa=True)
