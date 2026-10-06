"""Activations to int8 in groups of 32, as llama.cpp's q8_1, after RMSNorm when given its weight."""

import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.kernels.common import (
    GROUP,
    WARP,
    block_sum,
    carry,
    lane_range,
    load_vector,
    rounded,
    warp_max,
    warp_sum,
)

WARPS = 8  # per block, at most
SPREAD_ROWS = 16  # rows up to which a row's turns take blocks of their own


def _quantize_group(
    q: UOp, d: UOp, s: UOp, group: UOp, part: UOp, values: list[UOp], inside: UOp | None
) -> list[UOp]:
    # 8 lanes quantize a group of 32 values, 4 consecutive ones each, the lane's `part` of the
    # group: d = max|x| / 127, q = round(x / d), s = d * sum(q), with q packed four per int32 word
    # so matrix kernels read it without a copy; only where `inside` holds, if given. Custom
    # expressions keep both divisions exact: tinygrad would multiply by a reciprocal.
    amax = warp_max(functools.reduce(UOp.maximum, (v.maximum(-v) for v in values)), 8)
    scale = UOp(Ops.CUSTOMI, src=(amax,), arg=("({}/127.0f)", dtypes.float32))
    quants = []
    for v in values:
        quants.append((scale > 0).where(rounded(v, scale), 0.0).cast(dtypes.int32))
    word = functools.reduce(
        UOp.__or__, ((x & 0xFF).cast(dtypes.uint32) << (8 * i) for i, x in enumerate(quants))
    )
    total = warp_sum(sum(quants[1:], quants[0]), 8)
    at, first = group * (GROUP // 4) + part, part.eq(0)
    if inside is not None:
        at, first = at.valid(inside), first & inside
    first = group.valid(first)
    return [
        q[at].store(word.bitcast(dtypes.int32)),
        d[first].store(scale),
        s[first].store(scale * total.float()),
    ]


@functools.cache
def _quantize_q8_kernel(
    q: UOp, d: UOp, s: UOp, x: UOp, *weight: UOp, rows: int | UOp, eps: float
) -> UOp:
    # Each row takes a block, whose threads quantize 4 consecutive values at a time. Up to
    # SPREAD_ROWS rows, as the vector of a decode step, spread over blocks instead, though each
    # block then sums the squares of the whole row, from L2 after the first. Rows of whole groups
    # but not whole turns leave the last turn's extra threads idle.
    n, spread = int(x.shape[1]), int(x.shape[0]) <= SPREAD_ROWS
    warps = math.gcd(WARPS, n // (4 * WARP))
    threads = warps * WARP
    turns, ragged = -(-n // (4 * threads)), n % (4 * threads) != 0
    row = UOp.range(rows, 0, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(warps, 2, AxisType.LOCAL)
    turn = UOp.range(turns, 1, AxisType.GLOBAL if spread else AxisType.LOOP)
    thread = wave * WARP + lane
    zero = UOp.const(0.0, dtypes.float32)

    def load(at: UOp) -> list[UOp]:  # 4 values from at, or zeros past the row's end
        if not ragged:
            return list(load_vector(x[row, at], 4))
        return [(at < n).where(v, zero) for v in load_vector(x[row, at.minimum(n - 4)], 4)]

    at = (turn * threads + thread) * 4  # the thread's first value
    values = load(at)
    if weight:
        chunks = (load((i * threads + thread) * 4) for i in range(turns))
        squares = sum((v * v for chunk in chunks for v in chunk), zero)
        inv = (block_sum(warp_sum(squares), wave, lane) / n + eps).rsqrt()
        scale = load_vector(weight[0][at.minimum(n - 4) if ragged else at], 4)
        values = [v * inv * w for v, w in zip(values, scale, strict=True)]
    group = row * (n // GROUP) + at // GROUP
    inside = at < n if ragged else None
    stores = UOp.group(*_quantize_group(q, d, s, group, lane % 8, values, inside))
    if not spread:
        stores = stores.end(turn)
    info = KernelInfo(name="norm_quantize_q8" if weight else "quantize_q8", opts_to_apply=())
    return stores.end(*(row, turn) if spread else (row,), wave, lane).sink(arg=info)


def quantize_q8(
    x: Tensor, norm: tuple[Tensor, float] | None = None, rows: int | UOp | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantizes the rows of x (R, n) to int8 in groups of 32, after RMSNorm with `norm`'s weight
    and eps; only the first `rows` if given, which may be a bound variable.

    Returns the values packed four per int32 word, the scales d and the sums d * sum(q), each
    flattened row after row.
    """
    count, n = x.shape
    q = Tensor.empty(count * n // 4, dtype=dtypes.int32, device=x.device)
    d = Tensor.empty(count * n // GROUP, dtype=dtypes.float32, device=x.device)
    s = Tensor.empty(count * n // GROUP, dtype=dtypes.float32, device=x.device)
    x, rows = carry(x.float().contiguous(), count if rows is None else rows)
    weight = () if norm is None else (norm[0].float().contiguous(),)
    fxn = functools.partial(_quantize_q8_kernel, rows=rows, eps=0.0 if norm is None else norm[1])
    out = Tensor.custom_kernel(q, d, s, x, *weight, fxn=fxn)
    return out[0], out[1], out[2]
