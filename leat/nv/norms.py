"""Residual adds of RMSNormed outputs, as Gemma 4's blocks end, in one kernel rather than
tinygrad's reduction and elementwise kernels for each norm."""

import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from leat.nv.common import WARP, carry, lane_range, on_nvidia, warp_sum

WARPS = 8  # per block, at most


@functools.cache
def _add_normed_kernel(
    out: UOp, x: UOp, *srcs: UOp, rows: int | UOp, eps: float, parts: int, normed: bool,
    scaled: bool,
) -> UOp:  # fmt: skip
    # A block per row: out = (x + the sum of rms_norm(part, its weight), normed again with the
    # next weight if normed) times the scale if scaled; srcs holds the parts and their weights,
    # then those. Threads take every so-many values, and the block sums squares in shared memory.
    dim = int(x.shape[1])
    warps = math.gcd(WARPS, dim // WARP)
    row = UOp.range(rows, 0, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(warps, 1, AxisType.LOCAL)
    ats = [i * warps * WARP + wave * WARP + lane for i in range(dim // (warps * WARP))]
    zero = UOp.const(0.0, dtypes.float32)
    partial = UOp.alloc((parts + normed, warps), dtypes.float32, addrspace=AddrSpace.LOCAL)

    def norm(values: list[UOp], weight: UOp, n: int) -> list[UOp]:  # the block's n-th norm
        square = warp_sum(sum((v * v for v in values), zero))
        squares = partial[n, wave.valid(lane.eq(0))].store(square)
        total = sum((partial.after(squares)[n, w].load() for w in range(warps)), zero)
        inv = (total / dim + eps).rsqrt()
        return [v * inv * weight[at].load() for v, at in zip(values, ats, strict=True)]

    total = [zero] * len(ats)
    for p in range(parts):
        part = norm([srcs[2 * p][row, at].load() for at in ats], srcs[2 * p + 1], p)
        total = [t + v for t, v in zip(total, part, strict=True)]
    if normed:
        total = norm(total, srcs[2 * parts], parts)
    scale = srcs[-1][0].load() if scaled else 1.0
    stores = [
        out[row, at].store((x[row, at].load() + t) * scale)
        for t, at in zip(total, ats, strict=True)
    ]
    info = KernelInfo(name=f"add_normed_{parts}", opts_to_apply=())
    return UOp.group(*stores).end(row, wave, lane).sink(arg=info)


def supports_add_normed(x: Tensor) -> bool:
    single = all(isinstance(b, int) and b == 1 for b in x.shape[:-2])
    return on_nvidia(x) and single and int(x.shape[-1]) % WARP == 0


def add_normed(
    x: Tensor, parts: list[tuple[Tensor, Tensor]], weight: Tensor | None, eps: float,
    scale: Tensor | None = None,
) -> Tensor:  # fmt: skip
    """(x + rms_norm of the sum of rms_norm(part, its weight) with `weight`, or just the sum where
    it is None) * scale, for tokens x (1, T, dim) and parts like it."""
    _, tokens, dim = x.shape
    count = x.max_shape[1]

    def rows(t: Tensor) -> Tensor:
        return t.reshape(tokens, dim).float().pad_to((count, dim)).contiguous()

    out = Tensor.empty(count, dim, dtype=dtypes.float32, device=x.device)
    srcs = [t for part, w in parts for t in (rows(part), w.float().contiguous())]
    srcs += [w.float().contiguous() for w in (weight, scale) if w is not None]
    x, bound = carry(rows(x), tokens)
    fxn = functools.partial(
        _add_normed_kernel, rows=bound, eps=eps, parts=len(parts), normed=weight is not None,
        scaled=scale is not None,
    )  # fmt: skip
    out = Tensor.custom_kernel(out, x, *srcs, fxn=fxn)[0]
    return out[:tokens].reshape(1, tokens, dim)
