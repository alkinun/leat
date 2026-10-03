"""Argmax over rows of logits, in two passes: warps over slices of each row, then one per row."""

import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo

from leat.nv.common import WARP, lane_range, on_nvidia, shfl_xor

PARTS = 256  # warps per row in the first pass


def _argmax_step(best: UOp, index: UOp, value: UOp, at: UOp) -> tuple[UOp, UOp]:
    # keep the larger value, and on ties the lower index, as argmax does
    take = (value > best) | (value.eq(best) & (at < index))
    return take.where(value, best), take.where(at, index)


def _warp_argmax(best: UOp, index: UOp) -> tuple[UOp, UOp]:
    for mask in (16, 8, 4, 2, 1):
        best, index = _argmax_step(best, index, shfl_xor(best, mask), shfl_xor(index, mask))
    return best, index


@functools.cache
def _argmax_partial_kernel(values: UOp, indices: UOp, x: UOp) -> UOp:
    # one warp per PARTS-th of a row; lanes stride over its slice so loads coalesce
    rows, n = (int(d) for d in x.shape)
    per = -(-n // PARTS)
    row, part = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(PARTS, 1, AxisType.GLOBAL)
    lane = lane_range()
    best, index = UOp.const(-math.inf, dtypes.float32), UOp.const(0, dtypes.int32)
    for k in range(-(-per // WARP)):
        offset = k * WARP + lane
        at = (part * per + offset).cast(dtypes.int32)
        live = (offset < per) & (at < n)
        value = live.where(x[row, at.minimum(n - 1)].load(), -math.inf)
        best, index = _argmax_step(best, index, value, at)
    best, index = _warp_argmax(best, index)
    first = part.valid(lane.eq(0))
    stores = (values[row, first].store(best), indices[row, first].store(index))
    info = KernelInfo(name="argmax_partial", opts_to_apply=())
    return UOp.group(*stores).end(row, part, lane).sink(arg=info)


@functools.cache
def _argmax_final_kernel(out: UOp, values: UOp, indices: UOp) -> UOp:
    row, lane = UOp.range(int(out.shape[0]), 0, AxisType.GLOBAL), lane_range()
    best, index = UOp.const(-math.inf, dtypes.float32), UOp.const(0, dtypes.int32)
    for k in range(PARTS // WARP):
        part = k * WARP + lane
        best, index = _argmax_step(best, index, values[row, part].load(), indices[row, part].load())
    _, index = _warp_argmax(best, index)
    info = KernelInfo(name="argmax_final", opts_to_apply=())
    return out[row.valid(lane.eq(0))].store(index).end(row, lane).sink(arg=info)


def supports_argmax(x: Tensor) -> bool:
    return on_nvidia(x) and x.ndim == 2 and all(isinstance(d, int) for d in x.shape)


def argmax(x: Tensor) -> Tensor:
    """Index of the largest value in each row of x (B, V), the first one on ties: (B, 1) int32."""
    rows = x.shape[0]
    values = Tensor.empty(rows, PARTS, dtype=dtypes.float32, device=x.device)
    indices = Tensor.empty(rows, PARTS, dtype=dtypes.int32, device=x.device)
    x = x.float().contiguous()
    values, indices = Tensor.custom_kernel(values, indices, x, fxn=_argmax_partial_kernel)[:2]
    out = Tensor.empty(rows, dtype=dtypes.int32, device=x.device)
    return Tensor.custom_kernel(out, values, indices, fxn=_argmax_final_kernel)[0].reshape(rows, 1)
