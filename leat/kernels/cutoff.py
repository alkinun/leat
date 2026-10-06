"""Where top-k and then top-p cut each row of scores, read off a histogram of the row's scores in
BINS steps of STEP below its top score, each step's probability taken as that of its middle.

Two kernels: blocks over each row count their scores within RANGE of their own top score, in
steps of one grid, STEP-wide from 0; then a block per row adds their counts up, aligned on the
grid, and sums them from the top down to find the cuts."""

import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.kernels.common import (
    LOG2E,
    WARP,
    block_sum,
    either,
    lane_range,
    on_gpu,
    shfl_xor,
    tree,
    warp_max,
    warp_sum,
)

RANGE = 20.0  # below the top score, that the histogram spans: a token below is e^-20 as likely
BINS = 2048  # steps of STEP, 0.01: the cuts land within one below where they would exactly
STEP = RANGE / BINS
PARTS = 32  # blocks per row that count its scores
COUNT_WARPS = 8  # per block that counts
CUT_WARPS = 32  # per block that cuts


def _grid(score: UOp) -> UOp:
    # the step of the grid that holds a score
    return (score * (1 / STEP)).floor()


@functools.cache
def _histogram_kernel(counts: UOp, tops: UOp, scores: UOp) -> UOp:
    # A block per row and PARTS-th of it, its threads taking every so-many scores: the block's top
    # score, then the count of its scores in each of the BINS steps of the grid up to the top's
    rows, n = (int(d) for d in scores.shape)
    per = -(-n // PARTS)
    row, part = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(PARTS, 1, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(COUNT_WARPS, 2, AxisType.LOCAL)
    thread, threads = wave * WARP + lane, COUNT_WARPS * WARP
    offsets = [k * threads + thread for k in range(-(-per // threads))]
    ats = [part * per + offset for offset in offsets]
    values = [
        ((offset < per) & (at < n)).where(scores[row, at.minimum(n - 1)].load(), -math.inf)
        for offset, at in zip(offsets, ats, strict=True)
    ]
    top = block_sum(warp_max(tree(UOp.maximum, values)), wave, lane, UOp.maximum)
    first = _grid(top) - (BINS - 1)  # the grid's step of the histogram's first
    hist = UOp.alloc((BINS,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    zero, one = UOp.const(0, dtypes.uint32), UOp.const(1, dtypes.uint32)
    hist = hist.after(*(hist[i * threads + thread].store(zero) for i in range(BINS // threads)))
    added = []
    for value in values:
        step = _grid(value) - first
        added.append(_add(hist[step.maximum(0).cast(dtypes.int32)], one, step >= 0))
    hist = hist.after(UOp(Ops.BARRIER, src=tuple(added)))
    stores = [
        counts[row, part, i * threads + thread].store(hist[i * threads + thread].load())
        for i in range(BINS // threads)
    ]
    stores.append(tops[row, part.valid(thread.eq(0))].store(top))
    info = KernelInfo(name="cutoff_histogram", opts_to_apply=())
    return UOp.group(*stores).end(row, part, wave, lane).sink(arg=info)


@functools.cache
def _cut_kernel(out: UOp, counts: UOp, tops: UOp, options: UOp) -> UOp:
    # A block per row: the parts' counts added up in shared memory, each part's shifted down by
    # the steps between its top's and the row's top's. Each thread then takes BINS // threads
    # steps in turn, summing the counts and probabilities at or above each from the top down.
    # Top-k cuts below the last step whose count reaches k, top-p the last whose probability
    # reaches top_p of the top-k steps'.
    rows, parts = (int(d) for d in counts.shape[:2])
    row = UOp.range(rows, 0, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(CUT_WARPS, 1, AxisType.LOCAL)
    thread, threads = wave * WARP + lane, CUT_WARPS * WARP
    part_tops = [tops[row, p].load() for p in range(parts)]
    top = tree(UOp.maximum, part_tops)
    last = _grid(top)
    # at most BINS, past which a part adds nothing: a part of no scores, as a short row's last,
    # has a top of -inf, whose shift of +inf no int holds
    shifts = [(last - _grid(t)).minimum(BINS).cast(dtypes.int32) for t in part_tops]
    merged = UOp.alloc((BINS,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    sums = []
    for i in range(BINS // threads):
        step = i * threads + thread
        shifted = [(step + s, step + s < BINS) for s in shifts]
        parted = [
            ok.where(counts[row, p, at.minimum(BINS - 1)].load(), 0)
            for p, (at, ok) in enumerate(shifted)
        ]
        sums.append(merged[step].store(tree(UOp.__add__, parted).cast(dtypes.float32)))
    merged = merged.after(*sums)
    per = BINS // threads
    steps = [thread * per + i for i in range(per)]
    tallies = [merged[s].load() for s in steps]
    # each step's probability relative to the top score's, as if its scores sat at its middle
    middles = [(last - (BINS - 1) + s.cast(dtypes.float32) + 0.5) * STEP - top for s in steps]
    masses = [t * (m * LOG2E).exp2() for t, m in zip(tallies, middles, strict=True)]
    above_counts = _above(tallies, wave, lane)
    above_masses = _above(masses, wave, lane)

    def cut(above: list[UOp], target: UOp) -> UOp:  # the last step whose sum reaches target
        reached = tree(UOp.__add__, [(a >= target).cast(dtypes.float32) for a in above])
        return (block_sum(warp_sum(reached), wave, lane) - 1).maximum(0.0)

    k, p = options[row, 0].load(), options[row, 1].load()
    by_k = cut(above_counts, (k > 0).where(k, math.inf))  # 0 keeps every token
    picked = [
        s.cast(dtypes.float32).eq(by_k).where(m, 0.0)
        for s, m in zip(steps, above_masses, strict=True)
    ]
    kept = block_sum(warp_sum(tree(UOp.__add__, picked)), wave, lane)
    by_p = cut(above_masses, p * kept)
    first = (last - (BINS - 1)) * STEP
    cuts = (top, first + by_k * STEP, first + by_p * STEP)
    stores = [out[row.valid(thread.eq(0)), i].store(v) for i, v in enumerate(cuts)]
    info = KernelInfo(name="cutoff", opts_to_apply=())
    return UOp.group(*stores).end(row, wave, lane).sink(arg=info)


def _above(values: list[UOp], wave: UOp, lane: UOp) -> list[UOp]:
    # for each of a thread's values, in steps that its threads take in turn: the sum of it, those
    # after it, and all those of the threads after it
    sums, total = [], UOp.const(0.0, dtypes.float32)
    for v in reversed(values):
        total = total + v
        sums.append(total)
    sums.reverse()
    after, mask = UOp.const(0.0, dtypes.float32), 1
    while mask < WARP:  # the lanes after this one, in the warp
        other = shfl_xor(total, mask)
        after = (lane & mask).eq(0).where(after + other, after)
        total, mask = total + other, mask * 2
    warps = UOp.alloc((CUT_WARPS,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    warps = warps.after(warps[wave.valid(lane.eq(0))].store(total))
    later = [(wave < w).where(warps[w].load(), 0.0) for w in range(CUT_WARPS)]
    after = after + tree(UOp.__add__, later)
    return [s + after for s in sums]


def _add(at: UOp, value: UOp, gate: UOp) -> UOp:
    # an atomic add of value to shared memory at an index, where gate holds: a statement, which a
    # barrier must take as a source, as tinygrad drops ops whose value nothing uses
    add = either(
        "({2} ? atomicAdd({0}, {1}) : 0u)",
        "({2} ? __hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)"
        " : 0u)",
    )
    return UOp(Ops.CUSTOM, src=(at, value, gate), arg=(add + ";", dtypes.void))


def supports_cutoff(scores: Tensor) -> bool:
    # rows of a known number of scores
    return on_gpu(scores) and scores.ndim == 2 and all(isinstance(d, int) for d in scores.shape)


def cutoff(scores: Tensor, top_k: Tensor, top_p: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ops.cutoff()'s cuts for scores (B, V), each option (B, 1): each row's top score, where top_k
    cuts, and where top_p then does."""
    rows, device = scores.shape[0], scores.device
    scores, options = scores.float().contiguous(), top_k.cat(top_p, dim=1).float().contiguous()
    counts = Tensor.empty(rows, PARTS, BINS, dtype=dtypes.uint32, device=device)
    tops = Tensor.empty(rows, PARTS, dtype=dtypes.float32, device=device)
    counts, tops = Tensor.custom_kernel(counts, tops, scores, fxn=_histogram_kernel)[:2]
    out = Tensor.empty(rows, 3, dtype=dtypes.float32, device=device)
    out = Tensor.custom_kernel(out, counts, tops, options, fxn=_cut_kernel)[0]
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]
