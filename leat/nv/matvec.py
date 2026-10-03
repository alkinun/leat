"""Linear layers for one token, llama.cpp's MMVQ: the activation vector is quantized to int8, then
a warp computes each output row with __dp4a over the weights in their storage format."""

import functools
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo

from leat.nv.common import (
    WARP,
    WORD_TYPE,
    dp4a,
    f16,
    lane_range,
    minus_32,
    on_nvidia,
    silu,
    storage_words,
    warp_sum,
    word16,
)
from leat.nv.quantize import quantize_q8
from leat.quant import GGMLType, QTensor

Dot = Callable[[UOp, UOp], UOp]  # (row, unit) -> a unit's share of one row's dot product


def _rows(out: UOp, cols: int, name: str, dots: list[Dot], combine: Callable[..., UOp]) -> UOp:
    # one block of one warp per output row: lanes take units of 64 weights in turn, the warp sums
    # each dot product, and out[row] = combine(row, *sums). Grouping rows into wider blocks
    # measured slower on the 3090, by up to a quarter for Q6_K.
    rows = out.shape[0]
    row, lane = UOp.range(rows, 0, AxisType.GLOBAL), lane_range()
    units = range(cols // 64 // WARP)
    zero = UOp.const(0.0, dtypes.float32)
    sums = [warp_sum(sum((dot(row, it * WARP + lane) for it in units), zero)) for dot in dots]
    store = out[row.valid(lane.eq(0))].store(combine(row, *sums))
    info = KernelInfo(name=f"{name}_{rows}_{cols}", opts_to_apply=())
    return store.end(row, lane).sink(arg=info)


def _k_scale_min(w: UOp, base: UOp, sub: UOp) -> tuple[UOp, UOp]:
    # ggml's get_scale_min_k4 over the 12 bytes after a K-quant block's d and dmin.
    # sub ^ 4 is sub - 4 where that branch is taken, and stays in range where it is not.
    def byte(i: UOp) -> UOp:
        return (w[base + 1 + i // 4].load() >> ((i % 4) * 8).cast(dtypes.uint32)) & 0xFF

    low = sub < 4
    sc = low.where(byte(sub) & 63, (byte(sub + 4) & 15) | ((byte(sub ^ 4) >> 6) << 4))
    mn = low.where(byte(sub + 4) & 63, (byte(sub + 4) >> 4) | ((byte(sub) >> 6) << 4))
    return sc.float(), mn.float()


def _q4_k_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp) -> Dot:
    # Q4_K block, 36 words: d and dmin as f16, 12 bytes of 6-bit scales and mins, then 32 words of
    # nibbles. Sub-blocks 2j and 2j+1 are the low and high nibbles of words 4+8j .. 11+8j, so a
    # unit is such a pair: 64 weights against 16 words of activations.
    cols = 4 * int(xq.shape[0])

    def dot(row: UOp, pair: UOp) -> UOp:
        block, j = pair // 4, pair % 4
        base = (row * (cols // 256) + block) * 36
        dm = w[base].load()
        g = block * 8 + 2 * j  # activation group of the low sub-block; g + 1 is the high one
        dots = [UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)]
        for k in range(8):
            word = w[base + 4 + 8 * j + k].load()
            for h in range(2):
                weights = ((word >> (4 * h)) & 0x0F0F0F0F).bitcast(dtypes.int32)
                dots[h] = dp4a(weights, xq[(g + h) * 8 + k].load(), dots[h])
        (sc0, m0), (sc1, m1) = _k_scale_min(w, base, 2 * j), _k_scale_min(w, base, 2 * j + 1)
        scaled = sc0 * xd[g].load() * dots[0].float() + sc1 * xd[g + 1].load() * dots[1].float()
        mins = m0 * xs[g].load() + m1 * xs[g + 1].load()
        return f16(dm) * scaled - f16(dm >> 16) * mins

    return dot


def _q6_k_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp) -> Dot:
    # Q6_K block, 105 halfwords: ql[128] low nibbles, qh[64] high 2-bit pairs, 16 int8 scales and d
    # as f16. Each half n of 128 weights is 4 rows of 32: row k takes nibble k // 2 of
    # ql[64n + 32(k % 2):][:32] and bits 2k of qh[32n:][:32], minus 32, with a scale per 16. A unit
    # is rows k and k + 2 of a half, which share their ql bytes: 64 weights, as for Q4_K.
    cols = 4 * int(xq.shape[0])

    def dot(row: UOp, unit: UOp) -> UOp:
        block, n, k = unit // 4, unit % 4 // 2, unit % 2
        base = (row * (cols // 256) + block) * 105
        g = block * 8 + 4 * n + k  # activation group of row k; row k + 2 is group g + 2
        dots = [[UOp.const(0, dtypes.int32)] * 2 for _ in range(2)]  # [row k, k + 2][16 weights]
        for m in range(8):
            ql = word16(w, base + 32 * n + 16 * k + 2 * m)
            qh = word16(w, base + 64 + 16 * n + 2 * m)
            for r in range(2):
                high = (qh >> (2 * k + 4 * r).cast(dtypes.uint32)) & 0x03030303
                q = minus_32(((ql >> (4 * r)) & 0x0F0F0F0F) | (high << 4))
                dots[r][m // 4] = dp4a(q, xq[(g + 2 * r) * 8 + m].load(), dots[r][m // 4])
        total = UOp.const(0.0, dtypes.float32)
        for r in range(2):
            scales = w[base + 96 + 4 * n + k + 2 * r].load()  # both scales of row k + 2r
            for h in range(2):
                sc = ((scales >> (8 * h)) & 0xFF).cast(dtypes.uint8).bitcast(dtypes.int8).float()
                total = total + xd[g + 2 * r].load() * sc * dots[r][h].float()
        return f16(w[base + 104].load().cast(dtypes.uint32)) * total

    return dot


_DOTS = {GGMLType.Q4_K: _q4_k_dot, GGMLType.Q6_K: _q6_k_dot}


@functools.cache
def _matvec_kernel(
    out: UOp, w: UOp, xq: UOp, xd: UOp, xs: UOp, *residual: UOp, ggml_type: GGMLType
) -> UOp:
    def combine(row: UOp, total: UOp) -> UOp:
        return total + residual[0][row].load() if residual else total

    dot = _DOTS[ggml_type](w, xq, xd, xs)
    return _rows(out, 4 * int(xq.shape[0]), ggml_type.name.lower(), [dot], combine)


@functools.cache
def _swiglu_kernel(
    out: UOp, gate: UOp, up: UOp, xq: UOp, xd: UOp, xs: UOp, ggml_type: GGMLType
) -> UOp:
    def combine(row: UOp, g: UOp, u: UOp) -> UOp:
        return silu(g) * u

    dots = [_DOTS[ggml_type](w, xq, xd, xs) for w in (gate, up)]
    return _rows(out, 4 * int(xq.shape[0]), f"swiglu_{ggml_type.name.lower()}", dots, combine)


def supports_matvec(x: Tensor, w: QTensor) -> bool:
    # one token, and whole warps: each lane takes pairs of sub-blocks, 32 lanes per row
    one = isinstance(x.numel(), int) and x.numel() == x.shape[-1]
    return on_nvidia(x) and one and w.type in WORD_TYPE and w.shape[1] % (64 * WARP) == 0


def matvecs(
    x: Tensor,
    *ws: QTensor,
    norm: tuple[Tensor, float] | None = None,
    residual: Tensor | None = None,
) -> list[Tensor]:
    """x @ w.T for one token and each w, with the activations quantized to int8 once, after
    rms_norm(x, *norm) if given. `residual` is added inside the kernel; it needs a single w."""
    assert residual is None or len(ws) == 1, "a residual goes with one matrix"
    xq, xd, xs = quantize_q8(x.reshape(1, x.shape[-1]), norm)
    res = () if residual is None else (residual.reshape(ws[0].shape[0]).float().contiguous(),)
    outs = []
    for w in ws:
        out = Tensor.empty(w.shape[0], dtype=dtypes.float32, device=x.device)
        fxn = functools.partial(_matvec_kernel, ggml_type=w.type)
        out = Tensor.custom_kernel(out, storage_words(w), xq, xd, xs, *res, fxn=fxn)[0]
        outs.append(out.reshape(*x.shape[:-1], w.shape[0]))
    return outs


def swiglu(
    x: Tensor, gate: QTensor, up: QTensor, norm: tuple[Tensor, float] | None = None
) -> Tensor:
    """silu(x @ gate.T) * (x @ up.T) for one token, after rms_norm(x, *norm) if given, both
    matrices in one kernel; they share a type and shape."""
    xq, xd, xs = quantize_q8(x.reshape(1, x.shape[-1]), norm)
    out = Tensor.empty(gate.shape[0], dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(_swiglu_kernel, ggml_type=gate.type)
    words = storage_words(gate), storage_words(up)
    out = Tensor.custom_kernel(out, *words, xq, xd, xs, fxn=fxn)[0]
    return out.reshape(*x.shape[:-1], gate.shape[0])
