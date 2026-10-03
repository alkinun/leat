"""NVIDIA kernels for DEV=NV and DEV=CUDA, written in tinygrad's UOp DSL and rendered as CUDA C.

Decode-time linear layers follow llama.cpp's matrix-vector scheme: the activation vector is
quantized to int8 in groups of 32, then each warp computes one output row with __dp4a over the
weights in their storage format.
"""

import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.quant import GGMLType, QTensor

WARP = 32
ROWS = 4  # output rows per block, one warp each
GROUP = 32  # activations per int8 scale


def _lane() -> UOp:
    # tinygrad never splits a WARP axis and maps it to threadIdx.x, so lane i is hardware lane i
    return UOp.range(WARP, 1, AxisType.WARP)


def _dp4a(a: UOp, b: UOp, acc: UOp) -> UOp:
    # acc + dot product of the four signed bytes of a and b
    return UOp(Ops.CUSTOMI, src=(a, b, acc), arg=("__dp4a({}, {}, {})", dtypes.int32))


def _shfl_xor(value: UOp, mask: int) -> UOp:
    # a statement, not an inline expression: every lane of the warp must execute it
    fmt = f"__shfl_xor_sync(0xffffffffu, {{0}}, {mask})"
    return UOp(Ops.CUSTOM, src=(value,), arg=(fmt, value.dtype))


def _warp_sum(value: UOp) -> UOp:
    for mask in (16, 8, 4, 2, 1):
        value = value + _shfl_xor(value, mask)
    return value


def _warp_max(value: UOp) -> UOp:
    for mask in (16, 8, 4, 2, 1):
        value = value.maximum(_shfl_xor(value, mask))
    return value


def _half(bits: UOp) -> UOp:
    # the low 16 bits of a word as an f16, widened to f32
    return (bits & 0xFFFF).cast(dtypes.uint16).bitcast(dtypes.float16).float()


@functools.cache
def _quantize_q8_kernel(q: UOp, d: UOp, s: UOp, x: UOp) -> UOp:
    # one warp per group of 32: d = max|x| / 127, q = round(x / d), s = d * sum(q), with q packed
    # four per int32 word so matrix kernels read it without a copy
    group = UOp.range(x.shape[0] // GROUP, 0, AxisType.GLOBAL)
    lane = _lane()
    value = x[group * GROUP + lane].load()
    # custom expressions keep both divisions exact: tinygrad would multiply by a reciprocal
    amax = _warp_max(value.maximum(-value))
    scale = UOp(Ops.CUSTOMI, src=(amax,), arg=("({}/127.0f)", dtypes.float32))
    rounded = UOp(Ops.CUSTOMI, src=(value, scale), arg=("roundf({}/{})", dtypes.float32))
    quant = (scale > 0).where(rounded, 0.0).cast(dtypes.int32)
    total = _warp_sum(quant)
    # each lane shifts its byte into place, then the four lanes of a word OR theirs together
    word = (quant & 0xFF).cast(dtypes.uint32) << ((lane % 4) * 8).cast(dtypes.uint32)
    for mask in (1, 2):
        word = word | _shfl_xor(word, mask)
    word_index = group * (GROUP // 4) + (lane // 4).valid((lane % 4).eq(0))
    stores = (
        q[word_index].store(word.bitcast(dtypes.int32)),
        d[group.valid(lane.eq(0))].store(scale),
        s[group.valid(lane.eq(0))].store(scale * total.float()),
    )
    return (
        UOp.group(*stores)
        .end(group, lane)
        .sink(arg=KernelInfo(name="quantize_q8", opts_to_apply=()))
    )


def quantize_q8(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Quantizes a vector to int8 in groups of 32.

    Returns the values packed four per int32 word, the scales d and the sums d * sum(q).
    """
    n = x.shape[0]
    q = Tensor.empty(n // 4, dtype=dtypes.int32, device=x.device)
    d = Tensor.empty(n // GROUP, dtype=dtypes.float32, device=x.device)
    s = Tensor.empty(n // GROUP, dtype=dtypes.float32, device=x.device)
    q, d, s = Tensor.custom_kernel(q, d, s, x.float().contiguous(), fxn=_quantize_q8_kernel)[:3]
    return q, d, s


def _k_scale_min(w: UOp, base: UOp, sub: UOp) -> tuple[UOp, UOp]:
    # ggml's get_scale_min_k4 over the 12 bytes after a K-quant block's d and dmin.
    # sub ^ 4 is sub - 4 where that branch is taken, and stays in range where it is not.
    def byte(i: UOp) -> UOp:
        return (w[base + 1 + i // 4].load() >> ((i % 4) * 8).cast(dtypes.uint32)) & 0xFF

    low = sub < 4
    sc = low.where(byte(sub) & 63, (byte(sub + 4) & 15) | ((byte(sub ^ 4) >> 6) << 4))
    mn = low.where(byte(sub + 4) & 63, (byte(sub + 4) >> 4) | ((byte(sub) >> 6) << 4))
    return sc.float(), mn.float()


@functools.cache
def _q4_k_kernel(out: UOp, w: UOp, xq: UOp, xd: UOp, xs: UOp) -> UOp:
    # Q4_K block, 36 words: d and dmin as f16, 12 bytes of 6-bit scales and mins, then 32 words of
    # nibbles. Sub-blocks 2j and 2j+1 are the low and high nibbles of words 4+8j .. 11+8j, so each
    # lane takes such a pair: 64 weights against 16 words of activations.
    rows, cols = out.shape[0], xq.shape[0] * 4
    blocks, pairs = cols // 256, cols // 64
    blk = UOp.range(rows // ROWS, 0, AxisType.GLOBAL)
    lane = _lane()
    wave = UOp.range(ROWS, 2, AxisType.LOCAL)
    row = blk * ROWS + wave
    acc = UOp.const(0.0, dtypes.float32)
    for it in range(pairs // WARP):
        pair = it * WARP + lane
        block, j = pair // 4, pair % 4
        base = (row * blocks + block) * 36
        dm = w[base].load()
        g = block * 8 + 2 * j  # activation group of the low sub-block; g + 1 is the high one
        dots = [UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)]
        for k in range(8):
            word = w[base + 4 + 8 * j + k].load()
            for h in range(2):
                weights = ((word >> (4 * h)) & 0x0F0F0F0F).bitcast(dtypes.int32)
                dots[h] = _dp4a(weights, xq[(g + h) * 8 + k].load(), dots[h])
        (sc0, m0), (sc1, m1) = _k_scale_min(w, base, 2 * j), _k_scale_min(w, base, 2 * j + 1)
        dot = sc0 * xd[g].load() * dots[0].float() + sc1 * xd[g + 1].load() * dots[1].float()
        mins = m0 * xs[g].load() + m1 * xs[g + 1].load()
        acc = acc + _half(dm) * dot - _half(dm >> 16) * mins
    store = out[row.valid(lane.eq(0))].store(_warp_sum(acc))
    return store.end(blk, wave, lane).sink(
        arg=KernelInfo(name=f"q4_k_{rows}_{cols}", opts_to_apply=())
    )


def _word16(w: UOp, i: UOp) -> UOp:
    # four bytes from a halfword-aligned buffer: 32-bit loads must be 4-byte aligned
    return w[i].load().cast(dtypes.uint32) | (w[i + 1].load().cast(dtypes.uint32) << 16)


def _minus_32(q: UOp) -> UOp:
    # Q6_K's unsigned 0..63 to signed -32..31, in each byte of a word
    centered = UOp(Ops.CUSTOMI, src=(q,), arg=("__vsubss4({}, 0x20202020u)", dtypes.uint32))
    return centered.bitcast(dtypes.int32)


@functools.cache
def _q6_k_kernel(out: UOp, w: UOp, xq: UOp, xd: UOp, xs: UOp) -> UOp:
    # Q6_K block, 105 halfwords: ql[128] low nibbles, qh[64] high 2-bit pairs, 16 int8 scales and d
    # as f16. Each half n of 128 weights is 4 rows of 32: row k takes nibble k // 2 of
    # ql[64n + 32(k % 2):][:32] and bits 2k of qh[32n:][:32], minus 32, with a scale per 16. A lane
    # takes rows k and k + 2 of a half, which share their ql bytes: 64 weights, as for Q4_K.
    rows, cols = out.shape[0], xq.shape[0] * 4
    blocks, units = cols // 256, cols // 64
    blk = UOp.range(rows // ROWS, 0, AxisType.GLOBAL)
    lane = _lane()
    wave = UOp.range(ROWS, 2, AxisType.LOCAL)
    row = blk * ROWS + wave
    acc = UOp.const(0.0, dtypes.float32)
    for it in range(units // WARP):
        unit = it * WARP + lane
        block, n, k = unit // 4, unit % 4 // 2, unit % 2
        base = (row * blocks + block) * 105
        g = block * 8 + 4 * n + k  # activation group of row k; row k + 2 is group g + 2
        dots = [[UOp.const(0, dtypes.int32)] * 2 for _ in range(2)]  # [row k, k + 2][16 weights]
        for m in range(8):
            ql = _word16(w, base + 32 * n + 16 * k + 2 * m)
            qh = _word16(w, base + 64 + 16 * n + 2 * m)
            for r in range(2):
                high = (qh >> (2 * k + 4 * r).cast(dtypes.uint32)) & 0x03030303
                q = ((ql >> (4 * r)) & 0x0F0F0F0F) | (high << 4)
                dots[r][m // 4] = _dp4a(
                    _minus_32(q), xq[(g + 2 * r) * 8 + m].load(), dots[r][m // 4]
                )
        total = UOp.const(0.0, dtypes.float32)
        for r in range(2):
            scales = w[base + 96 + 4 * n + k + 2 * r].load()  # both scales of row k + 2r
            for h in range(2):
                sc = ((scales >> (8 * h)) & 0xFF).cast(dtypes.uint8).bitcast(dtypes.int8).float()
                total = total + xd[g + 2 * r].load() * sc * dots[r][h].float()
        acc = acc + _half(w[base + 104].load().cast(dtypes.uint32)) * total
    store = out[row.valid(lane.eq(0))].store(_warp_sum(acc))
    return store.end(blk, wave, lane).sink(
        arg=KernelInfo(name=f"q6_k_{rows}_{cols}", opts_to_apply=())
    )


# each kernel, and the word type it reads the weights as
_KERNELS = {
    GGMLType.Q4_K: (_q4_k_kernel, dtypes.uint32),
    GGMLType.Q6_K: (_q6_k_kernel, dtypes.uint16),
}


def supports(x: Tensor, w: QTensor) -> bool:
    if not isinstance(x.device, str) or x.device.split(":")[0] not in ("NV", "CUDA"):
        return False
    # one token, and whole warps: each lane takes pairs of sub-blocks, 32 lanes per row
    rows, cols = w.shape
    single = isinstance(x.numel(), int) and x.numel() == cols
    fits = (
        isinstance(cols, int)
        and cols % (64 * WARP) == 0
        and isinstance(rows, int)
        and rows % ROWS == 0
    )
    return w.type in _KERNELS and single and fits


def linears(x: Tensor, *ws: QTensor) -> list[Tensor]:
    """x @ w.T for one token and each w, with the activations quantized to int8 once."""
    xq, xd, xs = quantize_q8(x.reshape(x.shape[-1]))
    return [_matvec(w, xq, xd, xs).reshape(*x.shape[:-1], w.shape[0]) for w in ws]


def _matvec(w: QTensor, xq: Tensor, xd: Tensor, xs: Tensor) -> Tensor:
    out = Tensor.empty(w.shape[0], dtype=dtypes.float32, device=xq.device)
    kernel, word = _KERNELS[w.type]
    # .contiguous() on a bitcast of contiguous storage is a view; without it tinygrad copies
    words = w.data.flatten().bitcast(word).contiguous()
    return Tensor.custom_kernel(out, words, xq, xd, xs, fxn=kernel)[0]
