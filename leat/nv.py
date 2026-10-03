"""NVIDIA kernels for DEV=NV and DEV=CUDA, written in tinygrad's UOp DSL and rendered as CUDA C.

Decode-time linear layers follow llama.cpp's matrix-vector scheme: the activation vector is
quantized to int8 in groups of 32, then each warp computes one output row with __dp4a over the
weights in their storage format.
"""

import functools
import itertools
import math
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.quant import GGMLType, QTensor

WARP = 32
GROUP = 32  # activations per int8 scale
LOG2E = math.log2(math.e)


def _lane() -> UOp:
    # tinygrad never splits a WARP axis and maps it to threadIdx.x, so lane i is hardware lane i
    return UOp.range(WARP, -1, AxisType.WARP)


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


def _silu(x: UOp) -> UOp:
    return x * (1 + (x * -LOG2E).exp2()).reciprocal()  # as tinygrad's silu


def _half(bits: UOp) -> UOp:
    # the low 16 bits of a word as an f16, widened to f32
    return (bits & 0xFFFF).cast(dtypes.uint16).bitcast(dtypes.float16).float()


def _quantize_group(q: UOp, d: UOp, s: UOp, group: UOp, lane: UOp, value: UOp) -> list[UOp]:
    # one warp quantizes a group of 32 values, one per lane: d = max|x| / 127, q = round(x / d),
    # s = d * sum(q), with q packed four per int32 word so matrix kernels read it without a copy
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
    return [
        q[word_index].store(word.bitcast(dtypes.int32)),
        d[group.valid(lane.eq(0))].store(scale),
        s[group.valid(lane.eq(0))].store(scale * total.float()),
    ]


QUANT_WARPS = 8  # per block of the quantization kernel, each quantizing a group at a time


@functools.cache
def _quantize_q8_kernel(
    q: UOp, d: UOp, s: UOp, x: UOp, *weight: UOp, rows: int | UOp, eps: float, gated: bool
) -> UOp:
    # Quantization, after RMSNorm when given its weight, or of silu(gate) * up from rows holding
    # gate then up when gated. Each row takes a block, whose warps quantize one group after another.
    # A single row, the vector of a decode step, spreads over blocks of a group per warp instead:
    # on the 3090 that takes 3 us for 4096 values against 5 us for one block, though each block
    # sums the squares of the whole row, from L2 after the first.
    n, warps = int(x.shape[1]) // (2 if gated else 1), QUANT_WARPS
    groups = n // GROUP
    spread = isinstance(rows, int) and rows == 1
    row = UOp.range(rows, 0, AxisType.GLOBAL)
    lane, wave = _lane(), UOp.range(warps, 2, AxisType.LOCAL)
    turn = UOp.range(groups // warps, 1, AxisType.GLOBAL if spread else AxisType.LOOP)
    scale = UOp.const(1.0, dtypes.float32)
    if weight:
        thread, threads, zero = wave * WARP + lane, warps * WARP, UOp.const(0.0, dtypes.float32)
        squares = sum((x[row, i * threads + thread].load() ** 2 for i in range(n // threads)), zero)
        partial = UOp.alloc((warps,), dtypes.float32, addrspace=AddrSpace.LOCAL)
        partial = partial.after(partial[wave.valid(lane.eq(0))].store(_warp_sum(squares)))
        scale = (sum((partial[w].load() for w in range(warps)), zero) / n + eps).rsqrt()
    group = turn * warps + wave
    at = group * GROUP + lane
    value = x[row, at].load() * scale * (weight[0][at].load() if weight else 1.0)
    if gated:
        value = _silu(value) * x[row, n + at].load()
    stores = UOp.group(*_quantize_group(q, d, s, row * groups + group, lane, value))
    if not spread:
        stores = stores.end(turn)
    info = KernelInfo(name="norm_quantize_q8" if weight else "quantize_q8", opts_to_apply=())
    return stores.end(*(row, turn) if spread else (row,), wave, lane).sink(arg=info)


def quantize_q8(
    x: Tensor,
    norm: tuple[Tensor, float] | None = None,
    rows: int | UOp | None = None,
    gated: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantizes the rows of x (R, n) to int8 in groups of 32, after RMSNorm with `norm`'s weight
    and eps; only the first `rows` if given, which may be a bound variable. With `gated`, rows
    hold gate then up, and silu(gate) * up is quantized.

    Returns the values packed four per int32 word, the scales d and the sums d * sum(q), each
    flattened row after row.
    """
    count, n = x.shape[0], x.shape[1] // (2 if gated else 1)
    q = Tensor.empty(count * n // 4, dtype=dtypes.int32, device=x.device)
    d = Tensor.empty(count * n // GROUP, dtype=dtypes.float32, device=x.device)
    s = Tensor.empty(count * n // GROUP, dtype=dtypes.float32, device=x.device)
    x, rows = _carry(x.float().contiguous(), count if rows is None else rows)
    weight = () if norm is None else (norm[0].float().contiguous(),)
    eps = 0.0 if norm is None else norm[1]
    fxn = functools.partial(_quantize_q8_kernel, rows=rows, eps=eps, gated=gated)
    out = Tensor.custom_kernel(q, d, s, x, *weight, fxn=fxn)
    return out[0], out[1], out[2]


def _carry(t: Tensor, value: int | UOp) -> tuple[Tensor, int | UOp]:
    # a bound variable rides on one of the kernel's buffers, so the kernel's own copy stays unbound
    if isinstance(value, UOp):
        return Tensor(t.uop.after(value)), value.unbind_all()[0]
    return t, value


def _k_scale_min(w: UOp, base: UOp, sub: UOp) -> tuple[UOp, UOp]:
    # ggml's get_scale_min_k4 over the 12 bytes after a K-quant block's d and dmin.
    # sub ^ 4 is sub - 4 where that branch is taken, and stays in range where it is not.
    def byte(i: UOp) -> UOp:
        return (w[base + 1 + i // 4].load() >> ((i % 4) * 8).cast(dtypes.uint32)) & 0xFF

    low = sub < 4
    sc = low.where(byte(sub) & 63, (byte(sub + 4) & 15) | ((byte(sub ^ 4) >> 6) << 4))
    mn = low.where(byte(sub + 4) & 63, (byte(sub + 4) >> 4) | ((byte(sub) >> 6) << 4))
    return sc.float(), mn.float()


Dot = Callable[[UOp, UOp], UOp]  # (row, unit) -> a unit's share of one row's dot product


def _rows(out: UOp, cols: int, name: str, dots: list[Dot], combine: Callable[..., UOp]) -> UOp:
    # one block of one warp per output row: lanes take units of 64 weights in turn, the warp sums
    # each dot product, and out[row] = combine(row, *sums). Grouping rows into wider blocks
    # measured slower on the 3090, by up to a quarter for Q6_K.
    rows = out.shape[0]
    row, lane = UOp.range(rows, 0, AxisType.GLOBAL), _lane()
    units = range(cols // 64 // WARP)
    zero = UOp.const(0.0, dtypes.float32)
    sums = [_warp_sum(sum((dot(row, it * WARP + lane) for it in units), zero)) for dot in dots]
    store = out[row.valid(lane.eq(0))].store(combine(row, *sums))
    info = KernelInfo(name=f"{name}_{rows}_{cols}", opts_to_apply=())
    return store.end(row, lane).sink(arg=info)


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
                dots[h] = _dp4a(weights, xq[(g + h) * 8 + k].load(), dots[h])
        (sc0, m0), (sc1, m1) = _k_scale_min(w, base, 2 * j), _k_scale_min(w, base, 2 * j + 1)
        scaled = sc0 * xd[g].load() * dots[0].float() + sc1 * xd[g + 1].load() * dots[1].float()
        mins = m0 * xs[g].load() + m1 * xs[g + 1].load()
        return _half(dm) * scaled - _half(dm >> 16) * mins

    return dot


def _word16(w: UOp, i: UOp) -> UOp:
    # four bytes from a halfword-aligned buffer: 32-bit loads must be 4-byte aligned
    return w[i].load().cast(dtypes.uint32) | (w[i + 1].load().cast(dtypes.uint32) << 16)


def _minus_32(q: UOp) -> UOp:
    # Q6_K's unsigned 0..63 to signed -32..31, in each byte of a word: b + 96 stays in its byte,
    # and flipping its top bit makes it b - 32 in two's complement
    return ((q + 0x60606060) ^ 0x80808080).bitcast(dtypes.int32)


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
            ql = _word16(w, base + 32 * n + 16 * k + 2 * m)
            qh = _word16(w, base + 64 + 16 * n + 2 * m)
            for r in range(2):
                high = (qh >> (2 * k + 4 * r).cast(dtypes.uint32)) & 0x03030303
                q = _minus_32(((ql >> (4 * r)) & 0x0F0F0F0F) | (high << 4))
                dots[r][m // 4] = _dp4a(q, xq[(g + 2 * r) * 8 + m].load(), dots[r][m // 4])
        total = UOp.const(0.0, dtypes.float32)
        for r in range(2):
            scales = w[base + 96 + 4 * n + k + 2 * r].load()  # both scales of row k + 2r
            for h in range(2):
                sc = ((scales >> (8 * h)) & 0xFF).cast(dtypes.uint8).bitcast(dtypes.int8).float()
                total = total + xd[g + 2 * r].load() * sc * dots[r][h].float()
        return _half(w[base + 104].load().cast(dtypes.uint32)) * total

    return dot


# per type, the unit dot product and the word type it reads the weights as
_DOTS = {
    GGMLType.Q4_K: (_q4_k_dot, dtypes.uint32),
    GGMLType.Q6_K: (_q6_k_dot, dtypes.uint16),
}


@functools.cache
def _matvec_kernel(
    out: UOp, w: UOp, xq: UOp, xd: UOp, xs: UOp, *residual: UOp, ggml_type: GGMLType
) -> UOp:
    def combine(row: UOp, total: UOp) -> UOp:
        return total + residual[0][row].load() if residual else total

    dot = _DOTS[ggml_type][0](w, xq, xd, xs)
    return _rows(out, 4 * int(xq.shape[0]), ggml_type.name.lower(), [dot], combine)


@functools.cache
def _swiglu_kernel(
    out: UOp, gate: UOp, up: UOp, xq: UOp, xd: UOp, xs: UOp, ggml_type: GGMLType
) -> UOp:
    def combine(row: UOp, g: UOp, u: UOp) -> UOp:
        return _silu(g) * u

    dots = [_DOTS[ggml_type][0](w, xq, xd, xs) for w in (gate, up)]
    return _rows(out, 4 * int(xq.shape[0]), f"swiglu_{ggml_type.name.lower()}", dots, combine)


def supports(x: Tensor, w: QTensor) -> bool:
    if not isinstance(x.device, str) or x.device.split(":")[0] not in ("NV", "CUDA"):
        return False
    rows, cols = w.shape
    if w.type not in _DOTS or not isinstance(cols, int) or not isinstance(rows, int):
        return False
    if _one_token(x):
        # one token, and whole warps: each lane takes pairs of sub-blocks, 32 lanes per row
        return cols % (64 * WARP) == 0  # so also of GROUP * QUANT_WARPS
    # several tokens, in one sequence: whole tiles of rows, and whole steps along a row
    batch = x.shape[:-2]
    single = all(isinstance(b, int) and b == 1 for b in batch)
    return single and rows % 128 == 0 and cols % (GROUP * QUANT_WARPS) == 0


def linears(
    x: Tensor,
    *ws: QTensor,
    norm: tuple[Tensor, float] | None = None,
    residual: Tensor | None = None,
) -> list[Tensor]:
    """x @ w.T for each w, with the activations quantized to int8 once.

    `residual` is added to the result inside the kernel; it needs a single w.
    """
    assert residual is None or len(ws) == 1, "a residual goes with one matrix"
    if not _one_token(x):
        return _matmuls(x, ws, norm, residual)
    xq, xd, xs = quantize_q8(x.reshape(1, x.shape[-1]), norm)
    res = () if residual is None else (residual.reshape(ws[0].shape[0]).float().contiguous(),)
    outs = []
    for w in ws:
        out = Tensor.empty(w.shape[0], dtype=dtypes.float32, device=x.device)
        fxn = functools.partial(_matvec_kernel, ggml_type=w.type)
        out = Tensor.custom_kernel(out, _words(w), xq, xd, xs, *res, fxn=fxn)[0]
        outs.append(out.reshape(*x.shape[:-1], w.shape[0]))
    return outs


def swiglu(
    x: Tensor, gate: QTensor, up: QTensor, norm: tuple[Tensor, float] | None = None
) -> Tensor:
    """silu(x @ gate.T) * (x @ up.T); for one token, both matrices in one kernel."""
    if not _one_token(x):
        g, u = _matmuls(x, (gate, up), norm, None)
        return g.silu() * u
    xq, xd, xs = quantize_q8(x.reshape(1, x.shape[-1]), norm)
    out = Tensor.empty(gate.shape[0], dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(_swiglu_kernel, ggml_type=gate.type)
    out = Tensor.custom_kernel(out, _words(gate), _words(up), xq, xd, xs, fxn=fxn)[0]
    return out.reshape(*x.shape[:-1], gate.shape[0])


def _one_token(x: Tensor) -> bool:
    return isinstance(x.numel(), int) and x.numel() == x.shape[-1]


def _words(w: QTensor) -> Tensor:
    # .contiguous() on a bitcast of contiguous storage is a view; without it tinygrad copies
    return w.data.flatten().bitcast(_DOTS[w.type][1]).contiguous()


# ******** matrix products: several tokens on int8 tensor cores ********
# llama.cpp's MMQ scheme. The activations are quantized as for decode. A block stages a tile of
# weights, unpacked to int8, and 64 tokens in shared memory, half a weight block per step, while
# it fetches the next step into registers. Each of its warps multiplies 32 rows by the 64 tokens
# on tensor cores and scales every group of 32 in f32. Tiles of 256 rows reach 68 to 79 TOPS on
# 512 tokens, against 47 to 61 for 128; matrices with few rows take 128, to occupy more SMs.

TILE_TOKENS = 64
WARP_ROWS = 32  # a warp per 32 rows of the tile, and a thread per row to load its scales
STEP = 128  # weights per row per step
SUBTILES_M, SUBTILES_N = WARP_ROWS // 16, TILE_TOKENS // 8  # of 16 x 8 per warp

# mma.sync on int8: a 16 x k tile of weights times a k x 8 tile of activations, for k = 32 or 16.
# {0} points at 4 int32 registers for the lane's share of the result, then come the lane's 4 or 2
# words of weights and its 2 or 1 of activations. A CUSTOM op has one value, so the statement
# returns the first result and writes the other three through the pointer.
_MMA = {
    32: '[&]{{ int c0, c1, c2, c3; asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 '
    '{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%10,%10,%10,%10}};" '
    ': "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) '
    ': "r"({1}), "r"({2}), "r"({3}), "r"({4}), "r"({5}), "r"({6}), "r"(0)); '
    "{0}[1] = c1; {0}[2] = c2; {0}[3] = c3; return c0; }}()",
    16: '[&]{{ int c0, c1, c2, c3; asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 '
    '{{%0,%1,%2,%3}}, {{%4,%5}}, {{%6}}, {{%7,%7,%7,%7}};" '
    ': "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) '
    ': "r"({1}), "r"({2}), "r"({3}), "r"(0)); '
    "{0}[1] = c1; {0}[2] = c2; {0}[3] = c3; return c0; }}()",
}


def _mma(a: list[UOp], b: list[UOp]) -> list[UOp]:
    # lane 4g + t gets the results for rows g and g + 8 by tokens 2t and 2t + 1
    c = UOp.alloc((4,), dtypes.int32, addrspace=AddrSpace.REG)
    product = UOp(Ops.CUSTOM, src=(c[0], *a, *b), arg=(_MMA[8 * len(a)], dtypes.int32))
    c = c.after(c[0].store(product))
    return [c[i].load() for i in range(4)]


class _Stack:
    """Weight matrices of one type and width stacked by rows, as one: a tile of rows lies within
    one of them, which `load` reads with each matrix's load predicated on holding the tile."""

    def __init__(self, ws: tuple[UOp, ...], tiles: tuple[int, ...], tile: UOp, rows: int):
        self.row_words = int(ws[0].shape[0]) // (tiles[0] * rows)
        self.parts, first = [], 0
        for w, n in zip(ws, tiles, strict=True):
            mine = None if len(ws) == 1 else (tile >= first) & (tile < first + n)
            self.parts.append((w, (tile - first) * rows, mine))
            first += n

    def load(self, row: UOp, at: UOp | int) -> UOp:
        # word `at` of the tile's row `row`
        words = []
        for w, row0, mine in self.parts:
            index = (row0 + row) * self.row_words + at
            words.append(w[index if mine is None else index.valid(mine)].load())
        return functools.reduce(UOp.__or__, words)

    def word16(self, row: UOp, at: UOp | int) -> UOp:
        # four bytes from halfword `at` on, as _word16
        low, high = (self.load(row, at + i).cast(dtypes.uint32) for i in (0, 1))
        return low | (high << 16)


class _Q4KTile:
    """Q4_K weights, half a block per step: 16 words of nibbles per row, and its 4 groups' scales
    and mins, which get_scale_min_k4 takes from the block's 12 scale bytes."""

    def __init__(self, stack: _Stack, rows: int, tid: UOp):
        self.stack, self.rows, self.tid = stack, rows, tid
        self.quants = UOp.alloc((rows, 16 + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
        self.scales = UOp.alloc((2, 4, rows), dtypes.float32, addrspace=AddrSpace.LOCAL)

    def shared(self) -> list[UOp]:
        return [self.quants, self.scales]

    def fetch(self, step: UOp) -> list[UOp]:
        # the thread's words of the step: nibbles, then d and dmin and the scale bytes of its row
        words, block = [], step // 2 * 36
        for i in range(16):
            row, word = (i * self.rows + self.tid) // 16, (i * self.rows + self.tid) % 16
            words.append(self.stack.load(row, block + 4 + 16 * (step % 2) + word))
        return words + [self.stack.load(self.tid, block + i) for i in range(4)]

    def put(self, bufs: list[UOp], step: UOp, words: list[UOp]) -> list[UOp]:
        quants, scales = bufs
        stores = []
        for i, word in enumerate(words[:-4]):
            row, at = (i * self.rows + self.tid) // 16, (i * self.rows + self.tid) % 16
            stores.append(quants[row, at].store(word))
        dm, *packed = words[-4:]

        def byte(i: int) -> UOp:
            return (packed[i // 4] >> (8 * (i % 4))) & 0xFF

        first = (step % 2).eq(0)  # groups 0..3 of the block, else 4..7
        for g in range(4):
            sc = first.where(byte(g) & 63, (byte(g + 8) & 15) | ((byte(g) >> 6) << 4))
            mn = first.where(byte(g + 4) & 63, (byte(g + 8) >> 4) | ((byte(g + 4) >> 6) << 4))
            stores.append(scales[0, g, self.tid].store(_half(dm) * sc.float()))
            stores.append(scales[1, g, self.tid].store(_half(dm >> 16) * mn.float()))
        return stores

    @staticmethod
    def products(bufs, s, rows, t4, b, xd, xs) -> list[UOp]:
        # groups 2j and 2j + 1 are the low and high nibbles of words 8j .. 8j + 7
        quants, scales = bufs
        shift = (4 * (s % 2)).cast(dtypes.uint32)
        words = [quants[r, s // 2 * 8 + 4 * h + t4].load() for h in range(2) for r in rows]
        c = _mma([((x >> shift) & 0x0F0F0F0F).bitcast(dtypes.int32) for x in words], b)
        return [
            c[e].float() * (scales[0, s, rows[e // 2]].load() * xd[e % 2])
            - scales[1, s, rows[e // 2]].load() * xs[e % 2]
            for e in range(4)
        ]


class _Q6KTile:
    """Q6_K weights, half a block per step, unpacked to int8 as in _q6_k_dot: 4 groups of 32 per
    row, its 8 scales and d. Rows have no padding, to fit in shared memory; XOR-ing words with the
    row instead keeps a fragment's 8 rows in different banks."""

    def __init__(self, stack: _Stack, rows: int, tid: UOp):
        self.stack, self.rows, self.tid = stack, rows, tid
        self.quants = UOp.alloc((rows, 32), dtypes.int32, addrspace=AddrSpace.LOCAL)
        self.scales = UOp.alloc((2, rows), dtypes.uint32, addrspace=AddrSpace.LOCAL)
        self.d = UOp.alloc((rows,), dtypes.float32, addrspace=AddrSpace.LOCAL)

    def shared(self) -> list[UOp]:
        return [self.quants, self.scales, self.d]

    def fetch(self, step: UOp) -> list[UOp]:
        # for each of the thread's (row, word) pairs, the words of low and high bits behind it;
        # then the 8 scale bytes and d of its row
        words, block, half = [], step // 2 * 105, 32 * (step % 2)
        for i in range(8):
            row, m = (i * self.rows + self.tid) // 8, (i * self.rows + self.tid) % 8
            for at in (half + 2 * m, half + 16 + 2 * m, 64 + half // 2 + 2 * m):
                words.append(self.stack.word16(row, block + at))
        words += [self.stack.word16(self.tid, block + 96 + half // 8 + 2 * j) for j in range(2)]
        return words + [self.stack.load(self.tid, block + 104).cast(dtypes.uint32)]

    def put(self, bufs: list[UOp], step: UOp, words: list[UOp]) -> list[UOp]:
        quants, scales, d = bufs
        stores = []
        for i in range(8):
            row, m = (i * self.rows + self.tid) // 8, (i * self.rows + self.tid) % 8
            ql, qh = words[3 * i : 3 * i + 2], words[3 * i + 2]
            for k in range(4):
                low = (ql[k % 2] >> (4 * (k // 2))) & 0x0F0F0F0F
                high = ((qh >> (2 * k)) & 0x03030303) << 4
                stores.append(quants[row, _swizzle(row, 8 * k + m)].store(_minus_32(low | high)))
        stores += [scales[j, self.tid].store(words[-3 + j]) for j in range(2)]
        return stores + [d[self.tid].store(_half(words[-1]))]

    @staticmethod
    def products(bufs, s, rows, t4, b, xd, xs) -> list[UOp]:
        # a scale per 16 weights: two k = 16 products, combined in int32
        quants, scales, d = bufs
        a = [quants[r, _swizzle(r, 8 * s + 4 * h + t4)].load() for h in range(2) for r in rows]
        c = [_mma(a[2 * h : 2 * h + 2], [b[h]]) for h in range(2)]
        dots = []
        for e in range(4):
            packed = scales[s // 2, rows[e // 2]].load()  # the scales of groups 2s and 2s + 1
            sc = [_byte_i8(packed, 16 * (s % 2) + 8 * h) for h in range(2)]
            dots.append(c[0][e] * sc[0] + c[1][e] * sc[1])
        return [dots[e].float() * (d[rows[e // 2]].load() * xd[e % 2]) for e in range(4)]


def _swizzle(row: UOp, word: UOp | int) -> UOp:
    return (row % 8 * 4) ^ word


def _byte_i8(word: UOp, shift: UOp) -> UOp:
    return (
        ((word >> shift.cast(dtypes.uint32)) & 0xFF)
        .cast(dtypes.uint8)
        .bitcast(dtypes.int8)
        .cast(dtypes.int32)
    )


_TILES: dict[GGMLType, type[_Q4KTile] | type[_Q6KTile]] = {
    GGMLType.Q4_K: _Q4KTile,
    GGMLType.Q6_K: _Q6KTile,
}


@functools.cache
def _matmul_kernel(
    out: UOp, *srcs: UOp, tokens: int | UOp, ggml_type: GGMLType, rows: int, tiles: tuple[int, ...]
) -> UOp:
    # srcs: the stacked matrices, tiles of `rows` rows of each, then xq, xd, xs and a residual; a
    # block per tile of `rows` rows by TILE_TOKENS tokens
    ws, (xq, xd, xs, *residual) = srcs[: len(tiles)], srcs[len(tiles) :]
    count, n = (int(x) for x in out.shape)
    cols, threads = 4 * int(xq.shape[0]) // count, rows
    steps = cols // STEP
    tile_tokens = UOp.range((tokens + TILE_TOKENS - 1) // TILE_TOKENS, 0, AxisType.GLOBAL)
    tile_rows = UOp.range(n // rows, 1, AxisType.GLOBAL)
    lane, warp = _lane(), UOp.range(rows // WARP_ROWS, 2, AxisType.LOCAL)
    tid, g, t4 = warp * WARP + lane, lane // 4, lane % 4
    row0, token0 = tile_rows * rows, tile_tokens * TILE_TOKENS
    weights = _TILES[ggml_type](_Stack(ws, tiles, tile_rows, rows), rows, tid)
    # the tile's tokens for a step: 32 words each, and the d and d * sum(q) of 4 groups
    act = UOp.alloc((TILE_TOKENS, 32 + 4), dtypes.int32, addrspace=AddrSpace.LOCAL)
    act_scales = UOp.alloc((2, 4, TILE_TOKENS), dtypes.float32, addrspace=AddrSpace.LOCAL)
    words_a = 32 * TILE_TOKENS // threads
    pairs = [((i * threads + tid) // 32, (i * threads + tid) % 32) for i in range(words_a)]
    groups = [
        ((i * threads + tid) // TILE_TOKENS, (i * threads + tid) % TILE_TOKENS)
        for i in range(4 * TILE_TOKENS // threads)
    ]

    def fetch(step: UOp) -> list[UOp]:
        words = weights.fetch(step)
        for tok, word in pairs:
            words.append(
                xq[(token0 + tok) * (cols // 4) + step * 32 + word].load().bitcast(dtypes.uint32)
            )
        for grp, tok in groups:
            at = (token0 + tok) * (cols // GROUP) + step * 4 + grp
            words += [xd[at].load().bitcast(dtypes.uint32), xs[at].load().bitcast(dtypes.uint32)]
        return words

    def put(bufs: list[UOp], step: UOp, words: list[UOp]) -> list[UOp]:
        *mine, act, act_scales = bufs
        n_act = words_a + 2 * len(groups)
        stores = weights.put(mine, step, words[:-n_act])
        words = words[-n_act:]
        for (tok, word), value in zip(pairs, words[:words_a], strict=True):
            stores.append(act[tok, word].store(value.bitcast(dtypes.int32)))
        for i, (grp, tok) in enumerate(groups):
            for j in (0, 1):
                stores.append(
                    act_scales[j, grp, tok].store(
                        words[words_a + 2 * i + j].bitcast(dtypes.float32)
                    )
                )
        return stores

    # step 0 before the loop; in step i, the fetch of step i + 1 is staged in registers before
    # the products, so that its loads are in flight meanwhile, and stored after them
    bufs = [*weights.shared(), act, act_scales]
    first = put(bufs, UOp.const(0, dtypes.weakint), fetch(UOp.const(0, dtypes.weakint)))
    step = UOp.range(steps, 3, AxisType.LOOP)
    following = fetch((step + 1).minimum(steps - 1))
    stage = UOp.alloc((len(following),), dtypes.uint32, addrspace=AddrSpace.REG)
    staged = UOp.group(*(stage[i].store(v) for i, v in enumerate(following)))
    bufs = [buf.after(*first).after(step).after(staged) for buf in bufs]
    *mine, act, act_scales = bufs

    acc = _register((SUBTILES_M * SUBTILES_N * 4,), 0.0)
    s = UOp.range(STEP // GROUP, 4, AxisType.LOOP)
    prev, vals = acc.after(step, s), list[UOp]()
    for mi in range(SUBTILES_M):
        r = warp * WARP_ROWS + mi * 16 + g
        for ni in range(SUBTILES_N):
            tok = ni * 8
            b = [act[tok + g, 8 * s + 4 * h + t4].load() for h in range(2)]
            xd_, xs_ = ([act_scales[i, s, tok + 2 * t4 + j].load() for j in (0, 1)] for i in (0, 1))
            for inc in weights.products(mine, s, (r, r + 8), t4, b, xd_, xs_):
                vals.append(prev[len(vals)].load() + inc)
    computed = acc.store(UOp.stack(*vals)).end(s)
    # every warp is done with this step's tiles before they are overwritten
    done = UOp(Ops.BARRIER, src=(computed,))
    stage = stage.after(staged)
    later = put(
        [buf.after(done) for buf in bufs],
        step + 1,
        [stage[i].load() for i in range(len(following))],
    )
    acc = acc.after(UOp.group(computed, *later).end(step))

    results = []
    for mi in range(SUBTILES_M):
        for ni in range(SUBTILES_N):
            for e in range(4):
                row = row0 + warp * WARP_ROWS + mi * 16 + g + 8 * (e // 2)
                tok = token0 + ni * 8 + 2 * t4 + e % 2
                value = acc[(mi * SUBTILES_N + ni) * 4 + e].load()
                if residual:
                    value = value + residual[0][tok, row].load()
                results.append(out[tok, row].store(value))
    info = KernelInfo(name=f"matmul_{ggml_type.name.lower()}_{n}_{cols}", opts_to_apply=())
    return UOp.group(*results).end(tile_tokens, tile_rows, lane, warp).sink(arg=info)


def _matmuls(
    x: Tensor, ws: tuple[QTensor, ...], norm: tuple[Tensor, float] | None, residual: Tensor | None
) -> list[Tensor]:
    tokens = x.shape[-2]
    q8 = quantize_q8(_tiled(x), norm, rows=tokens)
    outs = []
    for out, heights in _products(q8, tokens, ws, residual):
        for start, h in zip(itertools.accumulate([0, *heights]), heights, strict=False):
            outs.append(out[:tokens, start : start + h].reshape(*x.shape[:-1], h))
    return outs


def _tiled(x: Tensor) -> Tensor:
    # Several tokens, while prefilling a bound count of them: buffers hold the most there may be,
    # rounded up to whole tiles, and the kernels stop after the tiles holding actual tokens
    count = -(-x.max_shape[-2] // TILE_TOKENS) * TILE_TOKENS
    return x.reshape(x.shape[-2], x.shape[-1]).float().pad_to((count, x.shape[-1])).contiguous()


def _products(
    q8: tuple[Tensor, Tensor, Tensor], tokens: int | UOp, ws: tuple[QTensor, ...],
    residual: Tensor | None,
) -> list[tuple[Tensor, list[int]]]:  # fmt: skip
    # the products of the quantized activations and each matrix, with consecutive matrices of one
    # type stacked in one kernel, as few rows leave SMs idle: each kernel's output and heights
    xq, xd, xs = q8
    count = int(xd.shape[0]) * GROUP // ws[0].shape[1]
    xq, bound = _carry(xq, tokens)
    res = () if residual is None else (_tiled(residual),)
    outs = []
    for _, group in itertools.groupby(ws, key=lambda w: w.type):
        stack = tuple(group)
        heights = [int(w.shape[0]) for w in stack]
        tile = 256 if sum(heights) >= 4096 and all(h % 256 == 0 for h in heights) else 128
        out = Tensor.empty(count, sum(heights), dtype=dtypes.float32, device=xq.device)
        fxn = functools.partial(
            _matmul_kernel, tokens=bound, ggml_type=stack[0].type, rows=tile,
            tiles=tuple(h // tile for h in heights),
        )  # fmt: skip
        out = Tensor.custom_kernel(out, *map(_words, stack), xq, xd, xs, *res, fxn=fxn)[0]
        outs.append((out, heights))
    return outs


def feed_forward(
    x: Tensor, gate: QTensor, up: QTensor, down: QTensor, norm: tuple[Tensor, float]
) -> Tensor:
    """x + silu(n @ gate.T) * (n @ up.T) @ down.T for n = rms_norm(x, *norm). For several tokens,
    the activations between the matrices go from the gate and up kernel to quantization unwritten
    in between."""
    if _one_token(x):
        return linears(swiglu(x, gate, up, norm), down, residual=x)[0]
    tokens = x.shape[-2]
    ((both, _),) = _products(quantize_q8(_tiled(x), norm, rows=tokens), tokens, (gate, up), None)
    q8 = quantize_q8(both, rows=tokens, gated=True)
    ((out, _),) = _products(q8, tokens, (down,), x)
    return out[:tokens].reshape(*x.shape[:-1], down.shape[0])


# ******** attention: one query token against the KV cache ********
# FlashDecoding, adapted from tinygrad/llm/kernels/amd.py: the cache is cut into chunks of KEYS
# keys, blocks reduce chunks with an online softmax, and a second kernel combines their partials.

KEYS = 64  # keys per chunk
PARTIALS = 48  # most blocks per kv head; longer caches loop over several chunks per block
SHARED = 49152  # bytes of shared memory a block may use without opting in to more
PAD = 8  # halves of shared memory after each warp's outputs, see _attention_partial_kernel


def _vector(ptr: UOp, lanes: int) -> tuple[UOp, ...]:
    # `lanes` consecutive values from an index, as one vector load, widened to f32
    buf, coords = ptr.src[0], ptr.src[1:]
    start = sum((c * math.prod(buf.shape[i + 1 :]) for i, c in enumerate(coords)), UOp.const(0))
    vec = UOp(Ops.SHRINK, src=(buf.flatten(), start, UOp.const(lanes))).load()
    return tuple(vec[i].float() for i in range(lanes))


def _min(a: int | UOp, b: int) -> int | UOp:
    return a.minimum(b) if isinstance(a, UOp) else min(a, b)


def _register(shape: tuple[int, ...], value: float) -> UOp:
    reg = UOp.alloc(shape, dtypes.float32, addrspace=AddrSpace.REG)
    return reg.after(reg.store(reg.const_like(value)))


@functools.cache
def _attention_partial_kernel(
    out: UOp, stats: UOp, q: UOp, cache: UOp, length: int | UOp, waves: int
) -> UOp:
    # A block takes one kv head and every PARTIALS-th chunk of its keys, for all query heads of
    # the GQA group. Each of `waves` warps scores KEYS / waves keys of a chunk; lanes hold
    # dim / 32 dimensions. The warps then merge through shared memory into one partial per block:
    # the unnormalized output, its running max and its sum of weights.
    kv_heads, dim = int(cache.shape[2]), int(cache.shape[4])
    group, per_lane, partials = int(q.shape[1]) // kv_heads, dim // WARP, int(out.shape[1])
    per_wave, zero = KEYS // waves, UOp.const(0.0, dtypes.float32)
    chunks = (length + KEYS - 1) // KEYS
    head = UOp.range(kv_heads, 0, AxisType.GLOBAL)
    block = UOp.range(_min(chunks, partials), 1, AxisType.GLOBAL)
    lane, wave = _lane(), UOp.range(waves, 3, AxisType.LOCAL)
    qs = [_vector(q[0, head * group + h, 0, lane * per_lane], per_lane) for h in range(group)]
    rounds = UOp.range((chunks - 1 - block) // partials + 1, 4, AxisType.LOOP)
    chunk = block + rounds * partials
    valid, scores, values = [], [], []
    for j in range(per_wave):
        key = chunk * KEYS + wave * per_wave + j
        valid.append(key < length)
        k = _vector(cache[0, 0, head, key, lane * per_lane], per_lane)
        # values load during scoring, so both streams are in flight together
        values.append(
            [
                valid[j].where(v, zero)
                for v in _vector(cache[1, 0, head, key, lane * per_lane], per_lane)
            ]
        )
        dots = [_warp_sum(sum((a * b for a, b in zip(qh, k, strict=True)), zero)) for qh in qs]
        scores.append([valid[j].where(d / math.sqrt(dim), -1e30) for d in dots])
    # a finite initial max keeps fully masked warps from computing exp(-inf - -inf)
    acc, mx, total = (
        _register((group, per_lane), 0.0),
        _register((group,), -1e30),
        _register((group,), 0.0),
    )
    prev_acc, prev_max, prev_sum = acc.after(rounds), mx.after(rounds), total.after(rounds)
    new_max = [
        functools.reduce(UOp.maximum, (sc[h] for sc in scores), prev_max[h].load())
        for h in range(group)
    ]
    scale = [((prev_max[h].load() - new_max[h]) * LOG2E).exp2() for h in range(group)]
    accs = [[scale[h] * prev_acc[h, i].load() for i in range(per_lane)] for h in range(group)]
    sums = [scale[h] * prev_sum[h].load() for h in range(group)]
    for j in range(per_wave):
        for h in range(group):
            weight = valid[j].where(((scores[j][h] - new_max[h]) * LOG2E).exp2(), zero)
            accs[h] = [a + weight * v for a, v in zip(accs[h], values[j], strict=True)]
            sums[h] = sums[h] + weight
    update = UOp.group(
        acc.store(UOp.stack(*(x for a in accs for x in a)).reshape(group, per_lane)),
        mx.store(UOp.stack(*new_max)),
        total.store(UOp.stack(*sums)),
    ).end(rounds)
    acc, mx, total = acc.after(update), mx.after(update), total.after(update)

    # merge the warps through shared memory: each writes its output normalized to f16 (sum >= 1
    # unless empty), as [head, dimension within lane, lane] plus PAD, then its max and sum.
    # tinygrad's codegen declares an index at its first use in the rounds loop and reuses it after
    # the loop, out of scope, if the same expression recurs there: this layout repeats neither the
    # lane offset of the cache loads nor, thanks to PAD, their stride per warp.
    width = group * dim + PAD
    shared = UOp.alloc((waves, width), dtypes.half, addrspace=AddrSpace.LOCAL)
    stat = UOp.alloc((waves, group, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
    stores = [
        shared[wave, (h * per_lane + i) * WARP + lane].store(
            (acc[h, i].load() / total[h].load().maximum(1)).cast(dtypes.half)
        )
        for h in range(group)
        for i in range(per_lane)
    ]
    stores += [
        stat[wave, h, i].store(x)
        for h in range(group)
        for i, x in enumerate((mx[h].load(), total[h].load()))
    ]
    shared, stat = shared.after(*stores), stat.after(*stores)
    thread, results = wave * WARP + lane, []
    for i in range(-(-group * dim // (waves * WARP))):
        flat = thread + i * waves * WARP
        h, d = flat // dim, flat % dim
        top = functools.reduce(UOp.maximum, (stat[w, h, 0].load() for w in range(waves)))
        val = sum((((stat[w, h, 0].load() - top) * LOG2E).exp2() * stat[w, h, 1].load()
                   * shared[w, (h * per_lane + d % per_lane) * WARP + d // per_lane].load().float()
                   for w in range(waves)), zero)  # fmt: skip
        live = (
            (head * group + h).valid(flat < group * dim)
            if group * dim % (waves * WARP)
            else head * group + h
        )
        results.append(out[live, block, d].store(val))
    top = functools.reduce(UOp.maximum, (stat[w, thread, 0].load() for w in range(waves)))
    weights = sum((((stat[w, thread, 0].load() - top) * LOG2E).exp2() * stat[w, thread, 1].load()
                   for w in range(waves)), zero)  # fmt: skip
    qh = (head * group + thread).valid(thread < group)
    results += [stats[qh, block, 0].store(top), stats[qh, block, 1].store(weights)]
    info = KernelInfo(name="attention_partial", opts_to_apply=())
    return UOp.group(*results).end(lane, wave, block, head).sink(arg=info)


@functools.cache
def _attention_combine_kernel(o: UOp, partial: UOp, stats: UOp, live: int | UOp) -> UOp:
    # one warp per query head and 64 dimensions: weights each block's partial by exp(max - max)
    heads, _, dim = partial.shape
    tile, per_lane = 64, 64 // WARP
    head, part = UOp.range(heads, 0, AxisType.GLOBAL), UOp.range(dim // tile, 1, AxisType.GLOBAL)
    lane = _lane()
    dims = [part * tile + lane * per_lane + i for i in range(per_lane)]
    c1 = UOp.range(live, 100, AxisType.LOOP)
    top = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
    top = top.after(top.store(top.const_like(-math.inf)))
    top = top.after(top.store(top.after(c1).maximum(stats[head, c1, 0].load())).end(c1))
    c2 = UOp.range(live, 101, AxisType.LOOP)
    acc, total = _register((per_lane,), 0.0), _register((1,), 0.0)
    weight = ((stats[head, c2, 0].load() - top) * LOG2E).exp2()
    update = UOp.group(
        *[
            acc[i].store(acc.after(c2)[i].load() + weight * partial[head, c2, d].load())
            for i, d in enumerate(dims)
        ],
        total[0].store(total.after(c2)[0].load() + weight * stats[head, c2, 1].load()),
    ).end(c2)
    acc, total = acc.after(update), total.after(update)
    stores = [o[0, head, 0, d].store(acc[i].load() / total[0].load()) for i, d in enumerate(dims)]
    info = KernelInfo(name="attention_combine", opts_to_apply=())
    return UOp.group(*stores).end(lane, part, head).sink(arg=info)


def supports_attention(q: Tensor, cache: Tensor) -> bool:
    if not isinstance(q.device, str) or q.device.split(":")[0] not in ("NV", "CUDA"):
        return False
    shape = (*cache.shape[1:], *q.shape[1:3])
    if not all(isinstance(x, int) for x in shape):
        return False
    batch, kv_heads, n, dim, heads, tokens = (int(x) for x in shape)
    group = heads // kv_heads
    fits = (group * dim + PAD) * 2 + group * 8 <= SHARED  # one warp's share of shared memory
    return batch == 1 and tokens == 1 and dim % 64 == 0 and n % KEYS == 0 and fits


def attention(q: Tensor, cache: Tensor, length: int | UOp) -> Tensor:
    """Attention of one query token (1, H, 1, D) over the first `length` cached positions."""
    cache, length = _carry(cache, length)
    heads, dim, group = q.shape[1], cache.shape[4], q.shape[1] // cache.shape[2]
    waves = 16
    while waves * ((group * dim + PAD) * 2 + group * 8) > SHARED:
        waves //= 2
    chunks = min(PARTIALS, int(cache.shape[3]) // KEYS)
    partial = Tensor.empty(heads, chunks, dim, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(heads, chunks, 2, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_attention_partial_kernel, length=length, waves=waves)
    partial, stats = Tensor.custom_kernel(partial, stats, q.float().contiguous(), cache, fxn=fxn)[
        :2
    ]
    live = _min((length + KEYS - 1) // KEYS, chunks)
    out = Tensor.empty(1, heads, 1, dim, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_attention_combine_kernel, live=live)
    return Tensor.custom_kernel(out, partial, stats, fxn=fxn)[0]


# ******** attention: a chunk of query tokens against the KV cache ********
# FlashAttention-2 on f16 tensor cores. A block takes 16 query tokens and one kv head, with a warp
# for each query head of the GQA group; the warps share tiles of 64 keys and values in shared
# memory, and each keeps its rows' scores, softmax statistics and outputs in registers.

QUERIES, KEY_TILE = 16, 64

# mma.sync on f16 with f32 accumulation: c (4 f32) + a 16 x 16 tile times a 16 x 8 tile, from the
# lane's 4 and 2 words of f16 pairs; results return as in _MMA
_MMA_F16 = (
    '[&]{{ float d0, d1, d2, d3; asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 '
    '{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%10,%11,%12,%13}};" '
    ': "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) '
    ': "r"({1}), "r"({2}), "r"({3}), "r"({4}), "r"({5}), "r"({6}), '
    '"f"({7}), "f"({8}), "f"({9}), "f"({10})); '
    "{0}[1] = d1; {0}[2] = d2; {0}[3] = d3; return d0; }}()"
)


def _mma_f16(a: list[UOp], b: list[UOp], c: list[UOp]) -> list[UOp]:
    d = UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.REG)
    product = UOp(Ops.CUSTOM, src=(d[0], *a, *b, *c), arg=(_MMA_F16, dtypes.float32))
    d = d.after(d[0].store(product))
    return [d[i].load() for i in range(4)]


def _f16_pair(lo: UOp, hi: UOp) -> UOp:
    # two f32 as one word of f16, lo in the low half; cvt puts its first source in the high half
    code = (
        '[&]{{ unsigned r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"({1}), "f"({0})); '
        "return r; }}()"
    )
    return UOp(Ops.CUSTOM, src=(lo, hi), arg=(code, dtypes.uint32))


def _halves_word(lo: UOp, hi: UOp) -> UOp:
    return lo.bitcast(dtypes.uint16).cast(dtypes.uint32) | (
        hi.bitcast(dtypes.uint16).cast(dtypes.uint32) << 16
    )


def _quad(value: UOp, op: Callable[[UOp, UOp], UOp]) -> UOp:
    # reduces over the 4 lanes that hold a row of an mma result
    for mask in (1, 2):
        value = op(value, _shfl_xor(value, mask))
    return value


@functools.cache
def _flash_attention_kernel(
    out: UOp, q: UOp, cache: UOp, start: int | UOp, tokens: int | UOp
) -> UOp:
    # q (heads, count, dim) in f16, scaled so that exp2 gives the softmax, and the f16 cache, read
    # as words of f16 pairs; query i is at position start + i and sees positions up to it
    heads, dim = int(q.shape[0]), int(q.shape[2])
    kv_heads, words = int(cache.shape[2]), dim // 2
    cache = cache.flatten().bitcast(dtypes.uint32).reshape(2, kv_heads, int(cache.shape[3]), words)
    group = heads // kv_heads
    threads = group * WARP
    tile = UOp.range((tokens + QUERIES - 1) // QUERIES, 0, AxisType.GLOBAL)
    kv_head = UOp.range(kv_heads, 1, AxisType.GLOBAL)
    lane, warp = _lane(), UOp.range(group, 2, AxisType.LOCAL)
    head, tid, g, t = kv_head * group + warp, warp * WARP + lane, lane // 4, lane % 4
    rows = (tile * QUERIES + g, tile * QUERIES + g + 8)  # the lane's rows of each mma result
    # queries as A fragments of 16 dimensions
    queries = [
        [
            _halves_word(*(q[head, r, 16 * k + 8 * h + 2 * t + i].load() for i in (0, 1)))
            for h in (0, 1)
            for r in rows
        ]
        for k in range(dim // 16)
    ]
    # the keys and values up to the tile's last query, KEY_TILE at a time: keys as they are in
    # the cache, values transposed so that B fragments of keys are words
    end = start + (tile * QUERIES + QUERIES).minimum(tokens)
    kt = UOp.range((end + KEY_TILE - 1) // KEY_TILE, 3, AxisType.LOOP)
    keys = UOp.alloc((KEY_TILE, words + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    values = UOp.alloc((dim, KEY_TILE // 2 + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    stores = []
    for i in range(KEY_TILE * words // threads):
        key, w = (i * threads + tid) // words, (i * threads + tid) % words
        stores.append(keys[key, w].store(cache[0, kv_head, kt * KEY_TILE + key, w].load()))
    for i in range(KEY_TILE // 2 * words // threads):
        pair, w = (i * threads + tid) % (KEY_TILE // 2), (i * threads + tid) // (KEY_TILE // 2)
        a, b = (cache[1, kv_head, kt * KEY_TILE + 2 * pair + j, w].load() for j in (0, 1))
        stores.append(values[2 * w, pair].store((a & 0xFFFF) | (b << 16)))
        stores.append(values[2 * w + 1, pair].store((a >> 16) | (b & 0xFFFF0000)))
    keys, values = keys.after(*stores), values.after(*stores)

    # a finite initial max keeps exp2 from seeing -inf - -inf
    acc, mx, total = (_register((dim // 2,), 0.0), _register((2,), -1e30), _register((2,), 0.0))
    prev_acc, prev_max, prev_total = acc.after(kt), mx.after(kt), total.after(kt)
    zero = UOp.const(0.0, dtypes.float32)
    scores = []  # 8 tiles of 8 keys, masked to the positions each row sees
    for j in range(KEY_TILE // 8):
        c = [zero] * 4
        for k in range(dim // 16):
            c = _mma_f16(queries[k], [keys[8 * j + g, 8 * k + 4 * h + t].load() for h in (0, 1)], c)
        position = kt * KEY_TILE + 8 * j + 2 * t
        scores.append(
            [(position + e % 2 <= start + rows[e // 2]).where(c[e], -math.inf) for e in range(4)]
        )
    row_max = [
        _quad(
            functools.reduce(UOp.maximum, (s[e] for s in scores for e in (2 * r, 2 * r + 1))),
            UOp.maximum,
        )
        for r in (0, 1)
    ]
    new_max = [prev_max[r].load().maximum(row_max[r]) for r in (0, 1)]
    rescale = [(prev_max[r].load() - new_max[r]).exp2() for r in (0, 1)]
    p = [[(s[e] - new_max[e // 2]).exp2() for e in range(4)] for s in scores]
    sums = [
        _quad(sum((x[e] for x in p for e in (2 * r, 2 * r + 1)), zero), UOp.__add__) for r in (0, 1)
    ]
    # the weights as A fragments: the results of two 8-key tiles make one of 16 keys
    weights = [
        [_f16_pair(*p[2 * k + i][2 * r : 2 * r + 2]) for i in (0, 1) for r in (0, 1)]
        for k in range(KEY_TILE // 16)
    ]
    outs: list[UOp] = []
    for n in range(dim // 8):
        c = [prev_acc[4 * n + e].load() * rescale[e // 2] for e in range(4)]
        for k in range(KEY_TILE // 16):
            c = _mma_f16(
                weights[k], [values[8 * n + g, 8 * k + 4 * h + t].load() for h in (0, 1)], c
            )
        outs += c
    update = UOp.group(
        acc.store(UOp.stack(*outs)),
        mx.store(UOp.stack(*new_max)),
        total.store(UOp.stack(*(prev_total[r].load() * rescale[r] + sums[r] for r in (0, 1)))),
    ).end(kt)
    acc, total = acc.after(update), total.after(update)
    results = [
        out[rows[e // 2], head * dim + 8 * n + 2 * t + e % 2].store(
            acc[4 * n + e].load() / total[e // 2].load()
        )
        for n in range(dim // 8)
        for e in range(4)
    ]
    info = KernelInfo(name="flash_attention", opts_to_apply=())
    return UOp.group(*results).end(tile, kv_head, lane, warp).sink(arg=info)


def supports_flash_attention(q: Tensor, cache: Tensor) -> bool:
    if not isinstance(q.device, str) or q.device.split(":")[0] not in ("NV", "CUDA"):
        return False
    shape = (*cache.shape[1:], q.shape[0], q.shape[1])
    if not all(isinstance(x, int) for x in shape):
        return False
    batch, kv_heads, n, dim, q_batch, heads = (int(x) for x in shape)
    threads, words = heads // kv_heads * WARP, dim // 2
    shared = 4 * (KEY_TILE * (words + 4) + dim * (KEY_TILE // 2 + 4))
    whole = KEY_TILE * words % threads == 0 and KEY_TILE // 2 * words % threads == 0
    return (
        batch == q_batch == 1 and dim % 16 == 0 and n % KEY_TILE == 0 and shared <= SHARED and whole
    )


def flash_attention(q: Tensor, cache: Tensor, start_pos: int | UOp) -> Tensor:
    """Causal attention of query tokens (1, H, T, D) at positions start_pos.. over the cache, which
    already holds their keys and values. Returns (1, T, H * D)."""
    _, heads, tokens, dim = q.shape
    count = -(-q.max_shape[2] // QUERIES) * QUERIES
    q = (q.reshape(heads, tokens, dim).float() * (LOG2E / math.sqrt(dim))).half()
    q = q.pad_to((heads, count, dim)).contiguous()
    q, start = _carry(q, start_pos)
    cache, length = _carry(cache, tokens)
    out = Tensor.empty(count, heads * dim, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_flash_attention_kernel, start=start, tokens=length)
    out = Tensor.custom_kernel(out, q, cache, fxn=fxn)[0]
    return out[:tokens].reshape(1, tokens, heads * dim)


# ******** argmax over a row of logits ********

PARTS = 256  # warps per row in the first pass


def _argmax_step(best: UOp, index: UOp, value: UOp, at: UOp) -> tuple[UOp, UOp]:
    # keep the larger value, and on ties the lower index, as argmax does
    take = (value > best) | (value.eq(best) & (at < index))
    return take.where(value, best), take.where(at, index)


def _warp_argmax(best: UOp, index: UOp) -> tuple[UOp, UOp]:
    for mask in (16, 8, 4, 2, 1):
        best, index = _argmax_step(best, index, _shfl_xor(best, mask), _shfl_xor(index, mask))
    return best, index


@functools.cache
def _argmax_partial_kernel(values: UOp, indices: UOp, x: UOp) -> UOp:
    # one warp per PARTS-th of a row; lanes stride over its slice so loads coalesce
    rows, n = (int(d) for d in x.shape)
    per = -(-n // PARTS)
    row, part, lane = (
        UOp.range(rows, 0, AxisType.GLOBAL),
        UOp.range(PARTS, 1, AxisType.GLOBAL),
        _lane(),
    )
    best, index = UOp.const(-math.inf, dtypes.float32), UOp.const(0, dtypes.int32)
    for k in range(-(-per // WARP)):
        offset = k * WARP + lane
        at = (part * per + offset).cast(dtypes.int32)
        live = (offset < per) & (at < n)
        value = live.where(x[row, at.minimum(n - 1)].load(), -math.inf)
        best, index = _argmax_step(best, index, value, at)
    best, index = _warp_argmax(best, index)
    first = lane.eq(0)
    stores = (
        values[row, part.valid(first)].store(best),
        indices[row, part.valid(first)].store(index),
    )
    info = KernelInfo(name="argmax_partial", opts_to_apply=())
    return UOp.group(*stores).end(row, part, lane).sink(arg=info)


@functools.cache
def _argmax_final_kernel(out: UOp, values: UOp, indices: UOp) -> UOp:
    row, lane = UOp.range(int(out.shape[0]), 0, AxisType.GLOBAL), _lane()
    best, index = UOp.const(-math.inf, dtypes.float32), UOp.const(0, dtypes.int32)
    for k in range(PARTS // WARP):
        part = k * WARP + lane
        best, index = _argmax_step(best, index, values[row, part].load(), indices[row, part].load())
    _, index = _warp_argmax(best, index)
    info = KernelInfo(name="argmax_final", opts_to_apply=())
    return out[row.valid(lane.eq(0))].store(index).end(row, lane).sink(arg=info)


def supports_argmax(x: Tensor) -> bool:
    on_nv = isinstance(x.device, str) and x.device.split(":")[0] in ("NV", "CUDA")
    return on_nv and x.ndim == 2 and all(isinstance(d, int) for d in x.shape)


def argmax(x: Tensor) -> Tensor:
    """Index of the largest value in each row of x (B, V), the first one on ties: (B, 1) int32."""
    rows = x.shape[0]
    values = Tensor.empty(rows, PARTS, dtype=dtypes.float32, device=x.device)
    indices = Tensor.empty(rows, PARTS, dtype=dtypes.int32, device=x.device)
    x = x.float().contiguous()
    values, indices = Tensor.custom_kernel(values, indices, x, fxn=_argmax_partial_kernel)[:2]
    out = Tensor.empty(rows, dtype=dtypes.int32, device=x.device)
    return Tensor.custom_kernel(out, values, indices, fxn=_argmax_final_kernel)[0].reshape(rows, 1)
