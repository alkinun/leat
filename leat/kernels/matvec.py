"""Linear layers for a few tokens, llama.cpp's MMVQ: the activations are quantized to int8, then a
warp computes each output row with __dp4a over the weights in their storage format, for every
token at once, so that each weight is read once however many tokens there are: a decode step of
several sequences costs little more than one's."""

import functools
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo

from leat.kernels.common import (
    WARP,
    dp4a,
    e8m0_half,
    f16,
    fifth_bits,
    funnel,
    glu,
    lane_range,
    load_words,
    minus,
    on_gpu,
    register,
    storage_words,
    table16,
    table_words,
    warp_sum,
    word16,
)
from leat.kernels.quantize import quantize_q8
from leat.quant import FP4_VALUES, IQ4_VALUES, GGMLType, QTensor

Dot = Callable[[UOp, UOp, UOp | int], UOp]  # (row, unit, x) -> a unit's share of row . x
MATVEC_TOKENS = 8  # tokens up to which the matrix-vector kernels run a layer


def _group(xq: UOp, g: UOp) -> list[UOp]:
    # the 8 words of int8 activations of group g, as two 16-byte loads rather than 8 of a word:
    # with several tokens, a unit loads more words of activations than of weights
    return [*load_words(xq, g * 8, 4), *load_words(xq, g * 8 + 4, 4)]


def rows_kernel(
    out: UOp, units: int, name: str, dots: list[Callable[[UOp, UOp], UOp]],
    combine: Callable[..., UOp | list[UOp]], rows: int | UOp | None = None, per_warp: int = 1,
) -> UOp:  # fmt: skip
    # one block of one warp per `per_warp` output rows, of `rows` if given: lanes take the rows'
    # `units` in turn, the warp sums each dot product, and out[row] = combine(row, *sums); where
    # combine gives a list, out[k * rows + row] its k-th value. A warp's rows share the loads of
    # their units' activations, which with several tokens cost more than the weights' reads. Whole
    # turns run in a loop; where a last turn has fewer units than lanes, the others repeat the
    # last unit, whose loads hit in cache, and drop its share, as rows past the last repeat it.
    # Grouping rows into wider blocks, a warp each, measured slower on the 3090, by up to a
    # quarter for Q6_K.
    count = out.shape[0] if rows is None else rows
    warp = UOp.range(-(-count // per_warp), 0, AxisType.GLOBAL)
    ragged = isinstance(count, int) and count % per_warp != 0
    owned = [warp * per_warp + r for r in range(per_warp)] if per_warp > 1 else [warp]
    clamped = [r.minimum(count - 1) for r in owned] if ragged else owned
    lane = lane_range()
    zero = UOp.const(0.0, dtypes.float32)
    whole, rest = divmod(units, WARP)
    pairs = [(row, dot) for row in clamped for dot in dots]
    acc = register((len(pairs),), 0.0)
    if whole:
        turn = UOp.range(whole, 1, AxisType.LOOP)
        prev = acc.after(turn)
        shares = [
            prev[i].load() + dot(row, turn * WARP + lane) for i, (row, dot) in enumerate(pairs)
        ]
        acc = acc.after(acc.store(UOp.stack(*shares)).end(turn))
    totals = [acc[i].load() for i in range(len(pairs))]
    if rest:
        unit = whole * WARP + lane
        totals = [
            t + (unit < units).where(dot(row, unit.minimum(units - 1)), zero)
            for t, (row, dot) in zip(totals, pairs, strict=True)
        ]
    sums, stores = [warp_sum(t) for t in totals], []
    for i, (row, own) in enumerate(zip(clamped, owned, strict=True)):
        live = lane.eq(0) & (own < count) if ragged else lane.eq(0)
        values = combine(row, *sums[i * len(dots) : (i + 1) * len(dots)])
        if isinstance(values, UOp):
            stores.append(out[row.valid(live)].store(values))
        else:
            stores += [out[(k * count + row).valid(live)].store(v) for k, v in enumerate(values)]
    info = KernelInfo(name=f"{name}_{out.shape[0]}_{units}", opts_to_apply=())
    return UOp.group(*stores).end(warp, lane).sink(arg=info)


def _k_scale_min(w: UOp, base: UOp, sub: UOp) -> tuple[UOp, UOp]:
    # ggml's get_scale_min_k4 over the 12 bytes after a K-quant block's d and dmin.
    # sub ^ 4 is sub - 4 where that branch is taken, and stays in range where it is not.
    def byte(i: UOp) -> UOp:
        return (w[base + 1 + i // 4].load() >> ((i % 4) * 8).cast(dtypes.uint32)) & 0xFF

    low = sub < 4
    sc = low.where(byte(sub) & 63, (byte(sub + 4) & 15) | ((byte(sub ^ 4) >> 6) << 4))
    mn = low.where(byte(sub + 4) & 63, (byte(sub + 4) >> 4) | ((byte(sub) >> 6) << 4))
    return sc.float(), mn.float()


def _k_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp, cols: int, high: bool) -> Dot:
    # Q4_K block, 36 words: d and dmin as f16, 12 bytes of 6-bit scales and mins, then 32 words of
    # nibbles. Sub-blocks 2j and 2j+1 are the low and high nibbles of words 4+8j .. 11+8j, so a
    # unit is such a pair: 64 weights against 16 words of activations. Q5_K, 44 words, has 8
    # words of high bits before the nibbles: bit s of byte l is bit 4 of value l of sub-block s.
    words, nibbles = (44, 12) if high else (36, 4)

    def dot(row: UOp, pair: UOp, x: UOp | int) -> UOp:
        block, j = pair // 4, pair % 4
        base = (row * (cols // 256) + block) * words
        dm = w[base].load()
        g = x * (cols // 32) + block * 8 + 2 * j  # activation group of the low sub-block
        dots = [UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)]
        groups = _group(xq, g), _group(xq, g + 1)
        for k in range(8):
            word = w[base + nibbles + 8 * j + k].load()
            for h in range(2):
                weights = (word >> (4 * h)) & 0x0F0F0F0F
                if high:
                    bits = w[base + 4 + k].load() >> (2 * j + h).cast(dtypes.uint32)
                    weights = weights | ((bits & 0x01010101) << 4)
                dots[h] = dp4a(weights.bitcast(dtypes.int32), groups[h][k], dots[h])
        (sc0, m0), (sc1, m1) = _k_scale_min(w, base, 2 * j), _k_scale_min(w, base, 2 * j + 1)
        scaled = sc0 * xd[g].load() * dots[0].float() + sc1 * xd[g + 1].load() * dots[1].float()
        mins = m0 * xs[g].load() + m1 * xs[g + 1].load()
        return f16(dm) * scaled - f16(dm >> 16) * mins

    return dot


def _q6_k_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp, cols: int) -> Dot:
    # Q6_K block, 105 halfwords: ql[128] low nibbles, qh[64] high 2-bit pairs, 16 int8 scales and d
    # as f16. Each half n of 128 weights is 4 rows of 32: row k takes nibble k // 2 of
    # ql[64n + 32(k % 2):][:32] and bits 2k of qh[32n:][:32], minus 32, with a scale per 16. A unit
    # is rows k and k + 2 of a half, which share their ql bytes: 64 weights, as for Q4_K.
    def dot(row: UOp, unit: UOp, x: UOp | int) -> UOp:
        block, n, k = unit // 4, unit % 4 // 2, unit % 2
        base = (row * (cols // 256) + block) * 105
        g = x * (cols // 32) + block * 8 + 4 * n + k  # group of row k; row k + 2 is group g + 2
        dots = [[UOp.const(0, dtypes.int32)] * 2 for _ in range(2)]  # [row k, k + 2][16 weights]
        groups = _group(xq, g), _group(xq, g + 2)
        for m in range(8):
            ql = word16(w, base + 32 * n + 16 * k + 2 * m)
            qh = word16(w, base + 64 + 16 * n + 2 * m)
            for r in range(2):
                high = (qh >> (2 * k + 4 * r).cast(dtypes.uint32)) & 0x03030303
                q = minus(((ql >> (4 * r)) & 0x0F0F0F0F) | (high << 4), 32)
                dots[r][m // 4] = dp4a(q, groups[r][m], dots[r][m // 4])
        total = UOp.const(0.0, dtypes.float32)
        for r in range(2):
            scales = w[base + 96 + 4 * n + k + 2 * r].load()  # both scales of row k + 2r
            for h in range(2):
                sc = ((scales >> (8 * h)) & 0xFF).cast(dtypes.uint8).bitcast(dtypes.int8).float()
                total = total + xd[g + 2 * r].load() * sc * dots[r][h].float()
        return f16(w[base + 104].load().cast(dtypes.uint32)) * total

    return dot


def _q8_0_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp, cols: int) -> Dot:
    # Q8_0 block, 17 halfwords: d as f16, then 32 int8 weights, the activation groups' size; a
    # unit is two blocks
    def dot(row: UOp, unit: UOp, x: UOp | int) -> UOp:
        total = UOp.const(0.0, dtypes.float32)
        for b in range(2):
            block = unit * 2 + b
            base, g = (row * (cols // 32) + block) * 17, x * (cols // 32) + block
            acc, group = UOp.const(0, dtypes.int32), _group(xq, g)
            for m in range(8):
                acc = dp4a(word16(w, base + 1 + 2 * m).bitcast(dtypes.int32), group[m], acc)
            d = f16(w[base].load().cast(dtypes.uint32))
            total = total + d * xd[g].load() * acc.float()
        return total

    return dot


def _q5_0_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp, cols: int) -> Dot:
    # Q5_0 block, 11 halfwords: d as f16, 32 high bits, then 16 bytes whose low nibbles are values
    # 0..15 and high ones 16..31; a weight is d * (q - 16). A unit is two blocks.
    def dot(row: UOp, unit: UOp, x: UOp | int) -> UOp:
        total = UOp.const(0.0, dtypes.float32)
        for b in range(2):
            block = unit * 2 + b
            base, g = (row * (cols // 32) + block) * 11, x * (cols // 32) + block
            high, acc, group = word16(w, base + 1), UOp.const(0, dtypes.int32), _group(xq, g)
            for m in range(4):
                word = word16(w, base + 3 + 2 * m)
                for h in range(2):  # values 4m.. and 16 + 4m..
                    q = ((word >> (4 * h)) & 0x0F0F0F0F) | fifth_bits(
                        (high >> (16 * h + 4 * m)) & 15
                    )
                    acc = dp4a(q.bitcast(dtypes.int32), group[4 * h + m], acc)
            d = f16(w[base].load().cast(dtypes.uint32))
            total = total + d * (xd[g].load() * acc.float() - 16 * xs[g].load())
        return total

    return dot


FP4_TABLE, IQ4_TABLE = table_words(FP4_VALUES), table_words(IQ4_VALUES)


def _q4_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp, cols: int, kind: GGMLType) -> Dot:
    # Blocks of 32 of 16 bytes of nibbles, the low ones values 0..15 and the high ones 16..31,
    # after f16 fields, read as halfwords: Q4_0's d, a weight d * (q - 8), 9 halfwords; Q4_1's d
    # and m, d * q + m, 10; IQ4_NL's d, d * IQ4_VALUES[q], 9; Q5_1's d, m and 32 high bits, a fifth
    # bit of each value, d * q + m, 12. A unit is two blocks.
    halves = {GGMLType.Q4_0: 9, GGMLType.Q4_1: 10, GGMLType.IQ4_NL: 9, GGMLType.Q5_1: 12}[kind]
    first = halves - 8  # the halfword the nibbles start at

    def dot(row: UOp, unit: UOp, x: UOp | int) -> UOp:
        total = UOp.const(0.0, dtypes.float32)
        for b in range(2):
            block = unit * 2 + b
            base, g = (row * (cols // 32) + block) * halves, x * (cols // 32) + block
            acc, group = UOp.const(0, dtypes.int32), _group(xq, g)
            high = word16(w, base + 2) if kind == GGMLType.Q5_1 else None
            for m in range(4):
                word, values = word16(w, base + first + 2 * m), list[UOp]()
                if kind == GGMLType.IQ4_NL:
                    values = list(table16(word, IQ4_TABLE))
                else:
                    values = [(word >> (4 * h)) & 0x0F0F0F0F for h in range(2)]
                    if high is not None:  # values 4m.. and 16 + 4m..
                        bits = [fifth_bits((high >> (16 * h + 4 * m)) & 15) for h in range(2)]
                        values = [v | b for v, b in zip(values, bits, strict=True)]
                    if kind == GGMLType.Q4_0:
                        values = [minus(v, 8) for v in values]
                for h, v in enumerate(values):
                    acc = dp4a(v.bitcast(dtypes.int32), group[4 * h + m], acc)
            d = f16(w[base].load().cast(dtypes.uint32))
            term = d * xd[g].load() * acc.float()
            if kind in (GGMLType.Q4_1, GGMLType.Q5_1):  # m times the group's sum
                term = term + f16(w[base + 1].load().cast(dtypes.uint32)) * xs[g].load()
            total = total + term
        return total

    return dot


def _iq4_xs_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp, cols: int) -> Dot:
    # IQ4_XS block, 34 words: d as f16 and 16 high bits of scales, 4 bytes of low ones, then
    # 128 bytes of nibbles, IQ4_NL's of 8 sub-blocks of 32; sub-block j's scale is d * (s - 32)
    # for s nibble j of the low bytes and bits 2j of the high ones. A unit is two sub-blocks.
    def dot(row: UOp, unit: UOp, x: UOp | int) -> UOp:
        block, pair = unit // 4, unit % 4
        base = (row * (cols // 256) + block) * 34
        head, lows = w[base].load(), w[base + 1].load()
        total = UOp.const(0.0, dtypes.float32)
        for b in range(2):
            j = pair * 2 + b
            g = x * (cols // 32) + block * 8 + j
            acc, group = UOp.const(0, dtypes.int32), _group(xq, g)
            for m in range(4):
                for h, v in enumerate(table16(w[base + 2 + 4 * j + m].load(), IQ4_TABLE)):
                    acc = dp4a(v.bitcast(dtypes.int32), group[4 * h + m], acc)
            low = (lows >> (j * 4).cast(dtypes.uint32)) & 15
            high = (head >> (16 + 2 * j).cast(dtypes.uint32)) & 3
            scale = (low | (high << 4)).cast(dtypes.int32) - 32
            total = total + scale.float() * xd[g].load() * acc.float()
        return f16(head) * total

    return dot


def _mxfp4_dot(w: UOp, xq: UOp, xd: UOp, xs: UOp, cols: int) -> Dot:
    # MXFP4 block, 17 bytes: an exponent byte e, then 16 bytes whose low nibbles index values
    # 0..15 and high ones 16..31 into FP4_VALUES, doubled E2M1 as int8; a weight is the value times
    # 2^(e - 128). Blocks are only byte aligned: a block's 5 aligned words hold it, from byte
    # `skew` of the first. A unit is two blocks.
    def dot(row: UOp, unit: UOp, x: UOp | int) -> UOp:
        total = UOp.const(0.0, dtypes.float32)
        for b in range(2):
            block = unit * 2 + b
            at, g = (row * (cols // 32) + block) * 17, x * (cols // 32) + block
            first, skew = at // 4, (at % 4).cast(dtypes.uint32)
            window = [w[(first + i).minimum(int(w.shape[0]) - 1)].load() for i in range(5)]
            e = (window[0] >> (skew * 8)) & 0xFF
            later, shift = skew.eq(3), ((skew + 1) % 4) * 8  # where the 16 bytes start
            acc, group = UOp.const(0, dtypes.int32), _group(xq, g)
            for m in range(4):  # where the bytes start in the second word, they end in the fifth
                lo, hi = (later.where(window[min(m + j + 1, 4)], window[m + j]) for j in (0, 1))
                low, high = table16(funnel(lo, hi, shift), FP4_TABLE)
                acc = dp4a(low.bitcast(dtypes.int32), group[m], acc)
                acc = dp4a(high.bitcast(dtypes.int32), group[4 + m], acc)
            total = total + e8m0_half(e) * xd[g].load() * acc.float()
        return total

    return dot


DOTS: dict[GGMLType, Callable[[UOp, UOp, UOp, UOp, int], Dot]] = {
    GGMLType.Q4_K: functools.partial(_k_dot, high=False),
    GGMLType.Q5_K: functools.partial(_k_dot, high=True),
    GGMLType.Q6_K: _q6_k_dot, GGMLType.Q5_0: _q5_0_dot, GGMLType.Q8_0: _q8_0_dot,
    GGMLType.MXFP4: _mxfp4_dot, GGMLType.IQ4_XS: _iq4_xs_dot,
    **{t: functools.partial(_q4_dot, kind=t)
       for t in (GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_1, GGMLType.IQ4_NL)},
}  # fmt: skip


@functools.cache
def _matvec_kernel(
    out: UOp, w: UOp, xq: UOp, xd: UOp, xs: UOp, *residual: UOp, ggml_type: GGMLType, tokens: int
) -> UOp:
    # out[t * rows + row] = w[row] . x[t], plus residual[t * rows + row] if given
    rows = int(out.shape[0]) // tokens

    def combine(row: UOp, *totals: UOp) -> list[UOp]:
        if residual:
            totals = tuple(t + residual[0][i * rows + row].load() for i, t in enumerate(totals))
        return list(totals)

    dot = DOTS[ggml_type](w, xq, xd, xs, cols := 4 * int(xq.shape[0]) // tokens)
    dots = [_token(dot, t) for t in range(tokens)]
    name = ggml_type.name.lower()
    return rows_kernel(out, cols // 64, name, dots, combine, rows, _per_warp(tokens))


@functools.cache
def _swiglu_kernel(
    out: UOp, gate: UOp, up: UOp, xq: UOp, xd: UOp, xs: UOp, ggml_type: GGMLType, kind: str,
    tokens: int,
) -> UOp:  # fmt: skip
    # out[t * rows + row] = glu(kind)(gate[row] . x[t], up[row] . x[t])
    def combine(row: UOp, *totals: UOp) -> list[UOp]:
        return [glu(kind)(g, u) for g, u in zip(totals[::2], totals[1::2], strict=True)]

    cols = 4 * int(xq.shape[0]) // tokens
    gate_dot, up_dot = (DOTS[ggml_type](w, xq, xd, xs, cols) for w in (gate, up))
    dots = [_token(dot, t) for t in range(tokens) for dot in (gate_dot, up_dot)]
    name, rows = f"glu_{kind}_{ggml_type.name.lower()}", int(out.shape[0]) // tokens
    return rows_kernel(out, cols // 64, name, dots, combine, rows, _per_warp(tokens, 2))


def _per_warp(tokens: int, matrices: int = 1) -> int:
    # rows a warp takes, of each of `matrices` that share the activations: the more tokens, the
    # more their activations' reads are worth sharing, against the registers each row's sums
    # take. Q4_K's and Q6_K's rows of 4096 and 14336 for 4 tokens took 0.66 to 0.77 as long with
    # 2 rows a warp, and for 8, 0.47 to 0.65 with 4; gate and up, 2 matrices, for 8 tokens 0.76
    # to 0.92 as long with 2, and spilled registers with 4.
    shared = 1 if tokens == 1 else 2 if tokens <= 4 else 4  # matrices' rows per activation load
    return max(shared // matrices, 1)


def _token(dot: Dot, t: int) -> Callable[[UOp, UOp], UOp]:
    # a dot product with token t's activations
    return lambda row, unit: dot(row, unit, t)


def supports_matvec(x: Tensor, w: QTensor) -> bool:
    # up to MATVEC_TOKENS tokens, known in advance, and whole units of 64 weights
    tokens = x.numel() // x.shape[-1] if isinstance(x.numel(), int) else None
    few = isinstance(tokens, int) and 0 < tokens <= MATVEC_TOKENS
    return on_gpu(x) and few and w.type in DOTS and w.shape[1] % 64 == 0


def matvecs(
    x: Tensor,
    *ws: QTensor,
    norm: tuple[Tensor, float] | None = None,
    residual: Tensor | None = None,
) -> list[Tensor]:
    """x @ w.T for a few tokens and each w, with the activations quantized to int8 once, after
    rms_norm(x, *norm) if given. `residual` is added inside the kernel; it needs a single w."""
    assert residual is None or len(ws) == 1, "a residual goes with one matrix"
    tokens = x.numel() // x.shape[-1]
    xq, xd, xs = quantize_q8(x.reshape(tokens, x.shape[-1]), norm)
    res = () if residual is None else (residual.flatten().float().contiguous(),)
    outs = []
    for w in ws:
        out = Tensor.empty(tokens * w.shape[0], dtype=dtypes.float32, device=x.device)
        fxn = functools.partial(_matvec_kernel, ggml_type=w.type, tokens=tokens)
        out = Tensor.custom_kernel(out, storage_words(w), xq, xd, xs, *res, fxn=fxn)[0]
        outs.append(out.reshape(*x.shape[:-1], w.shape[0]))
    return outs


def swiglu(
    x: Tensor, gate: QTensor, up: QTensor, norm: tuple[Tensor, float] | None = None,
    kind: str = "silu",
) -> Tensor:  # fmt: skip
    """glu(kind)(x @ gate.T, x @ up.T) for a few tokens, as common.glu has it, after
    rms_norm(x, *norm) if given, both matrices in one kernel; they share a type and shape."""
    tokens = x.numel() // x.shape[-1]
    xq, xd, xs = quantize_q8(x.reshape(tokens, x.shape[-1]), norm)
    out = Tensor.empty(tokens * gate.shape[0], dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(_swiglu_kernel, ggml_type=gate.type, kind=kind, tokens=tokens)
    words = storage_words(gate), storage_words(up)
    out = Tensor.custom_kernel(out, *words, xq, xd, xs, fxn=fxn)[0]
    return out.reshape(*x.shape[:-1], gate.shape[0])
