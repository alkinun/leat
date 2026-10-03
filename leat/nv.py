"""NVIDIA kernels for DEV=NV and DEV=CUDA, written in tinygrad's UOp DSL and rendered as CUDA C.

Decode-time linear layers follow llama.cpp's matrix-vector scheme: the activation vector is
quantized to int8 in groups of 32, then each warp computes one output row with __dp4a over the
weights in their storage format.
"""

import functools
import math
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.quant import GGMLType, QTensor

WARP = 32
GROUP = 32  # activations per int8 scale


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


@functools.cache
def _quantize_q8_kernel(q: UOp, d: UOp, s: UOp, x: UOp) -> UOp:
    group, lane = UOp.range(int(x.shape[0]) // GROUP, 0, AxisType.GLOBAL), _lane()
    stores = _quantize_group(q, d, s, group, lane, x[group * GROUP + lane].load())
    info = KernelInfo(name="quantize_q8", opts_to_apply=())
    return UOp.group(*stores).end(group, lane).sink(arg=info)


NORM_THREADS = 1024  # one block normalizes the whole vector


@functools.cache
def _norm_quantize_q8_kernel(q: UOp, d: UOp, s: UOp, x: UOp, weight: UOp, eps: float) -> UOp:
    # RMSNorm, then quantization, in one block: the warps' sums of squares meet in shared memory,
    # then each warp normalizes and quantizes every (NORM_THREADS / 32)-th group
    n, warps = int(x.shape[0]), NORM_THREADS // WARP
    lane, wave = _lane(), UOp.range(warps, 1, AxisType.LOCAL)
    thread, per_thread = wave * WARP + lane, n // NORM_THREADS
    squares = sum(
        (x[thread * per_thread + i].load() ** 2 for i in range(per_thread)),
        UOp.const(0.0, dtypes.float32),
    )
    partial = UOp.alloc((warps,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    partial = partial.after(partial[wave.valid(lane.eq(0))].store(_warp_sum(squares)))
    total = sum((partial[w].load() for w in range(warps)), UOp.const(0.0, dtypes.float32))
    inv = (total / n + eps).rsqrt()
    stores = []
    for k in range(n // GROUP // warps):
        group = wave + k * warps
        at = group * GROUP + lane
        stores += _quantize_group(q, d, s, group, lane, x[at].load() * inv * weight[at].load())
    info = KernelInfo(name="norm_quantize_q8", opts_to_apply=())
    return UOp.group(*stores).end(wave, lane).sink(arg=info)


def quantize_q8(
    x: Tensor, norm: tuple[Tensor, float] | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantizes a vector to int8 in groups of 32, after RMSNorm with `norm`'s weight and eps.

    Returns the values packed four per int32 word, the scales d and the sums d * sum(q).
    """
    n = x.shape[0]
    q = Tensor.empty(n // 4, dtype=dtypes.int32, device=x.device)
    d = Tensor.empty(n // GROUP, dtype=dtypes.float32, device=x.device)
    s = Tensor.empty(n // GROUP, dtype=dtypes.float32, device=x.device)
    x = x.float().contiguous()
    if norm is None:
        out = Tensor.custom_kernel(q, d, s, x, fxn=_quantize_q8_kernel)
    else:
        fxn = functools.partial(_norm_quantize_q8_kernel, eps=norm[1])
        out = Tensor.custom_kernel(q, d, s, x, norm[0].float().contiguous(), fxn=fxn)
    return out[0], out[1], out[2]


def _k_scale_min(w: UOp, base: UOp, sub: UOp) -> tuple[UOp, UOp]:
    # ggml's get_scale_min_k4 over the 12 bytes after a K-quant block's d and dmin.
    # sub ^ 4 is sub - 4 where that branch is taken, and stays in range where it is not.
    def byte(i: UOp) -> UOp:
        return (w[base + 1 + i // 4].load() >> ((i % 4) * 8).cast(dtypes.uint32)) & 0xFF

    low = sub < 4
    sc = low.where(byte(sub) & 63, (byte(sub + 4) & 15) | ((byte(sub ^ 4) >> 6) << 4))
    mn = low.where(byte(sub + 4) & 63, (byte(sub + 4) >> 4) | ((byte(sub) >> 6) << 4))
    return sc.float(), mn.float()


def _rows(
    out: UOp, cols: int, name: str, dot: Callable[[UOp, UOp], UOp], residual: tuple[UOp, ...]
) -> UOp:
    # one block of one warp per output row: lanes take units of 64 weights in turn, dot(row, unit)
    # gives a unit's contribution, and the warp sums them, plus residual[row] if given. Grouping
    # rows into wider blocks measured slower on the 3090, by up to a quarter for Q6_K.
    rows = out.shape[0]
    row, lane = UOp.range(rows, 0, AxisType.GLOBAL), _lane()
    units = (dot(row, it * WARP + lane) for it in range(cols // 64 // WARP))
    total = _warp_sum(sum(units, UOp.const(0.0, dtypes.float32)))
    if residual:
        total = total + residual[0][row].load()
    store = out[row.valid(lane.eq(0))].store(total)
    info = KernelInfo(name=f"{name}_{rows}_{cols}", opts_to_apply=())
    return store.end(row, lane).sink(arg=info)


@functools.cache
def _q4_k_kernel(out: UOp, w: UOp, xq: UOp, xd: UOp, xs: UOp, *residual: UOp) -> UOp:
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

    return _rows(out, cols, "q4_k", dot, residual)


def _word16(w: UOp, i: UOp) -> UOp:
    # four bytes from a halfword-aligned buffer: 32-bit loads must be 4-byte aligned
    return w[i].load().cast(dtypes.uint32) | (w[i + 1].load().cast(dtypes.uint32) << 16)


def _minus_32(q: UOp) -> UOp:
    # Q6_K's unsigned 0..63 to signed -32..31, in each byte of a word
    centered = UOp(Ops.CUSTOMI, src=(q,), arg=("__vsubss4({}, 0x20202020u)", dtypes.uint32))
    return centered.bitcast(dtypes.int32)


@functools.cache
def _q6_k_kernel(out: UOp, w: UOp, xq: UOp, xd: UOp, xs: UOp, *residual: UOp) -> UOp:
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

    return _rows(out, cols, "q6_k", dot, residual)


# each kernel, and the word type it reads the weights as
_KERNELS = {
    GGMLType.Q4_K: (_q4_k_kernel, dtypes.uint32),
    GGMLType.Q6_K: (_q6_k_kernel, dtypes.uint16),
}


def supports(x: Tensor, w: QTensor) -> bool:
    if not isinstance(x.device, str) or x.device.split(":")[0] not in ("NV", "CUDA"):
        return False
    # one token, and whole warps: each lane takes pairs of sub-blocks, 32 lanes per row
    cols = w.shape[1]
    single = isinstance(x.numel(), int) and x.numel() == cols
    fits = isinstance(cols, int) and cols % (64 * WARP) == 0  # also a multiple of NORM_THREADS
    return w.type in _KERNELS and single and fits


def linears(
    x: Tensor,
    *ws: QTensor,
    norm: tuple[Tensor, float] | None = None,
    residual: Tensor | None = None,
) -> list[Tensor]:
    """x @ w.T for one token and each w, with the activations quantized to int8 once.

    `residual` is added to the result inside the kernel; it needs a single w.
    """
    assert residual is None or len(ws) == 1, "a residual goes with one matrix"
    xq, xd, xs = quantize_q8(x.reshape(x.shape[-1]), norm)
    res = () if residual is None else (residual.reshape(ws[0].shape[0]).float().contiguous(),)
    return [_matvec(w, xq, xd, xs, *res).reshape(*x.shape[:-1], w.shape[0]) for w in ws]


def _matvec(w: QTensor, xq: Tensor, xd: Tensor, xs: Tensor, *residual: Tensor) -> Tensor:
    out = Tensor.empty(w.shape[0], dtype=dtypes.float32, device=xq.device)
    kernel, word = _KERNELS[w.type]
    # .contiguous() on a bitcast of contiguous storage is a view; without it tinygrad copies
    words = w.data.flatten().bitcast(word).contiguous()
    return Tensor.custom_kernel(out, words, xq, xd, xs, *residual, fxn=kernel)[0]


# ******** attention: one query token against the KV cache ********
# FlashDecoding, adapted from tinygrad/llm/kernels/amd.py: the cache is cut into chunks of KEYS
# keys, blocks reduce chunks with an online softmax, and a second kernel combines their partials.

KEYS = 64  # keys per chunk
PARTIALS = 48  # most blocks per kv head; longer caches loop over several chunks per block
SHARED = 49152  # bytes of shared memory a block may use without opting in to more
PAD = 8  # halves of shared memory after each warp's outputs, see _attention_partial_kernel
LOG2E = math.log2(math.e)


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
    # a bound length rides on the cache, so the kernels' own copy can stay unbound
    if isinstance(length, UOp):
        cache, length = Tensor(cache.uop.after(length)), length.unbind_all()[0]
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
