"""Attention over the KV cache: FlashDecoding for a query token of each sequence of a decode step,
FlashAttention-2 on f16 tensor cores for a sequence's several, and tokens' queries, keys and values
readied for it: biased, normed, rotated by RoPE, and the keys and values stored in the cache."""

import functools
import math
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.kernels.common import (
    LOG2E,
    SHARED,
    WARP,
    at_least,
    at_most,
    carry,
    carry_all,
    lane_range,
    load_vector,
    on_gpu,
    on_matrix_cores,
    on_rdna3,
    opaque,
    pick,
    register,
    shfl_xor,
    turns,
    warp_sum,
)

# ******** a query token per sequence: FlashDecoding ********
# Adapted from tinygrad/llm/kernels/amd.py: the cache is cut into chunks of KEYS keys, blocks
# reduce chunks with an online softmax, and a second kernel combines their partials.

KEYS = 64  # keys per chunk
PARTIALS = 48  # most blocks per kv head; longer caches loop over several chunks per block
GROUP_SPLIT = 4  # query heads per block at most: a block's time grows with them, 5.7 us for 4
# and 10 us for 8 at short context, while reading keys and values once per block costs little
GROUP_DIMS = 512  # and their dimensions: 2 heads of 512, Gemma 4's, spill registers on RDNA,
# while 1 decodes as fast on NVIDIA


@functools.cache
def _attention_partial_kernel(
    out: UOp, stats: UOp, q: UOp, cache: UOp, slots: tuple[int | UOp, ...],
    lengths: tuple[int | UOp, ...], most: int | UOp, waves: int, scale: float, window: int,
    split: int,
) -> UOp:  # fmt: skip
    # A block takes one row, a query token, and one kv head of the row's slot and every
    # PARTIALS-th chunk of its keys, of `most` at most for any row, for a `split`-th of the query
    # heads of its GQA group, from the chunk of the first key the window holds; blocks past a
    # row's chunks take none. Each of `waves` warps scores KEYS / waves keys of a chunk; lanes
    # hold dim / 32 dimensions. The warps then merge through shared memory into one partial per
    # block: the unnormalized output, its running max and its sum of weights.
    rows, kv_heads, dim = len(slots), int(cache.shape[2]), int(cache.shape[4])
    heads = int(q.shape[0])
    group, per_lane = heads // kv_heads // split, dim // WARP  # query heads per block
    partials = int(out.shape[1])
    per_wave, zero = KEYS // waves, UOp.const(0.0, dtypes.float32)
    token = UOp.range(rows, 2, AxisType.GLOBAL)
    slot, length = pick(token, slots), pick(token, lengths)
    since = _since(length, window)
    chunks = _chunks(length, window)
    head = UOp.range(kv_heads * split, 0, AxisType.GLOBAL)  # query heads from head * group
    block = UOp.range(at_most(most, partials), 1, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(waves, 3, AxisType.LOCAL)
    qs = [load_vector(q[head * group + h, token, lane * per_lane], per_lane) for h in range(group)]
    rounds = UOp.range(((chunks - block).maximum(0) + partials - 1) // partials, 4, AxisType.LOOP)
    chunk = since // KEYS + block + rounds * partials
    valid, scores, values = [], [], []
    for j in range(per_wave):
        key = chunk * KEYS + wave * per_wave + j
        valid.append((key < length) & (key >= since) if window else key < length)
        k = load_vector(cache[0, slot, head // split, key, lane * per_lane], per_lane)
        # values load during scoring, so both streams are in flight together
        v = load_vector(cache[1, slot, head // split, key, lane * per_lane], per_lane)
        values.append([valid[j].where(x, zero) for x in v])
        dots = [warp_sum(sum((a * b for a, b in zip(qh, k, strict=True)), zero)) for qh in qs]
        scores.append([valid[j].where(d * scale, -1e30) for d in dots])
    # a finite initial max keeps fully masked warps from computing exp(-inf - -inf)
    acc = register((group, per_lane), 0.0)
    mx, total = register((group,), -1e30), register((group,), 0.0)
    prev_acc, prev_max, prev_sum = acc.after(rounds), mx.after(rounds), total.after(rounds)
    new_max = [
        functools.reduce(UOp.maximum, (sc[h] for sc in scores), prev_max[h].load())
        for h in range(group)
    ]
    rescale = [((prev_max[h].load() - new_max[h]) * LOG2E).exp2() for h in range(group)]
    accs = [[rescale[h] * prev_acc[h, i].load() for i in range(per_lane)] for h in range(group)]
    sums = [rescale[h] * prev_sum[h].load() for h in range(group)]
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
    # unless empty), as [head, dimension within lane, lane], then its max and sum. Past the rounds
    # loop, indices of opaque copies of the coordinates: see common.opaque.
    ranges = (lane, wave, block, head, token)
    lane, wave, block, head, token = (opaque(u) for u in ranges)
    # a block past its row's chunks, which ran no rounds, holds the registers' first values: where
    # tinygrad works out that a block runs one round at most, as in a window, it writes those
    # inside the loop
    ran = block < _chunks(pick(token, lengths), window)
    sums = [ran.where(total[h].load(), 0.0) for h in range(group)]
    tops = [ran.where(mx[h].load(), -1e30) for h in range(group)]
    shared = UOp.alloc((waves, group * dim), dtypes.half, addrspace=AddrSpace.LOCAL)
    stat = UOp.alloc((waves, group, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
    stores = [
        shared[wave, (h * per_lane + i) * WARP + lane].store(
            (ran.where(acc[h, i].load(), 0.0) / sums[h].maximum(1)).cast(dtypes.half)
        )
        for h in range(group)
        for i in range(per_lane)
    ]
    stores += [
        stat[wave, h, i].store(x) for h in range(group) for i, x in enumerate((tops[h], sums[h]))
    ]
    shared, stat = shared.after(*stores), stat.after(*stores)
    thread, results, first = wave * WARP + lane, [], token * heads + head * group
    for i in range(-(-group * dim // (waves * WARP))):
        flat = thread + i * waves * WARP
        hq, d = flat // dim, flat % dim  # a query head of the block, and a dimension
        top = functools.reduce(UOp.maximum, (stat[w, hq, 0].load() for w in range(waves)))
        val = sum((((stat[w, hq, 0].load() - top) * LOG2E).exp2() * stat[w, hq, 1].load()
                   * shared[w, (hq * per_lane + d % per_lane) * WARP + d // per_lane].load().float()
                   for w in range(waves)), zero)  # fmt: skip
        live = first + hq
        if group * dim % (waves * WARP):
            live = live.valid(flat < group * dim)
        results.append(out[live, block, d].store(val))
    top = functools.reduce(UOp.maximum, (stat[w, thread, 0].load() for w in range(waves)))
    weights = sum((((stat[w, thread, 0].load() - top) * LOG2E).exp2() * stat[w, thread, 1].load()
                   for w in range(waves)), zero)  # fmt: skip
    qh = (first + thread).valid(thread < group)
    results += [stats[qh, block, 0].store(top), stats[qh, block, 1].store(weights)]
    info = KernelInfo(name="attention_partial", opts_to_apply=())
    return UOp.group(*results).end(*ranges).sink(arg=info)


@functools.cache
def _attention_combine_kernel(
    o: UOp, partial: UOp, stats: UOp, *sinks: UOp, live: int | UOp
) -> UOp:
    # one warp per query head of a row and tile of dimensions: weights each block's partial by
    # exp(max - max), counting each head's sink, if given, once: a score that adds no value.
    # Blocks that took no keys weigh nothing. Rows of several heads take the sink of row % heads.
    heads, _, dim = partial.shape
    tile = _combine_tile(int(dim))
    assert tile is not None, f"no tile of the combine kernel divides heads of {dim}"
    per_lane = tile // WARP
    head, part = UOp.range(heads, 0, AxisType.GLOBAL), UOp.range(dim // tile, 1, AxisType.GLOBAL)
    lane = lane_range()
    dims = [part * tile + lane * per_lane + i for i in range(per_lane)]
    c1 = UOp.range(live, 100, AxisType.LOOP)
    top = register((1,), -math.inf)
    top = top.after(top.store(top.after(c1).maximum(stats[head, c1, 0].load())).end(c1))
    sink = sinks[0][head % int(sinks[0].shape[0])].load() if sinks else None
    best = top[0].load() if sink is None else top[0].load().maximum(sink)
    c2 = UOp.range(live, 101, AxisType.LOOP)
    acc = register((per_lane,), 0.0)
    total = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
    start = 0.0 if sink is None else ((sink - best) * LOG2E).exp2()
    total = total.after(total[0].store(start))
    weight = ((stats[head, c2, 0].load() - best) * LOG2E).exp2()
    update = UOp.group(
        *[
            acc[i].store(acc.after(c2)[i].load() + weight * partial[head, c2, d].load())
            for i, d in enumerate(dims)
        ],
        total[0].store(total.after(c2)[0].load() + weight * stats[head, c2, 1].load()),
    ).end(c2)
    acc, total = acc.after(update), total.after(update)
    stores = [o[head, d].store(acc[i].load() / total[0].load()) for i, d in enumerate(dims)]
    info = KernelInfo(name="attention_combine", opts_to_apply=())
    return UOp.group(*stores).end(lane, part, head).sink(arg=info)


def _combine_tile(dim: int) -> int | None:
    # the dimensions a warp of the combine kernel takes, if any divide heads of `dim`: 64, two a
    # lane, or else one a lane, as Phi-3 mini's heads of 96 need
    return next((tile for tile in (64, WARP) if dim % tile == 0), None)


def _since(length: int | UOp, window: int) -> int | UOp:
    # the first of `length` positions that a window of the last `window` holds, 0 for no window
    return at_least(length - window, 0) if window else 0


def _chunks(length: int | UOp, window: int) -> int | UOp:
    # the chunks of KEYS keys that hold the positions a window of the last `window` of `length`
    # holds, all of them for no window
    return (length + KEYS - 1) // KEYS - _since(length, window) // KEYS


def _most(lengths: tuple[int | UOp, ...], window: int) -> int | UOp:
    # the most chunks any row's keys take
    return functools.reduce(_larger, (_chunks(n, window) for n in lengths))


def _larger(a: int | UOp, b: int | UOp) -> int | UOp:
    if isinstance(a, UOp):
        return a.maximum(b)
    return b.maximum(a) if isinstance(b, UOp) else max(a, b)


def _per_block(group: int, dim: int) -> int:
    # query heads per block, of a GQA group: as many as GROUP_SPLIT and GROUP_DIMS allow, and a
    # divisor of the group
    most = max(min(GROUP_SPLIT, GROUP_DIMS // dim), 1)
    return max(n for n in range(1, most + 1) if group % n == 0)


def supports_attention(q: Tensor, cache: Tensor) -> bool:
    # a token per row, as many rows as known in advance, over an f16 cache
    shape = (*cache.shape[2:], *q.shape)
    if not on_gpu(q) or cache.dtype != dtypes.half or not all(isinstance(x, int) for x in shape):
        return False
    kv_heads, n, dim, batch, heads, _, _ = (int(x) for x in shape)
    group = _per_block(heads // kv_heads, dim)
    fits = group * dim * 2 + group * 8 <= SHARED  # one warp's share of shared memory
    return batch == 1 and dim % 64 == 0 and n % KEYS == 0 and fits


def attention(
    q: Tensor, cache: Tensor, slots: list[int | UOp], lengths: list[int | UOp], scale: float,
    window: int = 0, sinks: Tensor | None = None, ends: list[int] | None = None,
) -> Tensor:  # fmt: skip
    """Attention of rows of one query token each, q (1, H, T, D): row t over the first
    `lengths[t]` positions of slot `slots[t]` of the cache (2, slots, KV_H, positions, D), or the
    last `window` of them, with scores q.k * scale, and a sink (H,) per head if given, a score
    that adds no value. Returns (1, T, H * D).

    Rows of several sequences' consecutive tokens may give `ends`, the rows of each one's last,
    the longest: the kernels size their work by those alone, as tinygrad's expression for the
    longest of many rows sharing a variable grows past rendering. A row before another of its
    sequence's in a window may take a chunk more than the last's."""
    _, heads, rows, dim = (int(x) for x in q.shape)
    kv_heads = int(cache.shape[2])
    group = _per_block(heads // kv_heads, dim)
    split = heads // kv_heads // group  # blocks per kv head
    q = q.reshape(heads, rows, dim).float().contiguous()
    q, slot_vars = carry_all(q, slots)
    q, length_vars = carry_all(q, lengths)
    waves = 16
    while waves * (group * dim * 2 + group * 8) > SHARED:
        waves //= 2
    chunks = min(PARTIALS, int(cache.shape[3]) // KEYS)
    partial = Tensor.empty(rows * heads, chunks, dim, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(rows * heads, chunks, 2, dtype=dtypes.float32, device=q.device)
    longest = [length_vars[i] for i in ends] if ends else length_vars
    most = _most(tuple(longest), window)
    if ends and window:
        most = most + 1
    fxn = functools.partial(
        _attention_partial_kernel, slots=slot_vars, lengths=length_vars, most=most, waves=waves,
        scale=scale, window=window, split=split,
    )  # fmt: skip
    outs = Tensor.custom_kernel(partial, stats, q, cache, fxn=fxn)
    live = at_most(most, chunks)
    out = Tensor.empty(rows * heads, dim, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_attention_combine_kernel, live=live)
    extra = () if sinks is None else (sinks.float().contiguous(),)
    out = Tensor.custom_kernel(out, outs[0], outs[1], *extra, fxn=fxn)[0]
    return out.reshape(1, rows, heads * dim)


# ******** several query tokens: FlashAttention-2 ********
# A block takes 16 query tokens and one kv head, with a warp for each query head of the GQA group;
# the warps share tiles of 32 keys and values in shared memory, and each keeps its rows' scores,
# softmax statistics and outputs in registers, on NVIDIA's tensor cores or RDNA 3's. Tiles of 32
# keys leave room for more blocks per SM than 64: 9 to 17% faster from 0 to 8k cached tokens. Heads
# wider than 256, Gemma 4's of 512, would not fit in registers: blocks take parts of 256 of their
# outputs, each working out every score, and read the queries as they go, with tiles of 16 keys to
# fit in shared memory. A single tile of queries, as while prefilling up to 16 tokens, would leave
# SMs idle: blocks split its key tiles, as FlashDecoding does, and FlashDecoding's combine kernel
# merges their outputs.

QUERIES, KEY_TILE, PART = 16, 32, 256
SPLIT_BLOCKS = 64  # blocks that split one tile of queries' keys aim for
HELD = 128  # the widest heads whose queries WMMA's warps hold in registers for the whole loop

# mma.sync on f16 with f32 accumulation: c (4 f32) + a 16 x 16 tile times a 16 x 8 tile, from the
# lane's 4 and 2 words of f16 pairs; results return as in matmul's _MMA
_MMA_F16 = (
    '[&]{{ float d0, d1, d2, d3; asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 '
    '{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%10,%11,%12,%13}};" '
    ': "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) '
    ': "r"({1}), "r"({2}), "r"({3}), "r"({4}), "r"({5}), "r"({6}), '
    '"f"({7}), "f"({8}), "f"({9}), "f"({10})); '
    "{0}[1] = d1; {0}[2] = d2; {0}[3] = d3; return d0; }}()"
)

# WMMA on f16 with f32 accumulation: c (8 f32) + a 16 x 16 tile times a 16 x 16 tile, k = 16, from
# the lane's 8 words of f16 pairs of each; results return as in matmul's _WMMA
_WMMA_F16 = (
    "[&]{{ typedef unsigned v8u __attribute__((ext_vector_type(8))); "
    "typedef _Float16 v16h __attribute__((ext_vector_type(16))); "
    "typedef float v8f __attribute__((ext_vector_type(8))); "
    "v8u a = {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}}; "
    "v8u b = {{{9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}}}; "
    "v8f c = {{{17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}}}; "
    "v8f d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32("
    "__builtin_bit_cast(v16h, a), __builtin_bit_cast(v16h, b), c); "
    "for (int i = 1; i < 8; i++) {0}[i] = d[i]; return d[0]; }}()"
)


def _products(code: str, a: list[UOp], b: list[UOp], c: list[UOp]) -> list[UOp]:
    d = UOp.alloc((len(c),), dtypes.float32, addrspace=AddrSpace.REG)
    product = UOp(Ops.CUSTOM, src=(d[0], *a, *b, *c), arg=(code, dtypes.float32))
    d = d.after(d[0].store(product))
    return [d[i].load() for i in range(len(c))]


def _f16_pair(lo: UOp, hi: UOp) -> UOp:
    # two f32 as one word of f16, lo in the low half; cvt puts its first source in the high half
    code = (
        '[&]{{ unsigned r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"({1}), "f"({0})); '
        "return r; }}()"
    )
    return UOp(Ops.CUSTOM, src=(lo, hi), arg=(code, dtypes.uint32))


def _halves_word(lo: UOp, hi: UOp) -> UOp:
    lo, hi = (x.bitcast(dtypes.uint16).cast(dtypes.uint32) for x in (lo, hi))
    return lo | (hi << 16)


class _MmaQueries:
    """A warp's products of 16 queries on mma.sync's fragments, of 8 keys or 8 dimensions: lane
    4g + t holds queries g and g + 8, the scores of keys 2t and 2t + 1 of each 8 and the outputs of
    dimensions 2t and 2t + 1 of each 8; the 4 lanes of a query share its max and sum. `queries`
    are the lane's of the tile, and `leads`, whether it writes their statistics."""

    def __init__(self, lane: UOp):
        self.g, self.t = lane // 4, lane % 4
        self.queries: tuple[UOp, ...] = (self.g, self.g + 8)
        self.leads = self.t.eq(0)

    def reduce(self, x: UOp, op: Callable[[UOp, UOp], UOp]) -> UOp:
        # op over the lanes that hold x's query
        return warp_sum(x, 4, op)

    def scores(
        self, query: Callable[[int, int | UOp], UOp], keys: UOp, dim: int, key_tile: int
    ) -> list[tuple[int, UOp, UOp]]:  # fmt: skip
        # the lane's q.k of a tile of keys, as (its query r, the key, the score), query(r, w)
        # giving word w of query r
        zero = UOp.const(0.0, dtypes.float32)
        products = [[zero] * 4 for _ in range(key_tile // 8)]  # by tiles of 8 keys
        for k in range(dim // 16):
            a = [query(r, 8 * k + 4 * h + self.t) for h in (0, 1) for r in (0, 1)]
            for j, c in enumerate(products):
                b = [keys[8 * j + self.g, 8 * k + 4 * h + self.t].load() for h in (0, 1)]
                products[j] = _products(_MMA_F16, a, b, c)
        keys_ = [8 * j + 2 * self.t + e % 2 for j in range(key_tile // 8) for e in range(4)]
        return [(e % 4 // 2, keys_[e], c) for e, c in enumerate(x for p in products for x in p)]

    def place(self, i: int) -> tuple[int, UOp]:
        # output i's query r and dimension, of the part
        n, e = divmod(i, 4)
        return e // 2, 8 * n + 2 * self.t + e % 2

    def weigh(
        self, p: list[UOp], values: UOp, acc: list[UOp], width: int, key_tile: int
    ) -> list[UOp]:  # fmt: skip
        # acc plus the weights p, in the order of scores, times the values: the results of two
        # tiles of 8 keys make one A fragment of 16
        weights = [
            [_f16_pair(*p[8 * k + 4 * i + 2 * r : 8 * k + 4 * i + 2 * r + 2])
             for i in (0, 1) for r in (0, 1)]
            for k in range(key_tile // 16)
        ]  # fmt: skip
        outs: list[UOp] = []
        for n in range(width // 8):
            c = acc[4 * n : 4 * n + 4]
            for k in range(key_tile // 16):
                b = [values[8 * n + self.g, 8 * k + 4 * h + self.t].load() for h in (0, 1)]
                c = _products(_MMA_F16, weights[k], b, c)
            outs += c
        return outs


class _WmmaQueries(_MmaQueries):
    """WMMA's, of 16 keys or 16 dimensions by the 16 queries, transposed so that a lane holds one
    query: lane l holds query l % 16, the scores of keys 8h + e of each 16, e < 8, for h = l // 16,
    lane l ^ 16 the others, and the outputs of dimensions 2e + h of each 16."""

    def __init__(self, lane: UOp):
        self.i, self.h = lane % 16, lane // 16
        self.queries, self.leads = (self.i,), self.h.eq(0)

    def reduce(self, x: UOp, op: Callable[[UOp, UOp], UOp]) -> UOp:
        return op(x, shfl_xor(x, 16))

    def scores(
        self, query: Callable[[int, int | UOp], UOp], keys: UOp, dim: int, key_tile: int
    ) -> list[tuple[int, UOp, UOp]]:  # fmt: skip
        # row i of a tile of 16 keys is its key 8 (i % 2) + i // 2, so that the results of rows
        # 2e + h are keys 8h + e
        zero = UOp.const(0.0, dtypes.float32)
        products = [[zero] * 8 for _ in range(key_tile // 16)]
        key = 8 * (self.i % 2) + self.i // 2
        for k in range(dim // 16):
            b = [query(0, 8 * k + w) for w in range(8)]
            for j, c in enumerate(products):
                a = [keys[16 * j + key, 8 * k + w].load() for w in range(8)]
                products[j] = _products(_WMMA_F16, a, b, c)
        return [
            (0, 16 * j + 8 * self.h + e, c[e]) for j, c in enumerate(products) for e in range(8)
        ]

    def place(self, i: int) -> tuple[int, UOp]:
        n, e = divmod(i, 8)
        return 0, 16 * n + 2 * e + self.h

    def weigh(
        self, p: list[UOp], values: UOp, acc: list[UOp], width: int, key_tile: int
    ) -> list[UOp]:  # fmt: skip
        # the B fragment of 16 keys is a lane's 4 words of weights and lane l ^ 16's, keys 0..7
        # first
        first, weights = self.h.eq(0), []
        for j in range(key_tile // 16):
            halves = [x.cast(dtypes.half) for x in p[8 * j : 8 * j + 8]]
            mine = [_halves_word(*halves[2 * m : 2 * m + 2]) for m in range(4)]
            theirs = [shfl_xor(w, 16) for w in mine]
            pairs = list(zip(mine, theirs, strict=True))
            weights.append(
                [first.where(a, b) for a, b in pairs] + [first.where(b, a) for a, b in pairs]
            )
        outs: list[UOp] = []
        for n in range(width // 16):
            c = acc[8 * n : 8 * n + 8]
            for j in range(key_tile // 16):
                a = [values[16 * n + self.i, 8 * j + w].load() for w in range(8)]
                c = _products(_WMMA_F16, a, weights[j], c)
            outs += c
        return outs


@functools.cache
def _flash_attention_kernel(
    out: UOp, *srcs: UOp, slot: int | UOp, start: int | UOp, tokens: int | UOp, window: int,
    key_tile: int, parts: int, splits: int, wmma: bool = False,
) -> UOp:  # fmt: skip
    # srcs: q (count, heads, dim) in f16, scaled so that exp2 gives the softmax, and the f16
    # cache, read as words of f16 pairs; query i is at position start + i and sees the slot's
    # positions up to it, or the last `window` of them. Blocks take key_tile keys at a time, and
    # 1 / parts of their outputs, which warps work out on WMMA's fragments if wmma, else on
    # mma.sync's. Split, the queries are one tile, whose key tiles blocks take in turn, each
    # writing out (count * heads, splits, dim) unnormalized and first in srcs the max and sum of
    # each row's weights, for _attention_combine_kernel to merge. A sink per head, last in srcs if
    # given, starts each row's max and sum where one block takes all its keys.
    stats, (q, cache, *sinks) = (srcs[0], srcs[1:]) if splits > 1 else (None, srcs)
    heads, dim = int(q.shape[1]), int(q.shape[2])
    _, slots, kv_heads, positions, _ = (int(d) for d in cache.shape)
    words, width = dim // 2, dim // parts  # the part's dimensions of the output
    cache = cache.flatten().bitcast(dtypes.uint32).reshape(2, slots, kv_heads, positions, words)
    group = heads // kv_heads
    threads = group * WARP
    tile: UOp = UOp.const(0, dtypes.weakint)
    if splits == 1:
        tile = UOp.range((tokens + QUERIES - 1) // QUERIES, 0, AxisType.GLOBAL)
    kv_head = UOp.range(kv_heads, 1, AxisType.GLOBAL)
    lane, warp = lane_range(), UOp.range(group, 2, AxisType.LOCAL)
    part = UOp.range(parts, 4, AxisType.GLOBAL) if parts > 1 else UOp.const(0, dtypes.weakint)
    head, tid = kv_head * group + warp, warp * WARP + lane
    frags = (_WmmaQueries if wmma else _MmaQueries)(lane)
    rows = [tile * QUERIES + r for r in frags.queries]  # the lane's queries
    # the keys and values from the tile's first query's window to its last query, key_tile at a
    # time: keys as they are in the cache, values transposed so that fragments of keys are words
    end = start + (tile * QUERIES + QUERIES).minimum(tokens)
    first = _since(start + tile * QUERIES + 1, window) // key_tile
    seen_tiles = (end + key_tile - 1) // key_tile - first
    split = UOp.const(0, dtypes.weakint)
    if splits > 1:  # every splits-th of them from the split's on
        split = UOp.range(at_most(seen_tiles, splits), 5, AxisType.GLOBAL)
        tiles = UOp.range((seen_tiles - 1 - split) // splits + 1, 3, AxisType.LOOP)
        kt = first + split + tiles * splits
    else:
        tiles = UOp.range(seen_tiles, 3, AxisType.LOOP)
        kt = first + tiles
    # the queries in registers for the whole loop with one part, if they fit; else read again
    # each time
    held = parts == 1 and (not wmma or dim <= HELD)
    src = q if held else q.after(tiles)

    def query(r: int, w: int | UOp) -> UOp:  # word w of the lane's query r
        return _halves_word(*(src[rows[r], head, 2 * w + i].load() for i in (0, 1)))

    keys = UOp.alloc((key_tile, words + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    values = UOp.alloc((width, key_tile // 2 + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    stores = []

    for item in turns(key_tile * words, threads, tid):
        key, w = item // words, item % words
        stores.append(keys[key, w].store(cache[0, slot, kv_head, kt * key_tile + key, w].load()))
    for item in turns(key_tile // 2 * width // 2, threads, tid):
        pair, w = item % (key_tile // 2), item // (key_tile // 2)
        a, b = (
            cache[1, slot, kv_head, kt * key_tile + 2 * pair + j, part * width // 2 + w].load()
            for j in (0, 1)
        )
        stores.append(values[2 * w, pair].store((a & 0xFFFF) | (b << 16)))
        stores.append(values[2 * w + 1, pair].store((a >> 16) | (b & 0xFFFF0000)))
    keys, values = keys.after(*stores), values.after(*stores)

    # a finite initial max keeps exp2 from seeing -inf - -inf
    n = len(rows)
    acc, mx, total = register((width // 2,), 0.0), register((n,), -1e30), register((n,), 0.0)
    if sinks and splits == 1:  # in the scores' units, of exp2
        sink = sinks[0][head].load() * LOG2E
        mx = UOp.alloc((n,), dtypes.float32, addrspace=AddrSpace.REG)
        mx = mx.after(mx.store(UOp.stack(*[sink] * n)))
        total = register((n,), 1.0)
    prev_acc, prev_max, prev_total = acc.after(tiles), mx.after(tiles), total.after(tiles)
    of, scores = [], []  # the query of each of the lane's scores, and the scores masked to the
    # positions it sees
    for r, key, score in frags.scores(query, keys, dim, key_tile):
        back = start + rows[r] - (kt * key_tile + key)  # how far the key is
        seen = (back >= 0) & (back < window) if window else back >= 0
        of.append(r)
        scores.append(seen.where(score, -math.inf))

    def per_query(xs: list[UOp], op: Callable[[UOp, UOp], UOp]) -> list[UOp]:
        # op over each of the lane's queries' xs, and over the lanes that hold the query
        mine = [
            functools.reduce(op, (x for i, x in zip(of, xs, strict=True) if i == r))
            for r in range(n)
        ]
        return [frags.reduce(x, op) for x in mine]

    new_max = [prev_max[r].load().maximum(x) for r, x in enumerate(per_query(scores, UOp.maximum))]
    rescale = [(prev_max[r].load() - new_max[r]).exp2() for r in range(n)]
    p = [(x - new_max[r]).exp2() for r, x in zip(of, scores, strict=True)]
    sums = per_query(p, UOp.__add__)
    before = [prev_acc[i].load() * rescale[frags.place(i)[0]] for i in range(width // 2)]
    update = UOp.group(
        acc.store(UOp.stack(*frags.weigh(p, values, before, width, key_tile))),
        mx.store(UOp.stack(*new_max)),
        total.store(UOp.stack(*(prev_total[r].load() * rescale[r] + sums[r] for r in range(n)))),
    ).end(tiles)
    acc, mx, total = acc.after(update), mx.after(update), total.after(update)
    # past the loop, the fragments of an opaque copy of the lane: see common.opaque
    frags = type(frags)(opaque(lane))
    rows = [tile * QUERIES + r for r in frags.queries]
    results = []
    for i in range(width // 2):
        r, d = frags.place(i)
        value, at = acc[i].load(), part * width + d
        if splits > 1:
            results.append(out[rows[r] * heads + head, split, at].store(value))
            continue
        results.append(out[rows[r], head * dim + at].store(value / total[r].load()))
    if stats is not None:  # the max in units of e, as the combine kernel takes it
        for r, row in enumerate(rows):
            for i, x in enumerate((mx[r].load() / LOG2E, total[r].load())):
                results.append(stats[(row * heads + head).valid(frags.leads), split, i].store(x))
    ranges = (kv_head, lane, warp, *(x for x in (tile, part, split) if x.op is Ops.RANGE))
    info = KernelInfo(name="flash_attention", opts_to_apply=())
    return UOp.group(*results).end(*ranges).sink(arg=info)


def _flash_shape(positions: int, dim: int) -> tuple[int, int] | None:
    # the key tile and parts the kernel takes for heads of `dim`, if it fits them: of 64 or more,
    # as tinygrad's rewrites of the kernel stall on narrower ones, which no supported model has
    if dim < 64:
        return None
    parts = -(-dim // PART)
    width = dim // parts
    for key_tile in (KEY_TILE, KEY_TILE // 2):
        shared = 4 * (key_tile * (dim // 2 + 4) + width * (key_tile // 2 + 4))
        if dim % (16 * parts) == 0 and positions % key_tile == 0 and shared <= SHARED:
            return key_tile, parts
    return None


def supports_flash_attention(q: Tensor, cache: Tensor) -> bool:
    # one sequence's queries, and heads and an f16 cache the kernel's tiles fit
    positions, dim = cache.shape[3:]
    if not on_matrix_cores(q) or cache.dtype != dtypes.half:
        return False
    if not isinstance(positions, int) or not isinstance(dim, int):
        return False
    return q.shape[0] == 1 and _flash_shape(positions, dim) is not None


def flash_attention(
    q: Tensor, cache: Tensor, slot: int | UOp, start_pos: int | UOp, scale: float, window: int = 0,
    sinks: Tensor | None = None,
) -> Tensor:  # fmt: skip
    """Causal attention of query tokens (1, H, T, D) at positions start_pos.. over a slot of the
    cache, which already holds their keys and values, or over the last `window` positions of
    each, with scores q.k * scale, and a sink (H,) per head if given. Returns (1, T, H * D)."""
    _, heads, tokens, dim = q.shape
    count = -(-q.max_shape[2] // QUERIES) * QUERIES
    shape = _flash_shape(int(cache.shape[3]), int(dim))
    assert shape is not None, "flash_attention needs supports_flash_attention"
    # in the layout of the projection that made q, where scaling and rounding it is a plain copy
    q = (q.transpose(1, 2).reshape(tokens, heads, dim).float() * (LOG2E * scale)).half()
    q = q.pad_to((count, heads, dim)).contiguous()
    q, start = carry(q, start_pos)
    cache, bound = carry(cache, tokens)
    # one tile of queries, while prefilling few tokens, leaves SMs idle: blocks split its keys,
    # where the combine kernel takes the heads. For Llama 3.1 8B's 8 kv heads, 8 blocks each took
    # 27 us at 2000 positions, 16 took 29 and one 170.
    split = count == QUERIES and _combine_tile(int(dim)) is not None
    splits = max(SPLIT_BLOCKS // (int(cache.shape[2]) * shape[1]), 1) if split else 1
    rows = count * heads  # of queries and heads
    shapes = [(rows, splits, n) for n in (dim, 2)] if splits > 1 else [(count, heads * dim)]
    outs = [Tensor.empty(*s, dtype=dtypes.float32, device=q.device) for s in shapes]
    outs[0], slot = carry(outs[0], slot)
    fxn = functools.partial(
        _flash_attention_kernel, slot=slot, start=start, tokens=bound, window=window,
        key_tile=shape[0], parts=shape[1], splits=splits, wmma=on_rdna3(q),
    )  # fmt: skip
    extra = () if sinks is None else (sinks.float().contiguous(),)
    outs = Tensor.custom_kernel(*outs, q, cache, *(extra if splits == 1 else ()), fxn=fxn)
    if splits > 1:
        end = (start + at_most(bound, QUERIES) + shape[0] - 1) // shape[0]
        live = at_most(end - _since(start + 1, window) // shape[0], splits)
        out = Tensor.empty(rows, dim, dtype=dtypes.float32, device=q.device)
        fxn = functools.partial(_attention_combine_kernel, live=live)
        outs = Tensor.custom_kernel(out, *outs[:2], *extra, fxn=fxn)
    return outs[0].reshape(count, heads * dim)[:tokens].reshape(1, tokens, heads * dim)


# ******** tokens' queries, keys and values: biases, norms, RoPE, and the cache ********


@functools.cache
def _rotate_kernel(
    out: UOp, cache: UOp, q: UOp, k: UOp, v: UOp, *extra: UOp,
    slots: tuple[int | UOp, ...] | int | UOp, positions: tuple[int | UOp, ...] | int | UOp,
    rotated: int, halves: bool, biased: bool, v_norm: bool, eps: float,
) -> UOp:  # fmt: skip
    # A warp per token and head of q, then of k and its v. Token t is in slot slots[t] at
    # positions[t], or for a single slot and position, in that slot at that position plus t.
    # extra holds RoPE's cos and sin (positions, rotated / 2) if any dimensions rotate, then each
    # of q's, k's and v's bias if biased, and q's and k's norm weights if any. Each of q, k and v
    # gets its bias; each head of q and k is then normed with its weight, and of v without, if
    # v_norm; the first `rotated` dimensions of q and k are rotated, q into out as (heads, tokens,
    # dim), the layout attention reads, and k into the cache with v. Lanes take pairs of
    # dimensions: those that rotate together, i and i + rotated / 2 or adjacent ones, each pair i
    # turning by angle cos[pos, i], sin[pos, i], and the others' adjacent pairs as they are.
    (cos, sin), extra = (extra[:2], extra[2:]) if rotated else ((None, None), extra)
    biases, norms = (extra[:3], extra[3:]) if biased else ((), extra)
    _, cache_slots, kv_heads, cache_positions, dim = (int(d) for d in cache.shape)
    tokens = int(k.shape[0]) // (kv_heads * dim)
    heads, half, turning = int(out.shape[0]) // (tokens * dim), dim // 2, rotated // 2
    warp, lane = UOp.range(heads + kv_heads, 0, AxisType.GLOBAL), lane_range()
    token = UOp.range(tokens, 1, AxisType.GLOBAL)
    slot = pick(token, slots) if isinstance(slots, tuple) else slots
    pos = pick(token, positions) if isinstance(positions, tuple) else positions + token
    # opaque: where k has a single head, tinygrad would find its warp's index a constant under
    # its gate, and then drop the warp from the launch or the gate from its stores
    head = opaque(warp)
    is_q, is_kv, kv = head < heads, head >= heads, head - heads
    zero = UOp.const(0.0, dtypes.float32)
    pairs = [lane + WARP * m for m in range(half // WARP)]

    def pair_dims(i: UOp, m: int) -> tuple[UOp, UOp]:  # the dimensions of pair i, of turn m
        kept = (2 * i, 2 * i + 1)
        if WARP * m >= turning or not halves:
            return kept
        both = i < turning
        return both.where(i, kept[0]), both.where(i + turning, kept[1])

    dims = [pair_dims(i, m) for m, i in enumerate(pairs)]
    flat = [d for pair in dims for d in pair]

    def of_q_or_k(qs: UOp, ks: UOp, d: UOp, per_token: bool = True) -> UOp:
        # dimension d of the warp's head, of q or k: of its token's, or of a bias
        at_q, at_kv = (token * heads + head, token * kv_heads + kv) if per_token else (head, kv)
        return qs[(at_q * dim + d).valid(is_q)].load() + ks[(at_kv * dim + d).valid(is_kv)].load()

    def of_v(vs: UOp, d: UOp, per_token: bool = True) -> UOp:
        at = token * kv_heads + kv if per_token else kv
        return vs[(at * dim + d).valid(is_kv)].load()

    def normed(values: list[UOp], weights: list[UOp] | None = None) -> list[UOp]:
        inv = (warp_sum(sum((x * x for x in values), zero)) / dim + eps).rsqrt()
        if weights is None:
            return [x * inv for x in values]
        return [x * inv * w for x, w in zip(values, weights, strict=True)]

    def stored(kind: int, d: UOp) -> UOp:  # where dimension d of the warp's key or value goes
        at = (((kind * cache_slots + slot) * kv_heads + kv) * cache_positions + pos) * dim + d
        return cache.flatten()[at.valid(is_kv)]

    x = [of_q_or_k(q, k, d) for d in flat]
    if biases:
        x = [a + of_q_or_k(biases[0], biases[1], d, False) for a, d in zip(x, flat, strict=True)]
    if norms:
        x = normed(x, [is_q.where(*(w[d].load() for w in norms)) for d in flat])
    stores = []
    for n, ((d0, d1), i) in enumerate(zip(dims, pairs, strict=True)):
        x0, x1 = x[2 * n], x[2 * n + 1]
        if WARP * n < turning:  # pairs past the rotated ones turn by 0
            assert cos is not None and sin is not None
            turns = i < turning if WARP * (n + 1) > turning else None
            at = i.minimum(turning - 1) if turns is not None else i
            c, s = cos[pos, at].load(), sin[pos, at].load()
            if turns is not None:
                c, s = turns.where(c, 1.0), turns.where(s, 0.0)
            x0, x1 = x0 * c - x1 * s, x0 * s + x1 * c
        for d, value in ((d0, x0), (d1, x1)):
            stores.append(out[((head * tokens + token) * dim + d).valid(is_q)].store(value))
            stores.append(stored(0, d).store(value.cast(dtypes.half)))
    values = [of_v(v, d) + of_v(biases[2], d, False) if biases else of_v(v, d) for d in flat]
    if v_norm:
        values = normed(values)
    for value, d in zip(values, flat, strict=True):
        stores.append(stored(1, d).store(value.cast(dtypes.half)))
    info = KernelInfo(name="rotate", opts_to_apply=())
    return UOp.group(*stores).end(warp, token, lane).sink(arg=info)


def supports_rotate(q: Tensor, cache: Tensor) -> bool:
    # tokens known in advance, and whole warps of pairs of dimensions of an f16 cache
    if not on_gpu(q) or cache.dtype != dtypes.half or not isinstance(q.numel(), int):
        return False
    return int(cache.shape[4]) % (2 * WARP) == 0


def rotate(
    q: Tensor, k: Tensor, v: Tensor, cache: Tensor, slots: list[int | UOp] | int | UOp,
    positions: list[int | UOp] | int | UOp, rope: tuple[tuple[Tensor, Tensor], int] | None,
    halves: bool, biases: tuple[Tensor, Tensor, Tensor] | None,
    norms: tuple[Tensor, Tensor] | None, v_norm: bool, eps: float,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    """For tokens' q (1, T, H, D), k and v (1, T, KV_H, D): adds their biases, if given; norms
    each head of q and k with its weight, if given, and of v without, if v_norm; rotates the first
    R dimensions of q and k by RoPE's tables (positions, R/2), for rope ((cos, sin), R) if given;
    and stores k and v in the cache. Token t is at positions[t] of slot slots[t], or for a single
    slot and position, at that position plus t of that slot. Returns q (1, H, T, D) and the
    cache."""
    _, tokens, heads, dim = (int(x) for x in q.shape)
    out = Tensor.empty(tokens * heads * dim, dtype=dtypes.float32, device=q.device)
    q, k, v = (t.reshape(-1).float().contiguous() for t in (q, k, v))
    q, slot_vars = carry_all(q, slots if isinstance(slots, list) else [slots])
    q, position_vars = carry_all(q, positions if isinstance(positions, list) else [positions])
    tables = () if rope is None else rope[0]
    extra = tuple(w.float().contiguous() for w in (*(biases or ()), *(norms or ())))
    fxn = functools.partial(
        _rotate_kernel, slots=slot_vars if isinstance(slots, list) else slot_vars[0],
        positions=position_vars if isinstance(positions, list) else position_vars[0],
        rotated=0 if rope is None else rope[1], halves=halves, biased=biases is not None,
        v_norm=v_norm, eps=eps,
    )  # fmt: skip
    out, cache = Tensor.custom_kernel(out, cache, q, k, v, *tables, *extra, fxn=fxn)[:2]
    return out.reshape(1, heads, tokens, dim), cache
