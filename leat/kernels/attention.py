"""Attention over the KV cache: FlashDecoding for one query token, FlashAttention-2 on f16 tensor
cores for several, and one token's queries, keys and values readied for it: biased, normed,
rotated by RoPE, and the keys and values stored in the cache."""

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
    at_most,
    carry,
    lane_range,
    load_vector,
    on_gpu,
    on_nvidia,
    opaque,
    register,
    shfl_xor,
    warp_sum,
)

# ******** one query token: FlashDecoding ********
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
    out: UOp, stats: UOp, q: UOp, cache: UOp, slot: int | UOp, length: int | UOp, waves: int,
    scale: float, window: int, split: int,
) -> UOp:  # fmt: skip
    # A block takes one kv head of the slot and every PARTIALS-th chunk of its keys, for a
    # `split`-th of the query heads of its GQA group, from the chunk of the first key the window
    # holds. Each of `waves` warps scores KEYS / waves keys of a chunk; lanes hold dim / 32
    # dimensions. The warps then merge through shared memory into one partial per block: the
    # unnormalized output, its running max and its sum of weights.
    kv_heads, dim = int(cache.shape[2]), int(cache.shape[4])
    group, per_lane = int(q.shape[1]) // kv_heads // split, dim // WARP  # query heads per block
    partials = int(out.shape[1])
    per_wave, zero = KEYS // waves, UOp.const(0.0, dtypes.float32)
    since = _since(length, window)
    chunks = (length + KEYS - 1) // KEYS - since // KEYS
    head = UOp.range(kv_heads * split, 0, AxisType.GLOBAL)  # query heads from head * group
    block = UOp.range(at_most(chunks, partials), 1, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(waves, 3, AxisType.LOCAL)
    qs = [load_vector(q[0, head * group + h, 0, lane * per_lane], per_lane) for h in range(group)]
    rounds = UOp.range((chunks - 1 - block) // partials + 1, 4, AxisType.LOOP)
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
    # unless empty), as [head, dimension within lane, lane], then its max and sum.
    # tinygrad's codegen declares an index at its first use in the rounds loop and reuses it after
    # the loop, out of scope, if the same expression recurs there: indices from here on are made
    # of opaque copies of the block's and thread's coordinates, which it cannot match.
    ranges = (lane, wave, block, head)
    lane, wave, block, head = (opaque(u) for u in ranges)
    shared = UOp.alloc((waves, group * dim), dtypes.half, addrspace=AddrSpace.LOCAL)
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
        hq, d = flat // dim, flat % dim  # a query head of the block, and a dimension
        top = functools.reduce(UOp.maximum, (stat[w, hq, 0].load() for w in range(waves)))
        val = sum((((stat[w, hq, 0].load() - top) * LOG2E).exp2() * stat[w, hq, 1].load()
                   * shared[w, (hq * per_lane + d % per_lane) * WARP + d // per_lane].load().float()
                   for w in range(waves)), zero)  # fmt: skip
        live = head * group + hq
        if group * dim % (waves * WARP):
            live = live.valid(flat < group * dim)
        results.append(out[live, block, d].store(val))
    top = functools.reduce(UOp.maximum, (stat[w, thread, 0].load() for w in range(waves)))
    weights = sum((((stat[w, thread, 0].load() - top) * LOG2E).exp2() * stat[w, thread, 1].load()
                   for w in range(waves)), zero)  # fmt: skip
    qh = (head * group + thread).valid(thread < group)
    results += [stats[qh, block, 0].store(top), stats[qh, block, 1].store(weights)]
    info = KernelInfo(name="attention_partial", opts_to_apply=())
    return UOp.group(*results).end(*ranges).sink(arg=info)


@functools.cache
def _attention_combine_kernel(
    o: UOp, partial: UOp, stats: UOp, *sinks: UOp, live: int | UOp
) -> UOp:
    # one warp per query head and 64 dimensions: weights each block's partial by exp(max - max),
    # counting each head's sink, if given, once: a score that adds no value. Rows of several
    # queries' heads take the sink of row % heads.
    heads, _, dim = partial.shape
    tile, per_lane = 64, 64 // WARP
    head, part = UOp.range(heads, 0, AxisType.GLOBAL), UOp.range(dim // tile, 1, AxisType.GLOBAL)
    lane = lane_range()
    dims = [part * tile + lane * per_lane + i for i in range(per_lane)]
    c1 = UOp.range(live, 100, AxisType.LOOP)
    top = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
    top = top.after(top.store(top.const_like(-math.inf)))
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
    stores = [o[0, head, 0, d].store(acc[i].load() / total[0].load()) for i, d in enumerate(dims)]
    info = KernelInfo(name="attention_combine", opts_to_apply=())
    return UOp.group(*stores).end(lane, part, head).sink(arg=info)


def _since(length: int | UOp, window: int) -> int | UOp:
    # the first of `length` positions that a window of the last `window` holds, 0 for no window
    if not window:
        return 0
    return (length - window).maximum(0) if isinstance(length, UOp) else max(length - window, 0)


def _per_block(group: int, dim: int) -> int:
    # query heads per block, of a GQA group: as many as GROUP_SPLIT and GROUP_DIMS allow, and a
    # divisor of the group
    most = max(min(GROUP_SPLIT, GROUP_DIMS // dim), 1)
    return max(n for n in range(1, most + 1) if group % n == 0)


def supports_attention(q: Tensor, cache: Tensor) -> bool:
    shape = (*cache.shape[2:], *q.shape[:3])
    if not on_gpu(q) or not all(isinstance(x, int) for x in shape):
        return False
    kv_heads, n, dim, batch, heads, tokens = (int(x) for x in shape)
    group = _per_block(heads // kv_heads, dim)
    fits = group * dim * 2 + group * 8 <= SHARED  # one warp's share of shared memory
    return batch == 1 and tokens == 1 and dim % 64 == 0 and n % KEYS == 0 and fits


def attention(
    q: Tensor, cache: Tensor, slot: int | UOp, length: int | UOp, scale: float, window: int = 0,
    sinks: Tensor | None = None,
) -> Tensor:  # fmt: skip
    """Attention of one query token (1, H, 1, D) over the first `length` positions of a slot of
    the cache (2, slots, KV_H, positions, D), or the last `window` of them, with scores
    q.k * scale, and a sink (H,) per head if given, a score that adds no value."""
    heads, dim = q.shape[1], cache.shape[4]
    kv_heads = int(cache.shape[2])
    group = _per_block(int(heads) // kv_heads, int(dim))
    split = int(heads) // kv_heads // group  # blocks per kv head
    q, slot = carry(q.float().contiguous(), slot)
    cache, length = carry(cache, length)
    waves = 16
    while waves * (group * dim * 2 + group * 8) > SHARED:
        waves //= 2
    chunks = min(PARTIALS, int(cache.shape[3]) // KEYS)
    partial = Tensor.empty(heads, chunks, dim, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(heads, chunks, 2, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(
        _attention_partial_kernel, slot=slot, length=length, waves=waves, scale=scale,
        window=window, split=split,
    )  # fmt: skip
    outs = Tensor.custom_kernel(partial, stats, q, cache, fxn=fxn)
    live = at_most((length + KEYS - 1) // KEYS - _since(length, window) // KEYS, chunks)
    out = Tensor.empty(1, heads, 1, dim, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_attention_combine_kernel, live=live)
    extra = () if sinks is None else (sinks.float().contiguous(),)
    return Tensor.custom_kernel(out, outs[0], outs[1], *extra, fxn=fxn)[0]


# ******** several query tokens: FlashAttention-2 ********
# A block takes 16 query tokens and one kv head, with a warp for each query head of the GQA group;
# the warps share tiles of 32 keys and values in shared memory, and each keeps its rows' scores,
# softmax statistics and outputs in registers. Tiles of 32 keys leave room for more blocks per SM
# than 64: 9 to 17% faster from 0 to 8k cached tokens. Heads wider than 256, Gemma 4's of 512,
# would not fit in registers: blocks take parts of 256 of their outputs, each working out every
# score, and read the queries as they go, with tiles of 16 keys to fit in shared memory. A single
# tile of queries, as while prefilling up to 16 tokens, would leave SMs idle: blocks split its key
# tiles, as FlashDecoding does, and FlashDecoding's combine kernel merges their outputs.

QUERIES, KEY_TILE, PART = 16, 32, 256
SPLIT_BLOCKS = 64  # blocks that split one tile of queries' keys aim for

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
    lo, hi = (x.bitcast(dtypes.uint16).cast(dtypes.uint32) for x in (lo, hi))
    return lo | (hi << 16)


def _quad(value: UOp, op: Callable[[UOp, UOp], UOp]) -> UOp:
    # reduces over the 4 lanes that hold a row of an mma result
    for mask in (1, 2):
        value = op(value, shfl_xor(value, mask))
    return value


@functools.cache
def _flash_attention_kernel(
    out: UOp, *srcs: UOp, slot: int | UOp, start: int | UOp, tokens: int | UOp, window: int,
    key_tile: int, parts: int, splits: int = 1,
) -> UOp:  # fmt: skip
    # srcs: q (count, heads, dim) in f16, scaled so that exp2 gives the softmax, and the f16
    # cache, read as words of f16 pairs; query i is at position start + i and sees the slot's
    # positions up to it, or the last `window` of them. Blocks take key_tile keys at a time, and
    # 1 / parts of their outputs. Split, the queries are one tile, whose key tiles blocks take in
    # turn, each writing out (count * heads, splits, dim) unnormalized and first in srcs the max
    # and sum of each row's weights, for _attention_combine_kernel to merge. A sink per head,
    # last in srcs if given, starts each row's max and sum where one block takes all its keys.
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
    head, tid, g, t = kv_head * group + warp, warp * WARP + lane, lane // 4, lane % 4
    rows = (tile * QUERIES + g, tile * QUERIES + g + 8)  # the lane's rows of each mma result
    # the keys and values from the tile's first query's window to its last query, key_tile at a
    # time: keys as they are in the cache, values transposed so that B fragments of keys are words
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

    def queries(k: int) -> list[UOp]:  # the A fragment of the queries' dimensions 16k..
        # in registers for the whole loop with one part; read again each time with more
        src = q if parts == 1 else q.after(tiles)
        return [
            _halves_word(*(src[r, head, 16 * k + 8 * h + 2 * t + i].load() for i in (0, 1)))
            for h in (0, 1)
            for r in rows
        ]

    keys = UOp.alloc((key_tile, words + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    values = UOp.alloc((width, key_tile // 2 + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    stores = []

    # threads take words in turn; where the last turn has more threads than words, the others
    # repeat the last word, storing what its thread stores
    def turns(words: int) -> list[UOp]:
        return [(i * threads + tid).minimum(words - 1) for i in range(-(-words // threads))]

    for item in turns(key_tile * words):
        key, w = item // words, item % words
        stores.append(keys[key, w].store(cache[0, slot, kv_head, kt * key_tile + key, w].load()))
    for item in turns(key_tile // 2 * width // 2):
        pair, w = item % (key_tile // 2), item // (key_tile // 2)
        a, b = (
            cache[1, slot, kv_head, kt * key_tile + 2 * pair + j, part * width // 2 + w].load()
            for j in (0, 1)
        )
        stores.append(values[2 * w, pair].store((a & 0xFFFF) | (b << 16)))
        stores.append(values[2 * w + 1, pair].store((a >> 16) | (b & 0xFFFF0000)))
    keys, values = keys.after(*stores), values.after(*stores)

    # a finite initial max keeps exp2 from seeing -inf - -inf
    acc, mx, total = register((width // 2,), 0.0), register((2,), -1e30), register((2,), 0.0)
    if sinks and splits == 1:  # in the scores' units, of exp2
        sink = sinks[0][head].load() * LOG2E
        mx = UOp.alloc((2,), dtypes.float32, addrspace=AddrSpace.REG)
        mx = mx.after(mx.store(UOp.stack(sink, sink)))
        total = UOp.alloc((2,), dtypes.float32, addrspace=AddrSpace.REG)
        total = total.after(total.store(UOp.stack(*(UOp.const(1.0, dtypes.float32),) * 2)))
    prev_acc, prev_max, prev_total = acc.after(tiles), mx.after(tiles), total.after(tiles)
    zero = UOp.const(0.0, dtypes.float32)
    products = [[zero] * 4 for _ in range(key_tile // 8)]  # by tiles of 8 keys
    for k in range(dim // 16):
        a = queries(k)
        for j, c in enumerate(products):
            b = [keys[8 * j + g, 8 * k + 4 * h + t].load() for h in (0, 1)]
            products[j] = _mma_f16(a, b, c)
    scores = []  # masked to the positions each row sees
    for j, c in enumerate(products):
        position = kt * key_tile + 8 * j + 2 * t
        back = [start + rows[e // 2] - position - e % 2 for e in range(4)]  # how far each key is
        seen = [(b >= 0) & (b < window) if window else b >= 0 for b in back]
        scores.append([seen[e].where(c[e], -math.inf) for e in range(4)])
    row_max = [
        _quad(functools.reduce(UOp.maximum, (s[e] for s in scores for e in (2 * r, 2 * r + 1))),
              UOp.maximum)
        for r in (0, 1)
    ]  # fmt: skip
    new_max = [prev_max[r].load().maximum(row_max[r]) for r in (0, 1)]
    rescale = [(prev_max[r].load() - new_max[r]).exp2() for r in (0, 1)]
    p = [[(s[e] - new_max[e // 2]).exp2() for e in range(4)] for s in scores]
    sums = [
        _quad(sum((x[e] for x in p for e in (2 * r, 2 * r + 1)), zero), UOp.__add__) for r in (0, 1)
    ]
    # the weights as A fragments: the results of two 8-key tiles make one of 16 keys
    weights = [
        [_f16_pair(*p[2 * k + i][2 * r : 2 * r + 2]) for i in (0, 1) for r in (0, 1)]
        for k in range(key_tile // 16)
    ]
    outs: list[UOp] = []
    for n in range(width // 8):
        c = [prev_acc[4 * n + e].load() * rescale[e // 2] for e in range(4)]
        for k in range(key_tile // 16):
            b = [values[8 * n + g, 8 * k + 4 * h + t].load() for h in (0, 1)]
            c = _mma_f16(weights[k], b, c)
        outs += c
    update = UOp.group(
        acc.store(UOp.stack(*outs)),
        mx.store(UOp.stack(*new_max)),
        total.store(UOp.stack(*(prev_total[r].load() * rescale[r] + sums[r] for r in (0, 1)))),
    ).end(tiles)
    acc, mx, total = acc.after(update), mx.after(update), total.after(update)
    results = []
    for n in range(width // 8):
        for e in range(4):
            value, at = acc[4 * n + e].load(), part * width + 8 * n + 2 * t + e % 2
            if splits > 1:
                results.append(out[rows[e // 2] * heads + head, split, at].store(value))
                continue
            value = value / total[e // 2].load()
            results.append(out[rows[e // 2], head * dim + at].store(value))
    if stats is not None:  # the max in units of e, as the combine kernel takes it
        for r, row in enumerate(rows):
            for i, x in enumerate((mx[r].load() / LOG2E, total[r].load())):
                results.append(stats[(row * heads + head).valid(t.eq(0)), split, i].store(x))
    ranges = (kv_head, lane, warp, *(x for x in (tile, part, split) if x.op is Ops.RANGE))
    info = KernelInfo(name="flash_attention", opts_to_apply=())
    return UOp.group(*results).end(*ranges).sink(arg=info)


def _flash_shape(positions: int, dim: int) -> tuple[int, int] | None:
    # the key tile and parts the kernel takes for heads of `dim`, if it fits them
    parts = -(-dim // PART)
    width = dim // parts
    for key_tile in (KEY_TILE, KEY_TILE // 2):
        shared = 4 * (key_tile * (dim // 2 + 4) + width * (key_tile // 2 + 4))
        if dim % (16 * parts) == 0 and positions % key_tile == 0 and shared <= SHARED:
            return key_tile, parts
    return None


def supports_flash_attention(q: Tensor, cache: Tensor) -> bool:
    positions, dim = cache.shape[3:]
    if not on_nvidia(q) or not isinstance(positions, int) or not isinstance(dim, int):
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
    # one tile of queries, while prefilling few tokens, leaves SMs idle: blocks split its keys.
    # For Llama 3.1 8B's 8 kv heads, 8 blocks each took 27 us at 2000 positions, 16 took 29 and
    # one 170.
    splits = max(SPLIT_BLOCKS // (int(cache.shape[2]) * shape[1]), 1) if count == QUERIES else 1
    rows = count * heads  # of queries and heads
    shapes = [(rows, splits, n) for n in (dim, 2)] if splits > 1 else [(count, heads * dim)]
    outs = [Tensor.empty(*s, dtype=dtypes.float32, device=q.device) for s in shapes]
    outs[0], slot = carry(outs[0], slot)
    fxn = functools.partial(
        _flash_attention_kernel, slot=slot, start=start, tokens=bound, window=window,
        key_tile=shape[0], parts=shape[1], splits=splits,
    )  # fmt: skip
    extra = () if sinks is None else (sinks.float().contiguous(),)
    outs = Tensor.custom_kernel(*outs, q, cache, *(extra if splits == 1 else ()), fxn=fxn)
    if splits > 1:
        end = (start + at_most(bound, QUERIES) + shape[0] - 1) // shape[0]
        live = at_most(end - _since(start + 1, window) // shape[0], splits)
        out = Tensor.empty(1, rows, 1, dim, dtype=dtypes.float32, device=q.device)
        fxn = functools.partial(_attention_combine_kernel, live=live)
        outs = Tensor.custom_kernel(out, *outs[:2], *extra, fxn=fxn)
    return outs[0].reshape(count, heads * dim)[:tokens].reshape(1, tokens, heads * dim)


# ******** one token's queries, keys and values: biases, norms, RoPE, and the cache ********


@functools.cache
def _rotate_kernel(
    out: UOp, cache: UOp, q: UOp, k: UOp, v: UOp, *extra: UOp, slot: int | UOp, pos: int | UOp,
    rotated: int, halves: bool, biased: bool, v_norm: bool, eps: float,
) -> UOp:  # fmt: skip
    # A warp per head of q, then of k and its v. extra holds RoPE's cos and sin (positions,
    # rotated / 2) if any dimensions rotate, then each of q's, k's and v's bias if biased, and q's
    # and k's norm weights if any. Each of q, k and v gets its bias; each head of q and k is then
    # normed with its weight, and of v without, if v_norm; the first `rotated` dimensions of q and
    # k are rotated, q into out and k into the cache with v. Lanes take pairs of dimensions: those
    # that rotate together, i and i + rotated / 2 or adjacent ones, each pair i turning by angle
    # cos[pos, i], sin[pos, i], and the others' adjacent pairs as they are.
    (cos, sin), extra = (extra[:2], extra[2:]) if rotated else ((None, None), extra)
    biases, norms = (extra[:3], extra[3:]) if biased else ((), extra)
    _, slots, kv_heads, positions, dim = (int(d) for d in cache.shape)
    heads, half, turning = int(out.shape[0]) // dim, dim // 2, rotated // 2
    head, lane = UOp.range(heads + kv_heads, 0, AxisType.GLOBAL), lane_range()
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

    def of_q_or_k(qs: UOp, ks: UOp, d: UOp) -> UOp:  # dimension d of the warp's head, of q or k
        return qs[(head * dim + d).valid(is_q)].load() + ks[(kv * dim + d).valid(is_kv)].load()

    def of_v(vs: UOp, d: UOp) -> UOp:
        return vs[(kv * dim + d).valid(is_kv)].load()

    def normed(values: list[UOp], weights: list[UOp] | None = None) -> list[UOp]:
        inv = (warp_sum(sum((x * x for x in values), zero)) / dim + eps).rsqrt()
        if weights is None:
            return [x * inv for x in values]
        return [x * inv * w for x, w in zip(values, weights, strict=True)]

    def stored(kind: int, d: UOp) -> UOp:  # where dimension d of the warp's key or value goes
        at = (((kind * slots + slot) * kv_heads + kv) * positions + pos) * dim + d
        return cache.flatten()[at.valid(is_kv)]

    x = [of_q_or_k(q, k, d) for d in flat]
    if biases:
        x = [a + of_q_or_k(biases[0], biases[1], d) for a, d in zip(x, flat, strict=True)]
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
            stores.append(out[(head * dim + d).valid(is_q)].store(value))
            stores.append(stored(0, d).store(value.cast(dtypes.half)))
    values = [of_v(v, d) + of_v(biases[2], d) if biases else of_v(v, d) for d in flat]
    if v_norm:
        values = normed(values)
    for value, d in zip(values, flat, strict=True):
        stores.append(stored(1, d).store(value.cast(dtypes.half)))
    info = KernelInfo(name="rotate", opts_to_apply=())
    return UOp.group(*stores).end(head, lane).sink(arg=info)


def supports_rotate(q: Tensor, cache: Tensor) -> bool:
    # one token, and whole warps of pairs of dimensions
    one = isinstance(q.numel(), int) and q.numel() == q.shape[-2] * q.shape[-1]
    return on_gpu(q) and one and int(cache.shape[4]) % (2 * WARP) == 0


def rotate(
    q: Tensor, k: Tensor, v: Tensor, cache: Tensor, slot: int | UOp, start_pos: int | UOp,
    rope: tuple[tuple[Tensor, Tensor], int] | None, halves: bool,
    biases: tuple[Tensor, Tensor, Tensor] | None, norms: tuple[Tensor, Tensor] | None,
    v_norm: bool, eps: float,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    """For one token's q (1, 1, H, D), k and v (1, 1, KV_H, D): adds their biases, if given;
    norms each head of q and k with its weight, if given, and of v without, if v_norm; rotates the
    first R dimensions of q and k by RoPE's tables (positions, R/2) at start_pos, for rope
    ((cos, sin), R) if given, and stores k and v at start_pos of a slot of the cache. Returns q
    (1, H, 1, D) and the cache."""
    heads, dim = q.shape[-2], q.shape[-1]
    out = Tensor.empty(heads * dim, dtype=dtypes.float32, device=q.device)
    q, k, v = (t.reshape(-1).float().contiguous() for t in (q, k, v))
    q, pos = carry(q, start_pos)
    k, slot = carry(k, slot)
    tables = () if rope is None else rope[0]
    extra = tuple(w.float().contiguous() for w in (*(biases or ()), *(norms or ())))
    fxn = functools.partial(
        _rotate_kernel, slot=slot, pos=pos, rotated=0 if rope is None else rope[1],
        halves=halves, biased=biases is not None, v_norm=v_norm, eps=eps,
    )  # fmt: skip
    out, cache = Tensor.custom_kernel(out, cache, q, k, v, *tables, *extra, fxn=fxn)[:2]
    return out.reshape(1, heads, 1, dim), cache
