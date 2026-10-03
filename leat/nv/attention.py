"""Attention over the KV cache: FlashDecoding for one query token, FlashAttention-2 on f16 tensor
cores for several."""

import functools
import math
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.nv.common import (
    LOG2E,
    SHARED,
    WARP,
    at_most,
    carry,
    lane_range,
    load_vector,
    on_nvidia,
    register,
    shfl_xor,
    warp_sum,
)

# ******** one query token: FlashDecoding ********
# Adapted from tinygrad/llm/kernels/amd.py: the cache is cut into chunks of KEYS keys, blocks
# reduce chunks with an online softmax, and a second kernel combines their partials.

KEYS = 64  # keys per chunk
PARTIALS = 48  # most blocks per kv head; longer caches loop over several chunks per block
PAD = 8  # halves of shared memory after each warp's outputs, see _attention_partial_kernel


@functools.cache
def _attention_partial_kernel(
    out: UOp, stats: UOp, q: UOp, cache: UOp, slot: int | UOp, length: int | UOp, waves: int
) -> UOp:
    # A block takes one kv head of the slot and every PARTIALS-th chunk of its keys, for all query
    # heads of the GQA group. Each of `waves` warps scores KEYS / waves keys of a chunk; lanes hold
    # dim / 32 dimensions. The warps then merge through shared memory into one partial per block:
    # the unnormalized output, its running max and its sum of weights.
    kv_heads, dim = int(cache.shape[2]), int(cache.shape[4])
    group, per_lane, partials = int(q.shape[1]) // kv_heads, dim // WARP, int(out.shape[1])
    per_wave, zero = KEYS // waves, UOp.const(0.0, dtypes.float32)
    chunks = (length + KEYS - 1) // KEYS
    head = UOp.range(kv_heads, 0, AxisType.GLOBAL)
    block = UOp.range(at_most(chunks, partials), 1, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(waves, 3, AxisType.LOCAL)
    qs = [load_vector(q[0, head * group + h, 0, lane * per_lane], per_lane) for h in range(group)]
    rounds = UOp.range((chunks - 1 - block) // partials + 1, 4, AxisType.LOOP)
    chunk = block + rounds * partials
    valid, scores, values = [], [], []
    for j in range(per_wave):
        key = chunk * KEYS + wave * per_wave + j
        valid.append(key < length)
        k = load_vector(cache[0, slot, head, key, lane * per_lane], per_lane)
        # values load during scoring, so both streams are in flight together
        v = load_vector(cache[1, slot, head, key, lane * per_lane], per_lane)
        values.append([valid[j].where(x, zero) for x in v])
        dots = [warp_sum(sum((a * b for a, b in zip(qh, k, strict=True)), zero)) for qh in qs]
        scores.append([valid[j].where(d / math.sqrt(dim), -1e30) for d in dots])
    # a finite initial max keeps fully masked warps from computing exp(-inf - -inf)
    acc = register((group, per_lane), 0.0)
    mx, total = register((group,), -1e30), register((group,), 0.0)
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
        live = head * group + h
        if group * dim % (waves * WARP):
            live = live.valid(flat < group * dim)
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
    lane = lane_range()
    dims = [part * tile + lane * per_lane + i for i in range(per_lane)]
    c1 = UOp.range(live, 100, AxisType.LOOP)
    top = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
    top = top.after(top.store(top.const_like(-math.inf)))
    top = top.after(top.store(top.after(c1).maximum(stats[head, c1, 0].load())).end(c1))
    c2 = UOp.range(live, 101, AxisType.LOOP)
    acc, total = register((per_lane,), 0.0), register((1,), 0.0)
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
    shape = (*cache.shape[2:], *q.shape[:3])
    if not on_nvidia(q) or not all(isinstance(x, int) for x in shape):
        return False
    kv_heads, n, dim, batch, heads, tokens = (int(x) for x in shape)
    group = heads // kv_heads
    fits = (group * dim + PAD) * 2 + group * 8 <= SHARED  # one warp's share of shared memory
    return batch == 1 and tokens == 1 and dim % 64 == 0 and n % KEYS == 0 and fits


def attention(q: Tensor, cache: Tensor, slot: int | UOp, length: int | UOp) -> Tensor:
    """Attention of one query token (1, H, 1, D) over the first `length` positions of a slot of
    the cache (2, slots, KV_H, positions, D)."""
    heads, dim, group = q.shape[1], cache.shape[4], q.shape[1] // cache.shape[2]
    q, slot = carry(q.float().contiguous(), slot)
    cache, length = carry(cache, length)
    waves = 16
    while waves * ((group * dim + PAD) * 2 + group * 8) > SHARED:
        waves //= 2
    chunks = min(PARTIALS, int(cache.shape[3]) // KEYS)
    partial = Tensor.empty(heads, chunks, dim, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(heads, chunks, 2, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_attention_partial_kernel, slot=slot, length=length, waves=waves)
    outs = Tensor.custom_kernel(partial, stats, q, cache, fxn=fxn)
    live = at_most((length + KEYS - 1) // KEYS, chunks)
    out = Tensor.empty(1, heads, 1, dim, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_attention_combine_kernel, live=live)
    return Tensor.custom_kernel(out, outs[0], outs[1], fxn=fxn)[0]


# ******** several query tokens: FlashAttention-2 ********
# A block takes 16 query tokens and one kv head, with a warp for each query head of the GQA group;
# the warps share tiles of 32 keys and values in shared memory, and each keeps its rows' scores,
# softmax statistics and outputs in registers. Tiles of 32 keys leave room for more blocks per SM
# than 64: 9 to 17% faster from 0 to 8k cached tokens.

QUERIES, KEY_TILE = 16, 32

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
    out: UOp, q: UOp, cache: UOp, slot: int | UOp, start: int | UOp, tokens: int | UOp
) -> UOp:
    # q (count, heads, dim) in f16, scaled so that exp2 gives the softmax, and the f16 cache, read
    # as words of f16 pairs; query i is at position start + i and sees the slot's positions up to it
    heads, dim = int(q.shape[1]), int(q.shape[2])
    _, slots, kv_heads, positions, _ = (int(d) for d in cache.shape)
    words = dim // 2
    cache = cache.flatten().bitcast(dtypes.uint32).reshape(2, slots, kv_heads, positions, words)
    group = heads // kv_heads
    threads = group * WARP
    tile = UOp.range((tokens + QUERIES - 1) // QUERIES, 0, AxisType.GLOBAL)
    kv_head = UOp.range(kv_heads, 1, AxisType.GLOBAL)
    lane, warp = lane_range(), UOp.range(group, 2, AxisType.LOCAL)
    head, tid, g, t = kv_head * group + warp, warp * WARP + lane, lane // 4, lane % 4
    rows = (tile * QUERIES + g, tile * QUERIES + g + 8)  # the lane's rows of each mma result
    # queries as A fragments of 16 dimensions
    queries = [
        [
            _halves_word(*(q[r, head, 16 * k + 8 * h + 2 * t + i].load() for i in (0, 1)))
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
        stores.append(keys[key, w].store(cache[0, slot, kv_head, kt * KEY_TILE + key, w].load()))
    for i in range(KEY_TILE // 2 * words // threads):
        pair, w = (i * threads + tid) % (KEY_TILE // 2), (i * threads + tid) // (KEY_TILE // 2)
        a, b = (cache[1, slot, kv_head, kt * KEY_TILE + 2 * pair + j, w].load() for j in (0, 1))
        stores.append(values[2 * w, pair].store((a & 0xFFFF) | (b << 16)))
        stores.append(values[2 * w + 1, pair].store((a >> 16) | (b & 0xFFFF0000)))
    keys, values = keys.after(*stores), values.after(*stores)

    # a finite initial max keeps exp2 from seeing -inf - -inf
    acc, mx, total = register((dim // 2,), 0.0), register((2,), -1e30), register((2,), 0.0)
    prev_acc, prev_max, prev_total = acc.after(kt), mx.after(kt), total.after(kt)
    zero = UOp.const(0.0, dtypes.float32)
    scores = []  # tiles of 8 keys, masked to the positions each row sees
    for j in range(KEY_TILE // 8):
        c = [zero] * 4
        for k in range(dim // 16):
            b = [keys[8 * j + g, 8 * k + 4 * h + t].load() for h in (0, 1)]
            c = _mma_f16(queries[k], b, c)
        position = kt * KEY_TILE + 8 * j + 2 * t
        scores.append(
            [(position + e % 2 <= start + rows[e // 2]).where(c[e], -math.inf) for e in range(4)]
        )
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
        for k in range(KEY_TILE // 16)
    ]
    outs: list[UOp] = []
    for n in range(dim // 8):
        c = [prev_acc[4 * n + e].load() * rescale[e // 2] for e in range(4)]
        for k in range(KEY_TILE // 16):
            b = [values[8 * n + g, 8 * k + 4 * h + t].load() for h in (0, 1)]
            c = _mma_f16(weights[k], b, c)
        outs += c
    update = UOp.group(
        acc.store(UOp.stack(*outs)),
        mx.store(UOp.stack(*new_max)),
        total.store(UOp.stack(*(prev_total[r].load() * rescale[r] + sums[r] for r in (0, 1)))),
    ).end(kt)
    acc, total = acc.after(update), total.after(update)
    results = []
    for n in range(dim // 8):
        for e in range(4):
            value = acc[4 * n + e].load() / total[e // 2].load()
            results.append(out[rows[e // 2], head * dim + 8 * n + 2 * t + e % 2].store(value))
    info = KernelInfo(name="flash_attention", opts_to_apply=())
    return UOp.group(*results).end(tile, kv_head, lane, warp).sink(arg=info)


def supports_flash_attention(q: Tensor, cache: Tensor) -> bool:
    shape = (*cache.shape[2:], q.shape[0], q.shape[1])
    if not on_nvidia(q) or not all(isinstance(x, int) for x in shape):
        return False
    kv_heads, n, dim, batch, heads = (int(x) for x in shape)
    threads, words = heads // kv_heads * WARP, dim // 2
    shared = 4 * (KEY_TILE * (words + 4) + dim * (KEY_TILE // 2 + 4))
    whole = KEY_TILE * words % threads == 0 and KEY_TILE // 2 * words % threads == 0
    fits = dim % 16 == 0 and n % KEY_TILE == 0 and shared <= SHARED and whole
    return batch == 1 and fits


def flash_attention(q: Tensor, cache: Tensor, slot: int | UOp, start_pos: int | UOp) -> Tensor:
    """Causal attention of query tokens (1, H, T, D) at positions start_pos.. over a slot of the
    cache, which already holds their keys and values. Returns (1, T, H * D)."""
    _, heads, tokens, dim = q.shape
    count = -(-q.max_shape[2] // QUERIES) * QUERIES
    # in the layout of the projection that made q, where scaling and rounding it is a plain copy
    q = (q.transpose(1, 2).reshape(tokens, heads, dim).float() * (LOG2E / math.sqrt(dim))).half()
    q = q.pad_to((count, heads, dim)).contiguous()
    q, start = carry(q, start_pos)
    cache, length = carry(cache, tokens)
    out = Tensor.empty(count, heads * dim, dtype=dtypes.float32, device=q.device)
    out, slot = carry(out, slot)
    fxn = functools.partial(_flash_attention_kernel, slot=slot, start=start, tokens=length)
    out = Tensor.custom_kernel(out, q, cache, fxn=fxn)[0]
    return out[:tokens].reshape(1, tokens, heads * dim)
