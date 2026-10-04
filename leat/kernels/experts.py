"""Mixtures of experts. A kernel routes each token to its best scoring experts. For few tokens, the
matrix-vector kernels then read the chosen experts' rows, as llama.cpp's MMVQ with expert ids; for
more, a kernel lists the tokens routed to each expert, and the tensor-core kernels take each
expert's list."""

import functools
import math
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from leat.kernels.argmax import argmax_step, warp_argmax
from leat.kernels.common import (
    LOG2E,
    WARP,
    ballot,
    carry,
    glu,
    lane_range,
    load_vector,
    on_gpu,
    on_nvidia,
    popcount,
    register,
    storage_words,
    warp_sum,
)
from leat.kernels.matmul import matmul_fits, routed_products, tiled
from leat.kernels.matvec import DOTS, rows_kernel
from leat.kernels.quantize import quantize_q8
from leat.quant import GGMLType, QTensor

MATVEC_TOKENS = 8  # tokens up to which the matrix-vector kernels run a mixture
TILE = 8  # tokens and experts per warp scoring many tokens


@functools.cache
def _scores_kernel(
    out: UOp, x: UOp, weight: UOp, router: UOp, tokens: int | UOp, eps: float
) -> UOp:
    # a warp per token and expert: the router's row . rms_norm(x, weight, eps), the norm worked
    # out by each warp from the token's row, which stays in cache; lanes take 4 values at a time,
    # and zeros past the row's end
    experts, dim = int(out.shape[1]), int(x.shape[1])
    token, expert = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(experts, 1, AxisType.GLOBAL)
    lane, zero = lane_range(), UOp.const(0.0, dtypes.float32)
    ats = [(i * WARP + lane) * 4 for i in range(-(-dim // (4 * WARP)))]
    xs = [v for at in ats for v in _row(x, token * dim, at, dim)]
    inv = (warp_sum(sum((v * v for v in xs), zero)) / dim + eps).rsqrt()
    ws = [v for at in ats for v in _row(weight, 0, at, dim)]
    rs = [v for at in ats for v in _row(router, expert * dim, at, dim)]
    dot = warp_sum(sum((a * b * c for a, b, c in zip(xs, ws, rs, strict=True)), zero))
    store = out[token, expert.valid(lane.eq(0))].store(dot * inv)
    return store.end(token, expert, lane).sink(arg=KernelInfo(name="router", opts_to_apply=()))


@functools.cache
def _scores_tiles_kernel(
    out: UOp, x: UOp, weight: UOp, router: UOp, tokens: int | UOp, eps: float
) -> UOp:
    # _scores_kernel for many tokens: a warp per tile of TILE tokens by TILE experts, so that each
    # row is read once per tile rather than once per pair, as the reads bound the kernel. Lanes
    # take 4 values of each row at a time, 128 a step, and sum the squares of the tokens' rows as
    # they go.
    experts, dim = int(out.shape[1]), int(x.shape[1])
    tile_t = UOp.range((tokens + TILE - 1) // TILE, 0, AxisType.GLOBAL)
    tile_e, lane = UOp.range(experts // TILE, 1, AxisType.GLOBAL), lane_range()
    step = UOp.range(-(-dim // (4 * WARP)), 2, AxisType.LOOP)
    at = (step * WARP + lane) * 4
    acc = register((TILE * TILE + TILE,), 0.0)
    prev, ws = acc.after(step), _row(weight, 0, at, dim)
    xs = [_row(x, (tile_t * TILE + t) * dim, at, dim) for t in range(TILE)]
    rs = [_row(router, (tile_e * TILE + e) * dim, at, dim) for e in range(TILE)]
    weighted = [[v * w for v, w in zip(row, ws, strict=True)] for row in xs]
    vals = [
        prev[t * TILE + e].load() + sum(a * b for a, b in zip(weighted[t], rs[e], strict=True))
        for t in range(TILE)
        for e in range(TILE)
    ]
    vals += [prev[TILE * TILE + t].load() + sum(v * v for v in xs[t]) for t in range(TILE)]
    acc = acc.after(acc.store(UOp.stack(*vals)).end(step))
    stores = []
    for t in range(TILE):
        inv = (warp_sum(acc[TILE * TILE + t].load()) / dim + eps).rsqrt()
        for e in range(TILE):
            at = (tile_t * TILE + t) * experts + tile_e * TILE + e
            dot = warp_sum(acc[t * TILE + e].load())
            stores.append(out.flatten()[at.valid(lane.eq(0))].store(dot * inv))
    info = KernelInfo(name="router_tiles", opts_to_apply=())
    return UOp.group(*stores).end(tile_t, tile_e, lane).sink(arg=info)


def _row(buf: UOp, start: UOp | int, at: UOp, n: int) -> tuple[UOp, ...]:
    # the 4 values from `at` on of a row of n from `start` in a flat buffer, or zeros past its end
    if n % (4 * WARP) == 0:
        return load_vector(buf.flatten()[start + at], 4)
    values = load_vector(buf.flatten()[start + at.minimum(n - 4)], 4)
    return tuple((at < n).where(v, 0.0) for v in values)


def supports_scores(x: Tensor, router: QTensor) -> bool:
    return on_gpu(x) and router.type == GGMLType.F32 and router.shape[1] % 4 == 0


def scores(x: Tensor, norm: tuple[Tensor, float], router: QTensor) -> Tensor:
    """The router's scores of the experts for tokens x (1, T, dim): rms_norm(x, *norm) @ router.T
    in f32, the precision llama.cpp scores in, where routing is sensitive to rounding."""
    _, tokens, dim = x.shape
    count, experts = x.max_shape[1], router.shape[0]
    tiles = count > MATVEC_TOKENS and experts % TILE == 0
    count = -(-count // TILE) * TILE if tiles else count
    out = Tensor.empty(count, experts, dtype=dtypes.float32, device=x.device)
    x, bound = carry(x.reshape(tokens, dim).float().pad_to((count, dim)).contiguous(), tokens)
    kernel = _scores_tiles_kernel if tiles else _scores_kernel
    fxn = functools.partial(kernel, tokens=bound, eps=norm[1])
    # .contiguous() on the router's storage is a view; a reshape of it tinygrad would copy
    weights = norm[0].float().contiguous(), router.data.contiguous()
    out = Tensor.custom_kernel(out, x, *weights, fxn=fxn)[0]
    return out[:tokens].reshape(1, tokens, experts)


@functools.cache
def _route_kernel(ids: UOp, weights: UOp, scores: UOp, tokens: int | UOp, used: int) -> UOp:
    # a warp per token: `used` rounds of argmax over the experts' scores, the first index on ties,
    # each taking its winner out of the next; then the softmax of the winners' scores
    token, lane = UOp.range(tokens, 0, AxisType.GLOBAL), lane_range()
    ats = [(lane + WARP * j).cast(dtypes.int32) for j in range(int(scores.shape[1]) // WARP)]
    values = [scores[token, at].load() for at in ats]
    best: list[tuple[UOp, UOp]] = []
    for _ in range(used):
        top, index = UOp.const(-math.inf, dtypes.float32), UOp.const(0, dtypes.int32)
        for value, at in zip(values, ats, strict=True):
            top, index = argmax_step(top, index, value, at)
        best.append(warp_argmax(top, index))
        values = [
            at.eq(best[-1][1]).where(-math.inf, value)
            for value, at in zip(values, ats, strict=True)
        ]
    exps = [((top - best[0][0]) * LOG2E).exp2() for top, _ in best]
    total = sum(exps[1:], exps[0])
    stores = []
    for i, ((_, index), e) in enumerate(zip(best, exps, strict=True)):
        at = (token * used + i).valid(lane.eq(0))
        stores += [ids[at].store(index), weights[at].store(e / total)]
    info = KernelInfo(name=f"route_{used}", opts_to_apply=())
    return UOp.group(*stores).end(token, lane).sink(arg=info)


def route(scores: Tensor, used: int) -> tuple[Tensor, Tensor]:
    """The `used` best scoring experts of each row of `scores` (T, experts), as ids (T * used)
    int32, and the softmax of their scores."""
    count, tokens = scores.max_shape[0], scores.shape[0]
    ids = Tensor.empty(count * used, dtype=dtypes.int32, device=scores.device)
    weights = Tensor.empty(count * used, dtype=dtypes.float32, device=scores.device)
    scores, tokens = carry(scores.float().pad_to(scores.max_shape).contiguous(), tokens)
    fxn = functools.partial(_route_kernel, tokens=tokens, used=used)
    ids, weights = Tensor.custom_kernel(ids, weights, scores, fxn=fxn)[:2]
    return ids, weights


@functools.cache
def _bucket_kernel(order: UOp, counts: UOp, ids: UOp, pairs: int | UOp) -> UOp:
    # a warp per expert lists the pairs routed to it, in order: in each turn of 32 pairs, a ballot
    # of the lanes holding one gives each its place after those listed before
    experts = int(counts.shape[0])
    per = int(order.shape[0]) // experts
    expert, lane = UOp.range(experts, 0, AxisType.GLOBAL), lane_range()
    turn = UOp.range((pairs + WARP - 1) // WARP, 1, AxisType.LOOP)
    pair = turn * WARP + lane
    routed = ids[pair.minimum(int(ids.shape[0]) - 1)].load().eq(expert.cast(dtypes.int32))
    mine = routed & (pair < pairs)
    votes = ballot(mine)
    below = (UOp.const(1, dtypes.uint32) << lane.cast(dtypes.uint32)) - 1
    listed = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    listed = listed.after(listed[0].store(0))
    before = listed.after(turn)[0].load()
    place = before + popcount(votes & below)
    put = order[(expert * per + place).valid(mine)].store(pair.cast(dtypes.int32))
    listed = listed.after(UOp.group(put, listed[0].store(before + popcount(votes))).end(turn))
    store = counts[expert.valid(lane.eq(0))].store(listed[0].load())
    return store.end(expert, lane).sink(arg=KernelInfo(name="bucket", opts_to_apply=()))


def _bucket(ids: Tensor, experts: int, per: int, pairs: int | UOp) -> tuple[Tensor, Tensor]:
    # each expert's pairs, the first `pairs` of ids, as order (experts, per) and counts (experts,)
    order = Tensor.empty(experts * per, dtype=dtypes.int32, device=ids.device)
    counts = Tensor.empty(experts, dtype=dtypes.int32, device=ids.device)
    ids, pairs = carry(ids, pairs)
    fxn = functools.partial(_bucket_kernel, pairs=pairs)
    order, counts = Tensor.custom_kernel(order, counts, ids, fxn=fxn)[:2]
    return order, counts


@functools.cache
def _experts_swiglu_kernel(
    out: UOp, gate: UOp, up: UOp, xq: UOp, xd: UOp, xs: UOp, ids: UOp, *biases: UOp,
    pairs: int | UOp, meta: tuple[GGMLType, int, int, int, str, bool],
) -> UOp:  # fmt: skip
    # out[p * rows + r] = glu(kind)(gate . x, up . x) for row r of expert ids[p], where pair p is
    # slot p % used of token p // used, plus gate's and up's biases (experts * rows) if given.
    # Fused, gate and up are one stack, each expert's gate rows before its up rows.
    ggml_type, cols, rows, used, kind, fused = meta

    def expert(w: UOp, first: int) -> Callable[[UOp, UOp], UOp]:
        dot, stride = DOTS[ggml_type](w, xq, xd, xs, cols), 2 * rows if fused else rows

        def chosen(row: UOp, unit: UOp) -> UOp:
            pair = row // rows
            return dot(ids[pair].load() * stride + first + row % rows, unit, pair // used)

        return chosen

    def combine(row: UOp, g: UOp, u: UOp) -> UOp:
        if biases:
            at = ids[row // rows].load() * rows + row % rows
            g, u = g + biases[0][at].load(), u + biases[1][at].load()
        return glu(kind)(g, u)

    dots = [expert(gate, 0), expert(up, rows if fused else 0)]
    name = f"experts_glu_{kind}_{ggml_type.name.lower()}"
    return rows_kernel(out, cols // 64, name, dots, combine, pairs * rows)


@functools.cache
def _experts_down_kernel(
    out: UOp, w: UOp, hq: UOp, hd: UOp, hs: UOp, ids: UOp, weights: UOp, *extra: UOp,
    tokens: int | UOp, meta: tuple[GGMLType, int, int, int, bool, bool, bool],
) -> UOp:  # fmt: skip
    # out[t * rows + r] = the sum over token t's pairs p of weights[p] times row r of expert ids[p]
    # . the pair's hidden activations, plus that row's bias; extra holds, as meta says, each
    # expert's scale of its output, the biases (experts * rows), and a residual to add. Lanes
    # take (slot, unit) items in turn, so experts with narrow rows still keep them busy; a pair's
    # first unit takes its bias.
    ggml_type, cols, rows, used, scaled, biased, residual = meta
    dot, units = DOTS[ggml_type](w, hq, hd, hs, cols), cols // 64

    def mixed(row: UOp, item: UOp) -> UOp:
        pair = row // rows * used + item // units
        expert, weight = ids[pair].load(), weights[pair].load()
        if scaled:
            weight = weight * extra[0][expert].load()
        value = dot(expert * rows + row % rows, item % units, pair)
        if biased:
            bias = extra[int(scaled)][expert * rows + row % rows].load()
            value = value + (item % units).eq(0).where(bias, 0.0)
        return weight * value

    def combine(row: UOp, total: UOp) -> UOp:
        return total + extra[-1][row].load() if residual else total

    name = f"experts_down_{ggml_type.name.lower()}"
    return rows_kernel(out, used * units, name, [mixed], combine, tokens * rows)


def supports_mixture(x: Tensor, gate: QTensor, up: QTensor | None, down: QTensor) -> bool:
    # whole warps of experts to route, and the matrix-vector kernels' units of 64 weights; where
    # up is None, gate stacks both
    experts, rows, cols = gate.shape
    rows //= 2 if up is None else 1
    same = up is None or (gate.type == up.type and gate.shape == up.shape)
    shapes = experts % WARP == 0 and rows % 64 == 0 and cols % 64 == 0
    single = all(isinstance(b, int) and b == 1 for b in x.shape[:-2])
    types = gate.type in DOTS and down.type in DOTS and down.shape == (experts, cols, rows)
    return on_gpu(x) and single and same and shapes and types


def mixture(
    x: Tensor, scores: Tensor, gate: QTensor, up: QTensor | None, down: QTensor, used: int,
    norm: tuple[Tensor, float], kind: str = "silu", scales: Tensor | None = None,
    residual: bool = True, biases: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:  # fmt: skip
    """x + the mixture of experts for tokens x (1, T, dim) and their router `scores` (1, T,
    experts): each token's `used` best scoring experts, weighted by the softmax of their scores
    and by each expert's scale if given, run glu(kind)(n @ gate.T, n @ up.T) @ down.T on
    n = rms_norm(x, *norm), as common.glu has it, with biases (experts, rows) of gate, up and
    down if given; without x if not residual. Where up is None, gate stacks both, each expert's
    gate rows first."""
    _, tokens, dim = x.shape
    ids, weights = route(scores.reshape(tokens, gate.shape[0]), used)
    # few tokens, or matrices the tensor-core kernels do not fit, take the matrix-vector kernels
    few = isinstance(tokens, int) and tokens <= MATVEC_TOKENS
    _, cols, rows = down.shape  # (experts, dim, hidden)
    fit = on_nvidia(x) and matmul_fits(gate.type, rows, cols) and matmul_fits(down.type, cols, rows)
    flat = None if biases is None else tuple(b.float().flatten().contiguous() for b in biases)
    if few or not fit:
        args = (used, norm, kind, scales, residual, flat)
        return _matvecs(x, ids, weights, gate, up, down, *args)
    # many tokens: on tensor cores, each expert taking the pairs routed to it
    xt = tiled(x)
    count = int(xt.shape[0])
    order, counts = _bucket(ids, gate.shape[0], count, tokens * used)
    q8 = quantize_q8(xt, norm, rows=tokens)
    ws = (gate,) if up is None else (gate, up)
    routes = (order, counts, used)
    hidden = routed_products(q8, tokens, ws, *routes, True, up is None, kind, flat and flat[:2])
    q8 = quantize_q8(hidden, rows=tokens * used)
    out = routed_products(q8, tokens, (down,), *routes, False, biases=flat and flat[2:])
    ids, weights = ids.reshape(-1, used)[:tokens], weights.reshape(-1, used)[:tokens]
    if scales is not None:
        weights = weights * scales[ids]
    mixed = (out.reshape(count, used, dim)[:tokens] * weights.unsqueeze(-1)).sum(1).reshape(x.shape)
    return x + mixed if residual else mixed


def _matvecs(
    x: Tensor, ids: Tensor, weights: Tensor, gate: QTensor, up: QTensor | None, down: QTensor,
    used: int, norm: tuple[Tensor, float], kind: str, scales: Tensor | None, residual: bool,
    biases: tuple[Tensor, ...] | None,
) -> Tensor:  # fmt: skip
    # the matrix-vector kernels, reading each chosen expert's rows once per token; biases, if
    # given, flat
    _, tokens, dim = x.shape
    count = x.max_shape[1]
    _, cols, rows = down.shape  # (experts, dim, hidden)
    xq, xd, xs = quantize_q8(x.reshape(tokens, dim).pad_to((count, dim)), norm, rows=tokens)
    hidden = Tensor.empty(count * used * rows, dtype=dtypes.float32, device=x.device)
    xq, pairs = carry(xq, tokens * used)
    meta = (gate.type, cols, rows, used, kind, up is None)
    fxn = functools.partial(_experts_swiglu_kernel, pairs=pairs, meta=meta)
    words = storage_words(gate), storage_words(gate if up is None else up)
    gate_up = biases[:2] if biases else ()
    hidden = Tensor.custom_kernel(hidden, *words, xq, xd, xs, ids, *gate_up, fxn=fxn)[0]
    hq, hd, hs = quantize_q8(hidden.reshape(count * used, rows), rows=tokens * used)
    out = Tensor.empty(count * dim, dtype=dtypes.float32, device=x.device)
    extra = [] if scales is None else [scales.float().contiguous()]
    if biases:
        extra.append(biases[2])
    if residual:
        extra.append(x.reshape(tokens, dim).float().pad_to((count, dim)).contiguous().flatten())
    hq, bound = carry(hq, tokens)
    down_meta = (down.type, rows, cols, used, scales is not None, biases is not None, residual)
    fxn = functools.partial(_experts_down_kernel, tokens=bound, meta=down_meta)
    srcs = storage_words(down), hq, hd, hs, ids, weights, *extra
    out = Tensor.custom_kernel(out, *srcs, fxn=fxn)[0]
    return out.reshape(count, dim)[:tokens].reshape(x.shape)
