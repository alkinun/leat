"""Mixtures of experts. A kernel routes each token to its best scoring experts. For few tokens, the
matrix-vector kernels then read the chosen experts' rows, as llama.cpp's MMVQ with expert ids; for
more, a kernel lists the tokens routed to each expert, and the tensor-core kernels take each
expert's list."""

import functools
import math
from collections.abc import Callable

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.nv.argmax import argmax_step, warp_argmax
from leat.nv.common import LOG2E, WARP, carry, lane_range, on_nvidia, silu, storage_words
from leat.nv.matmul import TILE_TOKENS, routed_products, tiled
from leat.nv.matvec import DOTS, rows_kernel
from leat.nv.quantize import quantize_q8
from leat.quant import GGMLType, QTensor

FEW = 8  # tokens up to which the matrix-vector kernels run a mixture


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


@functools.cache
def _experts_swiglu_kernel(
    out: UOp, gate: UOp, up: UOp, xq: UOp, xd: UOp, xs: UOp, ids: UOp, pairs: int | UOp,
    meta: tuple[GGMLType, int, int, int],
) -> UOp:  # fmt: skip
    # out[p * rows + r] = silu(gate . x) * (up . x) for row r of expert ids[p], where pair p is
    # slot p % used of token p // used
    ggml_type, cols, rows, used = meta

    def expert(w: UOp) -> Callable[[UOp, UOp], UOp]:
        dot = DOTS[ggml_type](w, xq, xd, xs, cols)

        def chosen(row: UOp, unit: UOp) -> UOp:
            pair = row // rows
            return dot(ids[pair].load() * rows + row % rows, unit, pair // used)

        return chosen

    def combine(row: UOp, g: UOp, u: UOp) -> UOp:
        return silu(g) * u

    name = f"experts_swiglu_{ggml_type.name.lower()}"
    return rows_kernel(out, cols // 64, name, [expert(gate), expert(up)], combine, pairs * rows)


@functools.cache
def _experts_down_kernel(
    out: UOp, w: UOp, hq: UOp, hd: UOp, hs: UOp, ids: UOp, weights: UOp, residual: UOp,
    tokens: int | UOp, meta: tuple[GGMLType, int, int, int],
) -> UOp:  # fmt: skip
    # out[t * rows + r] = residual + the sum over token t's pairs p of weights[p] times row r of
    # expert ids[p] . the pair's hidden activations. Lanes take (slot, unit) items in turn, so
    # experts with narrow rows still keep them busy.
    ggml_type, cols, rows, used = meta
    dot, units = DOTS[ggml_type](w, hq, hd, hs, cols), cols // 64

    def mixed(row: UOp, item: UOp) -> UOp:
        pair = row // rows * used + item // units
        return weights[pair].load() * dot(ids[pair].load() * rows + row % rows, item % units, pair)

    def combine(row: UOp, total: UOp) -> UOp:
        return total + residual[row].load()

    name = f"experts_down_{ggml_type.name.lower()}"
    return rows_kernel(out, used * units, name, [mixed], combine, tokens * rows)


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


def supports_mixture(x: Tensor, gate: QTensor, up: QTensor, down: QTensor) -> bool:
    # whole warps of experts to route, and rows the quantization kernel takes whole
    experts, rows, cols = gate.shape
    same = gate.type == up.type and gate.shape == up.shape and down.shape == (experts, cols, rows)
    single = all(isinstance(b, int) and b == 1 for b in x.shape[:-2])
    types = gate.type in DOTS and down.type in DOTS
    shapes = experts % WARP == 0 and rows % 128 == 0 and cols % 128 == 0
    return on_nvidia(x) and single and same and types and shapes


def mixture(
    x: Tensor, scores: Tensor, gate: QTensor, up: QTensor, down: QTensor, used: int,
    norm: tuple[Tensor, float],
) -> Tensor:  # fmt: skip
    """x + the mixture of experts for tokens x (1, T, dim) and their router `scores` (1, T,
    experts): each token's `used` best scoring experts, weighted by the softmax of their scores,
    run silu(n @ gate.T) * (n @ up.T) @ down.T on n = rms_norm(x, *norm)."""
    _, tokens, dim = x.shape
    ids, weights = route(scores.reshape(tokens, gate.shape[0]), used)
    if isinstance(tokens, int) and tokens <= FEW:
        return _few(x, ids, weights, gate, up, down, used, norm)
    # many tokens: on tensor cores, each expert taking the pairs routed to it
    count = -(-x.max_shape[1] // TILE_TOKENS) * TILE_TOKENS
    order, counts = _bucket(ids, gate.shape[0], count, tokens * used)
    q8 = quantize_q8(tiled(x), norm, rows=tokens)
    hidden = routed_products(q8, tokens, (gate, up), order, counts, used, by_token=True)
    q8 = quantize_q8(hidden, rows=tokens * used)
    out = routed_products(q8, tokens, (down,), order, counts, used, by_token=False)
    out = out.reshape(count, used, dim)[:tokens] * weights.reshape(-1, used, 1)[:tokens]
    return x + out.sum(1).reshape(x.shape)


def _few(
    x: Tensor, ids: Tensor, weights: Tensor, gate: QTensor, up: QTensor, down: QTensor,
    used: int, norm: tuple[Tensor, float],
) -> Tensor:  # fmt: skip
    # the matrix-vector kernels, reading each chosen expert's rows once per token
    _, tokens, dim = x.shape
    _, rows, cols = gate.shape
    xq, xd, xs = quantize_q8(x.reshape(tokens, dim), norm)
    hidden = Tensor.empty(tokens * used * rows, dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(
        _experts_swiglu_kernel, pairs=tokens * used, meta=(gate.type, cols, rows, used)
    )
    words = storage_words(gate), storage_words(up)
    hidden = Tensor.custom_kernel(hidden, *words, xq, xd, xs, ids, fxn=fxn)[0]
    hq, hd, hs = quantize_q8(hidden.reshape(tokens * used, rows))
    out = Tensor.empty(tokens * dim, dtype=dtypes.float32, device=x.device)
    residual = x.reshape(tokens * dim).float().contiguous()
    fxn = functools.partial(_experts_down_kernel, tokens=tokens, meta=(down.type, rows, cols, used))
    srcs = storage_words(down), hq, hd, hs, ids, weights, residual
    return Tensor.custom_kernel(out, *srcs, fxn=fxn)[0].reshape(x.shape)


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
    ballot = UOp(Ops.CUSTOM, src=(mine,), arg=("__ballot_sync(0xffffffffu, {0})", dtypes.uint32))
    below = (UOp.const(1, dtypes.uint32) << lane.cast(dtypes.uint32)) - 1
    listed = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    listed = listed.after(listed[0].store(0))
    before = listed.after(turn)[0].load()
    place = before + _popc(ballot & below)
    put = order[(expert * per + place).valid(mine)].store(pair.cast(dtypes.int32))
    listed = listed.after(UOp.group(put, listed[0].store(before + _popc(ballot))).end(turn))
    store = counts[expert.valid(lane.eq(0))].store(listed[0].load())
    return store.end(expert, lane).sink(arg=KernelInfo(name="bucket", opts_to_apply=()))


def _popc(x: UOp) -> UOp:
    return UOp(Ops.CUSTOMI, src=(x,), arg=("__popc({0})", dtypes.int32))


def _bucket(ids: Tensor, experts: int, per: int, pairs: int | UOp) -> tuple[Tensor, Tensor]:
    # each expert's pairs, the first `pairs` of ids, as order (experts, per) and counts (experts,)
    order = Tensor.empty(experts * per, dtype=dtypes.int32, device=ids.device)
    counts = Tensor.empty(experts, dtype=dtypes.int32, device=ids.device)
    ids, pairs = carry(ids, pairs)
    fxn = functools.partial(_bucket_kernel, pairs=pairs)
    order, counts = Tensor.custom_kernel(order, counts, ids, fxn=fxn)[:2]
    return order, counts
