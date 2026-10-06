"""Gated DeltaNet, Qwen3.5's linear attention, in two kernels per layer and step: each channel's
causal convolution over a sequence's tokens, then each value head's recurrence over them, its state
in registers throughout, a warp for every COLUMNS of its columns. Then the outputs' gated norm."""

import functools
import math
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from leat.kernels.common import (
    LOG2E,
    WARP,
    carry_all,
    fast_reciprocal,
    fast_rsqrt,
    lane_range,
    on_gpu,
    opaque,
    pick,
    shfl_xor,
    warp_sum,
)

MOST_DIMS = 128  # keys a column of the state holds, its lanes' shares in registers
COLUMNS = 8  # of the state a warp holds, each by WARP // COLUMNS lanes
CONV_WARPS = 4  # per block of the convolution
CONV_TILE = 8  # tokens the convolution takes at a time, their inputs loaded together


@functools.cache
def _conv_kernel(
    out: UOp, conv_state: UOp, mixed: UOp, conv: UOp, slots: tuple[int | UOp, ...],
    starts: tuple[int | UOp, ...], tokens: int | UOp,
) -> UOp:  # fmt: skip
    # A thread per row and channel: the channel's causal convolution over the row's tokens in
    # turn, then SiLU, its last inputs carried in registers and left in the conv state
    channels, width = (int(d) for d in conv.shape)
    history = width - 1
    row = UOp.range(len(slots), 0, AxisType.GLOBAL)
    block = UOp.range(channels // (CONV_WARPS * WARP), 1, AxisType.GLOBAL)
    lane, wave = lane_range(), UOp.range(CONV_WARPS, 2, AxisType.LOCAL)
    c = (block * CONV_WARPS + wave) * WARP + lane
    slot, kept = pick(row, slots), pick(row, starts) > 0  # from position 0, zero states
    weights = [conv[c, w].load() for w in range(width)]
    held = UOp.alloc((history,), dtypes.float32, addrspace=AddrSpace.REG)
    loads = [kept.where(conv_state[slot, w, c].load(), 0.0) for w in range(history)]
    held = held.after(held.store(UOp.stack(*loads)))
    n = UOp.range((tokens + CONV_TILE - 1) // CONV_TILE, 3, AxisType.LOOP)
    past, stores = [held.after(n)[w].load() for w in range(history)], []
    # indices of an opaque copy of the tile: a constant number of tokens may leave a token of
    # every tile to the first alone, whose loads tinygrad would then take out of the loop
    tile = opaque(n)
    for u in range(CONV_TILE):  # tokens past the row's leave the inputs held as they are
        real = n * CONV_TILE + u < tokens
        at = (row * tokens + tile * CONV_TILE + u).valid(real)
        x = mixed[at, c].load()
        products = [a * b for a, b in zip([*past, x], weights, strict=True)]
        stores.append(out[at, c].store(_silu(_sum(products))))
        past = [real.where(new, old) for new, old in zip([*past[1:], x], past, strict=True)]
    held = held.after(UOp.group(held.store(UOp.stack(*past)), *stores).end(n))
    # past the loop, indices of opaque copies of the coordinates: see common.opaque
    ranges = (lane, wave, block, row)
    lane, wave, block, row = (opaque(u) for u in ranges)
    c, slot = (block * CONV_WARPS + wave) * WARP + lane, pick(row, slots)
    stores = [conv_state[slot, w, c].store(held[w].load()) for w in range(history)]
    info = KernelInfo(name="delta_net_conv", opts_to_apply=())
    return UOp.group(*stores).end(*ranges).sink(arg=info)


@functools.cache
def _recurrence_kernel(
    out: UOp, state: UOp, conved: UOp, gates: UOp, decay: UOp, bias: UOp,
    slots: tuple[int | UOp, ...], starts: tuple[int | UOp, ...], tokens: int | UOp, eps: float,
) -> UOp:  # fmt: skip
    # A block, a warp, per row, value head and COLUMNS of its dimensions, whose columns of the
    # head's state, keys by values, its lanes hold in registers: the lanes of a column a share of
    # its keys each. For each token, the lanes L2-norm the queries and keys of the head's key
    # head, each taking every 32nd dimension, and share them through shared memory; each then
    # sums its share of its column's products with them, the column's lanes their shares, and
    # works out the column's update and output and its share of the new column.
    _, heads, dims, _ = (int(d) for d in state.shape)
    k_heads = (int(conved.shape[1]) - heads * dims) // (2 * dims)
    keys = dims // (WARP // COLUMNS)  # of a lane's share
    row, head = UOp.range(len(slots), 0, AxisType.GLOBAL), UOp.range(heads, 1, AxisType.GLOBAL)
    block, lane = UOp.range(dims // COLUMNS, 2, AxisType.GLOBAL), lane_range()
    slot, kept = pick(row, slots), pick(row, starts) > 0  # from position 0, zero states
    dim, share = block * COLUMNS + lane % COLUMNS, lane // COLUMNS
    k_head, rate, offset = head % k_heads, decay[head].load(), bias[head].load()
    cells = [UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(keys)]
    loads = [kept.where(state[slot, head, share * keys + i, dim].load(), 0.0) for i in range(keys)]
    cells = [cell.after(cell[0].store(x)) for cell, x in zip(cells, loads, strict=True)]
    shared = UOp.alloc((2, dims), dtypes.float32, addrspace=AddrSpace.LOCAL)

    t = UOp.range(tokens, 3, AxisType.LOOP)
    at, old = row * tokens + t, [cell.after(t)[0].load() for cell in cells]
    ats = [lane + j * WARP for j in range(dims // WARP)]  # the lane's dimensions of q and k
    q = [conved[at, k_head * dims + i].load() for i in ats]
    k = [conved[at, (k_heads + k_head) * dims + i].load() for i in ats]
    # 1 / max(|x|, eps), and the normed query's product with the normed key
    q_inv = fast_rsqrt(warp_sum(_sum([x * x for x in q])).maximum(eps * eps)) / math.sqrt(dims)
    k_inv = fast_rsqrt(warp_sum(_sum([x * x for x in k])).maximum(eps * eps))
    overlap = warp_sum(_sum([a * b for a, b in zip(q, k, strict=True)])) * q_inv * k_inv
    stores = [shared[0, i].store(x * q_inv) for i, x in zip(ats, q, strict=True)]
    stores += [shared[1, i].store(x * k_inv) for i, x in zip(ats, k, strict=True)]
    shared = shared.after(*stores)
    qs, ks = ([shared[n, share * keys + i].load() for i in range(keys)] for n in (0, 1))
    # the decay exp(a * softplus(alpha + bias)) and the update's gain sigmoid(beta)
    decayed = _exp(rate * _softplus(gates[at, head].load() + offset))
    gain = _sigmoid(gates[at, heads + head].load())
    by_key = _column_sum(_chains([s * x for s, x in zip(old, ks, strict=True)]))
    by_query = _column_sum(_chains([s * x for s, x in zip(old, qs, strict=True)]))
    delta = gain * (conved[at, (2 * k_heads + head) * dims + dim].load() - decayed * by_key)
    update = UOp.group(
        *(
            cell[0].store(decayed * s + x * delta)
            for cell, s, x in zip(cells, old, ks, strict=True)
        ),
        out[at, (head * dims + dim).valid(share.eq(0))].store(decayed * by_query + delta * overlap),
    ).end(t)
    cells = [cell.after(update) for cell in cells]
    # past the loop, indices of opaque copies of the coordinates: see common.opaque
    ranges = (lane, block, head, row)
    lane, block, head, row = (opaque(u) for u in ranges)
    slot, dim, share = pick(row, slots), block * COLUMNS + lane % COLUMNS, lane // COLUMNS
    stores = [
        state[slot, head, share * keys + i, dim].store(cell[0].load())
        for i, cell in enumerate(cells)
    ]
    info = KernelInfo(name="delta_net", opts_to_apply=())
    return UOp.group(*stores).end(*ranges).sink(arg=info)


def _column_sum(value: UOp) -> UOp:
    # over the lanes of a column: those COLUMNS apart
    mask = COLUMNS
    while mask < WARP:
        value, mask = value + shfl_xor(value, mask), 2 * mask
    return value


def _exp(x: UOp) -> UOp:
    return (x * LOG2E).exp2()


def _softplus(x: UOp) -> UOp:
    # log(1 + e^x), without overflow
    return x.maximum(0) + (1 + _exp(-x.abs())).log2() / LOG2E


def _sum(values: list[UOp]) -> UOp:
    return functools.reduce(UOp.__add__, values) if values else UOp.const(0.0, dtypes.float32)


def _chains(values: list[UOp], chains: int = 8) -> UOp:
    # the sum in interleaved chains, whose additions overlap, rather than one long chain
    return _sum([_sum(values[c::chains]) for c in range(min(chains, len(values)))])


def _sigmoid(x: UOp) -> UOp:
    return fast_reciprocal(1 + (x * -LOG2E).exp2())


def _silu(x: UOp) -> UOp:
    return x * _sigmoid(x)


def supports_delta_net(mixed: Tensor, state: Tensor) -> bool:
    # keys and values of the same dimensions, whole warps of them that fit in registers, and
    # channels in whole blocks of the convolution
    _, heads, k_dims, dims = (int(d) for d in state.shape)
    channels = int(mixed.shape[-1])
    fits = k_dims == dims and dims % WARP == 0 and dims <= MOST_DIMS
    return on_gpu(mixed) and fits and channels % (CONV_WARPS * WARP) == 0


def delta_net(
    mixed: Tensor, z: Tensor, gates: Tensor, conv: Tensor,
    decay: tuple[Tensor, Tensor], norm: tuple[Tensor, float], states: tuple[Tensor, Tensor],
    slots: list[int | UOp], starts: list[int | UOp], tokens: int | UOp,
) -> Tensor:  # fmt: skip
    """ops.delta_net() for rows of `tokens` tokens each, row r from position starts[r] of slot
    slots[r]: one row of a bound number of tokens, or several of one each."""
    T, count = mixed.shape[1], int(mixed.max_shape[1])
    heads, dims = int(states[1].shape[1]), int(states[1].shape[3])

    def rows(t: Tensor) -> Tensor:  # (count, width), past the tokens padded
        return t.reshape(T, t.shape[-1]).float().pad_to((count, t.shape[-1])).contiguous()

    def bound(t: Tensor) -> tuple[Tensor, dict[str, Any]]:  # t, carrying the rows' variables
        t, (*held, n) = carry_all(t, [*slots, *starts, tokens])
        return t, {
            "slots": tuple(held[: len(slots)]),
            "starts": tuple(held[len(slots) :]),
            "tokens": n,
        }

    conved = Tensor.empty(count, int(mixed.shape[-1]), dtype=dtypes.float32, device=mixed.device)
    mixed, args = bound(rows(mixed))
    fxn = functools.partial(_conv_kernel, **args)
    conved = Tensor.custom_kernel(conved, states[0], mixed, conv.float().contiguous(), fxn=fxn)[0]
    out = Tensor.empty(count, heads * dims, dtype=dtypes.float32, device=conved.device)
    conved, args = bound(conved)
    fxn = functools.partial(_recurrence_kernel, eps=norm[1], **args)
    small = (rows(gates), *(t.float().contiguous() for t in decay))
    out = Tensor.custom_kernel(out, states[1], conved, *small, fxn=fxn)[0]
    # each head's output normed, and gated by SiLU of z
    out = out[:T].reshape(T, heads, dims)
    gated = out * (out.square().mean(-1, keepdim=True) + norm[1]).rsqrt() * norm[0]
    gated = gated * z.reshape(T, heads, dims).silu()
    return gated.reshape(1, T, heads * dims)
