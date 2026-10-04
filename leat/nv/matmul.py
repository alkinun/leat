"""Linear layers for several tokens on int8 tensor cores, llama.cpp's MMQ.

The activations are quantized as for one token. A block stages a tile of weights, unpacked to
int8, and 64 tokens in shared memory, 128 weights of each row per step, while it fetches the next
step into registers. Each of its warps multiplies 32 rows by the 64 tokens on tensor cores and
scales every group of 32 in f32. Tiles of 256 rows of Q4_K reach 68 to 79 TOPS on 512 tokens,
against 54 to 67 for 128; matrices with few rows take 128, to occupy more SMs, and 64 where 128
do not divide them. A mixture of experts' tokens take the same kernel, a block per tile of an
expert's tokens.
"""

import functools
import itertools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from leat.nv.common import (
    GROUP,
    WARP,
    activation,
    carry,
    f16,
    fifth_bits,
    lane_range,
    minus,
    on_nvidia,
    register,
    storage_words,
)
from leat.nv.quantize import quantize_q8
from leat.quant import GGMLType, QTensor

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


# 0, once the threads for which a condition holds have exited
_EXIT = '[&]{{ if ({0}) asm volatile("exit;"); return 0; }}()'


def _mma(a: list[UOp], b: list[UOp]) -> list[UOp]:
    # lane 4g + t gets the results for rows g and g + 8 by tokens 2t and 2t + 1
    c = UOp.alloc((4,), dtypes.int32, addrspace=AddrSpace.REG)
    product = UOp(Ops.CUSTOM, src=(c[0], *a, *b), arg=(_MMA[8 * len(a)], dtypes.int32))
    c = c.after(c[0].store(product))
    return [c[i].load() for i in range(4)]


class _Stack:
    """Weight matrices of one type and width stacked by rows, as one: a tile of rows lies within
    one of them, which `load` reads with each matrix's load predicated on holding the tile. Paired,
    a tile holds the same rows of two matrices instead, in its first and second half."""

    def __init__(
        self, ws: tuple[UOp, ...], heights: tuple[int, ...], tile: UOp, rows: int, paired: bool,
        expert: UOp | None = None, experts: int = 1, fused: bool = False,
    ):  # fmt: skip
        # experts: each matrix stacks that many of `heights` rows, and the tile's rows are those
        # of `expert`; fused, the paired matrices are one stack, each expert's first rows then
        # the other's
        stacked = 2 if fused else 1
        self.row_words = int(ws[0].shape[0]) // (heights[0] * experts * stacked)
        self.half, first = rows // 2 if paired else 0, 0
        self.parts: list[tuple[UOp, UOp, UOp | None]] = []
        for i, (w, h) in enumerate(zip(ws, heights, strict=True)):
            base = 0 if expert is None else (expert * stacked + (i if fused else 0)) * h
            if paired:  # a matrix's rows from tile * half on are the tile's rows from first on
                self.parts.append((w, base + tile * self.half - first, None))
                first += self.half
                continue
            mine = None if len(ws) == 1 else (tile >= first) & (tile < first + h // rows)
            self.parts.append((w, base + (tile - first) * rows, mine))
            first += h // rows

    def load(self, row: UOp, at: UOp | int, inside: UOp | None = None) -> UOp:
        # word `at` of the tile's row `row`, or 0 where `inside` does not hold
        words = []
        for i, (w, row0, mine) in enumerate(self.parts):
            if self.half:
                mine = (row < self.half) if i == 0 else (row >= self.half)
            if inside is not None:
                mine = inside if mine is None else mine & inside
            index = (row0 + row) * self.row_words + at
            words.append(w[index if mine is None else index.valid(mine)].load())
        return functools.reduce(UOp.__or__, words)

    def word16(self, row: UOp, at: UOp | int, inside: UOp | None = None) -> UOp:
        # four bytes from halfword `at` on, as common.word16
        low, high = (self.load(row, at + i, inside).cast(dtypes.uint32) for i in (0, 1))
        return low | (high << 16)


# A weight type's part of the kernel: its shared buffers, `fetch` of a step's words into
# registers and `put` of them into the buffers, and `products` of group s. Those take the
# buffers, s, the lane's rows (r, r + 8) in the tile and t4, its activation words, its two
# tokens' d and d * sum(q), and its 4 sums so far; they return the sums with the group's
# products added, written so that the compiler fuses the additions into multiply-adds.


class _Q4KTile:
    """Q4_K weights, half a block per step: 16 words of nibbles per row, and its 4 groups' scales
    and mins, which get_scale_min_k4 takes from the block's 12 scale bytes."""

    max_rows, block = 256, 256  # rows per tile at most, for shared memory; weights per block

    def __init__(self, stack: _Stack, rows: int, tid: UOp, cols: int):
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
        return stores + self.put_scales(scales, step, words[-4:])

    def put_scales(self, scales: UOp, step: UOp, words: list[UOp]) -> list[UOp]:
        # d * scale and dmin * min of the thread's row, for the step's 4 groups
        dm, *packed = words

        def byte(i: int) -> UOp:
            return (packed[i // 4] >> (8 * (i % 4))) & 0xFF

        first, stores = (step % 2).eq(0), []  # groups 0..3 of the block, else 4..7
        for g in range(4):
            sc = first.where(byte(g) & 63, (byte(g + 8) & 15) | ((byte(g) >> 6) << 4))
            mn = first.where(byte(g + 4) & 63, (byte(g + 8) >> 4) | ((byte(g + 4) >> 6) << 4))
            stores.append(scales[0, g, self.tid].store(f16(dm) * sc.float()))
            stores.append(scales[1, g, self.tid].store(f16(dm >> 16) * mn.float()))
        return stores

    @staticmethod
    def products(bufs, s, rows, t4, b, xd, xs, acc) -> list[UOp]:
        # groups 2j and 2j + 1 are the low and high nibbles of words 8j .. 8j + 7
        quants, scales = bufs
        shift = (4 * (s % 2)).cast(dtypes.uint32)
        words = [quants[r, s // 2 * 8 + 4 * h + t4].load() for h in range(2) for r in rows]
        c = _mma([((x >> shift) & 0x0F0F0F0F).bitcast(dtypes.int32) for x in words], b)
        return _with_mins(scales, s, rows, xd, xs, acc, c)


class _Q5KTile(_Q4KTile):
    """Q5_K weights: Q4_K's nibbles each with a fifth bit from the block's 8 words of high bits,
    unpacked to int8 as Q6_K is: 4 groups of 32 per row, and their scales and mins."""

    max_rows = 128

    def __init__(self, stack: _Stack, rows: int, tid: UOp, cols: int):
        self.stack, self.rows, self.tid = stack, rows, tid
        self.quants = UOp.alloc((rows, 32), dtypes.int32, addrspace=AddrSpace.LOCAL)
        self.scales = UOp.alloc((2, 4, rows), dtypes.float32, addrspace=AddrSpace.LOCAL)

    def fetch(self, step: UOp) -> list[UOp]:
        # each of the thread's words of nibbles and the word of high bits behind it; then d and
        # dmin and the scale bytes of its row
        words, block = [], step // 2 * 44
        for i in range(16):
            row, word = (i * self.rows + self.tid) // 16, (i * self.rows + self.tid) % 16
            words.append(self.stack.load(row, block + 12 + 16 * (step % 2) + word))
            words.append(self.stack.load(row, block + 4 + word % 8))
        return words + [self.stack.load(self.tid, block + i) for i in range(4)]

    def put(self, bufs: list[UOp], step: UOp, words: list[UOp]) -> list[UOp]:
        quants, scales = bufs
        stores = []
        for i in range(16):
            row, at = (i * self.rows + self.tid) // 16, (i * self.rows + self.tid) % 16
            word, high = words[2 * i : 2 * i + 2]
            pair = 2 * (step % 2) + at // 8  # the word holds sub-blocks 2 * pair and 2 * pair + 1
            for h in range(2):
                bits = (high >> (2 * pair + h).cast(dtypes.uint32)) & 0x01010101
                value = ((word >> (4 * h)) & 0x0F0F0F0F) | (bits << 4)
                at_ = _swizzle(row, (2 * (at // 8) + h) * 8 + at % 8)
                stores.append(quants[row, at_].store(value.bitcast(dtypes.int32)))
        return stores + self.put_scales(scales, step, words[-4:])

    @staticmethod
    def products(bufs, s, rows, t4, b, xd, xs, acc) -> list[UOp]:
        quants, scales = bufs
        c = _mma(
            [quants[r, _swizzle(r, 8 * s + 4 * h + t4)].load() for h in range(2) for r in rows], b
        )
        return _with_mins(scales, s, rows, xd, xs, acc, c)


class _Q6KTile:
    """Q6_K weights, half a block per step, unpacked to int8 as in matvec's _q6_k_dot: 4 groups of
    32 per row, its 8 scales and d. Rows have no padding, to fit in shared memory; XOR-ing words
    with the row instead keeps a fragment's 8 rows in different banks."""

    max_rows, block = 256, 256

    def __init__(self, stack: _Stack, rows: int, tid: UOp, cols: int):
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
                stores.append(quants[row, _swizzle(row, 8 * k + m)].store(minus(low | high, 32)))
        stores += [scales[j, self.tid].store(words[-3 + j]) for j in range(2)]
        return stores + [d[self.tid].store(f16(words[-1]))]

    @staticmethod
    def products(bufs, s, rows, t4, b, xd, xs, acc) -> list[UOp]:
        # a scale per 16 weights: two k = 16 products, combined in int32
        quants, scales, d = bufs
        a = [quants[r, _swizzle(r, 8 * s + 4 * h + t4)].load() for h in range(2) for r in rows]
        c = [_mma(a[2 * h : 2 * h + 2], [b[h]]) for h in range(2)]
        sums = []
        for e in range(4):
            packed = scales[s // 2, rows[e // 2]].load()  # the scales of groups 2s and 2s + 1
            sc = [_signed_byte(packed, 16 * (s % 2) + 8 * h) for h in range(2)]
            dot = (c[0][e] * sc[0] + c[1][e] * sc[1]).float()
            sums.append(acc[e] + dot * (d[rows[e // 2]].load() * xd[e % 2]))
        return sums


class _Q80Tile:
    """Q8_0 weights, 4 blocks per step: 32 words of int8 per row, swizzled as for Q6_K, and the 4
    blocks' d. Rows of whole blocks but not whole steps read zeros past their end."""

    max_rows, block, words = 256, 32, 17  # weights and halfwords per block

    def __init__(self, stack: _Stack, rows: int, tid: UOp, cols: int):
        self.stack, self.rows, self.tid = stack, rows, tid
        self.blocks, self.ragged = cols // 32, cols % STEP != 0
        self.quants = UOp.alloc((rows, 32), dtypes.int32, addrspace=AddrSpace.LOCAL)
        self.d = UOp.alloc((4, rows), dtypes.float32, addrspace=AddrSpace.LOCAL)

    def shared(self) -> list[UOp]:
        return [self.quants, self.d]

    def inside(self, step: UOp, j: UOp | int) -> UOp | None:
        # whether the row holds the step's block j
        return step * 4 + j < self.blocks if self.ragged else None

    def fetch(self, step: UOp) -> list[UOp]:
        # the thread's words of the step's blocks of 17 halfwords, then the d of its row's 4
        words, base = [], step * 4 * self.words
        for i in range(32):
            row, at = (i * self.rows + self.tid) // 32, (i * self.rows + self.tid) % 32
            at, inside = base + at // 8 * self.words + 1 + 2 * (at % 8), self.inside(step, at // 8)
            words.append(self.stack.word16(row, at, inside))
        return words + self.scales(step)

    def scales(self, step: UOp) -> list[UOp]:
        # the d of the thread's row's 4 blocks of the step
        at = step * 4 * self.words
        return [
            self.stack.load(self.tid, at + self.words * j, self.inside(step, j)).cast(dtypes.uint32)
            for j in range(4)
        ]

    def put(self, bufs: list[UOp], step: UOp, words: list[UOp]) -> list[UOp]:
        quants, d = bufs
        stores = []
        for i, word in enumerate(words[:-4]):
            row, at = (i * self.rows + self.tid) // 32, (i * self.rows + self.tid) % 32
            stores.append(quants[row, _swizzle(row, at)].store(word.bitcast(dtypes.int32)))
        return stores + [d[j, self.tid].store(f16(word)) for j, word in enumerate(words[-4:])]

    @staticmethod
    def products(bufs, s, rows, t4, b, xd, xs, acc) -> list[UOp]:
        quants, d = bufs
        c = _mma(
            [quants[r, _swizzle(r, 8 * s + 4 * h + t4)].load() for h in range(2) for r in rows], b
        )
        return [acc[e] + c[e].float() * (d[s, rows[e // 2]].load() * xd[e % 2]) for e in range(4)]


class _Q50Tile(_Q80Tile):
    """Q5_0 weights, unpacked to Q8_0's layout less 16: a block's 16 bytes hold values 0..15 in
    their low nibbles and 16..31 in their high ones, each with a fifth bit from its 32 high bits."""

    words = 11

    def fetch(self, step: UOp) -> list[UOp]:
        # for each of the thread's (row, block j, word k) of the step, a word of the block's
        # nibbles and its high bits; then the d of its row's 4 blocks
        words, base = [], step * 4 * self.words
        for i in range(16):
            item = i * self.rows + self.tid
            row, j, k = item // 16, item % 16 // 4, item % 4
            at, inside = base + self.words * j, self.inside(step, j)
            words += [
                self.stack.word16(row, at + 3 + 2 * k, inside),
                self.stack.word16(row, at + 1, inside),
            ]
        return words + self.scales(step)

    def put(self, bufs: list[UOp], step: UOp, words: list[UOp]) -> list[UOp]:
        quants, d = bufs
        stores = []
        for i in range(16):
            item = i * self.rows + self.tid
            row, j, k = item // 16, item % 16 // 4, item % 4
            nibbles, high = words[2 * i], words[2 * i + 1]
            for h in range(2):  # values 4k.. of the block, then 16 + 4k..
                fifth = fifth_bits((high >> (16 * h + 4 * k)) & 15)
                q = ((nibbles >> (4 * h)) & 0x0F0F0F0F) | fifth
                stores.append(quants[row, _swizzle(row, 8 * j + 4 * h + k)].store(minus(q, 16)))
        return stores + [d[j, self.tid].store(f16(word)) for j, word in enumerate(words[-4:])]


def _swizzle(row: UOp, word: UOp | int) -> UOp:
    return (row % 8 * 4) ^ word


def _signed_byte(word: UOp, shift: UOp) -> UOp:
    byte = ((word >> shift.cast(dtypes.uint32)) & 0xFF).cast(dtypes.uint8)
    return byte.bitcast(dtypes.int8).cast(dtypes.int32)


def _with_mins(
    scales: UOp, s: UOp, rows: tuple[UOp, UOp], xd: list[UOp], xs: list[UOp], acc: list[UOp],
    c: list[UOp],
) -> list[UOp]:  # fmt: skip
    # the sums plus group s's products c of a K-quant with mins: c * d * scale - dmin * min * xs
    return [
        acc[e]
        + c[e].float() * (scales[0, s, rows[e // 2]].load() * xd[e % 2])
        - scales[1, s, rows[e // 2]].load() * xs[e % 2]
        for e in range(4)
    ]


_TILES: dict[GGMLType, type[_Q4KTile] | type[_Q6KTile] | type[_Q80Tile]] = {
    GGMLType.Q4_K: _Q4KTile,
    GGMLType.Q5_K: _Q5KTile,
    GGMLType.Q6_K: _Q6KTile,
    GGMLType.Q5_0: _Q50Tile,
    GGMLType.Q8_0: _Q80Tile,
}


@functools.cache
def _matmul_kernel(
    out: UOp, *srcs: UOp, tokens: int | UOp, ggml_type: GGMLType, rows: int,
    heights: tuple[int, ...], gated: bool, routed: tuple[int, bool] | None = None,
    fused: bool = False, gelu: bool = False,
) -> UOp:  # fmt: skip
    # srcs: the stacked matrices, of `heights` rows, then xq, xd, xs and a residual; a
    # block per tile of `rows` rows by TILE_TOKENS tokens. Gated, the matrices are gate and up,
    # a tile holds rows of both, each warp the same 16 of either, and out is act(gate) * up for
    # SiLU, or GELU if gelu; fused, gate and up are one stack of experts, as _Stack has it.
    # Routed, the matrices stack experts, and order and counts list the (token, slot) pairs routed
    # to each, of `used` slots per token: a block takes TILE_TOKENS pairs of an expert's list,
    # reading activation row pair // used, by token, or else pair, and writing output row pair.
    ws, (xq, xd, xs, *rest) = srcs[: len(heights)], srcs[len(heights) :]
    count, n = (int(x) for x in out.shape)
    sources = count // routed[0] if routed is not None and routed[1] else count
    cols, threads = 4 * int(xq.shape[0]) // sources, rows
    # rows of whole blocks but not whole steps: the last step reads zeros past their end
    out_rows, steps, ragged = rows // 2 if gated else rows, -(-cols // STEP), cols % STEP != 0
    tile_tokens = UOp.range((tokens + TILE_TOKENS - 1) // TILE_TOKENS, 0, AxisType.GLOBAL)
    tile_rows = UOp.range(n // out_rows, 1, AxisType.GLOBAL)
    lane, warp = lane_range(), UOp.range(rows // WARP_ROWS, 2, AxisType.LOCAL)
    tid, g, t4 = warp * WARP + lane, lane // 4, lane % 4
    row0, token0 = tile_rows * out_rows, tile_tokens * TILE_TOKENS
    ranges = [tile_tokens, tile_rows, lane, warp]
    if routed is None:
        residual, stack = rest, _Stack(ws, heights, tile_rows, rows, gated)

        def source(tok: UOp | int) -> UOp:  # the activation row of the tile's token tok
            return token0 + tok

        def target(tok: UOp | int, row: UOp) -> UOp:  # where its output for a row goes
            return out[token0 + tok, row]

    else:
        (order, counts), residual = rest, []
        (used, by_token), experts = routed, int(counts.shape[0])
        ranges.append(block_expert := UOp.range(experts, 5, AxisType.GLOBAL))
        # blocks past the end of their expert's list exit at once: all else depends on the expert
        expert = block_expert + UOp(Ops.CUSTOM, src=(token0 >= counts[block_expert].load(),),
                                    arg=(_EXIT, dtypes.int32))  # fmt: skip
        listed, per = counts[expert].load(), count // used
        stack = _Stack(ws, heights, tile_rows, rows, gated, expert, experts, fused)

        def pair(tok: UOp | int) -> UOp:
            at = token0 + tok
            return (at < listed).where(order[expert * per + at].load(), 0)

        def source(tok: UOp | int) -> UOp:
            return pair(tok) // used if by_token else pair(tok)

        def target(tok: UOp | int, row: UOp) -> UOp:
            return out.flatten()[(pair(tok) * n + row).valid(token0 + tok < listed)]

    weights = _TILES[ggml_type](stack, rows, tid, cols)
    # the first rows of the warp's two subtiles of 16 rows, within the tile
    firsts = [warp * 16 + mi * out_rows if gated else warp * WARP_ROWS + mi * 16 for mi in (0, 1)]
    # the tile's tokens for a step: 32 words each, and the d and d * sum(q) of 4 groups
    act = UOp.alloc((TILE_TOKENS, 32 + 4), dtypes.int32, addrspace=AddrSpace.LOCAL)
    act_scales = UOp.alloc((2, 4, TILE_TOKENS), dtypes.float32, addrspace=AddrSpace.LOCAL)
    pairs = [_split(i * threads + tid, 32) for i in range(32 * TILE_TOKENS // threads)]
    groups = [_split(i * threads + tid, TILE_TOKENS) for i in range(4 * TILE_TOKENS // threads)]

    def fetch(step: UOp) -> list[UOp]:
        words = weights.fetch(step)
        for tok, word in pairs:
            at = source(tok) * (cols // 4) + step * 32 + word
            at = at.valid(step * 32 + word < cols // 4) if ragged else at
            words.append(xq[at].load().bitcast(dtypes.uint32))
        for grp, tok in groups:
            at = source(tok) * (cols // GROUP) + step * 4 + grp
            at = at.valid(step * 4 + grp < cols // GROUP) if ragged else at
            words += [xd[at].load().bitcast(dtypes.uint32), xs[at].load().bitcast(dtypes.uint32)]
        return words

    def put(bufs: list[UOp], step: UOp, words: list[UOp]) -> list[UOp]:
        *mine, act, act_scales = bufs
        theirs = len(words) - len(pairs) - 2 * len(groups)
        stores = weights.put(mine, step, words[:theirs])
        quants, scales = words[theirs : theirs + len(pairs)], words[theirs + len(pairs) :]
        for (tok, word), value in zip(pairs, quants, strict=True):
            stores.append(act[tok, word].store(value.bitcast(dtypes.int32)))
        for i, (grp, tok) in enumerate(groups):
            for j in (0, 1):
                value = scales[2 * i + j].bitcast(dtypes.float32)
                stores.append(act_scales[j, grp, tok].store(value))
        return stores

    # step 0 before the loop; in step i, the fetch of step i + 1 is staged in registers before
    # the products, so that its loads are in flight meanwhile, and stored after them. tinygrad
    # orders operations by their dependencies alone, so the products depend on the staging.
    bufs = [*weights.shared(), act, act_scales]
    first = put(bufs, UOp.const(0, dtypes.weakint), fetch(UOp.const(0, dtypes.weakint)))
    step = UOp.range(steps, 3, AxisType.LOOP)
    following = fetch((step + 1).minimum(steps - 1))
    stage = UOp.alloc((len(following),), dtypes.uint32, addrspace=AddrSpace.REG)
    staged = UOp.group(*(stage[i].store(v) for i, v in enumerate(following)))
    bufs = [buf.after(*first).after(step).after(staged) for buf in bufs]
    *mine, act, act_scales = bufs

    acc = register((SUBTILES_M * SUBTILES_N * 4,), 0.0)
    s = UOp.range(STEP // GROUP, 4, AxisType.LOOP)
    prev, vals = acc.after(step, s), list[UOp]()
    for mi in range(SUBTILES_M):
        r = firsts[mi] + g
        for ni in range(SUBTILES_N):
            tok = ni * 8
            b = [act[tok + g, 8 * s + 4 * h + t4].load() for h in range(2)]
            xd_, xs_ = ([act_scales[i, s, tok + 2 * t4 + j].load() for j in (0, 1)] for i in (0, 1))
            sums = [prev[len(vals) + e].load() for e in range(4)]
            vals += weights.products(mine, s, (r, r + 8), t4, b, xd_, xs_, sums)
    computed = acc.store(UOp.stack(*vals)).end(s)
    # every warp is done with this step's tiles before they are overwritten
    done = UOp(Ops.BARRIER, src=(computed,))
    staged_words = [stage.after(staged)[i].load() for i in range(len(following))]
    later = put([buf.after(done) for buf in bufs], step + 1, staged_words)
    acc = acc.after(UOp.group(computed, *later).end(step))

    results = []
    for mi in range(1 if gated else SUBTILES_M):
        for ni in range(SUBTILES_N):
            for e in range(4):
                row = row0 + firsts[mi] + g + 8 * (e // 2)
                token = ni * 8 + 2 * t4 + e % 2
                value = acc[(mi * SUBTILES_N + ni) * 4 + e].load()
                if gated:
                    value = activation(gelu)(value) * acc[(SUBTILES_N + ni) * 4 + e].load()
                if residual:
                    value = value + residual[0][token0 + token, row].load()
                results.append(target(token, row).store(value))
    info = KernelInfo(name=f"matmul_{ggml_type.name.lower()}_{n}_{cols}", opts_to_apply=())
    return UOp.group(*results).end(*ranges).sink(arg=info)


def _split(a: UOp, b: int) -> tuple[UOp, UOp]:
    return a // b, a % b


def tiled(x: Tensor) -> Tensor:
    # Several tokens, while prefilling a bound count of them: buffers hold the most there may be,
    # rounded up to whole tiles, and the kernels stop after the tiles holding actual tokens. The
    # first rows of a kernel's output of that many rows are that output, whose other rows nobody
    # reads; anything else is padded, a copy.
    count, n = -(-x.max_shape[-2] // TILE_TOKENS) * TILE_TOKENS, x.shape[-1]
    view = x.uop
    while view.op is Ops.RESHAPE:
        view = view.src[0]
    if view.op is Ops.SHRINK and view.src[0].shape == (count, n) and x.dtype == dtypes.float32:
        starts, sizes = view.src[1].src, view.src[2].src
        if all(s.op is Ops.CONST and s.arg == 0 for s in starts) and sizes[1].arg == n:
            return Tensor(view.src[0])
    return x.reshape(x.shape[-2], n).float().pad_to((count, n)).contiguous()


def _tile(ggml_type: GGMLType, heights: list[int], gated: bool) -> int:
    # rows per tile: 256 where there are many, to reach more TOPS, else 128 to occupy more SMs, or
    # 64 where 128 do not divide the matrices; a gated tile holds half its rows of each
    per = 2 if gated else 1
    big = sum(heights) >= 4096 and all(h % (256 // per) == 0 for h in heights)
    if big and _TILES[ggml_type].max_rows == 256:
        return 256
    return 128 if all(h % (128 // per) == 0 for h in heights) else 64


def _products(
    q8: tuple[Tensor, Tensor, Tensor], tokens: int | UOp, ws: tuple[QTensor, ...],
    residual: Tensor | None, gated: bool = False, gelu: bool = False,
) -> list[tuple[Tensor, list[int]]]:  # fmt: skip
    # the products of the quantized activations and each matrix, with consecutive matrices of one
    # type stacked in one kernel, as few rows leave SMs idle: each kernel's output and heights.
    # Gated, ws are gate and up, and the one output is act(gate) * up, SiLU or GELU if gelu.
    xq, xd, xs = q8
    count = int(xd.shape[0]) * GROUP // ws[0].shape[1]
    xq, bound = carry(xq, tokens)
    res = () if residual is None else (tiled(residual),)
    outs = []
    for _, group in itertools.groupby(ws, key=lambda w: w.type):
        stack = tuple(group)
        heights = [w.shape[0] for w in stack]
        tile = _tile(stack[0].type, heights, gated)
        width = heights[0] if gated else sum(heights)
        out = Tensor.empty(count, width, dtype=dtypes.float32, device=xq.device)
        fxn = functools.partial(
            _matmul_kernel, tokens=bound, ggml_type=stack[0].type, rows=tile,
            heights=tuple(heights), gated=gated, gelu=gelu,
        )  # fmt: skip
        words = map(storage_words, stack)
        out = Tensor.custom_kernel(out, *words, xq, xd, xs, *res, fxn=fxn)[0]
        outs.append((out, [width] if gated else heights))
    return outs


def routed_products(
    q8: tuple[Tensor, Tensor, Tensor], tokens: int | UOp, ws: tuple[QTensor, ...], order: Tensor,
    counts: Tensor, used: int, by_token: bool, fused: bool = False, gelu: bool = False,
) -> Tensor:  # fmt: skip
    """The products of quantized activations and the experts that (token, slot) pairs are routed
    to, a row per pair, slot p % used of token p // used: ws stack experts (experts, rows, cols),
    and `order` (experts, n) and `counts` list each expert's pairs. The activations have a row per
    token if by_token, else per pair. Two matrices are gate and up, multiplied as act(gate) * up,
    SiLU or GELU if gelu; fused, one stack holds both, each expert's gate rows first.
    """
    experts, rows, _ = ws[0].shape
    ws = (ws[0], ws[0]) if fused else ws
    rows //= 2 if fused else 1
    tile = _tile(ws[0].type, [rows] * len(ws), len(ws) == 2)
    out = Tensor.empty(
        int(order.shape[0]) // experts * used, rows, dtype=dtypes.float32, device=q8[0].device
    )
    xq, bound = carry(q8[0], tokens)
    fxn = functools.partial(
        _matmul_kernel, tokens=bound, ggml_type=ws[0].type, rows=tile,
        heights=(rows,) * len(ws), gated=len(ws) == 2, routed=(used, by_token),
        fused=fused, gelu=gelu,
    )  # fmt: skip
    words = map(storage_words, ws)
    return Tensor.custom_kernel(out, *words, xq, *q8[1:], order, counts, fxn=fxn)[0]


def matmul_fits(ggml_type: GGMLType, rows: int, cols: int) -> bool:
    # whole tiles of rows, and whole blocks along a row
    return ggml_type in _TILES and rows % 64 == 0 and cols % _TILES[ggml_type].block == 0


def supports_matmul(x: Tensor, w: QTensor) -> bool:
    # tokens of one sequence, and a matrix the tiles fit
    single = all(isinstance(b, int) and b == 1 for b in x.shape[:-2])
    return on_nvidia(x) and single and matmul_fits(w.type, *w.shape)


def matmuls(
    x: Tensor,
    *ws: QTensor,
    norm: tuple[Tensor, float] | None = None,
    residual: Tensor | None = None,
) -> list[Tensor]:
    """x @ w.T for tokens x (1, T, n) and each w, with the activations quantized to int8 once,
    after rms_norm(x, *norm) if given. `residual` is added inside the kernel; it needs a single w.
    """
    assert residual is None or len(ws) == 1, "a residual goes with one matrix"
    tokens = x.shape[-2]
    q8 = quantize_q8(tiled(x), norm, rows=tokens)
    outs = []
    for out, heights in _products(q8, tokens, ws, residual):
        for start, h in zip(itertools.accumulate([0, *heights]), heights, strict=False):
            outs.append(out[:tokens, start : start + h].reshape(*x.shape[:-1], h))
    return outs


def feed_forward(
    x: Tensor, gate: QTensor, up: QTensor, down: QTensor, norm: tuple[Tensor, float],
    gelu: bool = False, residual: bool = True,
) -> Tensor:  # fmt: skip
    """x + act(n @ gate.T) * (n @ up.T) @ down.T for tokens x and n = rms_norm(x, *norm), act
    SiLU or GELU if gelu, without x if not residual; gate and up, which share a type and shape,
    in one kernel that also applies act."""
    tokens = x.shape[-2]
    q8 = quantize_q8(tiled(x), norm, rows=tokens)
    ((hidden, _),) = _products(q8, tokens, (gate, up), None, gated=True, gelu=gelu)
    q8 = quantize_q8(hidden, rows=tokens)
    ((out, _),) = _products(q8, tokens, (down,), x if residual else None)
    return out[:tokens].reshape(*x.shape[:-1], down.shape[0])
