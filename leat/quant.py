"""GGML storage formats and their reference dequantization in plain tinygrad ops.

Block layouts and arithmetic order follow ggml's `dequantize_row_*` (ggml-quants.c), so the
reference path matches llama.cpp; tests check them against gguf-py's.
"""

from dataclasses import dataclass
from enum import IntEnum

from tinygrad import Tensor, dtypes
from tinygrad.dtype import DType


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    MXFP4 = 39


# (elements, bytes) per block for every type a GGUF index may reference
BLOCK = {
    GGMLType.F32: (1, 4), GGMLType.F16: (1, 2), GGMLType.BF16: (1, 2), GGMLType.F64: (1, 8),
    GGMLType.I8: (1, 1), GGMLType.I16: (1, 2), GGMLType.I32: (1, 4), GGMLType.I64: (1, 8),
    GGMLType.Q4_0: (32, 18), GGMLType.Q4_1: (32, 20), GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24), GGMLType.Q8_0: (32, 34), GGMLType.IQ4_NL: (32, 18),
    GGMLType.MXFP4: (32, 17), GGMLType.Q2_K: (256, 84), GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144), GGMLType.Q5_K: (256, 176), GGMLType.Q6_K: (256, 210),
    GGMLType.IQ2_XXS: (256, 66), GGMLType.IQ2_XS: (256, 74), GGMLType.IQ3_XXS: (256, 98),
    GGMLType.IQ1_S: (256, 50), GGMLType.IQ3_S: (256, 110), GGMLType.IQ2_S: (256, 82),
    GGMLType.IQ4_XS: (256, 136), GGMLType.IQ1_M: (256, 56),
}  # fmt: skip

NATIVE: dict[GGMLType, DType] = {
    GGMLType.F32: dtypes.float32, GGMLType.F16: dtypes.float16, GGMLType.BF16: dtypes.bfloat16,
}  # fmt: skip


def _f16(b: Tensor) -> Tensor:
    return b.bitcast(dtypes.float16).cast(dtypes.float32)


def _fields(b: Tensor, width: int) -> Tensor:
    # each byte of b split into its fields of `width` bits, least significant first, along a new
    # last axis
    shifts = Tensor(list(range(0, 8, width)), dtype=dtypes.uint8, device=b.device)
    return (b.unsqueeze(-1) >> shifts) & ((1 << width) - 1)


def _nibbles(b: Tensor) -> Tensor:
    # bytes (n, 16) to their 32 nibbles: the low ones are values 0..15, the high ones 16..31
    return _fields(b, 4).transpose(1, 2).reshape(b.shape[0], -1)


def _lookup(table: tuple[int, ...], index: Tensor) -> Tensor:
    # table[index] for 4-bit indices, as f32
    out = Tensor.zeros(*index.shape, device=index.device)
    for i, v in enumerate(table):
        if v:
            out = (index == i).where(float(v), out)
    return out


def _q4_0(b: Tensor) -> Tensor:
    # d:f16, qs:u8[16]; minus 8
    return _f16(b[:, :2]) * (_nibbles(b[:, 2:]).cast(dtypes.int8) - 8).cast(dtypes.float32)


def _q4_1(b: Tensor) -> Tensor:
    # d:f16, m:f16, qs:u8[16]
    return _f16(b[:, :2]) * _nibbles(b[:, 4:]).cast(dtypes.float32) + _f16(b[:, 2:4])


def _q5_0(b: Tensor) -> Tensor:
    # d:f16, qh:u8[4], qs:u8[16]; value i < 16 is the low nibble of qs[i] and i + 16 its high one,
    # with bit i of qh as bit 4; minus 16
    q = (_nibbles(b[:, 6:]) | (_fields(b[:, 2:6], 1).flatten(1) << 4)).cast(dtypes.float32)
    return _f16(b[:, :2]) * (q - 16)


def _q5_1(b: Tensor) -> Tensor:
    # d:f16, m:f16, qh:u8[4], qs:u8[16]; Q5_0's values, plus m
    q = (_nibbles(b[:, 8:]) | (_fields(b[:, 4:8], 1).flatten(1) << 4)).cast(dtypes.float32)
    return _f16(b[:, :2]) * q + _f16(b[:, 2:4])


def _q8_0(b: Tensor) -> Tensor:
    # d:f16, qs:i8[32]
    return _f16(b[:, :2]) * b[:, 2:].bitcast(dtypes.int8).cast(dtypes.float32)


# IQ4_NL's and IQ4_XS's values, and MXFP4's E2M1 ones, doubled
IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
FP4_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _iq4_nl(b: Tensor) -> Tensor:
    # d:f16, qs:u8[16] of indices into IQ4_VALUES
    return _f16(b[:, :2]) * _lookup(IQ4_VALUES, _nibbles(b[:, 2:]))


def _mxfp4(b: Tensor) -> Tensor:
    # e:u8, qs:u8[16] of indices into FP4_VALUES; the scale is 2^(e - 128), as ggml's
    # e8m0_to_fp32_half writes it, with denormals for e < 2
    e = b[:, :1].cast(dtypes.uint32)
    denormal = Tensor(0x00200000, dtype=dtypes.uint32, device=b.device) << e
    scale = (e < 2).where(denormal, (e - 1) << 23).bitcast(dtypes.float32)
    return scale * _lookup(FP4_VALUES, _nibbles(b[:, 1:]))


def _iq4_xs(b: Tensor) -> Tensor:
    # d:f16, scales_h:u16, scales_l:u8[4], qs:u8[128]; sub-block j of 32 takes 16 bytes of qs, as
    # IQ4_NL, with scale d * (s - 32) of 6 bits: nibble j of scales_l, and bits 2j of scales_h
    n = b.shape[0]
    low = _fields(b[:, 4:8], 4).reshape(n, 8)
    high = _fields(b[:, 2:4], 2).reshape(n, 8)
    scales = ((low | (high << 4)).cast(dtypes.int8) - 32).cast(dtypes.float32)
    q = _lookup(IQ4_VALUES, _nibbles(b[:, 8:].reshape(n * 8, 16))).reshape(n, 8, 32)
    return ((_f16(b[:, :2]) * scales).reshape(n, 8, 1) * q).reshape(n, 256)


def _crumbs(qs: Tensor) -> Tensor:
    # Q2_K's and Q3_K's 64 bytes of 2-bit values: each half of 128 values takes 32 bytes, its rows
    # k of 32 their bits 2k: (n, 16, 16), by groups of 16 values
    n = qs.shape[0]
    return _fields(qs.reshape(n, 2, 32), 2).permute(0, 1, 3, 2).reshape(n, 16, 16)


def _q2_k(b: Tensor) -> Tensor:
    # scales:u8[16], qs:u8[64], d:f16, dmin:f16; each 16 values a scale d * (s & 15) and a min
    # dmin * (s >> 4)
    n, sc = b.shape[0], b[:, :16]
    dl = (_f16(b[:, 80:82]) * (sc & 15).cast(dtypes.float32)).reshape(n, 16, 1)
    ml = (_f16(b[:, 82:84]) * (sc >> 4).cast(dtypes.float32)).reshape(n, 16, 1)
    return (dl * _crumbs(b[:, 16:80]).cast(dtypes.float32) - ml).reshape(n, 256)


def _q3_k(b: Tensor) -> Tensor:
    # hmask:u8[32], qs:u8[64], scales:u8[12], d:f16; Q2_K's 2 bits, minus 4 where hmask's bit for
    # the value (bit j of byte l for value l of row j of 32) is clear; 16 scales of 6 bits, minus
    # 32: nibbles of the first 8 bytes, the low first, and bits 2(j / 4) of byte 8 + j % 4
    n = b.shape[0]
    low = _fields(b[:, 96:104], 4).transpose(1, 2).reshape(n, 16)
    high = _fields(b[:, 104:108], 2).transpose(1, 2).reshape(n, 16)
    scales = ((low | (high << 4)).cast(dtypes.int8) - 32).cast(dtypes.float32)
    dl = (_f16(b[:, 108:110]) * scales).reshape(n, 16, 1)
    clear = (_fields(b[:, :32], 1).permute(0, 2, 1) ^ 1).reshape(n, 16, 16)
    q = _crumbs(b[:, 32:96]).cast(dtypes.int8) - (clear << 2).cast(dtypes.int8)
    return (dl * q.cast(dtypes.float32)).reshape(n, 256)


def _k_scales(s: Tensor) -> tuple[Tensor, Tensor]:
    # 8 six-bit (scale, min) pairs packed in 12 bytes, ggml's get_scale_min_k4
    lo, mid, hi = s[:, 0:4], s[:, 4:8], s[:, 8:12]
    sc = (lo & 63).cat((hi & 15) | ((lo >> 6) << 4), dim=1)
    m = (mid & 63).cat((hi >> 4) | ((mid >> 6) << 4), dim=1)
    return sc.cast(dtypes.float32), m.cast(dtypes.float32)


def _q4_k(b: Tensor) -> Tensor:
    # d:f16, dmin:f16, scales:u8[12], qs:u8[128]; sub-block 2j+h is nibble h of qs[32j:32j+32]
    n = b.shape[0]
    sc, m = _k_scales(b[:, 4:16])
    qs = b[:, 16:].reshape(n, 4, 1, 32)
    q = (qs & 15).cat(qs >> 4, dim=2).reshape(n, 8, 32).cast(dtypes.float32)
    d, dmin = (_f16(b[:, :2]) * sc).unsqueeze(-1), (_f16(b[:, 2:4]) * m).unsqueeze(-1)
    return (d * q - dmin).reshape(n, 256)


def _q5_k(b: Tensor) -> Tensor:
    # d:f16, dmin:f16, scales:u8[12], qh:u8[32], qs:u8[128]; bit s of qh[l] is bit 4 of sub-block s
    n = b.shape[0]
    sc, m = _k_scales(b[:, 4:16])
    qs = b[:, 48:].reshape(n, 4, 1, 32)
    low = (qs & 15).cat(qs >> 4, dim=2).reshape(n, 8, 32)
    shifts = Tensor([1 << s for s in range(8)], dtype=dtypes.uint8, device=b.device)
    high = (b[:, 16:48].reshape(n, 1, 32) // shifts.reshape(1, 8, 1)) & 1
    q = (low | (high << 4)).cast(dtypes.float32)
    d, dmin = (_f16(b[:, :2]) * sc).unsqueeze(-1), (_f16(b[:, 2:4]) * m).unsqueeze(-1)
    return (d * q - dmin).reshape(n, 256)


def _q6_k(b: Tensor) -> Tensor:
    # ql:u8[128], qh:u8[64], scales:i8[16], d:f16; two halves of 128, each 4 rows k of 32 values:
    # row k takes nibble k//2 of ql[32*(k%2):][:32] and bits 2k..2k+1 of qh, one scale per 16 values
    n = b.shape[0]
    ql = b[:, :128].reshape(n, 2, 1, 2, 32)
    low = (ql & 15).cat(ql >> 4, dim=2).reshape(n, 2, 4, 32)
    shifts = Tensor([1 << (2 * k) for k in range(4)], dtype=dtypes.uint8, device=b.device)
    high = (b[:, 128:192].reshape(n, 2, 1, 32) // shifts.reshape(1, 1, 4, 1)) & 3
    q = ((low | (high << 4)).cast(dtypes.int8) - 32).cast(dtypes.float32).reshape(n, 2, 4, 2, 16)
    sc = b[:, 192:208].bitcast(dtypes.int8).cast(dtypes.float32).reshape(n, 2, 4, 2, 1)
    return (_f16(b[:, 208:210]).reshape(n, 1, 1, 1, 1) * sc * q).reshape(n, 256)


DEQUANT = {
    GGMLType.Q4_0: _q4_0, GGMLType.Q4_1: _q4_1, GGMLType.Q5_0: _q5_0, GGMLType.Q5_1: _q5_1,
    GGMLType.Q8_0: _q8_0, GGMLType.Q2_K: _q2_k, GGMLType.Q3_K: _q3_k, GGMLType.Q4_K: _q4_k,
    GGMLType.Q5_K: _q5_k, GGMLType.Q6_K: _q6_k, GGMLType.IQ4_NL: _iq4_nl,
    GGMLType.IQ4_XS: _iq4_xs, GGMLType.MXFP4: _mxfp4,
}  # fmt: skip


@dataclass(frozen=True, eq=False)
class QTensor:
    """A GGUF tensor in its storage format.

    `data` holds `(blocks, block_bytes)` uint8 for quantized types, or the values themselves for
    native float types. `shape` is the logical row-major shape.
    """

    data: Tensor
    type: GGMLType
    shape: tuple[int, ...]

    def dequant(self, dtype: DType = dtypes.float32) -> Tensor:
        if self.type in NATIVE:
            return self.data.reshape(self.shape).cast(dtype)
        if self.type not in DEQUANT:
            raise NotImplementedError(f"dequantizing {self.type.name} is not supported")
        return DEQUANT[self.type](self.data).reshape(self.shape).cast(dtype)
