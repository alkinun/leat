"""What the kernels share: warp intrinsics, loads, conversions and bound variables.

The intrinsics render as C for CUDA and for HIP alike, each choosing its own when compiled: the
warp-level kernels run on NVIDIA GPUs and on AMD's RDNA GPUs, whose waves of 32 lanes are warps.
The kernels on tensor cores, matmul's and FlashAttention's, are NVIDIA's alone.
"""

import functools
import math
import re
from collections.abc import Callable
from typing import Any

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops

from leat.quant import GGMLType, QTensor

WARP = 32
GROUP = 32  # activations per int8 scale
LOG2E = math.log2(math.e)
OAI_ALPHA, OAI_LIMIT = 1.702, 7.0  # gpt-oss's SwiGLU
SHARED = 49152  # bytes of shared memory a block may use without opting in to more

# the word type kernels read each storage type as: some blocks are only halfword aligned
WORD_TYPE = {
    GGMLType.Q4_K: dtypes.uint32, GGMLType.Q5_K: dtypes.uint32, GGMLType.Q6_K: dtypes.uint16,
    GGMLType.Q5_0: dtypes.uint16, GGMLType.Q8_0: dtypes.uint16, GGMLType.MXFP4: dtypes.uint32,
    GGMLType.Q4_0: dtypes.uint16, GGMLType.Q4_1: dtypes.uint16, GGMLType.Q5_1: dtypes.uint16,
    GGMLType.IQ4_NL: dtypes.uint16, GGMLType.IQ4_XS: dtypes.uint32,
}  # fmt: skip


def on_nvidia(t: Tensor) -> bool:
    return isinstance(t.device, str) and t.device.split(":")[0] in ("NV", "CUDA")


def on_gpu(t: Tensor) -> bool:
    # an NVIDIA GPU, or an AMD one of waves of 32 that tinygrad renders C for: RDNA 3 and 4, as
    # Strix Halo's
    return isinstance(t.device, str) and (on_nvidia(t) or _rdna(t.device.split(":")[0]))


@functools.cache
def _rdna(device: str) -> bool:
    if not device.endswith("AMD"):
        return False
    from tinygrad.renderer.cstyle import HIPRenderer

    dev: Any = Device[device]
    return isinstance(dev.renderer, HIPRenderer) and getattr(dev, "target", (0,))[0] in (11, 12)


def either(cuda: str, amd: str) -> str:
    # an expression of each backend's intrinsics over operands {0}, {1}...: HIP's clang defines
    # __AMDGCN__ for AMD GPUs. The operands render once, as the arguments of a lambda that picks:
    # written into both branches, an operand that is itself such an expression would double at
    # each level of nesting, as dp4a's accumulators do. Directives need lines of their own.
    n = 1 + max(int(i) for i in re.findall(r"\{(\d+)\}", cuda + amd))
    names = [f"_{i}" for i in range(n)]
    params, args = ", ".join(f"auto {x}" for x in names), ", ".join(f"{{{i}}}" for i in range(n))
    amd, cuda = amd.format(*names), cuda.format(*names)
    body = f"\n#if defined(__AMDGCN__)\nreturn {amd};\n#else\nreturn {cuda};\n#endif\n"
    return f"[]({params}) {{{{{body}}}}}({args})"


@functools.cache
def compute_units(device: str) -> int:
    # the GPU's SMs: those of the TPCs each GPC has enabled; the 3090's 82 where the backend
    # does not say
    dev: Any = Device[device]
    if not hasattr(dev, "num_gpcs"):
        return 82
    from tinygrad.runtime.ops_nv import nv_gpu

    masks = (
        dev.iface.rm_control(
            dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_TPC_MASK,
            nv_gpu.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS(gpcId=i),
        ).tpcMask
        for i in range(dev.num_gpcs)
    )  # fmt: skip
    return sum(bin(m).count("1") for m in masks) * dev.num_sm_per_tpc


def lane_range() -> UOp:
    # tinygrad never splits a WARP axis and maps it to threadIdx.x, so lane i is hardware lane i
    return UOp.range(WARP, -1, AxisType.WARP)


def dp4a(a: UOp, b: UOp, acc: UOp) -> UOp:
    # acc + dot product of the four signed bytes of a and b
    code = either(
        "__dp4a({0}, {1}, {2})", "__builtin_amdgcn_sudot4(true, {0}, true, {1}, {2}, false)"
    )
    return UOp(Ops.CUSTOMI, src=(a, b, acc), arg=(code, dtypes.int32))


_C_TYPES = {dtypes.float32: "float", dtypes.int32: "int", dtypes.uint32: "unsigned int"}


def shfl_xor(value: UOp, mask: int) -> UOp:
    # lane i's value of lane i ^ mask; a statement, not an inline expression: every lane of the
    # warp must execute it. AMD's ds_swizzle takes the lane as ((i & 31) | 0) ^ mask.
    t, pattern = _C_TYPES[value.dtype], 0x1F | mask << 10
    swizzled = f"__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {pattern})"
    code = either(
        f"__shfl_xor_sync(0xffffffffu, {{0}}, {mask})", f"__builtin_bit_cast({t}, {swizzled})"
    )
    return UOp(Ops.CUSTOM, src=(value,), arg=(code, value.dtype))


def ballot(predicate: UOp) -> UOp:
    # the lanes for which predicate holds, a bit each; a statement, as shfl_xor
    code = either("__ballot_sync(0xffffffffu, {0})", "__builtin_amdgcn_ballot_w32({0})")
    return UOp(Ops.CUSTOM, src=(predicate,), arg=(code, dtypes.uint32))


def popcount(x: UOp) -> UOp:
    code = either("__popc({0})", "__builtin_popcount({0})")
    return UOp(Ops.CUSTOMI, src=(x,), arg=(code, dtypes.int32))


def rounded(x: UOp, y: UOp) -> UOp:
    # x / y rounded half away from zero, the division exact: tinygrad would multiply by a
    # reciprocal
    code = either("roundf({0}/{1})", "__builtin_roundf({0}/{1})")
    return UOp(Ops.CUSTOMI, src=(x, y), arg=(code, dtypes.float32))


def warp_sum(value: UOp, lanes: int = WARP, op: Callable[[UOp, UOp], UOp] = UOp.__add__) -> UOp:
    # over each aligned run of `lanes` lanes
    for mask in (16, 8, 4, 2, 1)[5 - lanes.bit_length() + 1 :]:
        value = op(value, shfl_xor(value, mask))
    return value


def warp_max(value: UOp, lanes: int = WARP) -> UOp:
    return warp_sum(value, lanes, UOp.maximum)


def load_vector(ptr: UOp, lanes: int) -> tuple[UOp, ...]:
    # `lanes` consecutive values from an index, as one vector load, widened to f32
    buf, coords = ptr.src[0], ptr.src[1:]
    start = sum((c * math.prod(buf.shape[i + 1 :]) for i, c in enumerate(coords)), UOp.const(0))
    return tuple(v.float() for v in load_words(buf, start, lanes))


def load_words(buf: UOp, start: UOp | int, lanes: int) -> tuple[UOp, ...]:
    # `lanes` consecutive values of a buffer, flat, from `start`, as one vector load: start must
    # be a multiple of the vector's size
    start = start if isinstance(start, UOp) else UOp.const(start)
    vec = UOp(Ops.SHRINK, src=(buf.flatten(), start, UOp.const(lanes))).load()
    return tuple(vec[i] for i in range(lanes))


def word16(w: UOp, i: UOp) -> UOp:
    # four bytes from a halfword-aligned buffer: 32-bit loads must be 4-byte aligned
    return w[i].load().cast(dtypes.uint32) | (w[i + 1].load().cast(dtypes.uint32) << 16)


def byte_perm(a: UOp | int, b: UOp | int, selector: UOp | int) -> UOp:
    # the bytes of (b, a) that the low 4 nibbles of the selector pick, as CUDA's __byte_perm; AMD's
    # v_perm takes a byte per pick, from (its first, its second)
    srcs = tuple(x if isinstance(x, UOp) else UOp.const(x, dtypes.uint32) for x in (a, b, selector))
    picks = "|".join(f"((({{2}}) >> {4 * i}) & 7) << {8 * i}" for i in range(4))
    code = either("__byte_perm({0}, {1}, {2})", f"__builtin_amdgcn_perm({{1}}, {{0}}, {picks})")
    return UOp(Ops.CUSTOMI, src=srcs, arg=(code, dtypes.uint32))


def funnel(lo: UOp, hi: UOp, shift: UOp) -> UOp:
    # the 32 bits from bit `shift` on of (hi, lo), for shift < 32, as CUDA's __funnelshift_r
    code = either("__funnelshift_r({0}, {1}, {2})", "__builtin_amdgcn_alignbit({1}, {0}, {2})")
    return UOp(Ops.CUSTOMI, src=(lo, hi, shift), arg=(code, dtypes.uint32))


def table16(q: UOp, table: tuple[int, int, int, int]) -> tuple[UOp, UOp]:
    # a word of 8 nibbles to the int8 values a 16-entry table (as 4 words) gives them: those of
    # its low nibbles, then of its high ones, each a word of 4 bytes; as llama.cpp's
    # get_int_from_table_16, picking from each half of the table and then by the nibble's top bit
    halves, pick = [], (q & 0x88888888) >> 1 | 0x32103210
    for shift in (0, 16):
        low, high = (byte_perm(table[i], table[i + 1], q >> shift) for i in (0, 2))
        halves.append(byte_perm(low, high, pick >> shift))
    return byte_perm(halves[0], halves[1], 0x6420), byte_perm(halves[0], halves[1], 0x7531)


def table_words(values: tuple[int, ...]) -> tuple[int, int, int, int]:
    # 16 int8 values as the 4 words table16 takes
    raw = bytes(v & 0xFF for v in values)
    return tuple(int.from_bytes(raw[i : i + 4], "little") for i in range(0, 16, 4))  # type: ignore[return-value]


def e8m0_half(e: UOp) -> UOp:
    # 2^(e - 128) for an exponent byte, as f32: ggml's e8m0_to_fp32_half, denormal for e < 2
    e = e.cast(dtypes.uint32)
    bits = (e < 2).where(UOp.const(0x00200000, dtypes.uint32) << e, (e - 1) << 23)
    return bits.bitcast(dtypes.float32)


def f16(bits: UOp) -> UOp:
    # the low 16 bits of a word as an f16, widened to f32
    return (bits & 0xFFFF).cast(dtypes.uint16).bitcast(dtypes.float16).float()


def minus(q: UOp, offset: int) -> UOp:
    # each byte of a word less `offset`, as signed bytes, for unsigned bytes below 2 * offset, as
    # Q6_K's 0..63 to -32..31: b + 128 - offset stays in its byte, and flipping its top bit makes
    # it b - offset in two's complement
    return ((q + (128 - offset) * 0x01010101) ^ 0x80808080).bitcast(dtypes.int32)


def fifth_bits(bits: UOp) -> UOp:
    # 4 bits to bit 4 of each byte of a word
    return ((bits & 1) | ((bits & 2) << 7) | ((bits & 4) << 14) | ((bits & 8) << 21)) << 4


def silu(x: UOp) -> UOp:
    return x * (1 + (x * -LOG2E).exp2()).reciprocal()  # as tinygrad's silu


def glu(kind: str) -> Callable[[UOp, UOp], UOp]:
    # how an MLP's gate and up combine: act(gate) * up for act SiLU or GELU, or gpt-oss's
    # clamped SwiGLU, "oai"
    if kind == "oai":
        return _oai
    act = _gelu if kind == "gelu" else silu
    return lambda g, u: act(g) * u


def _oai(g: UOp, u: UOp) -> UOp:
    # min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clamp(u, -7, 7) + 1), as ggml's swiglu_oai
    g, u = g.minimum(OAI_LIMIT), u.maximum(-OAI_LIMIT).minimum(OAI_LIMIT)
    return g * (1 + (g * (-OAI_ALPHA * LOG2E)).exp2()).reciprocal() * (u + 1)


def _gelu(x: UOp) -> UOp:
    # tanh's approximation, as ggml's and tinygrad's: x * sigmoid(2 sqrt(2/pi) (x + 0.044715 x^3))
    z = x * (1 + 0.044715 * x * x) * (2 * math.sqrt(2 / math.pi))
    return x * (1 + (z * -LOG2E).exp2()).reciprocal()


def register(shape: tuple[int, ...], value: float) -> UOp:
    reg = UOp.alloc(shape, dtypes.float32, addrspace=AddrSpace.REG)
    return reg.after(reg.store(reg.const_like(value)))


def opaque(x: UOp) -> UOp:
    # x, as an expression tinygrad's codegen cannot match with another: it declares an index at
    # its first use inside a loop and reuses it after the loop, out of scope, where it recurs
    return UOp(Ops.CUSTOMI, src=(x.cast(dtypes.int32),), arg=("{0}", dtypes.int32))


def at_most(a: int | UOp, b: int) -> int | UOp:
    return a.minimum(b) if isinstance(a, UOp) else min(a, b)


def at_least(a: int | UOp, b: int) -> int | UOp:
    return a.maximum(b) if isinstance(a, UOp) else max(a, b)


def carry(t: Tensor, value: int | UOp) -> tuple[Tensor, int | UOp]:
    # a bound variable rides on one of the kernel's buffers, so the kernel's own copy stays unbound
    if isinstance(value, UOp):
        return Tensor(t.uop.after(value)), value.unbind_all()[0]
    return t, value


def storage_words(w: QTensor) -> Tensor:
    # .contiguous() on a bitcast of contiguous storage is a view; without it tinygrad copies
    return w.data.flatten().bitcast(WORD_TYPE[w.type]).contiguous()
