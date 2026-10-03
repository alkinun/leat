"""What the NVIDIA kernels share: warp intrinsics, loads, conversions and bound variables."""

import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops

from leat.quant import GGMLType, QTensor

WARP = 32
GROUP = 32  # activations per int8 scale
LOG2E = math.log2(math.e)
SHARED = 49152  # bytes of shared memory a block may use without opting in to more

# the word type kernels read each storage type as: some blocks are only halfword aligned
WORD_TYPE = {
    GGMLType.Q4_K: dtypes.uint32, GGMLType.Q5_K: dtypes.uint32, GGMLType.Q6_K: dtypes.uint16,
    GGMLType.Q8_0: dtypes.uint16,
}  # fmt: skip


def on_nvidia(t: Tensor) -> bool:
    return isinstance(t.device, str) and t.device.split(":")[0] in ("NV", "CUDA")


def lane_range() -> UOp:
    # tinygrad never splits a WARP axis and maps it to threadIdx.x, so lane i is hardware lane i
    return UOp.range(WARP, -1, AxisType.WARP)


def dp4a(a: UOp, b: UOp, acc: UOp) -> UOp:
    # acc + dot product of the four signed bytes of a and b
    return UOp(Ops.CUSTOMI, src=(a, b, acc), arg=("__dp4a({}, {}, {})", dtypes.int32))


def shfl_xor(value: UOp, mask: int) -> UOp:
    # a statement, not an inline expression: every lane of the warp must execute it
    fmt = f"__shfl_xor_sync(0xffffffffu, {{0}}, {mask})"
    return UOp(Ops.CUSTOM, src=(value,), arg=(fmt, value.dtype))


def warp_sum(value: UOp, lanes: int = WARP) -> UOp:
    # over each aligned run of `lanes` lanes
    for mask in (16, 8, 4, 2, 1)[5 - lanes.bit_length() + 1 :]:
        value = value + shfl_xor(value, mask)
    return value


def warp_max(value: UOp, lanes: int = WARP) -> UOp:
    for mask in (16, 8, 4, 2, 1)[5 - lanes.bit_length() + 1 :]:
        value = value.maximum(shfl_xor(value, mask))
    return value


def load_vector(ptr: UOp, lanes: int) -> tuple[UOp, ...]:
    # `lanes` consecutive values from an index, as one vector load, widened to f32
    buf, coords = ptr.src[0], ptr.src[1:]
    start = sum((c * math.prod(buf.shape[i + 1 :]) for i, c in enumerate(coords)), UOp.const(0))
    vec = UOp(Ops.SHRINK, src=(buf.flatten(), start, UOp.const(lanes))).load()
    return tuple(vec[i].float() for i in range(lanes))


def word16(w: UOp, i: UOp) -> UOp:
    # four bytes from a halfword-aligned buffer: 32-bit loads must be 4-byte aligned
    return w[i].load().cast(dtypes.uint32) | (w[i + 1].load().cast(dtypes.uint32) << 16)


def f16(bits: UOp) -> UOp:
    # the low 16 bits of a word as an f16, widened to f32
    return (bits & 0xFFFF).cast(dtypes.uint16).bitcast(dtypes.float16).float()


def minus_32(q: UOp) -> UOp:
    # Q6_K's unsigned 0..63 to signed -32..31, in each byte of a word: b + 96 stays in its byte,
    # and flipping its top bit makes it b - 32 in two's complement
    return ((q + 0x60606060) ^ 0x80808080).bitcast(dtypes.int32)


def silu(x: UOp) -> UOp:
    return x * (1 + (x * -LOG2E).exp2()).reciprocal()  # as tinygrad's silu


def register(shape: tuple[int, ...], value: float) -> UOp:
    reg = UOp.alloc(shape, dtypes.float32, addrspace=AddrSpace.REG)
    return reg.after(reg.store(reg.const_like(value)))


def at_most(a: int | UOp, b: int) -> int | UOp:
    return a.minimum(b) if isinstance(a, UOp) else min(a, b)


def carry(t: Tensor, value: int | UOp) -> tuple[Tensor, int | UOp]:
    # a bound variable rides on one of the kernel's buffers, so the kernel's own copy stays unbound
    if isinstance(value, UOp):
        return Tensor(t.uop.after(value)), value.unbind_all()[0]
    return t, value


def storage_words(w: QTensor) -> Tensor:
    # .contiguous() on a bitcast of contiguous storage is a view; without it tinygrad copies
    return w.data.flatten().bitcast(WORD_TYPE[w.type]).contiguous()
