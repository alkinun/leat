import os

import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("DEV", "").split(":")[0] not in ("NV", "CUDA"), reason="needs DEV=NV or CUDA"
    ),
]

WARP = 32


def dp4a(a: UOp, b: UOp, acc: UOp) -> UOp:
    return UOp(Ops.CUSTOMI, src=(a, b, acc), arg=("__dp4a({}, {}, {})", dtypes.int32))


def shfl_xor(value: UOp, mask: int) -> UOp:
    # CUSTOM, not CUSTOMI: the shuffle must be its own statement that every lane executes.
    fmt = f"__shfl_xor_sync(0xffffffffu, {{0}}, {mask})"
    return UOp(Ops.CUSTOM, src=(value,), arg=(fmt, value.dtype))


def int8_dot_kernel(out: UOp, a: UOp, b: UOp) -> UOp:
    # one warp per row: each lane dp4a-accumulates a strided slice, then a butterfly reduce
    rows, words = a.shape
    row = UOp.range(rows, 0, AxisType.GLOBAL)
    lane = UOp.range(WARP, 1, AxisType.LOCAL)
    acc = UOp.const(0, dtypes.int32)
    for i in range(words // WARP):
        acc = dp4a(a[row, i * WARP + lane].load(), b[row, i * WARP + lane].load(), acc)
    for mask in (16, 8, 4, 2, 1):
        acc = acc + shfl_xor(acc, mask)
    store = out[row.valid(lane.eq(0))].store(acc)
    return store.end(row, lane).sink(arg=KernelInfo(name="int8_dot", opts_to_apply=()))


def test_dp4a_warp_reduce():
    rng = np.random.default_rng(0)
    rows, n = 64, 4 * 4 * WARP
    a = rng.integers(-128, 128, (rows, n), dtype=np.int8)
    b = rng.integers(-128, 128, (rows, n), dtype=np.int8)
    out = Tensor.empty(rows, dtype=dtypes.int32)
    packed_a, packed_b = Tensor(a.view(np.int32)), Tensor(b.view(np.int32))
    out = Tensor.custom_kernel(out, packed_a, packed_b, fxn=int8_dot_kernel)[0]
    expected = (a.astype(np.int64) * b.astype(np.int64)).sum(axis=1)
    np.testing.assert_array_equal(out.numpy(), expected)
