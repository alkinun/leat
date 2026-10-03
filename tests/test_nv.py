import os

import gguf
import numpy as np
import pytest
from gguf.quants import dequantize
from tinygrad import Tensor, UOp

from leat import nv, ops
from leat.quant import BLOCK, GGMLType, QTensor
from tests.helpers import random_blocks

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("DEV", "").split(":")[0] not in ("NV", "CUDA"), reason="needs DEV=NV or CUDA"
    ),
]


def quantize_q8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # d = max|x| / 127 and q = roundf(x / d) per group of 32, all in f32 like the kernel
    groups = x.reshape(-1, nv.GROUP)
    d = (np.abs(groups).max(-1, keepdims=True) / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (groups / d).astype(np.float64)  # rounding half away from zero is exact in f64
    q = np.where(d > 0, np.sign(r) * np.floor(np.abs(r) + 0.5), 0).astype(np.int8)
    s = (d[:, 0] * q.sum(-1, dtype=np.int32).astype(np.float32)).astype(np.float32)
    return q.ravel(), d[:, 0], s


def test_quantize_q8():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(4096) * rng.uniform(0.01, 10, 4096)).astype(np.float32)
    x[64:96] = 0  # an all-zero group must give d = 0, not nan
    q, d, s = nv.quantize_q8(Tensor(x))
    for got, want in zip(
        (q.numpy().view(np.int8), d.numpy(), s.numpy()), quantize_q8(x), strict=True
    ):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("ggml_type", [GGMLType.Q4_K, GGMLType.Q6_K])
@pytest.mark.parametrize("shape", [(64, 4096), (8, 14336)])
def test_linear(ggml_type, shape):
    rng = np.random.default_rng(1)
    rows, cols = shape
    blocks = random_blocks(ggml_type, rows * cols // BLOCK[ggml_type][0], rng, scale=1e-3)
    weights = dequantize(blocks, gguf.GGMLQuantizationType(ggml_type)).reshape(shape)
    x = rng.standard_normal((1, 1, cols)).astype(np.float32)
    w = QTensor(Tensor(blocks), ggml_type, shape)
    assert nv.supports(Tensor(x), w)
    q, d, _ = quantize_q8(x.ravel())
    expected = weights.astype(np.float64) @ (q.reshape(-1, nv.GROUP) * d[:, None]).ravel()
    got = ops.linear(Tensor(x), w).numpy()
    assert got.shape == (1, 1, rows)
    np.testing.assert_allclose(got.ravel(), expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max())


def test_shared_input():
    rng = np.random.default_rng(3)
    ws = []
    for t in (GGMLType.Q4_K, GGMLType.Q6_K):
        ws.append(QTensor(Tensor(random_blocks(t, 8 * 2048 // 256, rng, 1e-3)), t, (8, 2048)))
    x = Tensor(rng.standard_normal((1, 1, 2048)).astype(np.float32))
    for got, w in zip(ops.linears(x, *ws), ws, strict=True):
        np.testing.assert_array_equal(got.numpy(), ops.linear(x, w).numpy())


def test_reference_switch(monkeypatch):
    rng = np.random.default_rng(2)
    blocks = random_blocks(GGMLType.Q4_K, 16 * 2048 // 256, rng, scale=1e-3)
    w = QTensor(Tensor(blocks), GGMLType.Q4_K, (16, 2048))
    x = Tensor(rng.standard_normal((1, 1, 2048)).astype(np.float32))
    monkeypatch.setenv("LEAT_KERNELS", "ref")
    np.testing.assert_array_equal(ops.linear(x, w).numpy(), (x @ w.dequant().T).numpy())


# cache sizes matter too: some strides trip a tinygrad codegen bug with symbolic lengths (nv.PAD)
@pytest.mark.parametrize(
    "n, length",
    [
        (64, 1),
        (64, 64),
        (1024, 63),
        (1024, 65),
        (1024, 1000),
        (3072, 3072),
        (4096, 3079),
        (4096, 4096),
    ],  # fmt: skip
)
@pytest.mark.parametrize("symbolic", [False, True])
def test_attention(n, length, symbolic):
    rng = np.random.default_rng(length)
    heads, kv_heads, dim = 32, 8, 128
    cache = rng.standard_normal((2, 1, kv_heads, n, dim)).astype(np.float16)
    q = rng.standard_normal((1, heads, 1, dim)).astype(np.float32)
    valid = UOp.variable("start_pos", 0, n - 1).bind(length - 1) + 1 if symbolic else length
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()  # the model's cache is a buffer
    assert nv.supports_attention(q_t, cache_t)
    got = nv.attention(q_t, cache_t, valid).numpy()
    k, v = (cache[i, 0, :, :length].astype(np.float64) for i in range(2))
    for h in range(heads):
        scores = k[h // (heads // kv_heads)] @ q[0, h, 0] / np.sqrt(dim)
        p = np.exp(scores - scores.max())
        expected = (p / p.sum()) @ v[h // (heads // kv_heads)]
        np.testing.assert_allclose(got[0, h, 0], expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows, n", [(1, 128256), (3, 1000), (2, 33), (1, 1)])
def test_argmax(rows, n):
    rng = np.random.default_rng(n)
    x = rng.integers(-50, 50, (rows, n)).astype(np.float32)  # many ties: the first one must win
    x[:, ::7] = -np.inf
    assert nv.supports_argmax(Tensor(x))
    np.testing.assert_array_equal(ops.argmax(Tensor(x)).numpy(), x.argmax(-1, keepdims=True))
