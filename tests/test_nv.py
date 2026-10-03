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


@pytest.mark.parametrize("rows", [None, 5, UOp.variable("rows", 1, 8).bind(3)])
def test_quantize_q8(rows):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((8, 4096)) * rng.uniform(0.01, 10, (8, 4096))).astype(np.float32)
    x[:, 64:96] = 0  # an all-zero group must give d = 0, not nan
    q, d, s = nv.quantize_q8(Tensor(x), rows=rows)
    Tensor.realize(q, d, s)  # in one schedule, as in the model: each alone would lose rows
    n = 8 if rows is None else rows if isinstance(rows, int) else rows.unbind()[1]
    outs = (q.numpy().view(np.int8)[: n * 4096], d.numpy()[: n * 128], s.numpy()[: n * 128])
    for got, want in zip(outs, quantize_q8(x[:n]), strict=True):
        np.testing.assert_array_equal(got, want)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x64 = x.astype(np.float64)
    return (x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + eps) * weight).astype(np.float32)


def test_norm_quantize_q8():
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((3, 4096)) * rng.uniform(1, 5, (3, 1))).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, 4096).astype(np.float32)
    q, d, s = (t.numpy() for t in nv.quantize_q8(Tensor(x), (Tensor(weight), 1e-5)))
    want_q, want_d, _ = quantize_q8(rms_norm(x, weight, 1e-5))
    # normalizing in f32 rather than f64 may move a value across a rounding boundary
    off = q.view(np.int8).astype(np.int32) - want_q
    assert np.abs(off).max() <= 1 and np.count_nonzero(off) <= 4 * len(x)
    np.testing.assert_allclose(d, want_d, rtol=1e-5)
    sums = q.view(np.int8).reshape(-1, nv.GROUP).sum(-1, dtype=np.int32).astype(np.float32)
    np.testing.assert_array_equal(s, (d * sums).astype(np.float32))


def test_linear_after_norm():
    rng = np.random.default_rng(5)
    blocks = random_blocks(GGMLType.Q4_K, 64 * 4096 // 256, rng, scale=1e-3)
    w = QTensor(Tensor(blocks), GGMLType.Q4_K, (64, 4096))
    x = (rng.standard_normal((1, 1, 4096)) * 3).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, 4096).astype(np.float32)
    got = ops.linears(Tensor(x), w, norm=(Tensor(weight), 1e-5))[0].numpy().ravel()
    q, d, _ = quantize_q8(rms_norm(x.ravel(), weight, 1e-5))
    weights = dequantize(blocks, gguf.GGMLQuantizationType.Q4_K).reshape(64, 4096)
    expected = weights.astype(np.float64) @ (q.reshape(-1, nv.GROUP) * d[:, None]).ravel()
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3 * np.abs(expected).max())


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


@pytest.mark.parametrize("types", [(GGMLType.Q4_K,) * 2, (GGMLType.Q6_K,) * 2,
                                   (GGMLType.Q4_K, GGMLType.Q6_K)])  # fmt: skip
def test_swiglu(types):
    rng = np.random.default_rng(7)
    rows, cols = 16, 4096
    blocks = [random_blocks(t, rows * cols // 256, rng, 1e-3) for t in types]
    gate, up = (QTensor(Tensor(b), t, (rows, cols)) for b, t in zip(blocks, types, strict=True))
    x = rng.standard_normal((1, 1, cols)).astype(np.float32)
    q, d, _ = quantize_q8(x.ravel())
    xq = (q.reshape(-1, nv.GROUP) * d[:, None]).ravel()
    g, u = (dequantize(b, gguf.GGMLQuantizationType(t)).reshape(rows, cols).astype(np.float64) @ xq
            for b, t in zip(blocks, types, strict=True))  # fmt: skip
    expected = g / (1 + np.exp(-g)) * u
    got = ops.swiglu(Tensor(x), gate, up).numpy().ravel()
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max())


@pytest.mark.parametrize("ggml_type", [GGMLType.Q4_K, GGMLType.Q6_K])
def test_residual(ggml_type):
    rng = np.random.default_rng(6)
    w = QTensor(Tensor(random_blocks(ggml_type, 8 * 2048 // 256, rng, 1e-3)), ggml_type, (8, 2048))
    x = Tensor(rng.standard_normal((1, 1, 2048)).astype(np.float32))
    r = Tensor(rng.standard_normal((1, 1, 8)).astype(np.float32))
    got = ops.linear(x, w, residual=r).numpy()
    np.testing.assert_array_equal(got, (ops.linear(x, w) + r).numpy())


def reference_matmul(x: np.ndarray, blocks: np.ndarray, ggml_type: GGMLType) -> np.ndarray:
    # x (T, cols) quantized per row as the kernels do, times the decoded weights, in f64
    q, d, _ = quantize_q8(x)
    xq = (q.reshape(-1, nv.GROUP) * d[:, None]).reshape(x.shape)
    weights = dequantize(blocks, gguf.GGMLQuantizationType(ggml_type)).reshape(-1, x.shape[1])
    return xq.astype(np.float64) @ weights.astype(np.float64).T


@pytest.mark.parametrize("ggml_type", [GGMLType.Q4_K, GGMLType.Q6_K])
@pytest.mark.parametrize("tokens", [64, 100, UOp.variable("tokens", 1, 128).bind(70)])
@pytest.mark.parametrize("shape", [(256, 2048), (4096, 512)])  # tiles of 128 and 256 rows
def test_matmul(ggml_type, tokens, shape):
    rng = np.random.default_rng(8)
    rows, cols = shape
    blocks = random_blocks(ggml_type, rows * cols // 256, rng, 1e-3)
    w = QTensor(Tensor(blocks), ggml_type, (rows, cols))
    if isinstance(tokens, int):
        x = rng.standard_normal((1, tokens, cols)).astype(np.float32)
        x_t, n = Tensor(x), tokens
    else:  # while prefilling, a bound number of tokens out of the most there may be
        x = rng.standard_normal((1, 128, cols)).astype(np.float32)
        x_t, n = Tensor(x)[:, :tokens], tokens.unbind()[1]
    assert nv.supports(x_t, w)
    got = ops.linear(x_t, w).pad_to((1, x.shape[1], rows)).numpy()[0, :n]
    expected = reference_matmul(x[0, :n], blocks, ggml_type)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max())


@pytest.mark.parametrize("heights", [(256, 128, 128), (4096, 1024, 1024)])  # tiles of 128, 256
def test_matmul_stacked(heights):
    # consecutive matrices of one type share a kernel: here the first two, as q and k
    rng = np.random.default_rng(10)
    cols, n = 512, 70
    types = (GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q6_K)
    blocks = [
        random_blocks(t, h * cols // 256, rng, 1e-3) for t, h in zip(types, heights, strict=True)
    ]
    ws = [QTensor(Tensor(b), t, (h, cols)) for b, t, h in zip(blocks, types, heights, strict=True)]
    x = rng.standard_normal((1, n, cols)).astype(np.float32)
    for got, b, t in zip(ops.linears(Tensor(x), *ws), blocks, types, strict=True):
        expected = reference_matmul(x[0], b, t)
        np.testing.assert_allclose(
            got.numpy()[0], expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max()
        )


def test_matmul_norm_residual():
    rng = np.random.default_rng(9)
    rows, cols, n = 256, 4096, 80
    blocks = random_blocks(GGMLType.Q4_K, rows * cols // 256, rng, 1e-3)
    w = QTensor(Tensor(blocks), GGMLType.Q4_K, (rows, cols))
    x = (rng.standard_normal((1, n, cols)) * 3).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, cols).astype(np.float32)
    r = rng.standard_normal((1, n, rows)).astype(np.float32)
    normed = ops.linears(Tensor(x), w, norm=(Tensor(weight), 1e-5))[0].numpy()[0]
    expected = reference_matmul(rms_norm(x[0], weight, 1e-5), blocks, GGMLType.Q4_K)
    # normalizing in f32 may round a few activations differently, as in test_norm_quantize_q8
    np.testing.assert_allclose(normed, expected, rtol=1e-3, atol=1e-3 * np.abs(expected).max())
    got = ops.linear(Tensor(x), w, residual=Tensor(r)).numpy()
    np.testing.assert_array_equal(got, (ops.linear(Tensor(x), w) + Tensor(r)).numpy())


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


@pytest.mark.parametrize("tokens, start", [(37, 0), (64, 0), (100, 300), (512, 3584)])
@pytest.mark.parametrize("symbolic", [False, True])
def test_flash_attention(tokens, start, symbolic):
    rng = np.random.default_rng(tokens + start)
    heads, kv_heads, dim, n = 32, 8, 128, 4096
    cache = rng.standard_normal((2, 1, kv_heads, n, dim)).astype(np.float16)
    q = rng.standard_normal((1, heads, 512, dim)).astype(np.float32)
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    if symbolic:  # as while prefilling: a bound start and number of tokens
        pos = UOp.variable("start_pos", 0, n - 1).bind(start)
        q_t = q_t[:, :, : UOp.variable("chunk_len", 1, 512).bind(tokens)]
    else:
        pos, q_t = start, q_t[:, :, :tokens]
    assert nv.supports_flash_attention(q_t, cache_t)
    got = nv.flash_attention(q_t, cache_t, pos).pad_to((1, 512, heads * dim)).numpy()
    got = got[0, :tokens].reshape(tokens, heads, dim)
    k, v = (cache[i, 0, :, : start + tokens].astype(np.float64) for i in range(2))
    causal = np.arange(start + tokens) > start + np.arange(tokens)[:, None]
    for h in range(heads):
        scores = q[0, h, :tokens] @ k[h // (heads // kv_heads)].T / np.sqrt(dim)
        p = np.exp(np.where(causal, -np.inf, scores - scores.max(-1, keepdims=True)))
        expected = (p / p.sum(-1, keepdims=True)) @ v[h // (heads // kv_heads)]
        # queries and weights are rounded to f16 for the tensor cores
        np.testing.assert_allclose(got[:, h], expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("rows, n", [(1, 128256), (3, 1000), (2, 33), (1, 1)])
def test_argmax(rows, n):
    rng = np.random.default_rng(n)
    x = rng.integers(-50, 50, (rows, n)).astype(np.float32)  # many ties: the first one must win
    x[:, ::7] = -np.inf
    assert nv.supports_argmax(Tensor(x))
    np.testing.assert_array_equal(ops.argmax(Tensor(x)).numpy(), x.argmax(-1, keepdims=True))
