import os

import gguf
import numpy as np
import pytest
from gguf.quants import dequantize
from tinygrad import Tensor, UOp, dtypes

from leat import nv, ops
from leat.quant import BLOCK, GGMLType, QTensor
from tests.helpers import random_blocks

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("DEV", "").split(":")[0] not in ("NV", "CUDA"), reason="needs DEV=NV or CUDA"
    ),
]
Q4_K, Q5_K, Q6_K, Q5_0, Q8_0 = (GGMLType[t] for t in ("Q4_K", "Q5_K", "Q6_K", "Q5_0", "Q8_0"))


def quantize_q8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # d = max|x| / 127 and q = roundf(x / d) per group of 32, all in f32 like the kernel
    groups = x.reshape(-1, nv.GROUP)
    d = (np.abs(groups).max(-1, keepdims=True) / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (groups / d).astype(np.float64)  # rounding half away from zero is exact in f64
    q = np.where(d > 0, np.sign(r) * np.floor(np.abs(r) + 0.5), 0).astype(np.int8)
    s = (d[:, 0] * q.sum(-1, dtype=np.int32).astype(np.float32)).astype(np.float32)
    return q.ravel(), d[:, 0], s


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x64 = x.astype(np.float64)
    return (x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + eps) * weight).astype(np.float32)


def random_matrix(
    ggml_type: GGMLType, rows: int, cols: int, rng: np.random.Generator
) -> tuple[QTensor, np.ndarray]:
    blocks = random_blocks(ggml_type, rows * cols // BLOCK[ggml_type][0], rng, 1e-3)
    return QTensor(Tensor(blocks), ggml_type, (rows, cols)), blocks


def reference_matmul(x: np.ndarray, blocks: np.ndarray, ggml_type: GGMLType) -> np.ndarray:
    # x (T, cols) quantized per row as the kernels do, times the decoded weights, in f64
    q, d, _ = quantize_q8(x)
    xq = (q.reshape(-1, nv.GROUP) * d[:, None]).reshape(x.shape)
    weights = dequantize(blocks, gguf.GGMLQuantizationType(ggml_type)).reshape(-1, x.shape[1])
    return xq.astype(np.float64) @ weights.astype(np.float64).T


def assert_close(got: np.ndarray, expected: np.ndarray, tolerance: float) -> None:
    # relative to the largest value: products of random weights have many near zero
    atol = tolerance * np.abs(expected).max()
    np.testing.assert_allclose(got, expected, rtol=tolerance, atol=atol)


# ******** quantization ********


# rows of 704 and 2112 values are not whole turns of the kernel's threads
@pytest.mark.parametrize("rows", [None, 5, UOp.variable("rows", 1, 8).bind(3)])
@pytest.mark.parametrize("width", [4096, 704, 2112])
def test_quantize_q8(rows, width):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((8, width)) * rng.uniform(0.01, 10, (8, width))).astype(np.float32)
    x[:, 64:96] = 0  # an all-zero group must give d = 0, not nan
    q, d, s = nv.quantize_q8(Tensor(x), rows=rows)
    Tensor.realize(q, d, s)  # in one schedule, as in the model: each alone would lose rows
    n = 8 if rows is None else rows if isinstance(rows, int) else rows.unbind()[1]
    groups = n * width // nv.GROUP
    outs = (q.numpy().view(np.int8)[: n * width], d.numpy()[:groups], s.numpy()[:groups])
    for got, want in zip(outs, quantize_q8(x[:n]), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("width", [4096, 2112])
def test_norm_quantize_q8(width):
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((3, width)) * rng.uniform(1, 5, (3, 1))).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, width).astype(np.float32)
    q, d, s = (t.numpy() for t in nv.quantize_q8(Tensor(x), (Tensor(weight), 1e-5)))
    want_q, want_d, _ = quantize_q8(rms_norm(x, weight, 1e-5))
    # normalizing in f32 rather than f64 may move a value across a rounding boundary
    off = q.view(np.int8).astype(np.int32) - want_q
    assert np.abs(off).max() <= 1 and np.count_nonzero(off) <= 4 * len(x)
    np.testing.assert_allclose(d, want_d, rtol=1e-5)
    sums = q.view(np.int8).reshape(-1, nv.GROUP).sum(-1, dtype=np.int32).astype(np.float32)
    np.testing.assert_array_equal(s, (d * sums).astype(np.float32))


# ******** one token ********


# rows of 768 weights leave lanes idle, and of 2816 some in a second turn
@pytest.mark.parametrize("ggml_type", [Q4_K, Q5_K, Q6_K, Q5_0, Q8_0])
@pytest.mark.parametrize("shape", [(64, 4096), (8, 14336), (16, 768), (24, 2816)])
def test_matvec(ggml_type, shape):
    rng = np.random.default_rng(1)
    w, blocks = random_matrix(ggml_type, *shape, rng)
    x = rng.standard_normal((1, 1, shape[1])).astype(np.float32)
    assert nv.supports_matvec(Tensor(x), w)
    got = ops.linear(Tensor(x), w).numpy()
    assert got.shape == (1, 1, shape[0])
    assert_close(got[0], reference_matmul(x[0], blocks, ggml_type), 1e-4)


def test_shared_input():
    # matrices of different types share one quantization of x
    rng = np.random.default_rng(3)
    ws = [random_matrix(t, 8, 2048, rng)[0] for t in (Q4_K, Q6_K)]
    x = Tensor(rng.standard_normal((1, 1, 2048)).astype(np.float32))
    for got, w in zip(ops.linears(x, *ws), ws, strict=True):
        np.testing.assert_array_equal(got.numpy(), ops.linear(x, w).numpy())


@pytest.mark.parametrize("ggml_type", [Q4_K, Q6_K])
@pytest.mark.parametrize("gelu", [False, True])
def test_swiglu(ggml_type, gelu):
    rng = np.random.default_rng(7)
    (gate, gate_blocks), (up, up_blocks) = (random_matrix(ggml_type, 16, 4096, rng) for _ in "gu")
    x = rng.standard_normal((1, 1, 4096)).astype(np.float32)
    g, u = (reference_matmul(x[0], b, ggml_type) for b in (gate_blocks, up_blocks))
    got = nv.swiglu(Tensor(x), gate, up, gelu=gelu).numpy()[0]
    assert_close(got, activation(g, gelu) * u, 1e-4)


def activation(x: np.ndarray, gelu: bool) -> np.ndarray:
    if gelu:  # tanh's approximation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return x / (1 + np.exp(-x))


# ******** several tokens ********


@pytest.mark.parametrize("ggml_type", [Q4_K, Q5_K, Q6_K, Q8_0])
@pytest.mark.parametrize("tokens", [64, 100, UOp.variable("tokens", 1, 128).bind(70)])
@pytest.mark.parametrize("shape", [(256, 2048), (4096, 512)])  # tiles of 128 and 256 rows
def test_matmul(ggml_type, tokens, shape):
    rng = np.random.default_rng(8)
    w, blocks = random_matrix(ggml_type, *shape, rng)
    if isinstance(tokens, int):
        x = rng.standard_normal((1, tokens, shape[1])).astype(np.float32)
        x_t, n = Tensor(x), tokens
    else:  # while prefilling, a bound number of tokens out of the most there may be
        x = rng.standard_normal((1, 128, shape[1])).astype(np.float32)
        x_t, n = Tensor(x)[:, :tokens], tokens.unbind()[1]
    assert nv.supports_matmul(x_t, w)
    got = ops.linear(x_t, w).pad_to((1, x.shape[1], shape[0])).numpy()[0, :n]
    assert_close(got, reference_matmul(x[0, :n], blocks, ggml_type), 1e-4)


# rows of whole blocks of 32 but not whole steps of 128, as Gemma 4's of 704 and 2112 weights
@pytest.mark.parametrize("ggml_type", [Q5_0, Q8_0])
@pytest.mark.parametrize("tokens", [64, UOp.variable("tokens", 1, 128).bind(70)])
@pytest.mark.parametrize("shape", [(256, 704), (128, 2112), (4096, 512)])
def test_matmul_blocks_of_32(ggml_type, tokens, shape):
    test_matmul(ggml_type, tokens, shape)


@pytest.mark.parametrize("heights", [(256, 128, 128), (4096, 1024, 1024)])  # tiles of 128, 256
def test_matmul_stacked(heights):
    # consecutive matrices of one type share a kernel: here the first two, as q and k
    rng = np.random.default_rng(10)
    types = (Q4_K, Q4_K, Q6_K)
    matrices = [random_matrix(t, h, 512, rng) for t, h in zip(types, heights, strict=True)]
    ws, blocks = zip(*matrices, strict=True)
    x = rng.standard_normal((1, 70, 512)).astype(np.float32)
    for got, b, t in zip(ops.linears(Tensor(x), *ws), blocks, types, strict=True):
        assert_close(got.numpy()[0], reference_matmul(x[0], b, t), 1e-4)


# ******** mixtures of experts ********


# a warp per token and expert, then for more than 8 tokens a warp per tile of them
@pytest.mark.parametrize(
    "tokens",
    [1, 7, UOp.variable("tokens", 1, 8).bind(5), 37, UOp.variable("tokens", 1, 64).bind(45)],
)
def test_scores(tokens):
    rng = np.random.default_rng(17)
    x = (rng.standard_normal((1, 64, 2048)) * 3).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, 2048).astype(np.float32)
    router = QTensor(
        Tensor(rng.standard_normal((128, 2048)).astype(np.float32)).flatten(),
        GGMLType.F32,
        (128, 2048),
    )
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    x_t = Tensor(x)[:, :tokens]
    assert nv.supports_scores(x_t, router)
    got = nv.scores(x_t, (Tensor(weight), 1e-6), router).pad_to((1, 64, 128)).numpy()[0, :n]
    expected = (
        rms_norm(x[0, :n], weight, 1e-6).astype(np.float64)
        @ router.data.numpy().reshape(128, 2048).T
    )
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tokens", [1, 7, UOp.variable("tokens", 1, 16).bind(5)])
def test_route(tokens):
    rng = np.random.default_rng(14)
    scores = rng.standard_normal((16, 128)).astype(np.float32)
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    ids, weights = (t.numpy()[: n * 8] for t in nv.route(Tensor(scores)[:tokens], 8))
    best = np.argsort(-scores[:n], -1, kind="stable")[:, :8]
    np.testing.assert_array_equal(ids.reshape(n, 8), best)
    top = np.take_along_axis(scores[:n], best, -1)
    p = np.exp(top - top[:, :1])
    np.testing.assert_allclose(weights.reshape(n, 8), p / p.sum(-1, keepdims=True), rtol=1e-6)


def random_experts(
    ggml_type: GGMLType, experts: int, rows: int, cols: int, rng: np.random.Generator
) -> tuple[QTensor, np.ndarray]:
    # a stack of matrices, and each one's blocks
    w, blocks = random_matrix(ggml_type, experts * rows, cols, rng)
    return QTensor(w.data, ggml_type, (experts, rows, cols)), blocks.reshape(
        experts, -1, blocks.shape[-1]
    )


def expected_mixture(normed: np.ndarray, scores: np.ndarray, used: int, expert) -> np.ndarray:
    # each token's `used` best scoring experts' outputs expert(e, normed row), weighted by the
    # softmax of their scores
    out = np.zeros(normed.shape, dtype=np.float64)
    for t in range(len(normed)):
        best = np.argsort(-scores[t], kind="stable")[:used]
        p = np.exp(scores[t, best] - scores[t, best].max())
        for e, pe in zip(best, p / p.sum(), strict=True):
            out[t] += pe * expert(e, normed[t : t + 1])[0]
    return out


# up to FEW tokens take the matrix-vector kernels, more the tensor cores; favored experts get
# more than a tile of tokens
@pytest.mark.parametrize("tokens", [1, 3, 70, UOp.variable("tokens", 1, 128).bind(37), 128])
@pytest.mark.parametrize("favored", [0, 3])
def test_mixture(tokens, favored):
    rng = np.random.default_rng(15)
    experts, used, dim, hidden = 32, 4, 512, 768
    (gate, gate_blocks), (up, up_blocks) = (
        random_experts(Q4_K, experts, hidden, dim, rng) for _ in "gu"
    )
    down, down_blocks = random_experts(Q6_K, experts, dim, hidden, rng)
    x = (rng.standard_normal((1, 128, dim)) * 3).astype(np.float32)
    scores = rng.standard_normal((1, 128, experts)).astype(np.float32)
    scores[..., :favored] += 4
    weight = rng.uniform(0.5, 1.5, dim).astype(np.float32)
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    x_t, scores_t = Tensor(x)[:, :tokens], Tensor(scores)[:, :tokens]
    assert nv.supports_mixture(x_t, gate, up, down)
    got = nv.mixture(x_t, scores_t, gate, up, down, used, (Tensor(weight), 1e-5))
    got = got.pad_to((1, 128, dim)).numpy()[0, :n]

    def expert(e, row):
        g, u = (reference_matmul(row, b[e], Q4_K) for b in (gate_blocks, up_blocks))
        return reference_matmul(activation(g, False).astype(np.float32) * u, down_blocks[e], Q6_K)

    normed = rms_norm(x[0, :n], weight, 1e-5)
    assert_close(got, x[0, :n] + expected_mixture(normed, scores[0, :n], used, expert), 2e-3)


# Gemma 4: gate and up in one stack, GELU, a scale per expert and no residual, on the
# matrix-vector kernels for any number of tokens; rows of 192 weights quantize in a ragged turn
@pytest.mark.parametrize("tokens", [1, 70, UOp.variable("tokens", 1, 128).bind(37)])
def test_mixture_gemma(tokens):
    rng = np.random.default_rng(16)
    experts, used, dim, hidden = 32, 4, 512, 192
    gate_up, gate_up_blocks = random_experts(Q4_K, experts, 2 * hidden, dim, rng)
    down, down_blocks = random_experts(Q5_0, experts, dim, hidden, rng)
    x = (rng.standard_normal((1, 128, dim)) * 3).astype(np.float32)
    scores = rng.standard_normal((1, 128, experts)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, dim).astype(np.float32)
    scales = rng.uniform(0.5, 2, experts).astype(np.float32)
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    x_t, scores_t = Tensor(x)[:, :tokens], Tensor(scores)[:, :tokens]
    assert nv.supports_mixture(x_t, gate_up, None, down)
    norm = (Tensor(weight), 1e-5)
    got = nv.mixture(x_t, scores_t, gate_up, None, down, used, norm, True, Tensor(scales), False)
    got = got.pad_to((1, 128, dim)).numpy()[0, :n]

    def expert(e, row):
        g, u = np.split(reference_matmul(row, gate_up_blocks[e], Q4_K), 2, axis=-1)
        h = activation(g, True).astype(np.float32) * u
        return reference_matmul(h, down_blocks[e], Q5_0) * scales[e]

    normed = rms_norm(x[0, :n], weight, 1e-5)
    assert_close(got, expected_mixture(normed, scores[0, :n], used, expert), 2e-3)


# ******** norms ********


# as Gemma 4's attention block ends, and its MLP block with experts, with the layer's scale
@pytest.mark.parametrize("parts, normed", [(1, False), (2, True)])
@pytest.mark.parametrize("tokens", [1, 5, UOp.variable("tokens", 1, 8).bind(3)])
def test_add_normed(monkeypatch, parts, normed, tokens):
    rng = np.random.default_rng(19)
    dim = 2816
    x, *ys = (
        Tensor((rng.standard_normal((1, 8, dim)) * 3).astype(np.float32)) for _ in range(parts + 1)
    )
    ws = [Tensor(rng.uniform(0.5, 1.5, dim).astype(np.float32)) for _ in range(parts + 1)]
    scale = Tensor(np.array([0.7], dtype=np.float32))
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    results = []
    for kernels in ("auto", "ref"):
        monkeypatch.setenv("LEAT_KERNELS", kernels)
        parts_ = [(y[:, :tokens], w) for y, w in zip(ys, ws, strict=False)]
        out = ops.add_normed(x[:, :tokens], parts_, ws[-1] if normed else None, 1e-6, scale)
        results.append(out.pad_to((1, 8, dim)).numpy()[0, :n])
    assert nv.supports_add_normed(x)
    np.testing.assert_allclose(results[0], results[1], rtol=1e-5, atol=1e-5)


# ******** ops on the kernels ********


@pytest.mark.parametrize("tokens", [1, 80])
def test_norm_and_residual(tokens):
    rng = np.random.default_rng(9)
    w, blocks = random_matrix(Q4_K, 256, 4096, rng)
    x = (rng.standard_normal((1, tokens, 4096)) * 3).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, 4096).astype(np.float32)
    r = Tensor(rng.standard_normal((1, tokens, 256)).astype(np.float32))
    normed = ops.linears(Tensor(x), w, norm=(Tensor(weight), 1e-5))[0].numpy()[0]
    # normalizing in f32 may round a few activations differently, as in test_norm_quantize_q8
    assert_close(normed, reference_matmul(rms_norm(x[0], weight, 1e-5), blocks, Q4_K), 1e-3)
    got = ops.linear(Tensor(x), w, residual=r).numpy()
    np.testing.assert_array_equal(got, (ops.linear(Tensor(x), w) + r).numpy())


# one token takes the matrix-vector kernels where the matrices are wide enough, else as several
# tokens do, with gate and up in tiles of 256 rows or, for few rows, 128, or 64 where 128 do not
# divide them: Gemma 4's MLP, of GELU and Q5_0 rows of 2112, without the residual
@pytest.mark.parametrize(
    "tokens, hidden, down_type, gelu",
    [(1, 4096, Q6_K, False), (1, 1024, Q6_K, False), (70, 4096, Q6_K, False),
     (70, 1024, Q6_K, False), (1, 2112, Q5_0, True), (70, 2112, Q5_0, True)],
)  # fmt: skip
def test_feed_forward(tokens, hidden, down_type, gelu):
    rng = np.random.default_rng(12)
    dim = 2048
    (gate, gate_blocks), (up, up_blocks) = (random_matrix(Q4_K, hidden, dim, rng) for _ in "gu")
    down, down_blocks = random_matrix(down_type, dim, hidden, rng)
    x = (rng.standard_normal((1, tokens, dim)) * 3).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, dim).astype(np.float32)
    norm = (Tensor(weight), 1e-5)
    got = ops.feed_forward(Tensor(x), gate, up, down, norm, gelu, residual=not gelu).numpy()[0]
    normed = rms_norm(x[0], weight, 1e-5)
    g, u = (reference_matmul(normed, b, Q4_K) for b in (gate_blocks, up_blocks))
    hidden_ = (activation(g, gelu) * u).astype(np.float32)
    expected = reference_matmul(hidden_, down_blocks, down_type) + (0 if gelu else x[0])
    # f32 rounding inside the kernels may move a few activations across a quantization step
    assert_close(got, expected, 2e-3)


def test_one_token_takes_matvec(monkeypatch):
    # the matrix kernels take one token too, several times slower than the matrix-vector kernels
    def fail(*args, **kwargs):
        raise AssertionError("one token went to a matrix kernel")

    monkeypatch.setattr(nv, "matmuls", fail)
    monkeypatch.setattr(nv, "feed_forward", fail)
    rng = np.random.default_rng(13)
    ws = [random_matrix(Q4_K, 2048, 2048, rng)[0] for _ in range(3)]
    x = Tensor(rng.standard_normal((1, 1, 2048)).astype(np.float32))
    norm = (Tensor.ones(2048), 1e-5)
    Tensor.realize(ops.linears(x, *ws, norm=norm)[0], ops.feed_forward(x, *ws, norm=norm))


def test_reference_switch(monkeypatch):
    rng = np.random.default_rng(2)
    w, _ = random_matrix(Q4_K, 16, 2048, rng)
    x = Tensor(rng.standard_normal((1, 1, 2048)).astype(np.float32))
    monkeypatch.setenv("LEAT_KERNELS", "ref")
    np.testing.assert_array_equal(ops.linear(x, w).numpy(), (x @ w.dequant().T).numpy())


# ******** attention ********

SLOTS, SLOT = 3, 1  # the kernels read one slot of a cache of several


def slot(symbolic: bool) -> int | UOp:
    return UOp.variable("slot", 0, SLOTS - 1).bind(SLOT) if symbolic else SLOT


def reference_attention(
    q: np.ndarray, cache: np.ndarray, start: int, window: int = 0, scale: float | None = None
) -> np.ndarray:
    # q (heads, T, dim) at positions start.. against slot SLOT, causally and within the window if
    # given, in f64: (T, heads, dim)
    heads, tokens, dim = q.shape
    group = heads // cache.shape[2]
    k, v = (cache[i, SLOT, :, : start + tokens].astype(np.float64) for i in range(2))
    back = start + np.arange(tokens)[:, None] - np.arange(start + tokens)
    hidden = (back < 0) | (back >= window) if window else back < 0
    out = np.empty((tokens, heads, dim))
    for h in range(heads):
        scores = q[h] @ k[h // group].T * (dim**-0.5 if scale is None else scale)
        p = np.exp(np.where(hidden, -np.inf, scores - scores.max(-1, keepdims=True)))
        out[:, h] = (p / p.sum(-1, keepdims=True)) @ v[h // group]
    return out


# cache sizes matter too: some strides trip a tinygrad codegen bug with symbolic lengths, see
# common.opaque
@pytest.mark.parametrize(
    "n, length",
    [(64, 1), (64, 64), (1024, 63), (1024, 65), (1024, 1000), (3072, 3072), (4096, 3079),
     (4096, 4096)],
)  # fmt: skip
@pytest.mark.parametrize("symbolic", [False, True])
def test_attention(n, length, symbolic):
    rng = np.random.default_rng(length)
    cache = rng.standard_normal((2, SLOTS, 8, n, 128)).astype(np.float16)
    q = rng.standard_normal((1, 32, 1, 128)).astype(np.float32)
    valid = UOp.variable("start_pos", 0, n - 1).bind(length - 1) + 1 if symbolic else length
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()  # the model's cache is a buffer
    assert nv.supports_attention(q_t, cache_t)
    got = nv.attention(q_t, cache_t, slot(symbolic), valid, 128**-0.5).numpy()[0, :, 0]
    expected = reference_attention(q[0], cache, length - 1)[0]
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


# GQA groups that blocks of 4 query heads do not divide: Qwen2.5 7B's of 7, and one of 9
@pytest.mark.parametrize("heads, kv_heads", [(28, 4), (18, 2)])
@pytest.mark.parametrize("length", [65, 1000])
@pytest.mark.parametrize("symbolic", [False, True])
def test_attention_groups(heads, kv_heads, length, symbolic):
    rng = np.random.default_rng(length + heads)
    cache = rng.standard_normal((2, SLOTS, kv_heads, 1024, 128)).astype(np.float16)
    q = rng.standard_normal((1, heads, 1, 128)).astype(np.float32)
    valid = UOp.variable("start_pos", 0, 1023).bind(length - 1) + 1 if symbolic else length
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    assert nv.supports_attention(q_t, cache_t)
    got = nv.attention(q_t, cache_t, slot(symbolic), valid, 128**-0.5).numpy()[0, :, 0]
    expected = reference_attention(q[0], cache, length - 1)[0]
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


# Llama 3.1 8B's groups of 4 query heads, and Qwen2.5 7B's of 7, which do not divide a tile's loads
@pytest.mark.parametrize("tokens, start", [(37, 0), (64, 0), (100, 300), (512, 3584)])
@pytest.mark.parametrize("heads, kv_heads", [(32, 8), (28, 4)])
@pytest.mark.parametrize("symbolic", [False, True])
def test_flash_attention(tokens, start, heads, kv_heads, symbolic):
    rng = np.random.default_rng(tokens + start)
    cache = rng.standard_normal((2, SLOTS, kv_heads, 4096, 128)).astype(np.float16)
    q = rng.standard_normal((1, heads, 512, 128)).astype(np.float32)
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    if symbolic:  # as while prefilling: a bound start and number of tokens
        pos = UOp.variable("start_pos", 0, 4095).bind(start)
        q_t = q_t[:, :, : UOp.variable("chunk_len", 1, 512).bind(tokens)]
    else:
        pos, q_t = start, q_t[:, :, :tokens]
    assert nv.supports_flash_attention(q_t, cache_t)
    got = nv.flash_attention(q_t, cache_t, slot(symbolic), pos, 128**-0.5)
    got = got.pad_to((1, 512, heads * 128)).numpy()[0, :tokens]
    expected = reference_attention(q[0, :, :tokens], cache, start)
    # queries and weights are rounded to f16 for the tensor cores
    np.testing.assert_allclose(got.reshape(expected.shape), expected, rtol=3e-3, atol=3e-3)


# Gemma 4's shapes: sliding-window layers of 8 kv heads of 256, the others of 2 kv heads of 512;
# caches of 2048 positions tripped the codegen bug that common.opaque avoids
@pytest.mark.parametrize("kv_heads, dim, window", [(8, 256, 1024), (2, 512, 0), (8, 128, 100)])
@pytest.mark.parametrize("n, length", [(4096, 70), (4096, 1100), (4096, 3079), (2048, 1500)])
@pytest.mark.parametrize("symbolic", [False, True])
def test_attention_window(kv_heads, dim, window, n, length, symbolic):
    rng = np.random.default_rng(length + dim)
    cache = rng.standard_normal((2, SLOTS, kv_heads, n, dim)).astype(np.float16)
    q = rng.standard_normal((1, 16, 1, dim)).astype(np.float32) * 0.2
    valid = UOp.variable("start_pos", 0, n - 1).bind(length - 1) + 1 if symbolic else length
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    assert nv.supports_attention(q_t, cache_t)
    got = nv.attention(q_t, cache_t, slot(symbolic), valid, 1.0, window).numpy()[0, :, 0]
    expected = reference_attention(q[0], cache, length - 1, window, 1.0)[0]
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("tokens, start", [(37, 0), (100, 1000), (512, 3584)])
@pytest.mark.parametrize("dim, window", [(256, 1024), (128, 100)])
@pytest.mark.parametrize("symbolic", [False, True])
def test_flash_attention_window(tokens, start, dim, window, symbolic):
    rng = np.random.default_rng(tokens + start + dim)
    cache = rng.standard_normal((2, SLOTS, 8, 4096, dim)).astype(np.float16)
    q = rng.standard_normal((1, 16, 512, dim)).astype(np.float32) * 0.2
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    if symbolic:
        pos = UOp.variable("start_pos", 0, 4095).bind(start)
        q_t = q_t[:, :, : UOp.variable("chunk_len", 1, 512).bind(tokens)]
    else:
        pos, q_t = start, q_t[:, :, :tokens]
    assert nv.supports_flash_attention(q_t, cache_t)
    got = nv.flash_attention(q_t, cache_t, slot(symbolic), pos, 1.0, window)
    got = got.pad_to((1, 512, 16 * dim)).numpy()[0, :tokens]
    expected = reference_attention(q[0, :, :tokens], cache, start, window, 1.0)
    np.testing.assert_allclose(got.reshape(expected.shape), expected, rtol=3e-3, atol=3e-3)


# llama: adjacent pairs; qwen2: halves and biases; qwen3: halves and norms of q and k; gemma4: v
# normed too, two kv heads
@pytest.mark.parametrize(
    "halves, biased, normed, v_norm, kv_heads",
    [
        (False, False, False, False, 8),
        (True, True, False, False, 8),
        (True, False, True, False, 8),
        (True, False, True, True, 2),
    ],
)
@pytest.mark.parametrize("symbolic", [False, True])
def test_rotate(monkeypatch, halves, biased, normed, v_norm, kv_heads, symbolic):
    rng = np.random.default_rng(18)
    dim, pos = 128, 300
    q, k, v = (
        Tensor(rng.standard_normal((1, 1, h, dim)).astype(np.float32)).realize()
        for h in (32, kv_heads, kv_heads)
    )
    angles = rng.uniform(0, 6, (512, dim // 2)).astype(np.float32)
    rope = (Tensor(np.cos(angles)).realize(), Tensor(np.sin(angles)).realize())
    norms = (
        tuple(Tensor(rng.uniform(0.5, 1.5, dim).astype(np.float32)) for _ in "qk")
        if normed
        else None
    )
    biases = (
        tuple(
            Tensor(rng.standard_normal(h * dim).astype(np.float32))
            for h in (32, kv_heads, kv_heads)
        )
        if biased
        else None
    )
    start = UOp.variable("start_pos", 0, 511).bind(pos) if symbolic else pos
    results = []
    for kernels in ("auto", "ref"):
        monkeypatch.setenv("LEAT_KERNELS", kernels)
        cache = Tensor.zeros(2, SLOTS, kv_heads, 512, dim, dtype=dtypes.half).contiguous().realize()
        assert nv.supports_rotate(q, cache)
        out, cache = ops.rotate(
            q, k, v, cache, slot(symbolic), start, rope, halves, biases, norms, v_norm, 1e-6
        )
        Tensor.realize(out, cache)  # in one schedule, as in the model: alone, either loses vars
        results.append((out.numpy(), cache.numpy()))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        results[0][1].astype(np.float32), results[1][1], rtol=1e-3, atol=1e-3
    )
    assert np.abs(results[0][1][:, SLOT, :, pos]).sum() > 0


# heads too wide for registers, Gemma 4's of 512: blocks take parts of their outputs
@pytest.mark.parametrize("tokens, start, window", [(37, 0, 0), (100, 1000, 0), (512, 1500, 1024)])
@pytest.mark.parametrize("symbolic", [False, True])
def test_flash_attention_wide(tokens, start, window, symbolic):
    rng = np.random.default_rng(tokens + start)
    cache = rng.standard_normal((2, SLOTS, 2, 2048, 512)).astype(np.float16) * np.float16(0.2)
    q = rng.standard_normal((1, 16, 512, 512)).astype(np.float32) * 0.2
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    if symbolic:
        pos = UOp.variable("start_pos", 0, 2047).bind(start)
        q_t = q_t[:, :, : UOp.variable("chunk_len", 1, 512).bind(tokens)]
    else:
        pos, q_t = start, q_t[:, :, :tokens]
    assert nv.supports_flash_attention(q_t, cache_t)
    got = nv.flash_attention(q_t, cache_t, slot(symbolic), pos, 1.0, window)
    got = got.pad_to((1, 512, 16 * 512)).numpy()[0, :tokens]
    expected = reference_attention(q[0, :, :tokens], cache, start, window, 1.0)
    np.testing.assert_allclose(got.reshape(expected.shape), expected, rtol=3e-3, atol=3e-3)


# ******** argmax ********


@pytest.mark.parametrize("rows, n", [(1, 128256), (3, 1000), (2, 33), (1, 1)])
def test_argmax(rows, n):
    rng = np.random.default_rng(n)
    x = rng.integers(-50, 50, (rows, n)).astype(np.float32)  # many ties: the first one must win
    x[:, ::7] = -np.inf
    assert nv.supports_argmax(Tensor(x))
    np.testing.assert_array_equal(ops.argmax(Tensor(x)).numpy(), x.argmax(-1, keepdims=True))
