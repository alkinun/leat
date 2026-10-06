import os

import gguf
import numpy as np
import pytest
from gguf.quants import dequantize
from tinygrad import Tensor, UOp, dtypes

from leat import kernels, ops
from leat.kernels import MATVEC_TOKENS
from leat.kernels.cutoff import BINS, RANGE
from leat.quant import BLOCK, GGMLType, QTensor
from tests.helpers import cuts, glu, random_blocks

# the warp-level kernels run on NVIDIA's GPUs and AMD's RDNA, tinygrad's emulated one too, and those
# on tensor cores on NVIDIA's and RDNA 3's
pytestmark = [pytest.mark.gpu]
MATRIX_CORES = os.environ.get("DEV", "").split(":")[0] in ("NV", "CUDA", "AMD", "MOCK+AMD")
matrix_cores = pytest.mark.skipif(not MATRIX_CORES, reason="tensor-core kernels need a GPU of them")
Q4_K, Q5_K, Q6_K, Q5_0, Q8_0, MXFP4 = (
    GGMLType[t] for t in ("Q4_K", "Q5_K", "Q6_K", "Q5_0", "Q8_0", "MXFP4")
)
# the matrix-vector kernels' other types
LEGACY = [GGMLType[t] for t in ("Q4_0", "Q4_1", "Q5_1", "IQ4_NL", "IQ4_XS")]
TILED = [GGMLType[t] for t in ("Q4_0", "IQ4_NL", "IQ4_XS")]  # of those, the tensor cores' too


def quantize_q8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # d = max|x| / 127 and q = roundf(x / d) per group of 32, all in f32 like the kernel
    groups = x.reshape(-1, kernels.GROUP)
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
    xq = (q.reshape(-1, kernels.GROUP) * d[:, None]).reshape(x.shape)
    weights = dequantize(blocks, gguf.GGMLQuantizationType(ggml_type)).reshape(-1, x.shape[1])
    return xq.astype(np.float64) @ weights.astype(np.float64).T


def assert_close(got: np.ndarray, expected: np.ndarray, tolerance: float) -> None:
    # relative to the largest value: products of random weights have many near zero
    atol = tolerance * np.abs(expected).max()
    np.testing.assert_allclose(got, expected, rtol=tolerance, atol=atol)


# ******** quantization ********


# rows of 704 and 2112 values are not whole turns of the kernel's threads; up to 16 rows spread
# over blocks, more take one each
@pytest.mark.parametrize("rows", [None, 5, UOp.variable("rows", 1, 8).bind(3)])
@pytest.mark.parametrize("width", [4096, 704, 2112])
@pytest.mark.parametrize("count", [8, 24])
def test_quantize_q8(rows, width, count):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((count, width)) * rng.uniform(0.01, 10, (count, width))
    x = x.astype(np.float32)
    x[:, 64:96] = 0  # an all-zero group must give d = 0, not nan
    q, d, s = kernels.quantize_q8(Tensor(x), rows=rows)
    Tensor.realize(q, d, s)  # in one schedule, as in the model: each alone would lose rows
    n = count if rows is None else rows if isinstance(rows, int) else rows.unbind()[1]
    groups = n * width // kernels.GROUP
    outs = (q.numpy().view(np.int8)[: n * width], d.numpy()[:groups], s.numpy()[:groups])
    for got, want in zip(outs, quantize_q8(x[:n]), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("width", [4096, 2112])
def test_norm_quantize_q8(width):
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((3, width)) * rng.uniform(1, 5, (3, 1))).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, width).astype(np.float32)
    q, d, s = (t.numpy() for t in kernels.quantize_q8(Tensor(x), (Tensor(weight), 1e-5)))
    want_q, want_d, _ = quantize_q8(rms_norm(x, weight, 1e-5))
    # normalizing in f32 rather than f64 may move a value across a rounding boundary
    off = q.view(np.int8).astype(np.int32) - want_q
    assert np.abs(off).max() <= 1 and np.count_nonzero(off) <= 4 * len(x)
    np.testing.assert_allclose(d, want_d, rtol=1e-5)
    sums = q.view(np.int8).reshape(-1, kernels.GROUP).sum(-1, dtype=np.int32).astype(np.float32)
    np.testing.assert_array_equal(s, (d * sums).astype(np.float32))


# ******** a few tokens: matrix-vector products ********


# rows of 768 weights leave lanes idle, and of 2816 some in a second turn
@pytest.mark.parametrize("ggml_type", [Q4_K, Q5_K, Q6_K, Q5_0, Q8_0, MXFP4, *LEGACY])
@pytest.mark.parametrize("shape", [(64, 4096), (8, 14336), (16, 768), (24, 2816)])
def test_matvec(ggml_type, shape):
    rng = np.random.default_rng(1)
    w, blocks = random_matrix(ggml_type, *shape, rng)
    x = rng.standard_normal((1, 1, shape[1])).astype(np.float32)
    assert kernels.supports_matvec(Tensor(x), w)
    got = ops.linear(Tensor(x), w).numpy()
    assert got.shape == (1, 1, shape[0])
    assert_close(got[0], reference_matmul(x[0], blocks, ggml_type), 1e-4)


@pytest.mark.parametrize("ggml_type", [Q4_K, Q5_K, Q6_K, Q5_0, Q8_0, MXFP4, *LEGACY])
@pytest.mark.parametrize("tokens", [3, MATVEC_TOKENS])
@pytest.mark.parametrize("rows", [24, 26])  # whole warps' rows, or a last warp's of fewer
def test_matvec_tokens(ggml_type, tokens, rows):
    # each row's weights read once for every token, a residual added per token
    rng = np.random.default_rng(4)
    w, blocks = random_matrix(ggml_type, rows, 2816, rng)
    x = rng.standard_normal((1, tokens, 2816)).astype(np.float32)
    r = rng.standard_normal((1, tokens, rows)).astype(np.float32)
    assert kernels.supports_matvec(Tensor(x), w)
    got = ops.linear(Tensor(x), w, residual=Tensor(r)).numpy()
    assert got.shape == (1, tokens, rows)
    assert_close(got[0], reference_matmul(x[0], blocks, ggml_type) + r[0], 1e-4)


def test_shared_input():
    # matrices of different types share one quantization of x
    rng = np.random.default_rng(3)
    ws = [random_matrix(t, 8, 2048, rng)[0] for t in (Q4_K, Q6_K)]
    x = Tensor(rng.standard_normal((1, 1, 2048)).astype(np.float32))
    for got, w in zip(ops.linears(x, *ws), ws, strict=True):
        np.testing.assert_array_equal(got.numpy(), ops.linear(x, w).numpy())


@pytest.mark.parametrize("ggml_type", [Q4_K, Q6_K])
@pytest.mark.parametrize("kind", ["silu", "gelu", "oai"])
@pytest.mark.parametrize("tokens", [1, 4])
def test_swiglu(ggml_type, kind, tokens):
    rng = np.random.default_rng(7)
    (gate, gate_blocks), (up, up_blocks) = (random_matrix(ggml_type, 16, 4096, rng) for _ in "gu")
    x = (rng.standard_normal((1, tokens, 4096)) * 3).astype(np.float32)
    g, u = (reference_matmul(x[0], b, ggml_type) for b in (gate_blocks, up_blocks))
    got = kernels.swiglu(Tensor(x), gate, up, kind=kind).numpy()[0]
    assert_close(got, glu(kind, g, u), 1e-4)


# ******** several tokens ********


# up to 16 tokens take tiles of 16; with few tiles, blocks split the steps along the rows
SHORT = [5, UOp.variable("tokens", 1, 16).bind(11)]


@pytest.mark.parametrize("ggml_type", [Q4_K, Q5_K, Q6_K, Q8_0, *TILED])
@pytest.mark.parametrize("tokens", [64, 100, UOp.variable("tokens", 1, 128).bind(70), *SHORT])
# tiles of 128 and 256 rows, and of 96, gpt-oss's 2880 rows, which 128 do not divide
@pytest.mark.parametrize("shape", [(256, 2048), (4096, 512), (2880, 512)])
@matrix_cores
def test_matmul(ggml_type, tokens, shape):
    rng = np.random.default_rng(8)
    w, blocks = random_matrix(ggml_type, *shape, rng)
    if isinstance(tokens, int):
        x = rng.standard_normal((1, tokens, shape[1])).astype(np.float32)
        x_t, n = Tensor(x), tokens
    else:  # while prefilling, a bound number of tokens out of the most there may be
        x = rng.standard_normal((1, 128, shape[1])).astype(np.float32)
        x_t, n = Tensor(x)[:, :tokens], tokens.unbind()[1]
    assert kernels.supports_matmul(x_t, w)
    got = ops.linear(x_t, w).pad_to((1, x.shape[1], shape[0])).numpy()[0, :n]
    assert_close(got, reference_matmul(x[0, :n], blocks, ggml_type), 1e-4)


# rows of whole blocks of 32 but not whole steps of 128, as Gemma 4's of 704 and 2112 weights
@pytest.mark.parametrize("ggml_type", [Q5_0, Q8_0, MXFP4, *TILED[:2]])
@pytest.mark.parametrize("tokens", [64, UOp.variable("tokens", 1, 128).bind(70), SHORT[1]])
@pytest.mark.parametrize("shape", [(256, 704), (128, 2112), (4096, 512)])
@matrix_cores
def test_matmul_blocks_of_32(ggml_type, tokens, shape):
    test_matmul(ggml_type, tokens, shape)


@pytest.mark.parametrize("heights", [(256, 128, 128), (4096, 1024, 1024)])  # tiles of 128, 256
@pytest.mark.parametrize("tokens", [70, 5])
@matrix_cores
def test_matmul_stacked(heights, tokens):
    # consecutive matrices of one type share a kernel: here the first two, as q and k
    rng = np.random.default_rng(10)
    types = (Q4_K, Q4_K, Q6_K)
    matrices = [random_matrix(t, h, 512, rng) for t, h in zip(types, heights, strict=True)]
    ws, blocks = zip(*matrices, strict=True)
    x = rng.standard_normal((1, tokens, 512)).astype(np.float32)
    for got, b, t in zip(ops.linears(Tensor(x), *ws), blocks, types, strict=True):
        assert_close(got.numpy()[0], reference_matmul(x[0], b, t), 1e-4)


# ******** mixtures of experts ********


# a warp per token and expert, then for more than 8 tokens a warp per tile of them
@pytest.mark.parametrize(
    "tokens",
    [1, 7, UOp.variable("tokens", 1, 8).bind(5), 37, UOp.variable("tokens", 1, 64).bind(45)],
)
@pytest.mark.parametrize("dim", [2048, 2880])  # gpt-oss's rows are no whole turns of a warp
def test_scores(tokens, dim):
    rng = np.random.default_rng(17)
    x = (rng.standard_normal((1, 64, dim)) * 3).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, dim).astype(np.float32)
    router = QTensor(
        Tensor(rng.standard_normal((128, dim)).astype(np.float32)).flatten(),
        GGMLType.F32,
        (128, dim),
    )
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    x_t = Tensor(x)[:, :tokens]
    assert kernels.supports_scores(x_t, router)
    got = kernels.scores(x_t, (Tensor(weight), 1e-6), router).pad_to((1, 64, 128)).numpy()[0, :n]
    expected = (
        rms_norm(x[0, :n], weight, 1e-6).astype(np.float64)
        @ router.data.numpy().reshape(128, dim).T
    )
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tokens", [1, 7, UOp.variable("tokens", 1, 16).bind(5)])
def test_route(tokens):
    rng = np.random.default_rng(14)
    scores = rng.standard_normal((16, 128)).astype(np.float32)
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    ids, weights = (t.numpy()[: n * 8] for t in kernels.route(Tensor(scores)[:tokens], 8))
    best = np.argsort(-scores[:n], -1, kind="stable")[:, :8]
    np.testing.assert_array_equal(ids.reshape(n, 8), best)
    top = np.take_along_axis(scores[:n], best, -1)
    p = np.exp(top - top[:, :1])
    np.testing.assert_allclose(weights.reshape(n, 8), p / p.sum(-1, keepdims=True), rtol=1e-6)


def test_route_distinct():
    # distinct experts, though a token scores fewer than it takes: the rest, of -inf or NaN, with
    # no weight
    scores = np.full((2, 128), -np.inf, dtype=np.float32)
    scores[:, [5, 70]] = [1.0, 2.0]
    scores[1, 9] = np.nan
    ids, weights = (t.numpy().reshape(2, 4) for t in kernels.route(Tensor(scores), 4))
    assert [len(set(row)) for row in ids] == [4, 4] and (ids[:, :2] == [70, 5]).all()
    p = np.exp([0.0, -1.0, -np.inf, -np.inf])
    np.testing.assert_allclose(weights, [p / p.sum()] * 2, rtol=1e-6)


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


# up to MATVEC_TOKENS tokens take the matrix-vector kernels, more the tensor cores, as do a bound
# number of up to 16, padded to 16 rows; favored experts get more than a tile of tokens
@pytest.mark.parametrize(
    "tokens", [1, 3, 70, UOp.variable("tokens", 1, 128).bind(37), 128, SHORT[1]]
)
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
    assert kernels.supports_mixture(x_t, gate, up, down)
    got = kernels.mixture(x_t, scores_t, gate, up, down, used, (Tensor(weight), 1e-5))
    got = got.pad_to((1, 128, dim)).numpy()[0, :n]

    def expert(e, row):
        g, u = (reference_matmul(row, b[e], Q4_K) for b in (gate_blocks, up_blocks))
        return reference_matmul(glu("silu", g, u).astype(np.float32), down_blocks[e], Q6_K)

    normed = rms_norm(x[0, :n], weight, 1e-5)
    assert_close(got, x[0, :n] + expected_mixture(normed, scores[0, :n], used, expert), 2e-3)


@pytest.mark.parametrize("residual", [True, False])
def test_mixture_live(residual):
    # a decode step's 4 rows, the last padding: the first 3 as alone, and the padding's x, or 0
    rng = np.random.default_rng(17)
    experts, used, dim, hidden = 32, 4, 512, 768
    gate, up = (random_experts(Q4_K, experts, hidden, dim, rng)[0] for _ in "gu")
    down = random_experts(Q6_K, experts, dim, hidden, rng)[0]
    x = Tensor((rng.standard_normal((1, 4, dim)) * 3).astype(np.float32)).realize()
    scores = Tensor(rng.standard_normal((1, 4, experts)).astype(np.float32)).realize()
    norm, live = (
        (Tensor(rng.uniform(0.5, 1.5, dim).astype(np.float32)), 1e-5),
        UOp.variable("live", 1, 4),
    )
    args = (gate, up, down, used, norm, "silu", None, residual)
    got = kernels.mixture(x, scores, *args, live=live.bind(3)).numpy()[0]
    alone = kernels.mixture(x[:, :3], scores[:, :3], *args).numpy()[0]
    np.testing.assert_allclose(got[:3], alone, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(got[3], x.numpy()[0, 3] if residual else 0)


# Gemma 4: gate and up in one stack, GELU, a scale per expert and no residual, for one token on
# the matrix-vector kernels and for more on tensor cores; rows of 192 weights quantize in a ragged
# turn
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
    assert kernels.supports_mixture(x_t, gate_up, None, down)
    norm = (Tensor(weight), 1e-5)
    args = (used, norm, "gelu", Tensor(scales), False)
    got = kernels.mixture(x_t, scores_t, gate_up, None, down, *args)
    got = got.pad_to((1, 128, dim)).numpy()[0, :n]

    def expert(e, row):
        g, u = np.split(reference_matmul(row, gate_up_blocks[e], Q4_K), 2, axis=-1)
        h = glu("gelu", g, u).astype(np.float32)
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
    for mode in ("auto", "ref"):
        monkeypatch.setenv("LEAT_KERNELS", mode)
        parts_ = [(y[:, :tokens], w) for y, w in zip(ys, ws, strict=False)]
        out = ops.add_normed(x[:, :tokens], parts_, ws[-1] if normed else None, 1e-6, scale)
        results.append(out.pad_to((1, 8, dim)).numpy()[0, :n])
    assert kernels.supports_add_normed(x)
    np.testing.assert_allclose(results[0], results[1], rtol=1e-5, atol=1e-5)


# ******** ops on the kernels ********


@pytest.mark.parametrize("tokens", [1, 80, 5])
def test_norm_and_residual(tokens):
    if tokens > MATVEC_TOKENS and not MATRIX_CORES:
        pytest.skip("many tokens take the tensor-core kernels")
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
# divide them, and 5 in tiles of 16 tokens, of 64 rows for Q6_K: Gemma 4's MLP, of GELU and Q5_0
# rows of 2112, without the residual
@pytest.mark.parametrize(
    "tokens, hidden, gate_type, down_type, kind",
    [(1, 4096, Q4_K, Q6_K, "silu"), (1, 1024, Q4_K, Q6_K, "silu"), (70, 4096, Q4_K, Q6_K, "silu"),
     (70, 1024, Q4_K, Q6_K, "silu"), (5, 4096, Q4_K, Q6_K, "silu"), (5, 1024, Q6_K, Q4_K, "silu"),
     (1, 2112, Q4_K, Q5_0, "gelu"), (70, 2112, Q4_K, Q5_0, "gelu"), (5, 2112, Q4_K, Q5_0, "gelu"),
     (70, 1024, Q8_0, Q8_0, "oai")],
)  # fmt: skip
def test_feed_forward(tokens, hidden, gate_type, down_type, kind):
    if tokens > MATVEC_TOKENS and not MATRIX_CORES:
        pytest.skip("many tokens take the tensor-core kernels")
    rng = np.random.default_rng(12)
    dim = 2048
    (gate, gate_blocks), (up, up_blocks) = (
        random_matrix(gate_type, hidden, dim, rng) for _ in "gu"
    )
    down, down_blocks = random_matrix(down_type, dim, hidden, rng)
    x = (rng.standard_normal((1, tokens, dim)) * 3).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, dim).astype(np.float32)
    norm = (Tensor(weight), 1e-5)
    residual = kind != "gelu"
    got = ops.feed_forward(Tensor(x), gate, up, down, norm, kind, residual=residual).numpy()[0]
    normed = rms_norm(x[0], weight, 1e-5)
    g, u = (reference_matmul(normed, b, gate_type) for b in (gate_blocks, up_blocks))
    hidden_ = glu(kind, g, u).astype(np.float32)
    expected = reference_matmul(hidden_, down_blocks, down_type) + (x[0] if residual else 0)
    # f32 rounding inside the kernels may move a few activations across a quantization step
    assert_close(got, expected, 2e-3)


@pytest.mark.parametrize("tokens", [1, MATVEC_TOKENS])
def test_few_tokens_take_matvec(monkeypatch, tokens):
    # the matrix kernels take a few tokens too, several times slower than the matrix-vector
    # kernels, which read the weights once for all of them
    def fail(*args, **kwargs):
        raise AssertionError("a few tokens went to a matrix kernel")

    monkeypatch.setattr(kernels, "matmuls", fail)
    monkeypatch.setattr(kernels, "feed_forward", fail)
    rng = np.random.default_rng(13)
    ws = [random_matrix(Q4_K, 2048, 2048, rng)[0] for _ in range(3)]
    x = Tensor(rng.standard_normal((1, tokens, 2048)).astype(np.float32))
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
    q: np.ndarray, cache: np.ndarray, start: int, window: int = 0, scale: float | None = None,
    sinks: np.ndarray | None = None, slot: int = SLOT,
) -> np.ndarray:  # fmt: skip
    # q (heads, T, dim) at positions start.. against a slot, causally and within the window if
    # given, with a sink per head if given, in f64: (T, heads, dim)
    heads, tokens, dim = q.shape
    group = heads // cache.shape[2]
    k, v = (cache[i, slot, :, : start + tokens].astype(np.float64) for i in range(2))
    back = start + np.arange(tokens)[:, None] - np.arange(start + tokens)
    hidden = (back < 0) | (back >= window) if window else back < 0
    out = np.empty((tokens, heads, dim))
    for h in range(heads):
        scores = q[h] @ k[h // group].T * (dim**-0.5 if scale is None else scale)
        scores = np.where(hidden, -np.inf, scores)
        sink = -np.inf if sinks is None else sinks[h]
        top = np.maximum(scores.max(-1, keepdims=True), sink)
        p = np.exp(scores - top)
        out[:, h] = (p / (p.sum(-1, keepdims=True) + np.exp(sink - top))) @ v[h // group]
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
    assert kernels.supports_attention(q_t, cache_t)
    got = kernels.attention(q_t, cache_t, [slot(symbolic)], [valid], 128**-0.5).numpy()
    got = got.reshape(-1, 128)
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
    assert kernels.supports_attention(q_t, cache_t)
    got = kernels.attention(q_t, cache_t, [slot(symbolic)], [valid], 128**-0.5).numpy()
    got = got.reshape(-1, 128)
    expected = reference_attention(q[0], cache, length - 1)[0]
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


# rows of a decode step: each a token in a slot of its own, or two in one, at lengths that take
# from one chunk to more chunks than blocks, Gemma 4's window and sinks too
@pytest.mark.parametrize("window, sinks", [(0, False), (1024, True)])
@pytest.mark.parametrize("symbolic", [False, True])
def test_attention_rows(window, sinks, symbolic):
    rows = [(0, 1), (2, 70), (1, 4000), (2, 3000)]
    rng = np.random.default_rng(len(rows) + window)
    cache = rng.standard_normal((2, SLOTS, 8, 4096, 128)).astype(np.float16)
    q = rng.standard_normal((1, 32, len(rows), 128)).astype(np.float32) * 0.3
    sink = rng.standard_normal(32).astype(np.float32) if sinks else None
    slots: list[int | UOp] = [s for s, _ in rows]
    lengths: list[int | UOp] = [n for _, n in rows]
    if symbolic:
        slots = [UOp.variable(f"slot{i}", 0, SLOTS - 1).bind(s) for i, (s, _) in enumerate(rows)]
        lengths = [
            UOp.variable(f"pos{i}", 0, 4095).bind(n - 1) + 1 for i, (_, n) in enumerate(rows)
        ]
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    assert kernels.supports_attention(q_t, cache_t)
    sink_t = None if sink is None else Tensor(sink)
    got = kernels.attention(q_t, cache_t, slots, lengths, 0.1, window, sink_t).numpy()[0]
    for t, (s, n) in enumerate(rows):
        expected = reference_attention(q[0, :, t : t + 1], cache, n - 1, window, 0.1, sink, s)
        np.testing.assert_allclose(got[t], expected.reshape(-1), rtol=2e-3, atol=2e-3)


def chunk_len(tokens: int) -> UOp:
    # a bound number of query tokens, as while prefilling; up to 16, bound to 16, they are one tile
    # of queries, whose keys blocks split
    return UOp.variable("chunk_len", 1, 16 if tokens <= 16 else 512).bind(tokens)


# Llama 3.1 8B's groups of 4 query heads, and Qwen2.5 7B's of 7, which do not divide a tile's
# loads, and Phi-3 mini's heads of 96, whose split keys the combine kernel takes 32 dims at a time
@pytest.mark.parametrize(
    "tokens, start", [(37, 0), (64, 0), (100, 300), (512, 3584), (5, 0), (16, 2000)]
)
@pytest.mark.parametrize("heads, kv_heads, dim", [(32, 8, 128), (28, 4, 128), (32, 32, 96)])
@pytest.mark.parametrize("symbolic", [False, True])
@matrix_cores
def test_flash_attention(tokens, start, heads, kv_heads, dim, symbolic):
    rng = np.random.default_rng(tokens + start)
    cache = rng.standard_normal((2, SLOTS, kv_heads, 4096, dim)).astype(np.float16)
    q = rng.standard_normal((1, heads, 512, dim)).astype(np.float32)
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    if symbolic:  # as while prefilling: a bound start and number of tokens
        pos = UOp.variable("start_pos", 0, 4095).bind(start)
        q_t = q_t[:, :, : chunk_len(tokens)]
    else:
        pos, q_t = start, q_t[:, :, :tokens]
    assert kernels.supports_flash_attention(q_t, cache_t)
    got = kernels.flash_attention(q_t, cache_t, slot(symbolic), pos, dim**-0.5)
    got = got.pad_to((1, 512, heads * dim)).numpy()[0, :tokens]
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
    assert kernels.supports_attention(q_t, cache_t)
    got = kernels.attention(q_t, cache_t, [slot(symbolic)], [valid], 1.0, window).numpy()
    got = got.reshape(-1, dim)
    expected = reference_attention(q[0], cache, length - 1, window, 1.0)[0]
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("tokens, start", [(37, 0), (100, 1000), (512, 3584), (11, 1500)])
@pytest.mark.parametrize("dim, window", [(256, 1024), (128, 100)])
@pytest.mark.parametrize("symbolic", [False, True])
@matrix_cores
def test_flash_attention_window(tokens, start, dim, window, symbolic):
    rng = np.random.default_rng(tokens + start + dim)
    cache = rng.standard_normal((2, SLOTS, 8, 4096, dim)).astype(np.float16)
    q = rng.standard_normal((1, 16, 512, dim)).astype(np.float32) * 0.2
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    if symbolic:
        pos = UOp.variable("start_pos", 0, 4095).bind(start)
        q_t = q_t[:, :, : chunk_len(tokens)]
    else:
        pos, q_t = start, q_t[:, :, :tokens]
    assert kernels.supports_flash_attention(q_t, cache_t)
    got = kernels.flash_attention(q_t, cache_t, slot(symbolic), pos, 1.0, window)
    got = got.pad_to((1, 512, 16 * dim)).numpy()[0, :tokens]
    expected = reference_attention(q[0, :, :tokens], cache, start, window, 1.0)
    np.testing.assert_allclose(got.reshape(expected.shape), expected, rtol=3e-3, atol=3e-3)


# llama: adjacent pairs; qwen2: halves and biases; qwen3: halves and norms of q and k; gemma4: v
# normed too, two kv heads; gemma3 1B: a single kv head
@pytest.mark.parametrize(
    "halves, biased, normed, v_norm, kv_heads",
    [
        (False, False, False, False, 8),
        (True, True, False, False, 8),
        (True, False, True, False, 8),
        (True, False, True, True, 2),
        (True, False, True, False, 1),
    ],
)
@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("rotated", [128, 96, 0])  # all, Phi-4-mini's 3/4, and SmolLM3's NoPE
def test_rotate(monkeypatch, halves, biased, normed, v_norm, kv_heads, symbolic, rotated):
    rng = np.random.default_rng(18)
    dim, pos = 128, 300
    q, k, v = (
        Tensor(rng.standard_normal((1, 1, h, dim)).astype(np.float32)).realize()
        for h in (32, kv_heads, kv_heads)
    )
    angles = rng.uniform(0, 6, (512, rotated // 2)).astype(np.float32)
    tables = (Tensor(np.cos(angles)).realize(), Tensor(np.sin(angles)).realize())
    rope = (tables, rotated) if rotated else None
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
    for mode in ("auto", "ref"):
        monkeypatch.setenv("LEAT_KERNELS", mode)
        cache = Tensor.zeros(2, SLOTS, kv_heads, 512, dim, dtype=dtypes.half).contiguous().realize()
        assert kernels.supports_rotate(q, cache)
        spans = [ops.Span(slot(symbolic), start)]
        out, cache = ops.rotate(q, k, v, cache, spans, rope, halves, biases, norms, v_norm, 1e-6)
        Tensor.realize(out, cache)  # in one schedule, as in the model: alone, either loses vars
        results.append((out.numpy(), cache.numpy()))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        results[0][1].astype(np.float32), results[1][1], rtol=1e-3, atol=1e-3
    )
    assert np.abs(results[0][1][:, SLOT, :, pos]).sum() > 0


@pytest.mark.parametrize("symbolic", [False, True])
def test_rotate_rows(monkeypatch, symbolic):
    # tokens of a decode step, each at a position of a slot of its own or two in one slot, with
    # Qwen2's halves and biases: the kernel as the reference ops, span by span
    rng = np.random.default_rng(19)
    rows, dim, kv_heads = [(0, 7), (2, 300), (1, 511), (2, 3)], 128, 8
    q, k, v = (
        Tensor(rng.standard_normal((1, len(rows), h, dim)).astype(np.float32)).realize()
        for h in (32, kv_heads, kv_heads)
    )
    angles = rng.uniform(0, 6, (512, dim // 2)).astype(np.float32)
    rope = ((Tensor(np.cos(angles)).realize(), Tensor(np.sin(angles)).realize()), dim)
    biases = tuple(Tensor(rng.standard_normal(h * dim).astype(np.float32)) for h in (32, 8, 8))
    spans = [ops.Span(s, p) for s, p in rows]
    if symbolic:
        spans = [
            ops.Span(UOp.variable(f"slot{i}", 0, SLOTS - 1).bind(s),
                     UOp.variable(f"pos{i}", 0, 511).bind(p))
            for i, (s, p) in enumerate(rows)
        ]  # fmt: skip
    results = []
    for mode in ("auto", "ref"):
        monkeypatch.setenv("LEAT_KERNELS", mode)
        cache = Tensor.zeros(2, SLOTS, kv_heads, 512, dim, dtype=dtypes.half).contiguous().realize()
        out, cache = ops.rotate(q, k, v, cache, spans, rope, True, biases, None, False, 1e-6)
        Tensor.realize(out, cache)
        results.append((out.numpy(), cache.numpy().astype(np.float32)))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-3, atol=1e-3)
    assert all(np.abs(results[0][1][:, s, :, p]).sum() > 0 for s, p in rows)


# heads too wide for registers, Gemma 4's of 512: blocks take parts of their outputs
@pytest.mark.parametrize(
    "tokens, start, window",
    [(37, 0, 0), (100, 1000, 0), (512, 1500, 1024), (7, 1000, 0), (3, 1900, 1024)],
)
@pytest.mark.parametrize("symbolic", [False, True])
@matrix_cores
def test_flash_attention_wide(tokens, start, window, symbolic):
    rng = np.random.default_rng(tokens + start)
    cache = rng.standard_normal((2, SLOTS, 2, 2048, 512)).astype(np.float16) * np.float16(0.2)
    q = rng.standard_normal((1, 16, 512, 512)).astype(np.float32) * 0.2
    q_t, cache_t = Tensor(q).realize(), Tensor(cache).realize()
    if symbolic:
        pos = UOp.variable("start_pos", 0, 2047).bind(start)
        q_t = q_t[:, :, : chunk_len(tokens)]
    else:
        pos, q_t = start, q_t[:, :, :tokens]
    assert kernels.supports_flash_attention(q_t, cache_t)
    got = kernels.flash_attention(q_t, cache_t, slot(symbolic), pos, 1.0, window)
    got = got.pad_to((1, 512, 16 * 512)).numpy()[0, :tokens]
    expected = reference_attention(q[0, :, :tokens], cache, start, window, 1.0)
    np.testing.assert_allclose(got.reshape(expected.shape), expected, rtol=3e-3, atol=3e-3)


# ******** argmax ********


@pytest.mark.parametrize("rows, n", [(1, 128256), (3, 1000), (2, 33), (1, 1)])
def test_argmax(rows, n):
    rng = np.random.default_rng(n)
    x = rng.integers(-50, 50, (rows, n)).astype(np.float32)  # many ties: the first one must win
    x[:, ::7] = -np.inf
    assert kernels.supports_argmax(Tensor(x))
    np.testing.assert_array_equal(ops.argmax(Tensor(x)).numpy(), x.argmax(-1, keepdims=True))


# ******** cutoff ********


@pytest.mark.parametrize("rows, n", [(1, 248320), (5, 32000), (2, 300)])
def test_cutoff(rows, n):
    # within a step below the exact cuts, of top_k alone, top_p alone or both; top_k 0 keeps
    # every token, which the kernel takes as the step of its grid RANGE below the top, and top_p
    # 0 the top one
    rng = np.random.default_rng(n)
    x = (rng.standard_normal((rows, n)) * 2.5).astype(np.float32)
    x[:, : n // 100] += 9  # a head of likely tokens
    top_k, top_p = np.array([20, 0, 5, 1, 0][:rows]), np.array([0.95, 0.9, 1.0, 0.5, 0.0][:rows])
    options = (Tensor(o.reshape(-1, 1).astype(np.float32)) for o in (top_k, top_p))
    assert kernels.supports_cutoff(Tensor(x))
    got = np.hstack([t.numpy() for t in kernels.cutoff(Tensor(x), *options)])
    expected = np.array([cuts(*row) for row in zip(x, top_k, top_p, strict=True)])
    step = RANGE / BINS
    expected[:, 1] = np.where(top_k > 0, expected[:, 1], expected[:, 0] - RANGE)
    above = np.where(top_k > 0, 1e-4, step).reshape(-1, 1)
    assert ((got <= expected + above) & (got >= expected - 1.5 * step)).all(), got - expected


# gpt-oss's attention sinks: 64 heads of 64 over 8 kv heads, for one token, a tile of few, and
# many, in a window of 128 positions or not
@pytest.mark.parametrize("tokens, start", [(1, 70), (1, 1000), (11, 300), (100, 1000)])
@pytest.mark.parametrize("window", [0, 128])
def test_attention_sinks(tokens, start, window):
    rng = np.random.default_rng(tokens + start + window)
    cache = rng.standard_normal((2, SLOTS, 8, 2048, 64)).astype(np.float16)
    q = rng.standard_normal((1, 64, tokens, 64)).astype(np.float32)
    sinks = rng.uniform(-2, 4, 64).astype(np.float32)
    q_t, cache_t, sinks_t = Tensor(q).realize(), Tensor(cache).realize(), Tensor(sinks)
    if tokens > 1 and not MATRIX_CORES:
        pytest.skip("FlashAttention is on tensor cores")
    if tokens == 1:
        assert kernels.supports_attention(q_t, cache_t)
        got = kernels.attention(q_t, cache_t, [SLOT], [start + 1], 0.125, window, sinks_t)
    else:
        assert kernels.supports_flash_attention(q_t, cache_t)
        got = kernels.flash_attention(q_t, cache_t, SLOT, start, 0.125, window, sinks_t)
    got = got.numpy().reshape(tokens, 64, 64)
    expected = reference_attention(q[0], cache, start, window, 0.125, sinks)
    np.testing.assert_allclose(got, expected, rtol=3e-3, atol=3e-3)


# gpt-oss: experts with biases of gate, up and down, and the clamped SwiGLU; of MXFP4 for one
# token and a few on the matrix-vector kernels, of Q8_0 for more on tensor cores too
@pytest.mark.parametrize(
    "ggml_type, tokens",
    [
        (MXFP4, 1),
        (MXFP4, 5),
        (MXFP4, 70),
        (MXFP4, UOp.variable("tokens", 1, 128).bind(37)),
        (Q8_0, 1),
        (Q8_0, 70),
    ],
)
def test_mixture_biases(ggml_type, tokens):
    rng = np.random.default_rng(17)
    experts, used, dim, hidden = 32, 4, 512, 768
    (gate, gate_blocks), (up, up_blocks) = (
        random_experts(ggml_type, experts, hidden, dim, rng) for _ in "gu"
    )
    down, down_blocks = random_experts(ggml_type, experts, dim, hidden, rng)
    biases = [
        rng.standard_normal((experts, rows)).astype(np.float32) for rows in (hidden, hidden, dim)
    ]
    x = (rng.standard_normal((1, 128, dim)) * 3).astype(np.float32)
    scores = rng.standard_normal((1, 128, experts)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, dim).astype(np.float32)
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    x_t, scores_t = Tensor(x)[:, :tokens], Tensor(scores)[:, :tokens]
    assert kernels.supports_mixture(x_t, gate, up, down)
    norm = (Tensor(weight), 1e-5)
    biases_t = tuple(Tensor(b) for b in biases)
    got = kernels.mixture(x_t, scores_t, gate, up, down, used, norm, "oai", None, True, biases_t)
    got = got.pad_to((1, 128, dim)).numpy()[0, :n]

    def expert(e, row):
        g, u = (reference_matmul(row, b[e], ggml_type) + bias[e] for b, bias in
                zip((gate_blocks, up_blocks), biases, strict=False))  # fmt: skip
        h = glu("oai", g, u).astype(np.float32)
        return reference_matmul(h, down_blocks[e], ggml_type) + biases[2][e]

    normed = rms_norm(x[0, :n], weight, 1e-5)
    assert_close(got, x[0, :n] + expected_mixture(normed, scores[0, :n], used, expert), 3e-3)


# ******** Gated DeltaNet ********

# Qwen3.6 35B A3B's: 16 heads of queries and keys, 32 of values, each of 128, and a convolution
# over 4 tokens
DN_K_HEADS, DN_HEADS, DN_DIMS, DN_WIDTH, DN_SLOTS = 16, 32, 128, 4, 3
DN_CHANNELS = (2 * DN_K_HEADS + DN_HEADS) * DN_DIMS


def delta_net_inputs(tokens: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    def normal(*shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return {
        "mixed": normal(tokens, DN_CHANNELS), "z": normal(tokens, DN_HEADS * DN_DIMS),
        "alpha": normal(tokens, DN_HEADS), "beta": normal(tokens, DN_HEADS),
        "conv": normal(DN_CHANNELS, DN_WIDTH, scale=0.5), "bias": normal(DN_HEADS),
        "a": -rng.uniform(0.1, 2.0, DN_HEADS).astype(np.float32),
        "norm": rng.uniform(0.5, 1.5, DN_DIMS).astype(np.float32),
    }  # fmt: skip


def reference_delta_net(x, conv_state, state, eps):
    # one sequence's tokens from its states, in f64: the outputs, then the states after
    T, qk = len(x["mixed"]), DN_K_HEADS * DN_DIMS
    inputs = np.concatenate([conv_state, x["mixed"]]).astype(np.float64)
    conved = np.stack([(inputs[t : t + DN_WIDTH] * x["conv"].T).sum(0) for t in range(T)])
    q, k, v = np.split(conved / (1 + np.exp(-conved)), [qk, 2 * qk], -1)

    def unit(z):
        z = z.reshape(T, DN_K_HEADS, DN_DIMS)
        return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), eps)

    q, k, v = unit(q) / np.sqrt(DN_DIMS), unit(k), v.reshape(T, DN_HEADS, DN_DIMS)
    decay = np.exp(x["a"] * np.logaddexp(0, x["alpha"] + x["bias"]))
    share = 1 / (1 + np.exp(-x["beta"].astype(np.float64)))
    s, out = state.astype(np.float64), np.zeros((T, DN_HEADS, DN_DIMS))
    for t in range(T):
        for h in range(DN_HEADS):
            key, query = k[t, h % DN_K_HEADS], q[t, h % DN_K_HEADS]
            s[h] *= decay[t, h]
            s[h] += np.outer(key, share[t, h] * (v[t, h] - s[h].T @ key))
            out[t, h] = s[h].T @ query
    z = x["z"].reshape(T, DN_HEADS, DN_DIMS)
    gated = (
        out / np.sqrt((out * out).mean(-1, keepdims=True) + eps) * x["norm"] * z / (1 + np.exp(-z))
    )
    return gated.reshape(T, -1), inputs[T:], s


def delta_net_states(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    conv_state = rng.standard_normal((DN_SLOTS, DN_WIDTH - 1, DN_CHANNELS)).astype(np.float32)
    state = (rng.standard_normal((DN_SLOTS, DN_HEADS, DN_DIMS, DN_DIMS)) * 0.1).astype(np.float32)
    return conv_state, state


def run_delta_net(x, rows, tokens, states):
    # ops.delta_net() on the kernel, rows (slot, start) of `tokens` tokens each, or of a bound
    # number for one row
    count = tokens if len(rows) == 1 else len(rows) * tokens

    def tensor(name):
        t = Tensor(x[name])
        return t.unsqueeze(0)[:, :count] if name in ("mixed", "z", "gates") else t

    spans = [ops.Span(slot, start, tokens) for slot, start in rows]
    x = x | {"gates": np.concatenate([x["alpha"], x["beta"]], -1)}
    mixed, z, gates, conv = (tensor(n) for n in ("mixed", "z", "gates", "conv"))
    assert kernels.supports_delta_net(mixed, states[1])
    decay, norm = (tensor("a"), tensor("bias")), (tensor("norm"), 1e-6)
    return ops.delta_net(mixed, z, gates, conv, decay, norm, states, spans)


@pytest.mark.parametrize("tokens", [1, 12, 37, UOp.variable("tokens", 1, 64).bind(37)])
@pytest.mark.parametrize("start", [0, 5])
def test_delta_net(tokens, start):
    # a sequence's tokens from its slot's states, or from zero ones at position 0; the states
    # after in its slot, the others' as they were. 12 leave the convolution's second tile 4.
    rng = np.random.default_rng(23)
    n = tokens if isinstance(tokens, int) else tokens.unbind()[1]
    x, (conv_state, state) = delta_net_inputs(64, rng), delta_net_states(rng)
    states = (Tensor(conv_state).contiguous().realize(), Tensor(state).contiguous().realize())
    out = run_delta_net(x, [(1, start)], tokens, states).pad_to((1, 64, DN_HEADS * DN_DIMS))
    out = out.numpy()[0, :n]
    first = {k: v[:n] if v.shape[0] == 64 else v for k, v in x.items()}
    kept = start > 0
    expected, conv_after, after = reference_delta_net(
        first, conv_state[1] * kept, state[1] * kept, 1e-6
    )
    assert_close(out, expected, 2e-4)
    got_conv, got = states[0].numpy(), states[1].numpy()
    np.testing.assert_allclose(got_conv[1], conv_after, rtol=1e-6, atol=1e-6)
    assert_close(got[1], after, 2e-4)
    for other in (0, 2):
        np.testing.assert_array_equal(got_conv[other], conv_state[other])
        np.testing.assert_array_equal(got[other], state[other])


def test_delta_net_rows():
    # a token of each row from its own slot, one row from position 0
    rng = np.random.default_rng(29)
    x, (conv_state, state) = delta_net_inputs(3, rng), delta_net_states(rng)
    states = (Tensor(conv_state).contiguous().realize(), Tensor(state).contiguous().realize())
    rows = [(2, 4), (0, 0), (1, 9)]
    out = run_delta_net(x, rows, 1, states).numpy()[0]
    for r, (slot, start) in enumerate(rows):
        one = {k: v[r : r + 1] if v.shape[0] == 3 and k != "conv" else v for k, v in x.items()}
        kept = start > 0
        expected, conv_after, after = reference_delta_net(
            one, conv_state[slot] * kept, state[slot] * kept, 1e-6
        )
        assert_close(out[r : r + 1], expected, 2e-4)
        np.testing.assert_allclose(states[0].numpy()[slot], conv_after, rtol=1e-6, atol=1e-6)
        assert_close(states[1].numpy()[slot], after, 2e-4)
