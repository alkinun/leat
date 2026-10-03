import gguf
import numpy as np
import pytest
from gguf.quants import dequantize
from tinygrad import Tensor, dtypes

from leat.engine import Engine
from leat.gguf import GGUF
from leat.model import Config, Transformer
from leat.quant import BLOCK, GGMLType
from leat.tokenizer import _BYTE_CHAR
from tests.test_quant import random_blocks

V, D, HIDDEN, HEADS, KV_HEADS, LAYERS, CONTEXT = 300, 256, 512, 4, 2, 2, 64
HEAD_DIM = D // HEADS
# (shape, storage type, scale of the block f16 fields) chosen so activations stay O(1)
TENSORS = {
    "token_embd.weight": ((V, D), GGMLType.Q4_K, 2e-4),
    "output.weight": ((V, D), GGMLType.Q6_K, 5e-5),
    "attn_q": ((D, D), GGMLType.Q4_K, 2e-4),
    "attn_k": ((KV_HEADS * HEAD_DIM, D), GGMLType.Q6_K, 5e-5),
    "attn_v": ((KV_HEADS * HEAD_DIM, D), GGMLType.Q8_0, 1e-3),
    "attn_output": ((D, D), GGMLType.F32, 0.05),
    "ffn_gate": ((HIDDEN, D), GGMLType.Q4_K, 2e-4),
    "ffn_up": ((HIDDEN, D), GGMLType.Q5_K, 2e-4),
    "ffn_down": ((D, HIDDEN), GGMLType.Q6_K, 5e-5),
}


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A random llama GGUF and its weights decoded independently by gguf-py."""
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("model") / "tiny.gguf"
    w = gguf.GGUFWriter(path, arch="llama")
    for key, value in [("block_count", LAYERS), ("context_length", CONTEXT),
                       ("embedding_length", D), ("feed_forward_length", HIDDEN),
                       ("attention.head_count", HEADS), ("attention.head_count_kv", KV_HEADS),
                       ("rope.dimension_count", HEAD_DIM)]:  # fmt: skip
        w.add_uint32(f"llama.{key}", value)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_float32("llama.rope.freq_base", 10000.0)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    w.add_array("tokenizer.ggml.tokens", [*_BYTE_CHAR.values()] + [f"t{i}" for i in range(V - 256)])
    w.add_array("tokenizer.ggml.token_type", [1] * V)
    w.add_array("tokenizer.ggml.merges", [])

    weights = {}

    def add(name, shape, ggml_type, scale):
        if ggml_type == GGMLType.F32:
            weights[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
            return w.add_tensor(name, weights[name])
        elements, nbytes = BLOCK[ggml_type]
        blocks = random_blocks(ggml_type, shape[0] * shape[1] // elements, rng, scale)
        blocks = blocks.reshape(shape[0], -1)
        weights[name] = dequantize(blocks, gguf.GGMLQuantizationType(ggml_type)).reshape(shape)
        w.add_tensor(name, blocks, raw_dtype=gguf.GGMLQuantizationType(ggml_type))

    add("token_embd.weight", *TENSORS["token_embd.weight"])
    add("output.weight", *TENSORS["output.weight"])
    norms = {"output_norm.weight": D, "rope_freqs.weight": HEAD_DIM // 2}
    for i in range(LAYERS):
        norms |= {f"blk.{i}.attn_norm.weight": D, f"blk.{i}.ffn_norm.weight": D}
        for name in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            add(f"blk.{i}.{name}.weight", *TENSORS[name])
    for name, n in norms.items():
        weights[name] = rng.uniform(1.0, 4.0 if "rope" in name else 1.5, n).astype(np.float32)
        w.add_tensor(name, weights[name])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, weights


def reference_logits(w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    # an independent float64 llama, with keys and values rounded to f16 like leat's cache
    def norm(x, weight):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * weight

    T = len(tokens)
    freqs = 10000.0 ** (-np.arange(0, HEAD_DIM, 2) / HEAD_DIM) / w["rope_freqs.weight"]
    angles = np.arange(T)[:, None, None] * freqs
    cos, sin = np.cos(angles), np.sin(angles)

    def rope(z):
        out = np.empty_like(z)
        out[..., 0::2] = z[..., 0::2] * cos - z[..., 1::2] * sin
        out[..., 1::2] = z[..., 0::2] * sin + z[..., 1::2] * cos
        return out

    x = w["token_embd.weight"][tokens].astype(np.float64)
    causal = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(LAYERS):
        lw = {n: w[f"blk.{i}.{n}.weight"] for n in TENSORS if not n.endswith(".weight")}
        h = norm(x, w[f"blk.{i}.attn_norm.weight"])
        q = rope((h @ lw["attn_q"].T).reshape(T, HEADS, HEAD_DIM))
        k = rope((h @ lw["attn_k"].T).reshape(T, KV_HEADS, HEAD_DIM)).astype(np.float16)
        v = (h @ lw["attn_v"].T).reshape(T, KV_HEADS, HEAD_DIM).astype(np.float16)
        heads = []
        for hd in range(HEADS):
            kv = hd // (HEADS // KV_HEADS)
            scores = q[:, hd] @ k[:, kv].T.astype(np.float64) / np.sqrt(HEAD_DIM) + causal
            p = np.exp(scores - scores.max(-1, keepdims=True))
            heads.append((p / p.sum(-1, keepdims=True)) @ v[:, kv].astype(np.float64))
        x = x + np.concatenate(heads, -1) @ lw["attn_output"].T
        h = norm(x, w[f"blk.{i}.ffn_norm.weight"])
        gate = h @ lw["ffn_gate"].T
        x = x + (gate / (1 + np.exp(-gate)) * (h @ lw["ffn_up"].T)) @ lw["ffn_down"].T
    return norm(x, w["output_norm.weight"]) @ w["output.weight"].T


PROMPT = [5, 77, 120, 3, 299, 42, 8, 150, 61, 200, 9, 33]


def test_forward_matches_reference(tiny_model):
    path, weights = tiny_model
    f = GGUF.open(path)
    model = Transformer(Config.from_gguf(f.metadata), f.load(), CONTEXT)
    tokens = Tensor([PROMPT], dtype=dtypes.int32)
    logits = model.logits(model(tokens, 0)).numpy()[0]
    np.testing.assert_allclose(logits, reference_logits(weights, PROMPT), rtol=2e-3, atol=2e-3)


def test_generate_matches_reference(tiny_model):
    path, weights = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=5)  # chunks of 5, 5 and 2
    out = list(engine.generate(PROMPT, 8))
    # every generated token is the reference argmax given all tokens before it
    expected = reference_logits(weights, PROMPT + out)[len(PROMPT) - 1 :].argmax(-1)
    assert out == expected[: len(out)].tolist()

    # a prompt that extends the previous one reuses the cache: only the new tokens are prefilled
    longer = PROMPT + out[:3] + [7]
    again = list(engine.generate(longer, 4))
    assert again == list(Engine(path, max_context=CONTEXT, prefill_chunk=5).generate(longer, 4))


def test_generate_stops_at_context_end(tiny_model):
    engine = Engine(tiny_model[0], max_context=16, prefill_chunk=8)
    assert len(list(engine.generate(PROMPT, 100))) == 16 - len(PROMPT) + 1
