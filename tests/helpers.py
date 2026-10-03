# Test data shared across test files: quantized blocks, tokenizer metadata and tiny models.

from pathlib import Path

import gguf
import numpy as np
from gguf.quants import dequantize

from leat.quant import BLOCK, GGMLType
from leat.tokenizer import _BYTE_CHAR, CONTROL, NORMAL, USER_DEFINED, Tokenizer

# byte offsets of each block's f16 scales; random bytes there would be inf/nan
F16_FIELDS = {
    GGMLType.Q8_0: (0,),
    GGMLType.Q4_K: (0, 2),
    GGMLType.Q5_K: (0, 2),
    GGMLType.Q6_K: (208,),
}


def random_blocks(
    ggml_type: GGMLType, n: int, rng: np.random.Generator, scale: float = 1.0
) -> np.ndarray:
    blocks = rng.integers(0, 256, (n, BLOCK[ggml_type][1]), dtype=np.uint8)
    for offset in F16_FIELDS[ggml_type]:
        d = rng.uniform(-scale, scale, (n, 1)).astype(np.float16)
        blocks[:, offset : offset + 2] = d.view(np.uint8)
    return blocks


def tiny_metadata(**overrides) -> dict:
    tokens = [*_BYTE_CHAR.values(), "bc", "ab", "he", "ll", "hell", "<s>", "<|eot|>", "<user>"]
    types = [NORMAL] * (len(tokens) - 3) + [CONTROL, CONTROL, USER_DEFINED]
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "llama-bpe",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": ["b c", "a b", "h e", "l l", "he ll"],
        "tokenizer.ggml.bos_token_id": tokens.index("<s>"),
        "tokenizer.ggml.eos_token_id": tokens.index("<|eot|>"),
    }
    return metadata | overrides


def ids(tok: Tokenizer, *pieces: str) -> list[int]:
    return [tok._vocab[p] if p in tok._vocab else tok._special[p] for p in pieces]


V, D, HIDDEN, HEADS, KV_HEADS, LAYERS, CONTEXT = 300, 256, 512, 4, 2, 2, 64
HEAD_DIM = D // HEADS
EXPERTS, USED, EXPERT_HIDDEN = 4, 2, 256  # qwen3moe's MLPs
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
    "ffn_gate_inp": ((EXPERTS, D), GGMLType.F32, 0.05),
    "ffn_gate_exps": ((EXPERTS, EXPERT_HIDDEN, D), GGMLType.Q4_K, 2e-4),
    "ffn_up_exps": ((EXPERTS, EXPERT_HIDDEN, D), GGMLType.Q5_K, 2e-4),
    "ffn_down_exps": ((EXPERTS, D, EXPERT_HIDDEN), GGMLType.Q6_K, 5e-5),
}
ATTENTION = ("attn_q", "attn_k", "attn_v", "attn_output")
MLP = ("ffn_gate", "ffn_up", "ffn_down")
MOE = ("ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def write_tiny_model(path: Path, arch: str = "llama") -> dict[str, np.ndarray]:
    # a random GGUF of a supported architecture; returns its weights as decoded independently by
    # gguf-py. llama has rope frequency factors, qwen3 RMSNorms of q and k, and qwen3moe also
    # a mixture of experts for MLP.
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(path, arch=arch)
    for key, value in [("block_count", LAYERS), ("context_length", CONTEXT),
                       ("embedding_length", D), ("feed_forward_length", HIDDEN),
                       ("attention.head_count", HEADS), ("attention.head_count_kv", KV_HEADS),
                       ("rope.dimension_count", HEAD_DIM)]:  # fmt: skip
        w.add_uint32(f"{arch}.{key}", value)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_float32(f"{arch}.rope.freq_base", 10000.0)
    if arch == "qwen3moe":
        w.add_uint32(f"{arch}.expert_count", EXPERTS)
        w.add_uint32(f"{arch}.expert_used_count", USED)
        w.add_uint32(f"{arch}.expert_feed_forward_length", EXPERT_HIDDEN)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    w.add_array("tokenizer.ggml.tokens", [*_BYTE_CHAR.values()] + [f"t{i}" for i in range(V - 256)])
    w.add_array("tokenizer.ggml.token_type", [1] * V)
    w.add_array("tokenizer.ggml.merges", [])
    w.add_chat_template("{{ messages[-1]['content'] }}")

    weights = {}

    def add(name, shape, ggml_type, scale):
        if ggml_type == GGMLType.F32:
            weights[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
            return w.add_tensor(name, weights[name])
        blocks = random_blocks(ggml_type, np.prod(shape) // BLOCK[ggml_type][0], rng, scale)
        blocks = blocks.reshape(*shape[:-1], -1)
        weights[name] = dequantize(blocks, gguf.GGMLQuantizationType(ggml_type)).reshape(shape)
        w.add_tensor(name, blocks, raw_dtype=gguf.GGMLQuantizationType(ggml_type))

    add("token_embd.weight", *TENSORS["token_embd.weight"])
    add("output.weight", *TENSORS["output.weight"])
    norms = {"output_norm.weight": D}
    if arch == "llama":
        norms["rope_freqs.weight"] = HEAD_DIM // 2
    for i in range(LAYERS):
        norms |= {f"blk.{i}.attn_norm.weight": D, f"blk.{i}.ffn_norm.weight": D}
        if arch != "llama":
            norms |= {f"blk.{i}.attn_{x}_norm.weight": HEAD_DIM for x in "qk"}
        for name in ATTENTION + (MOE if arch == "qwen3moe" else MLP):
            add(f"blk.{i}.{name}.weight", *TENSORS[name])
    for name, n in norms.items():
        weights[name] = rng.uniform(1.0, 4.0 if "rope" in name else 1.5, n).astype(np.float32)
        w.add_tensor(name, weights[name])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return weights


def reference_logits(
    w: dict[str, np.ndarray], tokens: list[int], arch: str = "llama"
) -> np.ndarray:
    # an independent float64 model, with keys and values rounded to f16 like leat's cache
    def norm(x, weight):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * weight

    T = len(tokens)
    freqs = 10000.0 ** (-np.arange(0, HEAD_DIM, 2) / HEAD_DIM) / w.get("rope_freqs.weight", 1.0)
    angles = np.arange(T)[:, None, None] * freqs
    cos, sin = np.cos(angles), np.sin(angles)

    # llama rotates adjacent pairs of dimensions, qwen3 dimension i with i + HEAD_DIM / 2
    half = HEAD_DIM // 2
    first, second = (np.s_[0::2], np.s_[1::2]) if arch == "llama" else (np.s_[:half], np.s_[half:])

    def rope(z):
        out = np.empty_like(z)
        out[..., first] = z[..., first] * cos - z[..., second] * sin
        out[..., second] = z[..., first] * sin + z[..., second] * cos
        return out

    def mlp(h, gate, up, down):
        g = h @ gate.T
        return (g / (1 + np.exp(-g)) * (h @ up.T)) @ down.T

    def mixture(h, router, gate, up, down):
        # each token's USED best scoring experts, weighted by the softmax of their scores
        out = np.zeros_like(h)
        for t, scores in enumerate(h @ router.T):
            best = np.argsort(-scores)[:USED]
            p = np.exp(scores[best] - scores[best].max())
            for e, weight in zip(best, p / p.sum(), strict=True):
                out[t] += weight * mlp(h[t], gate[e], up[e], down[e])
        return out

    x = w["token_embd.weight"][tokens].astype(np.float64)
    causal = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(LAYERS):
        lw = {n: w[f"blk.{i}.{n}.weight"] for n in ATTENTION + (MOE if arch == "qwen3moe" else MLP)}
        h = norm(x, w[f"blk.{i}.attn_norm.weight"])
        q = (h @ lw["attn_q"].T).reshape(T, HEADS, HEAD_DIM)
        k = (h @ lw["attn_k"].T).reshape(T, KV_HEADS, HEAD_DIM)
        if arch != "llama":
            q = norm(q, w[f"blk.{i}.attn_q_norm.weight"])
            k = norm(k, w[f"blk.{i}.attn_k_norm.weight"])
        q, k = rope(q), rope(k).astype(np.float16)
        v = (h @ lw["attn_v"].T).reshape(T, KV_HEADS, HEAD_DIM).astype(np.float16)
        heads = []
        for hd in range(HEADS):
            kv = hd // (HEADS // KV_HEADS)
            scores = q[:, hd] @ k[:, kv].T.astype(np.float64) / np.sqrt(HEAD_DIM) + causal
            p = np.exp(scores - scores.max(-1, keepdims=True))
            heads.append((p / p.sum(-1, keepdims=True)) @ v[:, kv].astype(np.float64))
        x = x + np.concatenate(heads, -1) @ lw["attn_output"].T
        h = norm(x, w[f"blk.{i}.ffn_norm.weight"])
        x = x + (mixture if arch == "qwen3moe" else mlp)(h, *(lw[n] for n in lw if "ffn" in n))
    return norm(x, w["output_norm.weight"]) @ w["output.weight"].T
