# Test data shared across test files: quantized blocks, tokenizer metadata and tiny models.

import functools
from pathlib import Path

import gguf
import numpy as np
from gguf.quants import dequantize

from leat.gguf import GGUF
from leat.quant import BLOCK, GGMLType
from leat.tokenizer import _BYTE_CHAR, CONTROL, NORMAL, USER_DEFINED, Tokenizer

# byte offsets of each block's f16 scales; random bytes there would be inf/nan
F16_FIELDS = {
    GGMLType.Q5_0: (0,),
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


def chat_template(model_path: Path) -> str:
    # the model's chat template, which says what its chats can hold: a system prompt, tools
    return GGUF.open(model_path).metadata.get("tokenizer.chat_template", "")


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
    # gguf-py. llama has rope frequency factors, qwen2 biases of q, k and v, qwen3 RMSNorms of q
    # and k, and qwen3moe also a mixture of experts for MLP.
    if arch.startswith("gemma4"):
        return _write_gemma4(path, experts=arch == "gemma4")
    w, weights, add = _writer(path, arch)
    for key, value in [("block_count", LAYERS),
                       ("embedding_length", D), ("feed_forward_length", HIDDEN),
                       ("attention.head_count", HEADS), ("attention.head_count_kv", KV_HEADS),
                       ("rope.dimension_count", HEAD_DIM)]:  # fmt: skip
        w.add_uint32(f"{arch}.{key}", value)
    w.add_float32(f"{arch}.rope.freq_base", 10000.0)
    if arch == "qwen3moe":
        w.add_uint32(f"{arch}.expert_count", EXPERTS)
        w.add_uint32(f"{arch}.expert_used_count", USED)
        w.add_uint32(f"{arch}.expert_feed_forward_length", EXPERT_HIDDEN)
    add("token_embd.weight", *TENSORS["token_embd.weight"])
    add("output.weight", *TENSORS["output.weight"])
    add("output_norm.weight", (D,))
    if arch == "llama":
        add("rope_freqs.weight", (HEAD_DIM // 2,), GGMLType.F32, 4.0)
    for i in range(LAYERS):
        qk = ("attn_q_norm", "attn_k_norm") if arch.startswith("qwen3") else ()
        for name in ("attn_norm", "ffn_norm", *qk):
            add(f"blk.{i}.{name}.weight", (HEAD_DIM if name in qk else D,))
        for name in ATTENTION + (MOE if arch == "qwen3moe" else MLP):
            add(f"blk.{i}.{name}.weight", *TENSORS[name])
        if arch == "qwen2":
            for name in ("attn_q", "attn_k", "attn_v"):
                add(f"blk.{i}.{name}.bias", (TENSORS[name][0][0],), GGMLType.F32, 2.0)
    _finish(w)
    return weights


def _writer(path: Path, arch: str):
    # a GGUF writer with the tiny tokenizer, its weights, and add(name, shape, type, scale), which
    # writes a random tensor and keeps it decoded; a norm weight by default, uniform from 1 to 1.5
    # (or to `scale` for F32 vectors), and else blocks whose f16 fields are up to `scale`
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(path, arch=arch)
    w.add_uint32(f"{arch}.context_length", CONTEXT)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    w.add_array("tokenizer.ggml.tokens", [*_BYTE_CHAR.values()] + [f"t{i}" for i in range(V - 256)])
    w.add_array("tokenizer.ggml.token_type", [1] * V)
    w.add_array("tokenizer.ggml.merges", [])
    w.add_chat_template("{{ prefix | default('') }}{{ messages[-1]['content'] }}")
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], ggml_type=GGMLType.F32, scale: float = 1.5) -> None:
        if ggml_type == GGMLType.F32:
            uniform = len(shape) == 1
            values = (
                rng.uniform(1.0, scale, shape) if uniform else rng.standard_normal(shape) * scale
            )
            weights[name] = values.astype(np.float32)
            return w.add_tensor(name, weights[name])
        blocks = random_blocks(ggml_type, np.prod(shape) // BLOCK[ggml_type][0], rng, scale)
        blocks = blocks.reshape(*shape[:-1], -1)
        weights[name] = dequantize(blocks, gguf.GGMLQuantizationType(ggml_type)).reshape(shape)
        w.add_tensor(name, blocks, raw_dtype=gguf.GGMLQuantizationType(ggml_type))

    return w, weights, add


def _finish(w: gguf.GGUFWriter) -> None:
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# Gemma 4: a sliding-window layer of 2 kv heads of 32, and a full-attention one of a kv head of
# 64 whose values are its keys; both with a shared MLP beside the experts, of gate and up stacked
G_WINDOW, G_DIMS, G_KV_HEADS, G_CAP = 4, (32, 64), (2, 1), 5.0


def _write_gemma4(path: Path, experts: bool) -> dict[str, np.ndarray]:
    # with experts beside each MLP, as Gemma 4 26B A4B, or without, as the dense models
    w, weights, add = _writer(path, "gemma4")
    a = "gemma4."
    for key, value in [("block_count", LAYERS), ("embedding_length", D),
                       ("feed_forward_length", HIDDEN), ("attention.head_count", HEADS),
                       ("attention.key_length", G_DIMS[1]), ("attention.key_length_swa", G_DIMS[0]),
                       ("attention.value_length", G_DIMS[1]),
                       ("attention.value_length_swa", G_DIMS[0]),
                       ("rope.dimension_count", G_DIMS[1]), ("rope.dimension_count_swa", G_DIMS[0]),
                       ("attention.sliding_window", G_WINDOW)]:  # fmt: skip
        w.add_uint32(a + key, value)
    if experts:
        w.add_uint32(a + "expert_count", EXPERTS)
        w.add_uint32(a + "expert_used_count", USED)
        w.add_uint32(a + "expert_feed_forward_length", EXPERT_HIDDEN)
    w.add_array(a + "attention.head_count_kv", list(G_KV_HEADS))
    w.add_array(a + "attention.sliding_window_pattern", [True, False])
    w.add_float32(a + "rope.freq_base", 10000.0)
    w.add_float32(a + "rope.freq_base_swa", 1000.0)
    w.add_float32(a + "final_logit_softcapping", G_CAP)
    add("token_embd.weight", (V, D), GGMLType.Q4_K, 2e-4)  # also the output, tied
    add("output_norm.weight", (D,))
    # the full-attention layer rotates the first quarter of its dimensions only
    weights["rope_freqs.weight"] = np.array([1.0] * 8 + [1e30] * 24, dtype=np.float32)
    w.add_tensor("rope_freqs.weight", weights["rope_freqs.weight"])
    for i, (dim, kv_heads) in enumerate(zip(G_DIMS, G_KV_HEADS, strict=True)):
        b = f"blk.{i}."
        add(b + "attn_q.weight", (HEADS * dim, D), GGMLType.Q4_K, 2e-4)
        add(b + "attn_k.weight", (kv_heads * dim, D), GGMLType.Q8_0, 1e-3)
        if i == 0:
            add(b + "attn_v.weight", (kv_heads * dim, D), GGMLType.Q8_0, 1e-3)
        add(b + "attn_output.weight", (D, HEADS * dim), GGMLType.F32, 0.05)
        for name in ("attn_q_norm", "attn_k_norm"):
            add(b + name + ".weight", (dim,))
        for name in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
            add(b + name + ".weight", (D,))
        add(b + "layer_output_scale.weight", (1,))
        for name in MLP:
            add(b + name + ".weight", *TENSORS[name])
        if not experts:
            continue
        for name in ("pre_ffw_norm_2", "post_ffw_norm_1", "post_ffw_norm_2"):
            add(b + name + ".weight", (D,))
        add(b + "ffn_gate_inp.weight", (EXPERTS, D), GGMLType.F32, 0.05)
        add(b + "ffn_gate_inp.scale", (D,))
        add(b + "ffn_gate_up_exps.weight", (EXPERTS, 2 * EXPERT_HIDDEN, D), GGMLType.Q4_K, 2e-4)
        add(b + "ffn_down_exps.weight", (EXPERTS, D, EXPERT_HIDDEN), GGMLType.Q6_K, 5e-5)
        add(b + "ffn_down_exps.scale", (EXPERTS,))
    _finish(w)
    return weights


def reference_logits(
    w: dict[str, np.ndarray], tokens: list[int], arch: str = "llama"
) -> np.ndarray:
    # an independent float64 model, with keys and values rounded to f16 like leat's cache
    if arch.startswith("gemma4"):
        return _reference_gemma4(w, tokens)

    T = len(tokens)
    freqs = 10000.0 ** (-np.arange(0, HEAD_DIM, 2) / HEAD_DIM) / w.get("rope_freqs.weight", 1.0)
    angles = np.arange(T)[:, None, None] * freqs
    cos, sin = np.cos(angles), np.sin(angles)

    # llama rotates adjacent pairs of dimensions, the others dimension i with i + HEAD_DIM / 2
    half = HEAD_DIM // 2
    first, second = (np.s_[0::2], np.s_[1::2]) if arch == "llama" else (np.s_[:half], np.s_[half:])

    def rope(z):
        out = np.empty_like(z)
        out[..., first] = z[..., first] * cos - z[..., second] * sin
        out[..., second] = z[..., first] * sin + z[..., second] * cos
        return out

    def mixture(h, router, gate, up, down):
        return experts(h, h @ router.T, lambda e, x: mlp(x, gate[e], up[e], down[e]))

    x = w["token_embd.weight"][tokens].astype(np.float64)
    causal = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(LAYERS):
        lw = {n: w[f"blk.{i}.{n}.weight"] for n in ATTENTION + (MOE if arch == "qwen3moe" else MLP)}
        h = norm(x, w[f"blk.{i}.attn_norm.weight"])
        q, k, v = (h @ lw[n].T + w.get(f"blk.{i}.{n}.bias", 0.0) for n in ATTENTION[:3])
        q, k, v = q.reshape(T, HEADS, HEAD_DIM), *(z.reshape(T, KV_HEADS, HEAD_DIM) for z in (k, v))
        if arch.startswith("qwen3"):
            q = norm(q, w[f"blk.{i}.attn_q_norm.weight"])
            k = norm(k, w[f"blk.{i}.attn_k_norm.weight"])
        out = attention(rope(q), rope(k), v, causal, 1 / np.sqrt(HEAD_DIM))
        x = x + out @ lw["attn_output"].T
        h = norm(x, w[f"blk.{i}.ffn_norm.weight"])
        x = x + (mixture if arch == "qwen3moe" else mlp)(h, *(lw[n] for n in lw if "ffn" in n))
    return norm(x, w["output_norm.weight"]) @ w["output.weight"].T


def norm(x, weight=1.0):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * weight


def mlp(h, gate, up, down, gelu=False):
    g = h @ gate.T
    act = (
        0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3)))
        if gelu
        else g / (1 + np.exp(-g))
    )
    return (act * (h @ up.T)) @ down.T


def experts(h, scores, expert):
    # the sum of each token's USED best scoring experts' outputs expert(e, h[t]), weighted by the
    # softmax of their scores
    out = np.zeros_like(h)
    for t in range(len(h)):
        best = np.argsort(-scores[t])[:USED]
        p = np.exp(scores[t, best] - scores[t, best].max())
        for e, weight in zip(best, p / p.sum(), strict=True):
            out[t] += weight * expert(e, h[t])
    return out


def attention(q, k, v, mask, scale):
    # q (T, heads, dim) and k, v (T, kv heads, dim), with keys and values rounded to f16 as in
    # leat's cache: (T, heads * dim)
    k, v = (z.astype(np.float16).astype(np.float64) for z in (k, v))
    group, heads = q.shape[1] // k.shape[1], []
    for hd in range(q.shape[1]):
        scores = q[:, hd] @ k[:, hd // group].T * scale + mask
        p = np.exp(scores - scores.max(-1, keepdims=True))
        heads.append((p / p.sum(-1, keepdims=True)) @ v[:, hd // group])
    return np.concatenate(heads, -1)


def _reference_gemma4(w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    T, positions = len(tokens), np.arange(len(tokens))
    x = w["token_embd.weight"][tokens].astype(np.float64) * np.sqrt(D)
    for i, (dim, kv_heads, sliding) in enumerate(
        zip(G_DIMS, G_KV_HEADS, (True, False), strict=True)
    ):
        lw = {
            n.removeprefix(f"blk.{i}.").removesuffix(".weight"): v
            for n, v in w.items()
            if n.startswith(f"blk.{i}.")
        }
        # rotations of dimension j with j + dim / 2, the full layer's frequencies scaled
        freqs = (1000.0 if sliding else 10000.0) ** (-np.arange(0, dim, 2) / dim)
        angles = positions[:, None, None] * (freqs if sliding else freqs / w["rope_freqs.weight"])
        cos, sin = np.cos(angles), np.sin(angles)

        h = norm(x, lw["attn_norm"])
        q = (h @ lw["attn_q"].T).reshape(T, HEADS, dim)
        k = (h @ lw["attn_k"].T).reshape(T, kv_heads, dim)
        v = norm((h @ lw["attn_v"].T).reshape(T, kv_heads, dim) if "attn_v" in lw else k)
        q, k = (
            rotate_halves(norm(z, lw[f"attn_{n}_norm"]), cos, sin) for z, n in ((q, "q"), (k, "k"))
        )
        back = positions[:, None] - positions
        mask = np.where((back < 0) | (sliding & (back >= G_WINDOW)), -np.inf, 0)
        out = attention(q, k, v, mask, 1.0) @ lw["attn_output"].T
        x = x + norm(out, lw["post_attention_norm"])

        out = mlp(norm(x, lw["ffn_norm"]), lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"], True)
        if "ffn_gate_inp" in lw:  # experts beside the MLP, each output normed
            scores = (norm(x) / np.sqrt(D) * lw["ffn_gate_inp.scale"]) @ lw["ffn_gate_inp"].T
            expert = functools.partial(_gemma4_expert, lw)
            mixed = experts(norm(x, lw["pre_ffw_norm_2"]), scores, expert)
            out = norm(out, lw["post_ffw_norm_1"]) + norm(mixed, lw["post_ffw_norm_2"])
        x = (x + norm(out, lw["post_ffw_norm"])) * lw["layer_output_scale"]
    logits = norm(x, w["output_norm.weight"]) @ w["token_embd.weight"].T
    return np.tanh(logits / G_CAP) * G_CAP


def rotate_halves(z, cos, sin):
    half = z.shape[-1] // 2
    a, b = z[..., :half], z[..., half:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], -1)


def _gemma4_expert(lw, e, h):
    g, u = np.split(lw["ffn_gate_up_exps"][e], 2)
    return mlp(h, g, u, lw["ffn_down_exps"][e], True) * lw["ffn_down_exps.scale"][e]
