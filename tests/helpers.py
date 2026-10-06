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
    GGMLType.Q4_0: (0,),
    GGMLType.Q4_1: (0, 2),
    GGMLType.Q5_0: (0,),
    GGMLType.Q5_1: (0, 2),
    GGMLType.Q8_0: (0,),
    GGMLType.Q2_K: (80, 82),
    GGMLType.Q3_K: (108,),
    GGMLType.Q4_K: (0, 2),
    GGMLType.Q5_K: (0, 2),
    GGMLType.Q6_K: (208,),
    GGMLType.IQ4_NL: (0,),
    GGMLType.IQ4_XS: (0,),
    GGMLType.MXFP4: (),
}


def random_blocks(
    ggml_type: GGMLType, n: int, rng: np.random.Generator, scale: float = 1.0
) -> np.ndarray:
    blocks = rng.integers(0, 256, (n, BLOCK[ggml_type][1]), dtype=np.uint8)
    if ggml_type == GGMLType.MXFP4:  # an exponent byte: scales of 2^-10 to 2^-1 times `scale`
        e = 128 + np.round(np.log2(scale)) + rng.integers(-10, 0, n)
        blocks[:, 0] = np.clip(e, 0, 254).astype(np.uint8)
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
EXPERTS, USED, EXPERT_HIDDEN = 4, 2, 256  # the mixtures of experts' MLPs
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
    if arch in _WRITERS:
        return _WRITERS[arch](path)
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
                add(f"blk.{i}.{name}.bias", (TENSORS[name][0][0],), GGMLType.F32)
    _finish(w)
    return weights


def _writer(path: Path, arch: str):
    # a GGUF writer with the tiny tokenizer, its weights, and add(name, shape, type, scale), which
    # writes a random tensor and keeps it decoded: a norm weight by default, an F32 vector uniform
    # from 1 to `scale`, 1.5; other F32 tensors, biases too, normal times `scale`; and else blocks
    # whose f16 fields are up to `scale`
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(path, arch=arch)
    w.add_uint32(f"{arch}.context_length", CONTEXT)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    w.add_array("tokenizer.ggml.tokens", [*_BYTE_CHAR.values()] + [f"t{i}" for i in range(V - 256)])
    w.add_array("tokenizer.ggml.token_type", [NORMAL] * V)
    w.add_array("tokenizer.ggml.merges", [])
    w.add_chat_template("{{ prefix | default('') }}{{ messages[-1]['content'] }}")
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], ggml_type=GGMLType.F32, scale: float = 1.5) -> None:
        if ggml_type == GGMLType.F32:
            uniform = len(shape) == 1 and not name.endswith(".bias")
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
# 64 whose values are its keys; both with an MLP, and beside it, as in 26B A4B, experts whose gate
# and up are stacked
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
    if arch in _REFERENCES:
        return _REFERENCES[arch](w, tokens)

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


def mlp(h, gate, up, down, kind="silu"):
    return glu(kind, h @ gate.T, h @ up.T) @ down.T


def glu(kind, g, u):
    # act(g) * u for SiLU or GELU, tanh's approximation, or gpt-oss's clamped SwiGLU, "oai"
    if kind == "oai":
        g, u = np.minimum(g, 7.0), np.clip(u, -7.0, 7.0)
        return g / (1 + np.exp(-1.702 * g)) * (u + 1)
    if kind == "gelu":
        return 0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3))) * u
    return g / (1 + np.exp(-g)) * u


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


def attention(q, k, v, mask, scale, sinks=None):
    # q (T, heads, dim) and k, v (T, kv heads, dim), with keys and values rounded to f16 as in
    # leat's cache: (T, heads * dim); with a sink per head, a score that takes its share of the
    # softmax and adds no value
    k, v = (z.astype(np.float16).astype(np.float64) for z in (k, v))
    group, heads = q.shape[1] // k.shape[1], []
    for hd in range(q.shape[1]):
        scores = q[:, hd] @ k[:, hd // group].T * scale + mask
        sink = -np.inf if sinks is None else sinks[hd]
        top = np.maximum(scores.max(-1, keepdims=True), sink)
        p = np.exp(scores - top)
        heads.append(p / (p.sum(-1, keepdims=True) + np.exp(sink - top)) @ v[:, hd // group])
    return np.concatenate(heads, -1)


def cuts(scores, top_k, top_p):
    # a row's top score, where top_k cuts it, at its k-th score or its last for 0, and where top_p
    # then does: at the last of the top k whose likelier ones' share of their probability falls
    # short of top_p, the top one at least
    ranked = np.sort(scores.astype(np.float64))[::-1][: int(top_k) or len(scores)]
    weights = np.exp(ranked - ranked[0])
    likelier = np.cumsum(weights) - weights
    return ranked[0], ranked[-1], ranked[max((likelier < top_p * weights.sum()).sum(), 1) - 1]


def _reference_gemma4(w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    T, positions = len(tokens), np.arange(len(tokens))
    x = w["token_embd.weight"][tokens].astype(np.float64) * np.sqrt(D)
    for i, (dim, kv_heads, sliding) in enumerate(
        zip(G_DIMS, G_KV_HEADS, (True, False), strict=True)
    ):
        lw = _layer(w, i)
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

        out = mlp(norm(x, lw["ffn_norm"]), lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"], "gelu")
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
    return mlp(h, g, u, lw["ffn_down_exps"][e], "gelu") * lw["ffn_down_exps.scale"][e]


# Gemma 3: six layers, the last of which sees all positions and the others a window of 4; the
# full layer's RoPE of another base, its positions scaled by 1/8; q, k and output norms; GELU
G3_LAYERS, G3_WINDOW, G3_THETAS, G3_SCALE = 6, 4, (10000.0, 1e6), 8.0


def _write_gemma3(path: Path) -> dict[str, np.ndarray]:
    w, weights, add = _writer(path, "gemma3")
    a = "gemma3."
    for key, value in [("block_count", G3_LAYERS), ("embedding_length", D),
                       ("feed_forward_length", HIDDEN), ("attention.head_count", HEADS),
                       ("attention.head_count_kv", KV_HEADS), ("attention.key_length", HEAD_DIM),
                       ("attention.value_length", HEAD_DIM),
                       ("attention.sliding_window", G3_WINDOW)]:  # fmt: skip
        w.add_uint32(a + key, value)
    w.add_float32(a + "rope.freq_base", G3_THETAS[1])
    w.add_string(a + "rope.scaling.type", "linear")
    w.add_float32(a + "rope.scaling.factor", G3_SCALE)
    add("token_embd.weight", *TENSORS["token_embd.weight"])  # also the output, tied
    add("output_norm.weight", (D,))
    for i in range(G3_LAYERS):
        b = f"blk.{i}."
        for name in ATTENTION + MLP:
            add(b + name + ".weight", *TENSORS[name])
        for name in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
            add(b + name + ".weight", (D,))
        for name in ("attn_q_norm", "attn_k_norm"):
            add(b + name + ".weight", (HEAD_DIM,))
    _finish(w)
    return weights


def _reference_gemma3(w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    T, positions = len(tokens), np.arange(len(tokens))
    x = w["token_embd.weight"][tokens].astype(np.float64) * np.sqrt(D)
    for i in range(G3_LAYERS):
        sliding = i % 6 < 5
        lw = _layer(w, i)
        freqs = G3_THETAS[not sliding] ** (-np.arange(0, HEAD_DIM, 2) / HEAD_DIM)
        angles = positions[:, None, None] * freqs / (1.0 if sliding else G3_SCALE)
        cos, sin = np.cos(angles), np.sin(angles)
        h = norm(x, lw["attn_norm"])
        q = (h @ lw["attn_q"].T).reshape(T, HEADS, HEAD_DIM)
        k, v = ((h @ lw[n].T).reshape(T, KV_HEADS, HEAD_DIM) for n in ("attn_k", "attn_v"))
        q, k = (
            rotate_halves(norm(z, lw[f"attn_{n}_norm"]), cos, sin) for z, n in ((q, "q"), (k, "k"))
        )
        out = attention(q, k, v, _mask(T, G3_WINDOW if sliding else 0), 1 / np.sqrt(HEAD_DIM))
        x = x + norm(out @ lw["attn_output"].T, lw["post_attention_norm"])
        out = mlp(norm(x, lw["ffn_norm"]), lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"], "gelu")
        x = x + norm(out, lw["post_ffw_norm"])
    return norm(x, w["output_norm.weight"]) @ w["token_embd.weight"].T


# gpt-oss: a layer that sees a window of 4 positions and one that sees all, both with YaRN's RoPE;
# biases of q, k, v and the output, attention sinks; experts of MXFP4 with biases, clamped SwiGLU
# and a router with a bias, the softmax over each token's chosen ones
OSS_WINDOW, OSS_THETA, OSS_YARN = 4, 150000.0, (8.0, 16)  # factor, original context


def _write_gpt_oss(path: Path) -> dict[str, np.ndarray]:
    w, weights, add = _writer(path, "gpt-oss")
    a = "gpt-oss."
    for key, value in [("block_count", LAYERS), ("embedding_length", D),
                       ("feed_forward_length", EXPERT_HIDDEN), ("attention.head_count", HEADS),
                       ("attention.head_count_kv", KV_HEADS), ("attention.key_length", HEAD_DIM),
                       ("attention.value_length", HEAD_DIM),
                       ("attention.sliding_window", OSS_WINDOW), ("expert_count", EXPERTS),
                       ("expert_used_count", USED),
                       ("expert_feed_forward_length", EXPERT_HIDDEN)]:  # fmt: skip
        w.add_uint32(a + key, value)
    w.add_float32(a + "rope.freq_base", OSS_THETA)
    w.add_string(a + "rope.scaling.type", "yarn")
    w.add_float32(a + "rope.scaling.factor", OSS_YARN[0])
    w.add_uint32(a + "rope.scaling.original_context_length", OSS_YARN[1])
    add("token_embd.weight", (V, D), GGMLType.Q8_0, 1e-3)
    add("output.weight", (V, D), GGMLType.Q8_0, 1e-3)
    add("output_norm.weight", (D,))
    for i in range(LAYERS):
        b = f"blk.{i}."
        add(b + "attn_norm.weight", (D,))
        add(b + "post_attention_norm.weight", (D,))  # the experts' input norm, despite its name
        for name, rows in (
            ("attn_q", D),
            ("attn_k", KV_HEADS * HEAD_DIM),
            ("attn_v", KV_HEADS * HEAD_DIM),
        ):
            add(b + name + ".weight", (rows, D), GGMLType.Q8_0, 1e-3)
            add(b + name + ".bias", (rows,), GGMLType.F32, 0.5)
        add(b + "attn_output.weight", (D, D), GGMLType.Q8_0, 1e-3)
        add(b + "attn_output.bias", (D,), GGMLType.F32, 0.5)
        add(b + "attn_sinks.weight", (HEADS,), GGMLType.F32, 2.0)
        add(b + "ffn_gate_inp.weight", (EXPERTS, D), GGMLType.F32, 0.05)
        add(b + "ffn_gate_inp.bias", (EXPERTS,), GGMLType.F32, 0.5)
        for name, shape in (
            ("gate", (EXPERT_HIDDEN, D)),
            ("up", (EXPERT_HIDDEN, D)),
            ("down", (D, EXPERT_HIDDEN)),
        ):
            add(b + f"ffn_{name}_exps.weight", (EXPERTS, *shape), GGMLType.MXFP4, 0.25)
            add(b + f"ffn_{name}_exps.bias", (EXPERTS, shape[0]), GGMLType.F32, 0.2)
    _finish(w)
    return weights


def _yarn(positions: np.ndarray, dim: int, theta: float, factor: float, original: int):
    # cos and sin of YaRN's angles, scaled as llama.cpp's: a blend of the scaled and unscaled
    # frequencies by a ramp over the dimensions, and both times 1 + 0.1 ln(factor)
    freqs = theta ** (-np.arange(0, dim, 2) / dim)

    def corr(beta):
        return dim * np.log(original / (beta * 2 * np.pi)) / (2 * np.log(theta))

    low, high = max(0.0, np.floor(corr(32.0))), min(dim - 1.0, np.ceil(corr(1.0)))
    ramp = 1 - np.clip((np.arange(dim // 2) - low) / max(0.001, high - low), 0, 1)
    freqs = freqs / factor * (1 - ramp) + freqs * ramp
    angles, mscale = positions[:, None, None] * freqs, 1 + 0.1 * np.log(factor)
    return np.cos(angles) * mscale, np.sin(angles) * mscale


def _reference_gpt_oss(w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    T, positions = len(tokens), np.arange(len(tokens))
    cos, sin = _yarn(positions, HEAD_DIM, OSS_THETA, *OSS_YARN)
    x = w["token_embd.weight"][tokens].astype(np.float64)
    for i in range(LAYERS):
        lw = _layer(w, i)
        h = norm(x, lw["attn_norm"])
        q, k, v = (h @ lw[n].T + lw[f"{n}.bias"] for n in ("attn_q", "attn_k", "attn_v"))
        q = rotate_halves(q.reshape(T, HEADS, HEAD_DIM), cos, sin)
        k = rotate_halves(k.reshape(T, KV_HEADS, HEAD_DIM), cos, sin)
        v = v.reshape(T, KV_HEADS, HEAD_DIM)
        mask = _mask(T, OSS_WINDOW if i % 2 == 0 else 0)
        out = attention(q, k, v, mask, 1 / np.sqrt(HEAD_DIM), lw["attn_sinks"])
        x = x + out @ lw["attn_output"].T + lw["attn_output.bias"]
        h = norm(x, lw["post_attention_norm"])
        scores = h @ lw["ffn_gate_inp"].T + lw["ffn_gate_inp.bias"]

        def expert(e, row, lw=lw):
            g, u = (
                row @ lw[f"ffn_{n}_exps"][e].T + lw[f"ffn_{n}_exps.bias"][e] for n in ("gate", "up")
            )
            return glu("oai", g, u) @ lw["ffn_down_exps"][e].T + lw["ffn_down_exps.bias"][e]

        x = x + experts(h, scores, expert)
    return norm(x, w["output_norm.weight"]) @ w["output.weight"].T


# Phi-3: q, k and v in one matrix and gate and up in one; RoPE over 3/4 of each head, with
# LongRoPE's long factors past an original context of 16 and its cos and sin scaled
P3_ROTATED, P3_ORIGINAL, P3_MSCALE = 3 * HEAD_DIM // 4, 16, 1.19


def _write_phi3(path: Path) -> dict[str, np.ndarray]:
    w, weights, add = _writer(path, "phi3")
    a = "phi3."
    for key, value in [("block_count", LAYERS), ("embedding_length", D),
                       ("feed_forward_length", HIDDEN), ("attention.head_count", HEADS),
                       ("attention.head_count_kv", KV_HEADS), ("rope.dimension_count", P3_ROTATED),
                       ("rope.scaling.original_context_length", P3_ORIGINAL)]:  # fmt: skip
        w.add_uint32(a + key, value)
    w.add_float32(a + "rope.freq_base", 10000.0)
    w.add_float32(a + "rope.scaling.attn_factor", P3_MSCALE)
    add("token_embd.weight", *TENSORS["token_embd.weight"])  # also the output, tied
    add("output_norm.weight", (D,))
    for kind in ("long", "short"):
        add(f"rope_factors_{kind}.weight", (P3_ROTATED // 2,), GGMLType.F32, 4.0)
    for i in range(LAYERS):
        b = f"blk.{i}."
        add(b + "attn_norm.weight", (D,))
        add(b + "ffn_norm.weight", (D,))
        add(b + "attn_qkv.weight", (D + 2 * KV_HEADS * HEAD_DIM, D), GGMLType.Q4_K, 2e-4)
        add(b + "attn_output.weight", *TENSORS["attn_output"])
        add(b + "ffn_up.weight", (2 * HIDDEN, D), GGMLType.Q4_K, 2e-4)  # gate's rows, then up's
        add(b + "ffn_down.weight", *TENSORS["ffn_down"])
    _finish(w)
    return weights


def _reference_phi3(w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    # with max_context 64, past the original 16: the long factors
    T, positions = len(tokens), np.arange(len(tokens))
    freqs = 10000.0 ** (-np.arange(0, P3_ROTATED, 2) / P3_ROTATED) / w["rope_factors_long.weight"]
    angles = positions[:, None, None] * freqs
    cos, sin = np.cos(angles) * P3_MSCALE, np.sin(angles) * P3_MSCALE

    def rope(z):  # the first P3_ROTATED dimensions, i with i + P3_ROTATED / 2
        return np.concatenate(
            [rotate_halves(z[..., :P3_ROTATED], cos, sin), z[..., P3_ROTATED:]], -1
        )

    x = w["token_embd.weight"][tokens].astype(np.float64)
    kv = KV_HEADS * HEAD_DIM
    for i in range(LAYERS):
        lw = _layer(w, i)
        q, k, v = np.split(norm(x, lw["attn_norm"]) @ lw["attn_qkv"].T, [D, D + kv], -1)
        q, k = rope(q.reshape(T, HEADS, HEAD_DIM)), rope(k.reshape(T, KV_HEADS, HEAD_DIM))
        out = attention(q, k, v.reshape(T, KV_HEADS, HEAD_DIM), _mask(T, 0), 1 / np.sqrt(HEAD_DIM))
        x = x + out @ lw["attn_output"].T
        gate, up = np.split(lw["ffn_up"], 2)
        x = x + mlp(norm(x, lw["ffn_norm"]), gate, up, lw["ffn_down"])
    return norm(x, w["output_norm.weight"]) @ w["token_embd.weight"].T


# Qwen3.5's mixture of experts: two Gated DeltaNet layers, then an attention layer whose q also
# gives each head's output a gate, with q and k norms and RoPE over 16 of 64 dimensions; experts
# beside a shared one with a gate of its own; and a layer past the others, for predicting
# further tokens, which leat leaves unread
Q35_EVERY, Q35_ROTATED, Q35_HEAD = 3, 16, 64  # every 3rd layer attention; heads of 64
Q35_K_HEADS, Q35_V_HEADS, Q35_DIM, Q35_CONV, Q35_SHARED = 2, 4, 32, 4, 128
Q35_CHANNELS = (2 * Q35_K_HEADS + Q35_V_HEADS) * Q35_DIM


def _write_qwen35moe(path: Path) -> dict[str, np.ndarray]:
    w, weights, add = _writer(path, "qwen35moe")
    a = "qwen35moe."
    for key, value in [("block_count", Q35_EVERY + 1), ("nextn_predict_layers", 1),
                       ("embedding_length", D), ("attention.head_count", HEADS),
                       ("attention.head_count_kv", KV_HEADS), ("attention.key_length", Q35_HEAD),
                       ("attention.value_length", Q35_HEAD),
                       ("rope.dimension_count", Q35_ROTATED), ("expert_count", EXPERTS),
                       ("expert_used_count", USED), ("expert_feed_forward_length", EXPERT_HIDDEN),
                       ("expert_shared_feed_forward_length", Q35_SHARED),
                       ("ssm.conv_kernel", Q35_CONV), ("ssm.state_size", Q35_DIM),
                       ("ssm.group_count", Q35_K_HEADS), ("ssm.time_step_rank", Q35_V_HEADS),
                       ("ssm.inner_size", Q35_V_HEADS * Q35_DIM),
                       ("full_attention_interval", Q35_EVERY)]:  # fmt: skip
        w.add_uint32(a + key, value)
    w.add_float32(a + "rope.freq_base", 1e7)
    add("token_embd.weight", *TENSORS["token_embd.weight"])
    add("output.weight", *TENSORS["output.weight"])
    add("output_norm.weight", (D,))
    rng = np.random.default_rng(1)
    for i in range(Q35_EVERY):
        b = f"blk.{i}."
        for name in ("attn_norm", "post_attention_norm"):
            add(b + name + ".weight", (D,))
        if i < Q35_EVERY - 1:  # Gated DeltaNet
            inner = Q35_V_HEADS * Q35_DIM
            add(b + "attn_qkv.weight", (Q35_CHANNELS, D), GGMLType.Q4_K, 2e-4)
            add(b + "attn_gate.weight", (inner, D), GGMLType.Q5_K, 2e-4)
            for name in ("ssm_alpha", "ssm_beta"):
                add(b + name + ".weight", (Q35_V_HEADS, D), GGMLType.F32, 0.1)
            add(b + "ssm_conv1d.weight", (Q35_CHANNELS, Q35_CONV), GGMLType.F32, 0.5)
            add(b + "ssm_dt.bias", (Q35_V_HEADS,), GGMLType.F32)
            add(b + "ssm_norm.weight", (Q35_DIM,))
            add(b + "ssm_out.weight", (D, inner), GGMLType.Q8_0, 1e-3)
            # -exp(A_log), as GGUF holds it
            weights[b + "ssm_a"] = -rng.uniform(0.2, 2.0, Q35_V_HEADS).astype(np.float32)
            w.add_tensor(b + "ssm_a", weights[b + "ssm_a"])
        else:
            add(b + "attn_q.weight", (2 * HEADS * Q35_HEAD, D), GGMLType.Q4_K, 2e-4)
            for name in ("attn_k", "attn_v"):
                add(b + name + ".weight", (KV_HEADS * Q35_HEAD, D), GGMLType.Q8_0, 1e-3)
            add(b + "attn_output.weight", (D, HEADS * Q35_HEAD), GGMLType.Q4_K, 2e-4)
            for name in ("attn_q_norm", "attn_k_norm"):
                add(b + name + ".weight", (Q35_HEAD,))
        for name in MOE:
            add(b + name + ".weight", *TENSORS[name])
        add(b + "ffn_gate_inp_shexp.weight", (D,), GGMLType.F32, 1.2)
        add(b + "ffn_gate_shexp.weight", (Q35_SHARED, D), GGMLType.Q4_K, 2e-4)
        add(b + "ffn_up_shexp.weight", (Q35_SHARED, D), GGMLType.Q6_K, 5e-5)
        add(b + "ffn_down_shexp.weight", (D, Q35_SHARED), GGMLType.Q8_0, 1e-3)
    add(f"blk.{Q35_EVERY}.nextn.enorm.weight", (D,))
    _finish(w)
    return weights


def _reference_qwen35moe(w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    T, positions = len(tokens), np.arange(len(tokens))
    freqs = 1e7 ** (-np.arange(0, Q35_ROTATED, 2) / Q35_ROTATED)
    angles = positions[:, None, None] * freqs
    cos, sin = np.cos(angles), np.sin(angles)

    def rope(z):  # the first Q35_ROTATED dimensions, i with i + Q35_ROTATED / 2
        return np.concatenate(
            [rotate_halves(z[..., :Q35_ROTATED], cos, sin), z[..., Q35_ROTATED:]], -1
        )

    x = w["token_embd.weight"][tokens].astype(np.float64)
    for i in range(Q35_EVERY):
        lw = _layer(w, i)
        h = norm(x, lw["attn_norm"])
        if i < Q35_EVERY - 1:
            out = _gated_delta_net(lw, h)
        else:
            q, gate = np.split((h @ lw["attn_q"].T).reshape(T, HEADS, 2 * Q35_HEAD), 2, -1)
            k, v = ((h @ lw[n].T).reshape(T, KV_HEADS, Q35_HEAD) for n in ("attn_k", "attn_v"))
            q, k = rope(norm(q, lw["attn_q_norm"])), rope(norm(k, lw["attn_k_norm"]))
            out = attention(q, k, v, _mask(T, 0), 1 / np.sqrt(Q35_HEAD))
            out = out * sigmoid(gate.reshape(T, -1)) @ lw["attn_output"].T
        x = x + out
        h = norm(x, lw["post_attention_norm"])
        x = x + experts(
            h,
            h @ lw["ffn_gate_inp"].T,
            lambda e, row, lw=lw: mlp(
                row, lw["ffn_gate_exps"][e], lw["ffn_up_exps"][e], lw["ffn_down_exps"][e]
            ),
        )
        shared = mlp(h, lw["ffn_gate_shexp"], lw["ffn_up_shexp"], lw["ffn_down_shexp"])
        x = x + shared * sigmoid(h @ lw["ffn_gate_inp_shexp"])[:, None]
    return norm(x, w["output_norm.weight"]) @ w["output.weight"].T


def _gated_delta_net(lw: dict[str, np.ndarray], h: np.ndarray) -> np.ndarray:
    # each value head's state, keys by values, from zero: for each token, decayed, then moved a
    # share toward the token's values for its key; its output, the state's values for the query
    T, qk = len(h), Q35_K_HEADS * Q35_DIM
    inputs = np.concatenate([np.zeros((Q35_CONV - 1, Q35_CHANNELS)), h @ lw["attn_qkv"].T])
    conved = np.stack([(inputs[t : t + Q35_CONV] * lw["ssm_conv1d"].T).sum(0) for t in range(T)])
    q, k, v = np.split(conved / (1 + np.exp(-conved)), [qk, 2 * qk], -1)

    def unit(z):
        z = z.reshape(T, Q35_K_HEADS, Q35_DIM)
        return z / np.sqrt((z * z).sum(-1, keepdims=True) + 1e-5)  # as llama.cpp

    q, k, v = unit(q) / np.sqrt(Q35_DIM), unit(k), v.reshape(T, Q35_V_HEADS, Q35_DIM)
    rate = h @ lw["ssm_alpha"].T + lw["ssm_dt.bias"]
    decay = np.exp(lw["ssm_a"] * np.logaddexp(0, rate))
    share = sigmoid(h @ lw["ssm_beta"].T)
    state, out = np.zeros((Q35_V_HEADS, Q35_DIM, Q35_DIM)), np.zeros((T, Q35_V_HEADS, Q35_DIM))
    for t in range(T):
        for hd in range(Q35_V_HEADS):
            key, query = k[t, hd % Q35_K_HEADS], q[t, hd % Q35_K_HEADS]
            state[hd] *= decay[t, hd]
            state[hd] += np.outer(key, share[t, hd] * (v[t, hd] - state[hd].T @ key))
            out[t, hd] = state[hd].T @ query
    z = (h @ lw["attn_gate"].T).reshape(T, Q35_V_HEADS, Q35_DIM)
    gated = norm(out, lw["ssm_norm"]) * z / (1 + np.exp(-z))
    return gated.reshape(T, -1) @ lw["ssm_out"].T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _layer(w: dict[str, np.ndarray], i: int) -> dict[str, np.ndarray]:
    # a layer's tensors by name, without "blk.{i}." and ".weight"
    return {
        n.removeprefix(f"blk.{i}.").removesuffix(".weight"): v
        for n, v in w.items()
        if n.startswith(f"blk.{i}.")
    }


def _mask(T: int, window: int) -> np.ndarray:
    # causal, and over the last `window` positions if given
    back = np.arange(T)[:, None] - np.arange(T)
    return np.where((back < 0) | (window > 0) & (back >= window), -np.inf, 0)


_WRITERS = {
    "gemma3": _write_gemma3, "gpt-oss": _write_gpt_oss, "phi3": _write_phi3,
    "qwen35moe": _write_qwen35moe,
}  # fmt: skip
_REFERENCES = {
    "gemma3": _reference_gemma3, "gpt-oss": _reference_gpt_oss, "phi3": _reference_phi3,
    "qwen35moe": _reference_qwen35moe,
}  # fmt: skip
