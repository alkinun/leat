# Test data shared across test files: quantized blocks, tokenizer metadata and tiny models.

import functools
import math
from pathlib import Path

import gguf
import numpy as np
from gguf.quants import dequantize
from tinygrad import Tensor, UOp, dtypes

from leat.draft import mtp
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
    # llama's vocab with the tokens of Mistral Small 3's images, as it is of the llama kind
    w, weights, add = _writer(path, arch, ("[IMG]", "[IMG_END]") if arch == "llama" else ())
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


def _writer(path: Path, arch: str, specials: tuple[str, ...] = ()):
    # a GGUF writer with the tiny tokenizer, its last tokens `specials`, control tokens, its
    # weights, and add(name, shape, type, scale), which
    # writes a random tensor and keeps it decoded: a norm weight by default, an F32 vector uniform
    # from 1 to `scale`, 1.5; other F32 tensors, biases too, normal times `scale`; and else blocks
    # whose f16 fields are up to `scale`
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(path, arch=arch)
    w.add_uint32(f"{arch}.context_length", CONTEXT)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    names = [f"t{i}" for i in range(V - 256 - len(specials))] + list(specials)
    w.add_array("tokenizer.ggml.tokens", [*_BYTE_CHAR.values()] + names)
    types = [NORMAL] * (V - len(specials)) + [CONTROL] * len(specials)
    w.add_array("tokenizer.ggml.token_type", types)
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
# and up are stacked. Its vocab ends with the tokens that open, fill and close an image.
G_WINDOW, G_DIMS, G_KV_HEADS, G_CAP = 4, (32, 64), (2, 1), 5.0
G_IMAGE = ("<|image>", "<|image|>", "<image|>")


def _write_gemma4(path: Path, experts: bool) -> dict[str, np.ndarray]:
    # with experts beside each MLP, as Gemma 4 26B A4B, or without, as the dense models
    w, weights, add = _writer(path, "gemma4", G_IMAGE)
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
    w: dict[str, np.ndarray], tokens: list[int], arch: str = "llama",
    images: dict[int, np.ndarray] | None = None, grids: dict[int, tuple[int, int]] | None = None,
) -> np.ndarray:  # fmt: skip
    # an independent float64 model, with keys and values rounded to f16 like leat's cache;
    # images' embeddings by key in place of the tokens that hold it, of grids by key, which
    # Qwen3.5's M-RoPE places
    if arch.startswith("gemma4"):
        return _reference_gemma4(w, tokens, images)
    if arch == "gemma3":
        return _reference_gemma3(w, tokens, images)
    if arch in ("qwen35", "qwen35moe"):
        return _reference_qwen35(w, tokens, images, grids)
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

    x = _embedded(w, tokens, images, 1.0)  # images' too, read causally, as Mistral Small 3's
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


def _reference_gemma4(
    w: dict[str, np.ndarray], tokens: list[int], images: dict[int, np.ndarray] | None = None
) -> np.ndarray:
    logits = _gemma4_hidden(w, tokens, images)[0] @ w["token_embd.weight"].T
    return np.tanh(logits / G_CAP) * G_CAP


def _gemma4_rope(dim: int, sliding: bool, positions: np.ndarray, factors: np.ndarray):
    # cos and sin of a layer's rotations of dimension j with j + dim / 2, the full layer's
    # frequencies divided by rope_freqs' factors
    freqs = (1000.0 if sliding else 10000.0) ** (-np.arange(0, dim, 2) / dim)
    angles = positions[:, None, None] * (freqs if sliding else freqs / factors)
    return np.cos(angles), np.sin(angles)


def _embedded(
    w: dict[str, np.ndarray], tokens: list[int], images: dict[int, np.ndarray] | None,
    scale: float = np.sqrt(D),
) -> np.ndarray:  # fmt: skip
    # the tokens' embeddings times `scale`, Gemma's sqrt(D), and images' (n, D) by key in place of
    # the tokens that hold them, as they are
    return np.stack([
        images[t][tokens[:i].count(t) % len(images[t])] if images and t < 0
        else w["token_embd.weight"][t] * scale
        for i, t in enumerate(tokens)
    ]).astype(np.float64)  # fmt: skip


def _same_image(tokens: list[int]) -> np.ndarray:
    # (T, T): whether two positions are of the same image, counted along the prompt
    run = np.cumsum([t < 0 and (i == 0 or tokens[i - 1] != t) for i, t in enumerate(tokens)])
    run = np.where(np.array(tokens) < 0, run, -1)
    return (run[:, None] == run) & (run[:, None] >= 0)


def _gemma4_hidden(
    w: dict[str, np.ndarray], tokens: list[int], images: dict[int, np.ndarray] | None = None
):  # fmt: skip
    # the normed hidden states (T, D) and each layer's keys and values, as in the cache; images'
    # embeddings (n, D) by key in place of the tokens that hold it, which see each other
    T, positions, cache = len(tokens), np.arange(len(tokens)), []
    x, image = _embedded(w, tokens, images), _same_image(tokens)
    for i, (dim, kv_heads, sliding) in enumerate(
        zip(G_DIMS, G_KV_HEADS, (True, False), strict=True)
    ):
        lw = _layer(w, i)
        cos, sin = _gemma4_rope(dim, sliding, positions, w["rope_freqs.weight"])

        h = norm(x, lw["attn_norm"])
        q = (h @ lw["attn_q"].T).reshape(T, HEADS, dim)
        k = (h @ lw["attn_k"].T).reshape(T, kv_heads, dim)
        v = norm((h @ lw["attn_v"].T).reshape(T, kv_heads, dim) if "attn_v" in lw else k)
        q, k = (
            rotate_halves(norm(z, lw[f"attn_{n}_norm"]), cos, sin) for z, n in ((q, "q"), (k, "k"))
        )
        back = positions[:, None] - positions
        mask = np.where((back < 0) & ~image | (sliding & (back >= G_WINDOW)), -np.inf, 0)
        cache.append((k, v))
        out = attention(q, k, v, mask, 1.0) @ lw["attn_output"].T
        x = x + norm(out, lw["post_attention_norm"])

        out = mlp(norm(x, lw["ffn_norm"]), lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"], "gelu")
        if "ffn_gate_inp" in lw:  # experts beside the MLP, each output normed
            scores = (norm(x) / np.sqrt(D) * lw["ffn_gate_inp.scale"]) @ lw["ffn_gate_inp"].T
            expert = functools.partial(_gemma4_expert, lw)
            mixed = experts(norm(x, lw["pre_ffw_norm_2"]), scores, expert)
            out = norm(out, lw["post_ffw_norm_1"]) + norm(mixed, lw["post_ffw_norm_2"])
        x = (x + norm(out, lw["post_ffw_norm"])) * lw["layer_output_scale"]
    return norm(x, w["output_norm.weight"]), cache


# Gemma 4's assistant for the tiny Gemma 4: as wide as half the target, a layer of a window over
# the target's first layer's keys and values and one over all of its second's
A_D, A_HIDDEN = 128, 256


def write_tiny_assistant(path: Path) -> dict[str, np.ndarray]:
    w, weights, add = _writer(path, "gemma4-assistant")
    a = "gemma4-assistant."
    for key, value in [("block_count", 2), ("embedding_length", A_D),
                       ("embedding_length_out", D), ("feed_forward_length", A_HIDDEN),
                       ("attention.head_count", HEADS), ("attention.key_length", G_DIMS[1]),
                       ("attention.key_length_swa", G_DIMS[0]),
                       ("attention.value_length", G_DIMS[1]),
                       ("attention.value_length_swa", G_DIMS[0]),
                       ("attention.sliding_window", G_WINDOW), ("nextn_predict_layers", 2),
                       ("attention.shared_kv_layers", 2)]:  # fmt: skip
        w.add_uint32(a + key, value)
    w.add_array(a + "attention.head_count_kv", list(G_KV_HEADS))
    w.add_array(a + "attention.sliding_window_pattern", [True, False])
    w.add_float32(a + "rope.freq_base", 10000.0)
    w.add_float32(a + "rope.freq_base_swa", 1000.0)
    add("token_embd.weight", (V, A_D), GGMLType.Q8_0, 1e-3)
    add("output_norm.weight", (A_D,))
    add("nextn.pre_projection.weight", (A_D, 2 * D), GGMLType.Q8_0, 1e-3)
    add("nextn.post_projection.weight", (D, A_D), GGMLType.Q8_0, 1e-3)
    weights["rope_freqs.weight"] = np.array([1.0] * 8 + [1e30] * 24, dtype=np.float32)
    w.add_tensor("rope_freqs.weight", weights["rope_freqs.weight"])
    for i, dim in enumerate(G_DIMS):
        b = f"blk.{i}."
        add(b + "attn_q.weight", (HEADS * dim, A_D), GGMLType.Q8_0, 1e-3)
        add(b + "attn_output.weight", (A_D, HEADS * dim), GGMLType.F32, 0.05)
        add(b + "attn_q_norm.weight", (dim,))
        for name in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
            add(b + name + ".weight", (A_D,))
        add(b + "layer_output_scale.weight", (1,))
        add(b + "ffn_gate.weight", (A_HIDDEN, A_D), GGMLType.Q8_0, 1e-3)
        add(b + "ffn_up.weight", (A_HIDDEN, A_D), GGMLType.Q8_0, 1e-3)
        add(b + "ffn_down.weight", (A_D, A_HIDDEN), GGMLType.Q8_0, 1e-3)
    _finish(w)
    return weights


class Oracle:
    """A drafter that drafts the tokens lists hold, one per slot, at the positions after a draft's,
    as one that guessed them would."""

    sequences = 3

    def __init__(self, *tokens: list[int]):
        rows = [t + [0] * (2 * CONTEXT - len(t)) for t in tokens]
        self.tokens = Tensor(rows, dtype=dtypes.int32).realize()

    def draft(self, tokens: Tensor, hidden: Tensor, slots: list, positions: list, count: int):
        rows = zip(slots, positions, strict=True)
        return Tensor.cat(*(self.tokens[s : s + 1, p + 1 : p + 1 + count] for s, p in rows))

    def follow(self, tokens: Tensor, hidden: Tensor, spans: list) -> None:
        pass

    def copy(self, source: UOp, slot: UOp) -> None:
        pass


def reference_drafts(
    target: dict[str, np.ndarray], assistant: dict[str, np.ndarray], tokens: list[int],
    count: int,
) -> list[int]:  # fmt: skip
    # the assistant's `count` greedy drafts after `tokens`, the last at a position the target has
    # not run: each from the token before it and the hidden state the step before gave, all at
    # that position, attending over the target's keys and values before it
    hidden, cache = _gemma4_hidden(target, tokens[:-1])
    pos, token, h, drafts = len(tokens) - 1, tokens[-1], hidden[-1], []
    for _ in range(count):
        e = target["token_embd.weight"][token].astype(np.float64) * np.sqrt(D)
        x = np.concatenate([e, h]) @ assistant["nextn.pre_projection.weight"].T
        for i, (dim, sliding) in enumerate(zip(G_DIMS, (True, False), strict=True)):
            lw, (k, v) = _layer(assistant, i), cache[i]
            cos, sin = _gemma4_rope(dim, sliding, np.array([pos]), assistant["rope_freqs.weight"])
            q = (norm(x, lw["attn_norm"]) @ lw["attn_q"].T).reshape(1, HEADS, dim)
            q = rotate_halves(norm(q, lw["attn_q_norm"]), cos, sin)
            back = pos - np.arange(pos)
            mask = np.where(sliding & (back >= G_WINDOW), -np.inf, 0)[None]
            out = attention(q, k, v, mask, 1.0)[0] @ lw["attn_output"].T
            x = x + norm(out, lw["post_attention_norm"])
            out = mlp(norm(x, lw["ffn_norm"]), lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"], "gelu")
            x = (x + norm(out, lw["post_ffw_norm"])) * lw["layer_output_scale"]
        x = norm(x, assistant["output_norm.weight"])
        token = int((x @ assistant["token_embd.weight"].T).argmax())
        h = x @ assistant["nextn.post_projection.weight"].T
        drafts.append(token)
    return drafts


# Gemma 4's vision encoder for the tiny Gemma 4: 2 layers of 2 heads of 16, over patches of 4 by 4
# pixels whose embeddings each pool 3 by 3
V_WIDTH, V_HEADS, V_HIDDEN, V_PATCH, V_LAYERS, V_COLUMNS = 32, 2, 64, 4, 2, 64
V_EPS = 1e-6
G3_IMAGE = 16  # the side of Gemma 3's tiny SigLIP's images: 4 by 4 patches, 2 by 2 embeddings
# the longest side of Pixtral's tiny images, and their normalization, CLIP's
P_IMAGE = 16
P_MEAN, P_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def write_tiny_mmproj(path: Path, kind: str = "gemma4v") -> dict[str, np.ndarray]:
    # Gemma 4's encoder, or Gemma 3's SigLIP of an image of 4 by 4 patches pooled 2 by 2, its
    # metadata of the older keys and its MLP's matrices named the other way round, as real ones
    rng = np.random.default_rng(1)
    w = gguf.GGUFWriter(path, arch="clip")
    siglip = kind == "gemma3"
    w.add_string("clip.projector_type" if siglip else "clip.vision.projector_type", kind)
    for key, value in [("patch_size", V_PATCH), ("embedding_length", V_WIDTH),
                       ("feed_forward_length", V_HIDDEN), ("block_count", V_LAYERS),
                       ("attention.head_count", V_HEADS), ("projection_dim", D)]:  # fmt: skip
        w.add_uint32(f"clip.vision.{key}", value)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", V_EPS)
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], scale: float = 0.0) -> None:
        # matrices f16, normal times `scale`; biases f32, normal times 0.1; and other vectors f32
        # uniform from 0.5 to 1.5
        if scale:
            weights[name] = (rng.standard_normal(shape) * scale).astype(np.float16)
        elif name.endswith(".bias"):
            weights[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        else:
            weights[name] = rng.uniform(0.5, 1.5, shape).astype(np.float32)
        w.add_tensor(name, weights[name])

    def matrix(name: str, shape: tuple[int, ...]) -> None:  # with a bias, in SigLIP
        add(name + ".weight", shape, 0.2)
        if siglip:
            add(name + ".bias", shape[:1])

    head = V_WIDTH // V_HEADS
    matrix("v.patch_embd", (V_WIDTH, 3, V_PATCH, V_PATCH))
    if siglip:
        w.add_uint32("clip.vision.image_size", G3_IMAGE)
        w.add_uint32("clip.vision.projector.scale_factor", 2)
        w.add_array("clip.vision.image_mean", [0.5] * 3)
        w.add_array("clip.vision.image_std", [0.5] * 3)
        positions = (G3_IMAGE // V_PATCH) ** 2
        weights["v.position_embd.weight"] = rng.standard_normal((positions, V_WIDTH)).astype(
            np.float32)  # fmt: skip
    else:
        weights["v.position_embd.weight"] = rng.standard_normal((2, V_COLUMNS, V_WIDTH)).astype(
            np.float32)  # fmt: skip
    w.add_tensor("v.position_embd.weight", weights["v.position_embd.weight"])
    for i in range(V_LAYERS):
        b = f"v.blk.{i}."
        for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
            matrix(b + name, (V_WIDTH, V_WIDTH))
        if siglip:  # fc1 named down, fc2 up
            matrix(b + "ffn_down", (V_HIDDEN, V_WIDTH))
            matrix(b + "ffn_up", (V_WIDTH, V_HIDDEN))
        else:
            matrix(b + "ffn_gate", (V_HIDDEN, V_WIDTH))
            matrix(b + "ffn_up", (V_HIDDEN, V_WIDTH))
            matrix(b + "ffn_down", (V_WIDTH, V_HIDDEN))
        for name in ("ln1", "ln2") if siglip else ("ln1", "attn_post_norm", "ln2", "ffn_post_norm"):
            add(b + name + ".weight", (V_WIDTH,))
            if siglip:
                add(b + name + ".bias", (V_WIDTH,))
        if not siglip:
            add(b + "attn_q_norm.weight", (head,))
            add(b + "attn_k_norm.weight", (head,))
    if siglip:
        add("v.post_ln.weight", (V_WIDTH,))
        add("v.post_ln.bias", (V_WIDTH,))
        add("mm.soft_emb_norm.weight", (V_WIDTH,))
        add("mm.input_projection.weight", (V_WIDTH, D), 0.2)  # (width, dim), as Gemma 3's
    else:
        add("v.std_bias", (V_WIDTH,))
        add("v.std_scale", (V_WIDTH,))
        add("mm.input_projection.weight", (D, V_WIDTH), 0.2)
    _finish(w)
    return {n: v.astype(np.float64) for n, v in weights.items()}


def write_tiny_pixtral(path: Path) -> dict[str, np.ndarray]:
    # Mistral Small 3's Pixtral over images of up to 4 by 4 patches, of 2 by 2 cells each merged
    rng = np.random.default_rng(2)
    w = gguf.GGUFWriter(path, arch="clip")
    w.add_string("clip.projector_type", "pixtral")
    for key, value in [("patch_size", V_PATCH), ("embedding_length", V_WIDTH),
                       ("feed_forward_length", V_HIDDEN), ("block_count", V_LAYERS),
                       ("attention.head_count", V_HEADS), ("projection_dim", D),
                       ("image_size", P_IMAGE), ("spatial_merge_size", 2)]:  # fmt: skip
        w.add_uint32(f"clip.vision.{key}", value)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", V_EPS)
    w.add_bool("clip.use_silu", True)
    w.add_array("clip.vision.image_mean", list(P_MEAN))
    w.add_array("clip.vision.image_std", list(P_STD))
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], scale: float = 0.0) -> None:
        # matrices f16, normal times `scale`; vectors f32, uniform from 0.5 to 1.5
        values = rng.standard_normal(shape) * scale if scale else rng.uniform(0.5, 1.5, shape)
        weights[name] = values.astype(np.float16 if scale else np.float32)
        w.add_tensor(name, weights[name])

    add("v.patch_embd.weight", (V_WIDTH, 3, V_PATCH, V_PATCH), 0.2)
    add("v.pre_ln.weight", (V_WIDTH,))
    for i in range(V_LAYERS):
        b = f"v.blk.{i}."
        for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
            add(b + name + ".weight", (V_WIDTH, V_WIDTH), 0.2)
        add(b + "ffn_gate.weight", (V_HIDDEN, V_WIDTH), 0.2)
        add(b + "ffn_up.weight", (V_HIDDEN, V_WIDTH), 0.2)
        add(b + "ffn_down.weight", (V_WIDTH, V_HIDDEN), 0.2)
        add(b + "ln1.weight", (V_WIDTH,))
        add(b + "ln2.weight", (V_WIDTH,))
    add("mm.input_norm.weight", (V_WIDTH,))
    add("mm.patch_merger.weight", (V_WIDTH, 4 * V_WIDTH), 0.1)
    add("mm.1.weight", (D, V_WIDTH), 0.2)
    add("mm.2.weight", (D, D), 0.1)
    weights["v.token_embd.img_break"] = rng.standard_normal(D).astype(np.float32)
    w.add_tensor("v.token_embd.img_break", weights["v.token_embd.img_break"])
    _finish(w)
    return {n: v.astype(np.float64) for n, v in weights.items()}


def reference_pixtral(w: dict[str, np.ndarray], pixels: np.ndarray) -> np.ndarray:
    # the embeddings of an image's pixels (H, W, 3), as transformers' Mistral 3: its patches,
    # normalized and projected, RMSNormed, through the layers, whose q and k, of adjacent pairs
    # in the GGUF, turn the first half of each head's pairs by the patch's row at RoPE's even
    # frequencies and the others by its column at the odd; each cell's 2 by 2 patches, normed, as
    # a row of their channels' patches, merged and projected; each row of cells ended by a break
    # but the last
    rows, columns = pixels.shape[0] // V_PATCH, pixels.shape[1] // V_PATCH
    patches = pixels.reshape(rows, V_PATCH, columns, V_PATCH, 3).transpose(0, 2, 1, 3, 4)
    x = (patches.reshape(rows * columns, -1, 3) / 255 - P_MEAN) / P_STD
    x = x.reshape(rows * columns, -1) @ w["v.patch_embd.weight"].transpose(0, 2, 3, 1).reshape(
        V_WIDTH, -1).T  # fmt: skip

    def rms(z, weight):
        return z / np.sqrt((z * z).mean(-1, keepdims=True) + V_EPS) * weight

    head = V_WIDTH // V_HEADS
    freqs = 10000.0 ** (-np.arange(0, head, 2) / head)  # a pair's
    row, column = np.repeat(np.arange(rows), columns), np.tile(np.arange(columns), rows)
    pairs = head // 2  # the first half's turned by row, the second's by column
    angles = np.concatenate([row[:, None] * freqs[0::2], column[:, None] * freqs[1::2]], -1)
    assert angles.shape[-1] == pairs
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]

    def rope(z):
        a, b = z[..., 0::2], z[..., 1::2]
        out = np.empty_like(z)
        out[..., 0::2], out[..., 1::2] = a * cos - b * sin, a * sin + b * cos
        return out

    x = rms(x, w["v.pre_ln.weight"])
    for i in range(V_LAYERS):
        lw = _layer({k[2:]: v for k, v in w.items() if k.startswith("v.")}, i)
        h = rms(x, lw["ln1"])
        q, k, v = ((h @ lw[f"attn_{c}"].T).reshape(-1, V_HEADS, head) for c in "qkv")
        x = x + _full_attention(rope(q), rope(k), v, head**-0.5) @ lw["attn_out"].T
        x = x + mlp(rms(x, lw["ln2"]), lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"])
    grid = rms(x, w["mm.input_norm.weight"]).reshape(rows // 2, 2, columns // 2, 2, V_WIDTH)
    cells = grid.transpose(0, 2, 4, 1, 3).reshape((rows // 2) * (columns // 2), -1)
    x = cells @ w["mm.patch_merger.weight"].T @ w["mm.1.weight"].T
    x = 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2))) @ w["mm.2.weight"].T
    lines = x.reshape(rows // 2, columns // 2, D)
    breaks = np.broadcast_to(w["v.token_embd.img_break"], (rows // 2, 1, D))
    return np.concatenate([lines, breaks], 1).reshape(-1, D)[:-1]


def write_tiny_qwen_vl(path: Path) -> dict[str, np.ndarray]:
    # Qwen3.5's encoder, Qwen3-VL's merger, of patches of 4 by 4 pixels, each cell of 2 by 2
    rng = np.random.default_rng(3)
    w = gguf.GGUFWriter(path, arch="clip")
    w.add_string("clip.projector_type", "qwen3vl_merger")
    for key, value in [("patch_size", V_PATCH), ("embedding_length", V_WIDTH),
                       ("feed_forward_length", V_HIDDEN), ("block_count", V_LAYERS),
                       ("attention.head_count", V_HEADS), ("projection_dim", D),
                       ("spatial_merge_size", 2)]:  # fmt: skip
        w.add_uint32(f"clip.vision.{key}", value)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", V_EPS)
    w.add_bool("clip.use_gelu", True)
    w.add_array("clip.vision.image_mean", [0.5] * 3)
    w.add_array("clip.vision.image_std", [0.5] * 3)
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], scale: float = 0.0) -> None:
        # matrices f16, normal times `scale`; biases f32, normal times 0.1; other vectors f32,
        # uniform from 0.5 to 1.5
        if scale:
            values = (rng.standard_normal(shape) * scale).astype(np.float16)
        elif name.endswith(".bias"):
            values = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        else:
            values = rng.uniform(0.5, 1.5, shape).astype(np.float32)
        weights[name] = values
        w.add_tensor(name, values)

    def matrix(name: str, shape: tuple[int, ...], scale: float = 0.2) -> None:
        add(name + ".weight", shape, scale)
        add(name + ".bias", shape[:1])

    kernel = (V_WIDTH, 3, V_PATCH, V_PATCH)
    matrix("v.patch_embd", kernel)
    add("v.patch_embd.weight.1", kernel, 0.2)  # the second frame's
    weights["v.position_embd.weight"] = rng.standard_normal((Q_SIDE**2, V_WIDTH)).astype(
        np.float32)  # fmt: skip
    w.add_tensor("v.position_embd.weight", weights["v.position_embd.weight"])
    for i in range(V_LAYERS):
        b = f"v.blk.{i}."
        matrix(b + "attn_qkv", (3 * V_WIDTH, V_WIDTH))
        matrix(b + "attn_out", (V_WIDTH, V_WIDTH))
        matrix(b + "ffn_up", (V_HIDDEN, V_WIDTH))
        matrix(b + "ffn_down", (V_WIDTH, V_HIDDEN))
        for name in ("ln1", "ln2"):
            add(b + name + ".weight", (V_WIDTH,))
            add(b + name + ".bias", (V_WIDTH,))
    add("v.post_ln.weight", (V_WIDTH,))
    add("v.post_ln.bias", (V_WIDTH,))
    matrix("mm.0", (4 * V_WIDTH, 4 * V_WIDTH), 0.1)
    matrix("mm.2", (D, 4 * V_WIDTH), 0.1)
    _finish(w)
    return {n: v.astype(np.float64) for n, v in weights.items()}


def reference_qwen_vl(w: dict[str, np.ndarray], pixels: np.ndarray) -> np.ndarray:
    # the embeddings of an image's pixels (H, W, 3), as transformers' Qwen3.5: its patches, cell
    # by cell, normalized and projected by both frames' kernels, with the learned embeddings of
    # their places bilinearly interpolated, as transformers' linspace spreads the image's grid over
    # the learned one; through the layers, q and k rotated by row over the first quarter of each
    # head's frequencies and by column over the second, as transformers' rotate_half does; normed,
    # each cell's patches joined and projected
    rows, columns = pixels.shape[0] // V_PATCH, pixels.shape[1] // V_PATCH
    grid = pixels.reshape(rows // 2, 2, V_PATCH, columns // 2, 2, V_PATCH, 3)
    patches = grid.transpose(0, 3, 1, 4, 2, 5, 6).reshape(rows * columns, -1, 3)
    order = [(2 * r + i, 2 * c + j) for r in range(rows // 2) for c in range(columns // 2)
             for i in range(2) for j in range(2)]  # fmt: skip
    row, column = (np.array([o[k] for o in order]) for k in (0, 1))
    kernel = w["v.patch_embd.weight"] + w["v.patch_embd.weight.1"]
    x = ((patches / 255 - 0.5) / 0.5).reshape(rows * columns, -1)
    x = x @ kernel.transpose(0, 2, 3, 1).reshape(V_WIDTH, -1).T + w["v.patch_embd.bias"]

    def spread(at, size):  # the learned rows before and after, and how far between
        spot = np.linspace(0, Q_SIDE - 1, size)[at]
        low = np.floor(spot).astype(int)
        return low, np.minimum(low + 1, Q_SIDE - 1), spot - low

    (r0, r1, fr), (c0, c1, fc) = spread(row, rows), spread(column, columns)
    table = w["v.position_embd.weight"]
    x = x + sum(table[a * Q_SIDE + b] * wt[:, None] for a, b, wt in [
        (r0, c0, (1 - fr) * (1 - fc)), (r0, c1, (1 - fr) * fc),
        (r1, c0, fr * (1 - fc)), (r1, c1, fr * fc)])  # fmt: skip
    head = V_WIDTH // V_HEADS
    freqs = 10000.0 ** (-np.arange(0, head // 2, 2) / (head // 2))
    angles = np.concatenate([row[:, None] * freqs, column[:, None] * freqs], -1)
    cos, sin = (f(np.concatenate([angles, angles], -1))[:, None] for f in (np.cos, np.sin))

    def rope(z):
        return z * cos + np.concatenate([-z[..., head // 2 :], z[..., : head // 2]], -1) * sin

    def layernorm(z, name):
        z = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + V_EPS)
        return z * w[f"{name}.weight"] + w[f"{name}.bias"]

    def linear(z, name):
        return z @ w[f"{name}.weight"].T + w[f"{name}.bias"]

    for i in range(V_LAYERS):
        b = f"v.blk.{i}."
        q, k, v = np.split(linear(layernorm(x, b + "ln1"), b + "attn_qkv"), 3, -1)
        q, k, v = (z.reshape(-1, V_HEADS, head) for z in (q, k, v))
        x = x + linear(_full_attention(rope(q), rope(k), v, head**-0.5), b + "attn_out")
        g = linear(layernorm(x, b + "ln2"), b + "ffn_up")
        x = x + linear(glu("gelu", g, 1.0), b + "ffn_down")
    x = layernorm(x, "v.post_ln").reshape(-1, 4 * V_WIDTH)
    x = linear(x, "mm.0")
    return linear(0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2))), "mm.2")


def reference_siglip(w: dict[str, np.ndarray], pixels: np.ndarray) -> np.ndarray:
    # the embeddings (tokens, D) of an image's pixels (G3_IMAGE square, 3), as transformers'
    # Gemma 3: SigLIP over its patches, normalized to [-1, 1], with a learned position each, then
    # pooled 2 by 2, RMSNormed and projected
    side = G3_IMAGE // V_PATCH
    patches = pixels.reshape(side, V_PATCH, side, V_PATCH, 3).transpose(0, 2, 1, 3, 4)
    x = (patches.reshape(side * side, -1) / 255 - 0.5) / 0.5
    kernel = w["v.patch_embd.weight"].transpose(0, 2, 3, 1).reshape(V_WIDTH, -1)
    x = x @ kernel.T + w["v.patch_embd.bias"] + w["v.position_embd.weight"]
    head = V_WIDTH // V_HEADS

    def layernorm(z, name):
        z = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + V_EPS)
        return z * w[f"{name}.weight"] + w[f"{name}.bias"]

    def linear(z, name):
        return z @ w[f"{name}.weight"].T + w[f"{name}.bias"]

    for i in range(V_LAYERS):
        b = f"v.blk.{i}."
        h = layernorm(x, b + "ln1")
        q, k, v = (linear(h, b + f"attn_{c}").reshape(-1, V_HEADS, head) for c in "qkv")
        out = _full_attention(q, k, v, head**-0.5)
        x = x + linear(out, b + "attn_out")
        g = linear(layernorm(x, b + "ln2"), b + "ffn_down")  # fc1
        x = x + linear(glu("gelu", g, 1.0), b + "ffn_up")
    x = layernorm(x, "v.post_ln")
    x = x.reshape(side // 2, 2, side // 2, 2, V_WIDTH).mean((1, 3)).reshape(-1, V_WIDTH)
    x = x / np.sqrt((x * x).mean(-1, keepdims=True) + V_EPS) * w["mm.soft_emb_norm.weight"]
    return x @ w["mm.input_projection.weight"]


def _full_attention(q, k, v, scale):
    # every position attending over every other, in f64: (n, heads * head)
    out = []
    for hd in range(q.shape[1]):
        scores = q[:, hd] @ k[:, hd].T * scale
        p = np.exp(scores - scores.max(-1, keepdims=True))
        out.append(p / p.sum(-1, keepdims=True) @ v[:, hd])
    return np.concatenate(out, -1)


def reference_image(w: dict[str, np.ndarray], pixels: np.ndarray, pool: int = 3) -> np.ndarray:
    # the embeddings (tokens, D) of an image's pixels (H, W, 3), as transformers' Gemma 4: its
    # patches, rows of RGB, projected, with learned embeddings of their columns and rows, through
    # the layers, then pooled each `pool` by `pool`, scaled, standardized, normed and projected
    rows, columns = pixels.shape[0] // V_PATCH, pixels.shape[1] // V_PATCH
    patches = pixels.reshape(rows, V_PATCH, columns, V_PATCH, 3).transpose(0, 2, 1, 3, 4)
    x = (patches.reshape(rows * columns, -1) / 255 - 0.5) * 2
    x = x @ w["v.patch_embd.weight"].transpose(0, 2, 3, 1).reshape(V_WIDTH, -1).T
    column, row = np.tile(np.arange(columns), rows), np.repeat(np.arange(rows), columns)
    x = x + w["v.position_embd.weight"][0][column] + w["v.position_embd.weight"][1][row]
    head, n = V_WIDTH // V_HEADS, len(x)
    freqs = 100.0 ** (-np.arange(0, head // 2, 2) / (head // 2))

    def rope(z):  # each half of a head's dimensions rotated by the column, then by the row
        halves = []
        for part, at in zip(np.split(z, 2, axis=-1), (column, row), strict=True):
            angles = at[:, None, None] * freqs
            halves.append(rotate_halves(part, np.cos(angles), np.sin(angles)))
        return np.concatenate(halves, -1)

    def vnorm(z, weight=1.0):
        return z / np.sqrt((z * z).mean(-1, keepdims=True) + V_EPS) * weight

    for i in range(V_LAYERS):
        lw = _layer({k[2:]: v for k, v in w.items() if k.startswith("v.")}, i)
        h = vnorm(x, lw["ln1"])
        q, k, v = ((h @ lw[f"attn_{c}"].T).reshape(n, V_HEADS, head) for c in "qkv")
        q, k = rope(vnorm(q, lw["attn_q_norm"])), rope(vnorm(k, lw["attn_k_norm"]))
        out = _full_attention(q, k, vnorm(v), 1.0)
        x = x + vnorm(out @ lw["attn_out"].T, lw["attn_post_norm"])
        out = mlp(vnorm(x, lw["ln2"]), lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"], "gelu")
        x = x + vnorm(out, lw["ffn_post_norm"])
    cells = x.reshape(rows // pool, pool, columns // pool, pool, V_WIDTH).mean((1, 3))
    x = cells.reshape(-1, V_WIDTH) * np.sqrt(V_WIDTH)
    x = (x - w["v.std_bias"]) * w["v.std_scale"]
    return vnorm(x) @ w["mm.input_projection.weight"].T


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
    w, weights, add = _writer(path, "gemma3", ("<start_of_image>", "<end_of_image>"))
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


def _reference_gemma3(
    w: dict[str, np.ndarray], tokens: list[int], images: dict[int, np.ndarray] | None = None
) -> np.ndarray:
    T, positions = len(tokens), np.arange(len(tokens))
    x, image = _embedded(w, tokens, images), _same_image(tokens)
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
        mask = _mask(T, G3_WINDOW if sliding else 0, image)
        out = attention(q, k, v, mask, 1 / np.sqrt(HEAD_DIM))
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
Q35_SECTIONS = (3, 3, 2)  # of M-RoPE's 8 frequencies: of time, height and width
Q_IMAGE = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
Q_SIDE = 4  # of the learned grid of the tiny Qwen3-VL's places
Q35_K_HEADS, Q35_V_HEADS, Q35_DIM, Q35_CONV, Q35_SHARED = 2, 4, 32, 4, 128
Q35_CHANNELS = (2 * Q35_K_HEADS + Q35_V_HEADS) * Q35_DIM


def _write_qwen35(path: Path, arch: str) -> dict[str, np.ndarray]:
    # Qwen3.5's and Qwen3.6's, of an MLP in each layer, qwen35, or of experts beside a shared one,
    # qwen35moe
    w, weights, add = _writer(path, arch, Q_IMAGE)
    a, moe = f"{arch}.", arch == "qwen35moe"
    feed_forward = [("expert_count", EXPERTS), ("expert_used_count", USED),
                    ("expert_feed_forward_length", EXPERT_HIDDEN),
                    ("expert_shared_feed_forward_length", Q35_SHARED),
                    ] if moe else [("feed_forward_length", HIDDEN)]  # fmt: skip
    for key, value in [("block_count", Q35_EVERY + 1), ("nextn_predict_layers", 1),
                       ("embedding_length", D), ("attention.head_count", HEADS),
                       ("attention.head_count_kv", KV_HEADS), ("attention.key_length", Q35_HEAD),
                       ("attention.value_length", Q35_HEAD),
                       ("rope.dimension_count", Q35_ROTATED), *feed_forward,
                       ("ssm.conv_kernel", Q35_CONV), ("ssm.state_size", Q35_DIM),
                       ("ssm.group_count", Q35_K_HEADS), ("ssm.time_step_rank", Q35_V_HEADS),
                       ("ssm.inner_size", Q35_V_HEADS * Q35_DIM),
                       ("full_attention_interval", Q35_EVERY)]:  # fmt: skip
        w.add_uint32(a + key, value)
    w.add_float32(a + "rope.freq_base", 1e7)
    w.add_array(a + "rope.dimension_sections", [*Q35_SECTIONS, 0])
    add("token_embd.weight", *TENSORS["token_embd.weight"])
    add("output.weight", *TENSORS["output.weight"])
    add("output_norm.weight", (D,))
    rng = np.random.default_rng(1)
    # the model's layers, then its MTP layer, of full attention, which drafts its next tokens
    for i in range(Q35_EVERY + 1):
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
        for name in MOE if moe else MLP:
            add(b + name + ".weight", *TENSORS[name])
        if moe:
            add(b + "ffn_gate_inp_shexp.weight", (D,), GGMLType.F32, 1.2)
            add(b + "ffn_gate_shexp.weight", (Q35_SHARED, D), GGMLType.Q4_K, 2e-4)
            add(b + "ffn_up_shexp.weight", (Q35_SHARED, D), GGMLType.Q6_K, 5e-5)
            add(b + "ffn_down_shexp.weight", (D, Q35_SHARED), GGMLType.Q8_0, 1e-3)
    b = f"blk.{Q35_EVERY}.nextn."
    for name in ("enorm", "hnorm", "shared_head_norm"):
        add(b + name + ".weight", (D,))
    add(b + "eh_proj.weight", (D, 2 * D), GGMLType.Q8_0, 1e-3)
    _finish(w)
    return weights


def split_mtp(path: Path, folder: Path, name: str) -> tuple[Path, Path]:
    # a Qwen3.5 GGUF with its MTP layer as llama.cpp's converters write Qwen3.8's: the model's
    # file without the layer, of its layers alone, and mtp-*.gguf of the layer, the embeddings and
    # the output, of the whole's metadata; each of general.name `name`
    reader = gguf.GGUFReader(path)
    arch = reader.fields["general.architecture"].contents()
    blocks, nextn = f"{arch}.block_count", f"{arch}.nextn_predict_layers"
    layers = reader.fields[blocks].contents() - 1
    layer = f"blk.{layers}."
    files = {
        folder / f"{name}.gguf": lambda n: not n.startswith(layer),
        folder / f"mtp-{name}.gguf": lambda n: n.startswith((layer, "token_embd", "output")),
    }
    for file, keep in files.items():
        w = gguf.GGUFWriter(file, arch=arch)
        w.add_name(name)
        alone = not mtp(file)  # the model's layers, none past them
        if alone:
            w.add_uint32(blocks, layers)
        for field in reader.fields.values():
            if field.name.startswith("GGUF.") or field.name == "general.architecture":
                continue
            if alone and field.name in (blocks, nextn):
                continue
            sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), field.types[0], sub)
        for t in reader.tensors:
            if keep(t.name):
                w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        _finish(w)
    return tuple(files)  # type: ignore[return-value]


def _reference_qwen35(
    w: dict[str, np.ndarray], tokens: list[int], images: dict[int, np.ndarray] | None = None,
    grids: dict[int, tuple[int, int]] | None = None,
) -> np.ndarray:  # fmt: skip
    return _qwen35_hidden(w, tokens, images, grids) @ w["output.weight"].T


def _qwen35_hidden(
    w: dict[str, np.ndarray], tokens: list[int], images: dict[int, np.ndarray] | None = None,
    grids: dict[int, tuple[int, int]] | None = None,
) -> np.ndarray:  # fmt: skip
    # the normed hidden states; images' embeddings by key in place of the tokens that hold them,
    # at M-RoPE's positions of their grids, by key
    x, places = _embedded(w, tokens, images, 1.0), _mrope_places(tokens, grids or {})
    for i in range(Q35_EVERY):
        x = _qwen35_layer(_layer(w, i), x, i < Q35_EVERY - 1, places)
    return norm(x, w["output_norm.weight"])


def _mrope_places(tokens: list[int], grids: dict[int, tuple[int, int]]) -> np.ndarray:
    # each token's time, height and width, as transformers' Qwen3.5 has them: of text, the next
    # position on each; of an image, its first's on time, and that plus its row and column, after
    # which text goes on past its longer side
    places, at, i = [], 0, 0
    while i < len(tokens):
        if tokens[i] >= 0:
            places.append((at, at, at))
            at, i = at + 1, i + 1
            continue
        rows, columns = grids[tokens[i]]
        places += [(at, at + n // columns, at + n % columns) for n in range(rows * columns)]
        at, i = at + max(rows, columns), i + rows * columns
    return np.array(places)


def _qwen35_layer(
    lw: dict[str, np.ndarray], x: np.ndarray, recurrent: bool, places: np.ndarray
) -> np.ndarray:
    # a Gated DeltaNet or full-attention layer of tokens at M-RoPE's places (T, 3), then its MLP,
    # or its experts and shared expert; frequency j turns by height if j % 3 is 1 within the
    # height's section, by width if 2 within the width's, else by time
    T = len(x)
    half = Q35_ROTATED // 2
    axes = [1 if j % 3 == 1 and j < 3 * Q35_SECTIONS[1] else 2 if j % 3 == 2 and
            j < 3 * Q35_SECTIONS[2] else 0 for j in range(half)]  # fmt: skip
    freqs = 1e7 ** (-np.arange(0, Q35_ROTATED, 2) / Q35_ROTATED)
    angles = (places[:, axes] * freqs)[:, None, :]
    cos, sin = np.cos(angles), np.sin(angles)

    def rope(z):  # the first Q35_ROTATED dimensions, i with i + Q35_ROTATED / 2
        return np.concatenate(
            [rotate_halves(z[..., :Q35_ROTATED], cos, sin), z[..., Q35_ROTATED:]], -1
        )

    h = norm(x, lw["attn_norm"])
    if recurrent:
        out = _gated_delta_net(lw, h)
    else:
        q, gate = np.split((h @ lw["attn_q"].T).reshape(T, HEADS, 2 * Q35_HEAD), 2, -1)
        k, v = ((h @ lw[n].T).reshape(T, KV_HEADS, Q35_HEAD) for n in ("attn_k", "attn_v"))
        q, k = rope(norm(q, lw["attn_q_norm"])), rope(norm(k, lw["attn_k_norm"]))
        out = attention(q, k, v, _mask(T, 0), 1 / np.sqrt(Q35_HEAD))
        out = out * sigmoid(gate.reshape(T, -1)) @ lw["attn_output"].T
    x = x + out
    h = norm(x, lw["post_attention_norm"])
    if "ffn_gate_inp" not in lw:
        return x + mlp(h, lw["ffn_gate"], lw["ffn_up"], lw["ffn_down"])
    x = x + experts(
        h,
        h @ lw["ffn_gate_inp"].T,
        lambda e, row: mlp(
            row, lw["ffn_gate_exps"][e], lw["ffn_up_exps"][e], lw["ffn_down_exps"][e]
        ),
    )
    shared = mlp(h, lw["ffn_gate_shexp"], lw["ffn_up_shexp"], lw["ffn_down_shexp"])
    return x + shared * sigmoid(h @ lw["ffn_gate_inp_shexp"])[:, None]


def reference_mtp_drafts(w: dict[str, np.ndarray], tokens: list[int], count: int) -> list[int]:
    # the MTP layer's `count` greedy drafts after `tokens`, the last at a position the model has
    # not run: the layer runs every position, of each token with the model's hidden state before
    # it, none before the first; then the drafts', each with the layer's own state before it
    hidden, lw = _qwen35_hidden(w, tokens[:-1]), _layer(w, Q35_EVERY)
    tokens, before, drafts = list(tokens), [np.zeros(D), *hidden], []
    for _ in range(count):
        e = norm(w["token_embd.weight"][tokens].astype(np.float64), lw["nextn.enorm"])
        h = norm(np.stack(before), lw["nextn.hnorm"])
        x = np.concatenate([e, h], -1) @ lw["nextn.eh_proj"].T
        places = _mrope_places(tokens, {})
        out = norm(_qwen35_layer(lw, x, False, places), lw["nextn.shared_head_norm"])[-1]
        drafts.append(int((out @ w["output.weight"].T).argmax()))
        tokens.append(drafts[-1])
        before.append(out)
    return drafts


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


def _mask(T: int, window: int, image: np.ndarray | None = None) -> np.ndarray:
    # causal but for the positions of the same image, as `image` (T, T) says, which see each
    # other; and over the last `window` positions if given
    back = np.arange(T)[:, None] - np.arange(T)
    later = (back < 0) & ~image if image is not None else back < 0
    return np.where(later | (window > 0) & (back >= window), -np.inf, 0)


_WRITERS = {
    "gemma3": _write_gemma3, "gpt-oss": _write_gpt_oss, "phi3": _write_phi3,
    "qwen35": lambda path: _write_qwen35(path, "qwen35"),
    "qwen35moe": lambda path: _write_qwen35(path, "qwen35moe"),
}  # fmt: skip
_REFERENCES = {
    "gpt-oss": _reference_gpt_oss, "phi3": _reference_phi3,
}  # fmt: skip
