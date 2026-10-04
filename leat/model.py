"""Decoder-only transformer, configured entirely from GGUF metadata."""

import math
from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, UOp, dtypes

from leat import ops
from leat.quant import QTensor

CACHE_TILE = 256  # positions
_LAYER = ("attn_norm", "attn_q", "attn_k", "attn_output", "ffn_norm")
_MLP = ("ffn_gate", "ffn_up", "ffn_down")
_EXPERTS = ("ffn_gate_inp", "ffn_down_exps")  # the router, and the experts' down projections
# the supported architectures, and whether their RoPE rotates dimension i with i + D/2, ggml's
# "neox" mode, rather than adjacent pairs, its "normal" mode, as GGUF lays out llama's q and k
_ROPE_HALVES = {"llama": False, "qwen2": True, "qwen3": True, "qwen3moe": True, "gemma4": True}


@dataclass(frozen=True)
class Config:
    n_layers: int
    dim: int
    hidden_dim: int
    n_heads: int
    vocab_size: int
    norm_eps: float
    context_length: int
    rope_halves: bool
    # each layer's attention: kv heads, head size, how many positions back it sees (0 for all)
    # and the base of its RoPE frequencies
    kv_heads: tuple[int, ...]
    head_dims: tuple[int, ...]
    windows: tuple[int, ...]
    rope_thetas: tuple[float, ...]
    experts: int = 0  # MLPs per layer of a mixture of experts, 0 for one MLP
    experts_used: int = 0  # experts each token takes
    # Gemma 4: embeddings scaled by sqrt(dim), unit attention scale rather than 1/sqrt(head
    # size), values normed like keys, GELU rather than SiLU gating its MLPs, and logits capped
    gemma: bool = False
    logit_cap: float = 0.0

    @staticmethod
    def from_gguf(metadata: dict[str, Any]) -> "Config":
        if (arch := metadata["general.architecture"]) not in _ROPE_HALVES:
            raise NotImplementedError(f"architecture {arch!r} is not supported")
        m = {k.removeprefix(f"{arch}."): v for k, v in metadata.items()}
        n_layers, n_heads = m["block_count"], m["attention.head_count"]

        def per_layer(value: Any) -> tuple:
            return tuple(value) if isinstance(value, list) else (value,) * n_layers

        # Gemma 4's sliding-window layers have their own head size and RoPE
        sliding = per_layer(m.get("attention.sliding_window_pattern", False))
        head_dim = m.get("attention.key_length", m["embedding_length"] // n_heads)
        rotated = m.get("rope.dimension_count", head_dim), m.get("rope.dimension_count_swa")
        if rotated[0] != head_dim or rotated[1] not in (None, m.get("attention.key_length_swa")):
            raise NotImplementedError("partial rotary embeddings are not supported")
        theta = m.get("rope.freq_base", 10000.0)
        return Config(
            n_layers=n_layers,
            dim=m["embedding_length"],
            hidden_dim=m["feed_forward_length"],
            n_heads=n_heads,
            vocab_size=len(metadata["tokenizer.ggml.tokens"]),
            norm_eps=m["attention.layer_norm_rms_epsilon"],
            context_length=m["context_length"],
            rope_halves=_ROPE_HALVES[arch],
            kv_heads=per_layer(m.get("attention.head_count_kv", n_heads)),
            head_dims=tuple(m["attention.key_length_swa"] if s else head_dim for s in sliding),
            windows=tuple(m["attention.sliding_window"] if s else 0 for s in sliding),
            rope_thetas=tuple(m["rope.freq_base_swa"] if s else theta for s in sliding),
            experts=m.get("expert_count", 0),
            experts_used=m.get("expert_used_count", 0),
            gemma=arch == "gemma4",
            logit_cap=m.get("final_logit_softcapping", 0.0),
        )


class Transformer:
    """Weights, RoPE tables and a KV cache of `slots` sequences of up to `max_context` tokens.

    Optional parts are used where the GGUF has their tensors: biases of q, k and v, RMSNorms of q
    and k, of the attention and MLP outputs, a shared MLP beside the experts, and a scale per layer
    output.
    """

    def __init__(
        self, config: Config, weights: dict[str, QTensor], max_context: int, slots: int = 1
    ):
        if not 0 < max_context <= config.context_length:
            raise ValueError(
                f"max_context must be in [1, {config.context_length}], got {max_context}"
            )
        self.config, self.max_context = config, max_context
        # each layer's tensors by name, without "blk.{i}." and ".weight"
        self.layers: list[dict[str, QTensor]] = [{} for _ in range(config.n_layers)]
        for name, w in weights.items():
            if name.startswith("blk."):
                i, part = name.removeprefix("blk.").split(".", 1)
                self.layers[int(i)][part.removesuffix(".weight")] = w
        needed = _LAYER + (_EXPERTS if config.experts else _MLP)
        missing = [
            f"blk.{i}.{n}" for i, layer in enumerate(self.layers) for n in needed if n not in layer
        ]
        if missing:
            raise ValueError(f"missing {len(missing)} tensors, first: {missing[0]}")
        # norm weights and scales, decoded once
        self.small = [
            {n: w.dequant() for n, w in layer.items() if len(w.shape) == 1} for layer in self.layers
        ]
        # Gemma 4 routes from x normed with a weight of its own, over sqrt(dim)
        for s in self.small:
            if "ffn_gate_inp.scale" in s:
                s["router_norm"] = (s["ffn_gate_inp.scale"] / math.sqrt(config.dim)).realize()
        self.embed = weights["token_embd.weight"]
        self.output = weights.get("output.weight", self.embed)  # tied embeddings when absent
        self.output_norm = weights["output_norm.weight"].dequant()
        # RoPE tables by base and head size; rope_freqs, where there is one, divides the
        # frequencies of the layers that see all positions: all of Llama 3.1's, few of Gemma 4's
        factors = weights.get("rope_freqs.weight")
        tables: dict[tuple, tuple[Tensor, Tensor]] = {}
        self.rope: list[tuple[Tensor, Tensor]] = []
        layers = zip(config.rope_thetas, config.head_dims, config.windows, strict=True)
        for theta, dim, window in layers:
            scaled = None if factors is None or window else factors.dequant()
            if (key := (theta, dim, scaled is None)) not in tables:
                tables[key] = _rope_table(theta, dim, max_context, scaled)
            self.rope.append(tables[key])
        # whole tiles of positions, which the attention kernels need; the rest stay unused
        positions = -(-max_context // CACHE_TILE) * CACHE_TILE
        self.cache = [
            Tensor.zeros(2, slots, kv_heads, positions, dim, dtype=dtypes.half)
            .contiguous()
            .realize()
            for kv_heads, dim in zip(config.kv_heads, config.head_dims, strict=True)
        ]

    def __call__(self, tokens: Tensor, start_pos: int | UOp, slot: int | UOp = 0) -> Tensor:
        """Runs `tokens` (1, T) at positions `start_pos...` of cache slot `slot` and returns normed
        hidden states."""
        x = ops.embedding(tokens, self.embed)
        if self.config.gemma:
            x = x * math.sqrt(self.config.dim)
        for i in range(self.config.n_layers):
            x = self._feed_forward(i, self._attention(i, x, start_pos, slot))
        return ops.rms_norm(x, self.output_norm, self.config.norm_eps)

    def logits(self, hidden: Tensor) -> Tensor:
        logits = ops.linear(hidden, self.output)
        if cap := self.config.logit_cap:
            logits = (logits / cap).tanh() * cap
        return logits

    def copy(self, source: int | UOp, slot: int | UOp) -> None:
        """Copies the cache of slot `source` to slot `slot`."""
        # Whole slots, as tinygrad runs a copy of a bound number of positions on few threads: for
        # Llama 3.1 8B on the 3090, 2130 positions took 12.3 ms, whole slots of 4096 3.1 ms.
        for cache in self.cache:
            cache[:, slot : slot + 1].assign(cache[:, source : source + 1])
        Tensor.realize(*self.cache)

    def _attention(self, i: int, x: Tensor, start_pos: int | UOp, slot: int | UOp) -> Tensor:
        # x + the attention block's output
        c, w, s = self.config, self.layers[i], self.small[i]
        B, T, _ = x.shape
        kv_heads, dim, eps = c.kv_heads[i], c.head_dims[i], c.norm_eps
        # Gemma 4's full-attention layers have no values of their own: the keys are, before norm
        proj = [w["attn_q"], w["attn_k"]] + ([w["attn_v"]] if "attn_v" in w else [])
        q, k, *values = ops.linears(x, *proj, norm=(s["attn_norm"], eps))
        q, k = q.reshape(B, T, c.n_heads, dim), k.reshape(B, T, kv_heads, dim)
        v = (values[0] if values else k).reshape(B, T, kv_heads, dim)
        biases = None
        if "attn_q.bias" in s:  # Qwen2's
            biases = (s["attn_q.bias"], s["attn_k.bias"], s["attn_v.bias"])
        norms = (s["attn_q_norm"], s["attn_k_norm"]) if "attn_q_norm" in s else None
        q, cache = ops.rotate(q, k, v, self.cache[i], slot, start_pos, self.rope[i],
                              c.rope_halves, biases, norms, c.gemma, eps)  # fmt: skip
        scale = 1.0 if c.gemma else 1 / math.sqrt(dim)
        out = ops.attention(q, cache, slot, start_pos, scale, c.windows[i])
        if "post_attention_norm" not in s:
            return ops.linear(out, w["attn_output"], residual=x)
        return ops.add_normed(
            x, [(ops.linear(out, w["attn_output"]), s["post_attention_norm"])], None, eps
        )

    def _feed_forward(self, i: int, x: Tensor) -> Tensor:
        # x + the MLP block's output: an MLP, a mixture of experts, or as in Gemma 4 both, each
        # output normed and then their sum, and the layer's output scaled
        c, w, s = self.config, self.layers[i], self.small[i]
        norm, eps, scale = (s["ffn_norm"], c.norm_eps), c.norm_eps, s.get("layer_output_scale")
        mlp = (w["ffn_gate"], w["ffn_up"], w["ffn_down"]) if "ffn_gate" in w else None
        if mlp and not c.experts:
            if "post_ffw_norm" not in s:
                return ops.feed_forward(x, *mlp, norm)
            out = ops.feed_forward(x, *mlp, norm, c.gemma, residual=False)
            return ops.add_normed(x, [(out, s["post_ffw_norm"])], None, eps, scale)
        scores = ops.router(x, (s.get("router_norm", s["ffn_norm"]), eps), w["ffn_gate_inp"])
        # stacked gate and up matrices, or one stack of both, the gate's rows first
        gate, up = (w["ffn_gate_up_exps"], None) if "ffn_gate_up_exps" in w else (
            w["ffn_gate_exps"], w["ffn_up_exps"])  # fmt: skip
        experts = (scores, gate, up, w["ffn_down_exps"], c.experts_used)
        if not mlp:
            return ops.mixture(x, *experts, norm)
        scales = s["ffn_down_exps.scale"]
        mixed = ops.mixture(x, *experts, (s["pre_ffw_norm_2"], eps), c.gemma, scales, False)
        shared = ops.feed_forward(x, *mlp, norm, c.gemma, residual=False)
        parts = [(shared, s["post_ffw_norm_1"]), (mixed, s["post_ffw_norm_2"])]
        return ops.add_normed(x, parts, s["post_ffw_norm"], eps, scale)


def _rope_table(
    theta: float, dim: int, length: int, factors: Tensor | None
) -> tuple[Tensor, Tensor]:
    # angle = pos * theta^(-2i/d) / factor_i in f32, the order llama.cpp's rope kernels use
    freqs = Tensor([theta ** (-2 * i / dim) for i in range(dim // 2)])
    angles = Tensor.arange(length).float().unsqueeze(1) * freqs.unsqueeze(0)
    if factors is not None:
        angles = angles / factors.unsqueeze(0)
    return angles.cos().contiguous().realize(), angles.sin().contiguous().realize()
