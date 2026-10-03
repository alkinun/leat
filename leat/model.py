"""Decoder-only transformer, configured entirely from GGUF metadata."""

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, UOp, dtypes

from leat import ops
from leat.quant import QTensor

CACHE_TILE = 256  # positions
_LAYER = ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm")
_MLP = ("ffn_gate", "ffn_up", "ffn_down")
_EXPERTS = ("ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")  # router first
_QK_NORM = ("attn_q_norm", "attn_k_norm")  # RMSNorm of each head of q and k, before RoPE
# the supported architectures, and whether their RoPE rotates dimension i with i + D/2, ggml's
# "neox" mode, rather than adjacent pairs, its "normal" mode, as GGUF lays out llama's q and k
_ROPE_HALVES = {"llama": False, "qwen3": True, "qwen3moe": True}


@dataclass(frozen=True)
class Config:
    n_layers: int
    dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    norm_eps: float
    rope_theta: float
    rope_halves: bool
    context_length: int
    experts: int = 0  # MLPs per layer of a mixture of experts, 0 for one MLP
    experts_used: int = 0  # experts each token takes

    @staticmethod
    def from_gguf(metadata: dict[str, Any]) -> "Config":
        if (arch := metadata["general.architecture"]) not in _ROPE_HALVES:
            raise NotImplementedError(f"architecture {arch!r} is not supported")
        m = {k.removeprefix(f"{arch}."): v for k, v in metadata.items()}
        n_heads = m["attention.head_count"]
        head_dim = m.get("attention.key_length", m["embedding_length"] // n_heads)
        if m.get("rope.dimension_count", head_dim) != head_dim:
            raise NotImplementedError("partial rotary embeddings are not supported")
        return Config(
            n_layers=m["block_count"],
            dim=m["embedding_length"],
            hidden_dim=m["feed_forward_length"],
            n_heads=n_heads,
            n_kv_heads=m.get("attention.head_count_kv", n_heads),
            head_dim=head_dim,
            vocab_size=len(metadata["tokenizer.ggml.tokens"]),
            norm_eps=m["attention.layer_norm_rms_epsilon"],
            rope_theta=m.get("rope.freq_base", 10000.0),
            rope_halves=_ROPE_HALVES[arch],
            context_length=m["context_length"],
            experts=m.get("expert_count", 0),
            experts_used=m.get("expert_used_count", 0),
        )


class Transformer:
    """Weights, RoPE tables and a KV cache of `slots` sequences of up to `max_context` tokens."""

    def __init__(
        self, config: Config, weights: dict[str, QTensor], max_context: int, slots: int = 1
    ):
        if not 0 < max_context <= config.context_length:
            raise ValueError(f"max_context must be in [1, {config.context_length}]")
        self.config, self.max_context = config, max_context
        self.qk_norm = "blk.0.attn_q_norm.weight" in weights
        names = _LAYER + (_EXPERTS if config.experts else _MLP) + (_QK_NORM if self.qk_norm else ())
        expected = [f"blk.{i}.{n}.weight" for i in range(config.n_layers) for n in names]
        if missing := [name for name in expected if name not in weights]:
            raise ValueError(f"missing {len(missing)} tensors, first: {missing[0]}")
        self.layers = [
            {n: weights[f"blk.{i}.{n}.weight"] for n in names} for i in range(config.n_layers)
        ]
        self.norms = [
            {n: w.dequant() for n, w in layer.items() if n.endswith("norm")}
            for layer in self.layers
        ]
        self.embed = weights["token_embd.weight"]
        self.output = weights.get("output.weight", self.embed)  # tied embeddings when absent
        self.output_norm = weights["output_norm.weight"].dequant()
        factors = (
            None if "rope_freqs.weight" not in weights else weights["rope_freqs.weight"].dequant()
        )
        self.cos, self.sin = _rope_table(config, max_context, factors)
        # whole tiles of positions, which the attention kernels need; the rest stay unused
        positions = -(-max_context // CACHE_TILE) * CACHE_TILE
        shape = (2, slots, config.n_kv_heads, positions, config.head_dim)
        self.cache = [
            Tensor.zeros(shape, dtype=dtypes.half).contiguous().realize()
            for _ in range(config.n_layers)
        ]

    def __call__(self, tokens: Tensor, start_pos: int | UOp, slot: int | UOp = 0) -> Tensor:
        """Runs `tokens` (1, T) at positions `start_pos...` of cache slot `slot` and returns normed
        hidden states."""
        x = ops.embedding(tokens, self.embed)
        for i in range(self.config.n_layers):
            x = self._block(i, x, start_pos, slot)
        return ops.rms_norm(x, self.output_norm, self.config.norm_eps)

    def logits(self, hidden: Tensor) -> Tensor:
        return ops.linear(hidden, self.output)

    def copy(self, source: int | UOp, slot: int | UOp) -> None:
        """Copies the cache of slot `source` to slot `slot`."""
        # Whole slots, as tinygrad runs a copy of a bound number of positions on few threads: for
        # Llama 3.1 8B on the 3090, 2130 positions took 12.3 ms, whole slots of 4096 3.1 ms.
        for cache in self.cache:
            cache[:, slot : slot + 1].assign(cache[:, source : source + 1])
        Tensor.realize(*self.cache)

    def _block(self, i: int, x: Tensor, start_pos: int | UOp, slot: int | UOp) -> Tensor:
        c, w, norms = self.config, self.layers[i], self.norms[i]
        B, T, _ = x.shape
        q, k, v = ops.linears(
            x, w["attn_q"], w["attn_k"], w["attn_v"], norm=(norms["attn_norm"], c.norm_eps)
        )
        q = q.reshape(B, T, c.n_heads, c.head_dim)
        k = k.reshape(B, T, c.n_kv_heads, c.head_dim)
        if self.qk_norm:
            q = ops.rms_norm(q, norms["attn_q_norm"], c.norm_eps)
            k = ops.rms_norm(k, norms["attn_k_norm"], c.norm_eps)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        cos, sin = self.cos[start_pos : start_pos + T], self.sin[start_pos : start_pos + T]
        q, k = ops.rope(q, cos, sin, c.rope_halves), ops.rope(k, cos, sin, c.rope_halves)

        cache = self.cache[i]
        new = Tensor.stack(k, v).cast(cache.dtype)
        cache[:, slot : slot + 1, :, start_pos : start_pos + T].assign(new)
        x = ops.linear(ops.attention(q, cache, slot, start_pos), w["attn_output"], residual=x)

        norm = (norms["ffn_norm"], c.norm_eps)
        if c.experts:
            router, gate, up, down = (w[n] for n in _EXPERTS)
            return ops.mixture(x, router, gate, up, down, c.experts_used, norm)
        return ops.feed_forward(x, w["ffn_gate"], w["ffn_up"], w["ffn_down"], norm=norm)


def _rope_table(config: Config, length: int, factors: Tensor | None) -> tuple[Tensor, Tensor]:
    # angle = pos * theta^(-2i/d) / factor_i in f32, the order llama.cpp's rope kernels use
    half = config.head_dim // 2
    freqs = Tensor([config.rope_theta ** (-2 * i / config.head_dim) for i in range(half)])
    angles = Tensor.arange(length).float().unsqueeze(1) * freqs.unsqueeze(0)
    if factors is not None:
        angles = angles / factors.unsqueeze(0)
    return angles.cos().contiguous().realize(), angles.sin().contiguous().realize()
