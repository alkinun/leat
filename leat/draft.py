"""Drafters for speculative decoding: small models that guess the tokens a target model will
generate next, which the target then checks all at once, reading its weights once for them all."""

from pathlib import Path

from tinygrad import Tensor, UOp

from leat import ops
from leat.gguf import GGUF
from leat.model import Transformer, rope_table
from leat.ops import Span

_LAYER = (
    "attn_norm", "attn_q", "attn_q_norm", "attn_output", "post_attention_norm",
    "ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm", "layer_output_scale",
)  # fmt: skip


class Gemma4Assistant:
    """Gemma 4's assistant, which drafts for a Gemma 4 target: a few layers of their own, whose
    attention has queries alone and reads the target's keys and values, of its last sliding-window
    layer for layers of a window and of its last layer for the others. A step takes a token at a
    position and the target's normed hidden state before it, their embeddings joined and
    projected, and gives the next token and a hidden state as the target's, from which the next
    step drafts another at the same position, as llama.cpp's."""

    def __init__(self, path: str | Path, target: Transformer):
        gguf = GGUF.open(path)
        arch, c = gguf.metadata["general.architecture"], target.config
        if arch != "gemma4-assistant" or c.arch != "gemma4":
            raise ValueError(f"a {arch} model drafts for no {c.arch} model")
        m = {k.removeprefix(f"{arch}."): v for k, v in gguf.metadata.items()}
        if (width := m["embedding_length_out"]) != c.dim:
            raise ValueError(f"the drafter is for a model {width} wide, not {c.dim}")
        self.target, self.eps = target, m["attention.layer_norm_rms_epsilon"]
        self.heads = m["attention.head_count"]
        sliding = m["attention.sliding_window_pattern"][: m["block_count"]]
        swa = max(i for i in range(c.n_layers) if c.windows[i])
        self.sources = [swa if s else c.n_layers - 1 for s in sliding]
        w = gguf.load()
        self.layers = [
            {part: w[f"blk.{i}.{part}.weight"] for part in _LAYER} for i in range(len(sliding))
        ]
        for i, (layer, source) in enumerate(zip(self.layers, self.sources, strict=True)):
            if layer["attn_q"].shape[0] != self.heads * c.head_dims[source]:
                raise ValueError(f"the drafter's layer {i} has heads unlike its target layer's")
        # norms and scales, decoded once
        self.small = [
            {k: v.dequant().realize() for k, v in layer.items() if k.endswith(("norm", "scale"))}
            for layer in self.layers
        ]
        self.pre, self.post = w["nextn.pre_projection.weight"], w["nextn.post_projection.weight"]
        self.embed, self.norm = w["token_embd.weight"], w["output_norm.weight"].dequant().realize()
        # RoPE as the target layers', by the drafter's own factors for those that see all
        factors = w["rope_freqs.weight"].dequant() if "rope_freqs.weight" in w else None
        self.rope: dict[int, tuple[Tensor, Tensor]] = {}
        for source in set(self.sources):
            if (rope := c.ropes[source]).dims:
                own = factors if rope.freqs else None
                self.rope[source] = rope_table(rope, target.max_context, own)

    def draft(
        self, token: Tensor, hidden: Tensor, slot: int | UOp, pos: int | UOp, count: int
    ) -> Tensor:
        """`count` tokens drafted greedily after `token` (1, 1) at position `pos` of cache slot
        `slot`, which the target has run up to pos, given its normed hidden state (1, 1, dim)
        at pos - 1: (1, count) int32."""
        scale, drafted = self.target.config.embed_scale, []
        for _ in range(count):
            x = ops.embedding(token, self.target.embed) * scale
            hidden, logits = self._step(ops.linear(x.cat(hidden, dim=-1), self.pre), slot, pos)
            token = ops.argmax(logits.reshape(1, -1)).reshape(1, 1)
            drafted.append(token)
        return drafted[0].cat(*drafted[1:], dim=1)

    def _step(self, x: Tensor, slot: int | UOp, pos: int | UOp) -> tuple[Tensor, Tensor]:
        # the hidden state for the next step and the logits, after x (1, 1, width)
        c, eps = self.target.config, self.eps
        for layer, small, source in zip(self.layers, self.small, self.sources, strict=True):
            q = ops.linear(ops.rms_norm(x, small["attn_norm"], eps), layer["attn_q"])
            q = q.reshape(1, 1, self.heads, c.head_dims[source])
            q = ops.rms_norm(q, small["attn_q_norm"], eps).transpose(1, 2)
            if source in self.rope:
                cos, sin = (table[pos : pos + 1] for table in self.rope[source])
                q = ops.rotary(q, cos, sin, c.rope_halves)
            # a query at pos sees the target's positions before it, as one at pos - 1 sees those
            # and its own, and one fewer back for a window
            cache, window = self.target.cache[source], max(c.windows[source] - 1, 0)
            assert cache is not None
            out = ops.attention(q, cache, [Span(slot, pos - 1)], c.scales[source], window)
            out = ops.linear(out, layer["attn_output"])
            x = x + ops.rms_norm(out, small["post_attention_norm"], eps)
            mlp = (layer["ffn_gate"], layer["ffn_up"], layer["ffn_down"])
            out = ops.feed_forward(x, *mlp, (small["ffn_norm"], eps), c.glu, residual=False)
            x = (x + ops.rms_norm(out, small["post_ffw_norm"], eps)) * small["layer_output_scale"]
        x = ops.rms_norm(x, self.norm, eps)
        return ops.linear(x, self.post), ops.linear(x, self.embed)
