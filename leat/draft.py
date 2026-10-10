"""Drafters for speculative decoding: small models that guess the tokens a target model will
generate next, which the target then checks all at once, reading its weights once for them all."""

from dataclasses import replace
from pathlib import Path
from typing import Protocol

from tinygrad import Tensor, UOp

from leat import ops
from leat.gguf import GGUF
from leat.model import QWEN35, Transformer, rope_table
from leat.ops import Span

Rows = list[int | UOp]  # a value for each sequence of a draft


class Drafter(Protocol):
    # sequences it drafts for at once at most, past which plain steps decode faster
    sequences: int

    def draft(
        self, tokens: Tensor, hidden: Tensor, slots: Rows, positions: Rows, count: int
    ) -> Tensor:
        """`count` tokens drafted greedily after each of several sequences' `tokens` (1, n),
        sequence i's at position positions[i] of cache slot slots[i], which the target has run up
        to there, given its normed hidden states (1, n, dim) at the positions before: (n, count)
        int32."""
        ...

    def follow(self, tokens: Tensor, hidden: Tensor, spans: list[Span]) -> None:
        """Takes in what the target ran: tokens (1, T), its spans', and the target's normed
        hidden states (1, T, dim) at the positions before theirs."""
        ...

    def copy(self, source: int | UOp, slot: int | UOp) -> None:
        """Copies what the drafter holds of slot `source` to slot `slot`."""
        ...

    def shift(self, slot: int | UOp, offset: int | UOp) -> None:
        """Turns what it runs of slot `slot` from here on as the target's Transformer.shift()."""
        ...


def load(path: str | Path, target: Transformer) -> Drafter:
    """The drafter in a GGUF for a target model: Gemma 4's assistant, or the target's own file
    for Qwen3.5's MTP layer."""
    gguf = GGUF.open(path)
    arch = gguf.metadata["general.architecture"]
    if arch == "gemma4-assistant":
        return Gemma4Assistant(gguf, target)
    if gguf.metadata.get(f"{arch}.nextn_predict_layers") and arch in QWEN35:
        return Qwen35Mtp(gguf, target)
    raise ValueError(f"a {arch} model drafts for no {target.config.arch} model")


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

    # for Gemma 4 26B A4B on the 3090, 2 sequences decoded at 1.11 times the speed of plain steps,
    # and 3 at 1.04
    sequences = 3

    def __init__(self, gguf: GGUF, target: Transformer):
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
        self, tokens: Tensor, hidden: Tensor, slots: Rows, positions: Rows, count: int
    ) -> Tensor:
        scale, n, drafted = self.target.config.embed_scale, len(slots), []
        for _ in range(count):
            x = ops.embedding(tokens, self.target.embed) * scale
            x = ops.linear(x.cat(hidden, dim=-1), self.pre)
            hidden, logits = self._step(x, slots, positions)
            tokens = ops.argmax(logits.reshape(n, -1)).reshape(1, n)
            drafted.append(tokens.reshape(n, 1))
        return drafted[0].cat(*drafted[1:], dim=1)

    def follow(self, tokens: Tensor, hidden: Tensor, spans: list[Span]) -> None:
        pass  # it reads the target's own keys and values

    def copy(self, source: int | UOp, slot: int | UOp) -> None:
        pass

    def shift(self, slot: int | UOp, offset: int | UOp) -> None:
        pass  # of Gemma 4, whose RoPE is plain

    def _step(self, x: Tensor, slots: Rows, positions: Rows) -> tuple[Tensor, Tensor]:
        # the hidden states for the next step and the logits, after x (1, n, width)
        c, eps, n = self.target.config, self.eps, len(slots)
        spans = [Span(slot, pos - 1) for slot, pos in zip(slots, positions, strict=True)]
        for layer, small, source in zip(self.layers, self.small, self.sources, strict=True):
            q = ops.linear(ops.rms_norm(x, small["attn_norm"], eps), layer["attn_q"])
            q = q.reshape(1, n, self.heads, c.head_dims[source])
            q = ops.rms_norm(q, small["attn_q_norm"], eps).transpose(1, 2)
            if source in self.rope:
                cos, sin = (
                    Tensor.cat(*(table[pos : pos + 1] for pos in positions))
                    for table in self.rope[source]
                )
                q = ops.rotary(q, cos, sin, c.rope_halves)
            # a query at pos sees the target's positions before it, as one at pos - 1 sees those
            # and its own, and one fewer back for a window
            cache, window = self.target.cache[source], max(c.windows[source] - 1, 0)
            assert cache is not None
            out = ops.attention(q, cache, spans, c.scales[source], window, None,
                                self.target.rings[source])  # fmt: skip
            out = ops.linear(out, layer["attn_output"])
            x = x + ops.rms_norm(out, small["post_attention_norm"], eps)
            mlp = (layer["ffn_gate"], layer["ffn_up"], layer["ffn_down"])
            out = ops.feed_forward(x, *mlp, (small["ffn_norm"], eps), c.glu, residual=False)
            x = (x + ops.rms_norm(out, small["post_ffw_norm"], eps)) * small["layer_output_scale"]
        x = ops.rms_norm(x, self.norm, eps)
        return ops.linear(x, self.post), ops.linear(x, self.embed)


class Qwen35Mtp:
    """Qwen3.5's and Qwen3.6's MTP layer, past the model's own in its GGUF: a full-attention layer
    of the model's kind, with its own keys and values, whose inputs are a token's embedding and
    the model's hidden state at the position before, each normed, joined and projected, and whose
    output, normed by a weight of its own, the model's output head reads. A step takes a token
    and the hidden state before it and gives the next token and its own hidden state, from which
    the next step drafts another at the next position, as llama.cpp's. The keys and values of
    every position the target runs come from its tokens and its hidden states."""

    # for Qwen3.6 35B A3B on the 3090, 2 sequences decoded at 1.25 times the speed of plain
    # steps, and 3 at 1.16
    sequences = 3

    def __init__(self, gguf: GGUF, target: Transformer):
        c, prefix = target.config, f"blk.{target.config.n_layers}."
        w = {n.removeprefix(prefix): t for n, t in gguf.load(
            names=[n for n in gguf.tensors if n.startswith(prefix)]).items()}  # fmt: skip
        self.target, self.eps = target, c.norm_eps
        self.proj = w.pop("nextn.eh_proj.weight")
        self.e_norm, self.h_norm = (
            w.pop(f"nextn.{n}.weight").dequant().realize() for n in ("enorm", "hnorm")
        )
        head_norm = w.pop("nextn.shared_head_norm.weight")
        # the layer as a model of its own, of the target's attention layers' shape, its output
        # normed with the shared head's norm
        full = max(i for i in range(c.n_layers) if not c.recurrent[i])
        one = {
            f: getattr(c, f)[full : full + 1]
            for f in ("kv_heads", "head_dims", "windows", "ropes", "scales", "recurrent")
        }
        weights = {f"blk.0.{n}": t for n, t in w.items()}
        weights |= {"token_embd.weight": target.embed, "output.weight": target.output}
        weights["output_norm.weight"] = head_norm
        config = replace(c, n_layers=1, **one)
        self.layer = Transformer(config, weights, target.max_context, target.slots)

    def draft(
        self, tokens: Tensor, hidden: Tensor, slots: Rows, positions: Rows, count: int
    ) -> Tensor:
        n, drafted = len(slots), []
        for i in range(count):
            spans = [Span(slot, pos + i) for slot, pos in zip(slots, positions, strict=True)]
            hidden = self.layer.forward(self._inputs(tokens, hidden), spans)
            tokens = ops.argmax(self.target.logits(hidden).reshape(n, -1)).reshape(1, n)
            drafted.append(tokens.reshape(n, 1))
        return drafted[0].cat(*drafted[1:], dim=1)

    def follow(self, tokens: Tensor, hidden: Tensor, spans: list[Span]) -> None:
        self.layer.store(0, self._inputs(tokens, hidden), spans)

    def copy(self, source: int | UOp, slot: int | UOp) -> None:
        self.layer.copy(source, slot)

    def shift(self, slot: int | UOp, offset: int | UOp) -> None:
        self.layer.shift(slot, offset)

    def _inputs(self, tokens: Tensor, hidden: Tensor) -> Tensor:
        # the layer's inputs: the tokens' embeddings and the hidden states before them
        e = ops.rms_norm(ops.embedding(tokens, self.target.embed), self.e_norm, self.eps)
        h = ops.rms_norm(hidden, self.h_norm, self.eps)
        return ops.linear(e.cat(h, dim=-1), self.proj)
