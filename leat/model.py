"""Decoder-only transformer, configured entirely from GGUF metadata."""

import math
from dataclasses import dataclass, replace
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import DType

from leat import ops
from leat.ops import Span
from leat.quant import BLOCK, NATIVE, GGMLType, QTensor

CACHE_TILE = 256  # positions
_LAYER = ("attn_norm", "attn_q", "attn_k", "attn_output", "ffn_norm")
# the tensors of a Gated DeltaNet layer, Qwen3.5's in place of attention: its input norm, the
# projections of queries, keys and values and of the output's gate z, those of the decay's alpha
# and of beta, the decay's rate and bias, the causal convolution, the output's norm and projection
_DELTA_NET = ("attn_norm", "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_a", "ssm_dt.bias",
              "ssm_conv1d", "ssm_norm", "ssm_out", "ffn_norm")  # fmt: skip
_MLP = ("ffn_gate", "ffn_up", "ffn_down")
_EXPERTS = ("ffn_gate_inp", "ffn_down_exps")  # the router, and the experts' down projections
# the supported architectures, and whether their RoPE rotates dimension i with i + D/2, ggml's
# "neox" mode, rather than adjacent pairs, its "normal" mode, as GGUF lays out llama's q and k
_ROPE_HALVES = {
    "llama": False, "qwen2": True, "qwen3": True, "qwen3moe": True, "gemma3": True,
    "gemma4": True, "gpt-oss": True, "phi3": True, "qwen35moe": True,
}  # fmt: skip
# the period of attention.sliding_window_pattern where a GGUF has none, as llama.cpp's
_SLIDING_EVERY = {"gemma3": 6, "gpt-oss": 2}


@dataclass(frozen=True)
class Rope:
    """How a layer rotates its queries and keys: RoPE of base `theta` over the first `dims`
    dimensions of each head, none for 0, as ggml's rope ops: frequencies divided by rope_freqs'
    factors where `freqs` and the GGUF has them, or by LongRoPE's, its long ones for contexts
    past `longrope` tokens, if given; positions scaled by `scale`; YaRN's blend of scaled and
    unscaled frequencies if `yarn`, (original context, beta_fast, beta_slow); and cos and sin
    times `mscale`."""

    theta: float
    dims: int
    freqs: bool = False
    longrope: int = 0
    scale: float = 1.0
    yarn: tuple[int, float, float] | None = None
    mscale: float = 1.0


@dataclass(frozen=True)
class DeltaNet:
    """The sizes of Gated DeltaNet, Qwen3.5's linear attention: heads of queries and keys and of
    values, the dimensions of each, and the tokens its causal convolution spans."""

    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv: int


@dataclass(frozen=True)
class Config:
    arch: str
    n_layers: int
    dim: int
    n_heads: int
    vocab_size: int
    norm_eps: float
    context_length: int
    rope_halves: bool
    # each layer's attention: kv heads, head size, how many positions back it sees (0 for all),
    # its RoPE and the scale of its scores
    kv_heads: tuple[int, ...]
    head_dims: tuple[int, ...]
    windows: tuple[int, ...]
    ropes: tuple[Rope, ...]
    scales: tuple[float, ...]
    # the layers that are Gated DeltaNet's rather than attention, and its sizes
    recurrent: tuple[bool, ...]
    delta_net: DeltaNet | None = None
    q_gate: bool = False  # q's projection also gives each head a gate for its output, as Qwen3.5's
    experts: int = 0  # MLPs per layer of a mixture of experts, 0 for one MLP
    experts_used: int = 0  # experts each token takes
    glu: str = "silu"  # how an MLP's gate and up combine, as ops.glu has it
    embed_scale: float = 1.0
    v_norm: bool = False  # values RMSNormed without a weight, as Gemma 4's
    logit_cap: float = 0.0
    # M-RoPE's sections of the frequencies, interleaved, that turn by an image's time, height and
    # width, as Qwen3.5's: text, of the same position on each, turns as by plain RoPE
    mrope: tuple[int, ...] = ()

    @staticmethod
    def from_gguf(metadata: dict[str, Any]) -> "Config":
        if (arch := metadata["general.architecture"]) not in _ROPE_HALVES:
            raise NotImplementedError(f"architecture {arch!r} is not supported")
        m = {k.removeprefix(f"{arch}."): v for k, v in metadata.items()}
        n_heads, dim = m["attention.head_count"], m["embedding_length"]
        # without the layers past the others that predict further tokens, as Qwen3.5's
        n_layers = m["block_count"] - m.get("nextn_predict_layers", 0)

        def per_layer(value: Any) -> tuple:
            return tuple(value[:n_layers]) if isinstance(value, list) else (value,) * n_layers

        # whether each layer sees the window: a list, or as llama.cpp a period n, every n-th layer
        # seeing all positions, the others the window, all of them for 0
        pattern = m.get("attention.sliding_window_pattern")
        if pattern is None and m.get("attention.sliding_window"):
            pattern = _SLIDING_EVERY.get(arch)
        if isinstance(pattern, int) and not isinstance(pattern, bool):
            n = pattern
            pattern = [n == 0 or i % n < n - 1 for i in range(n_layers)]
        sliding = per_layer(bool(pattern) if not isinstance(pattern, list) else pattern)
        head_dim = m.get("attention.key_length", dim // n_heads)
        # Gemma 4's sliding-window layers have their own head size and RoPE
        swa_dim = m.get("attention.key_length_swa", head_dim)
        head_dims = tuple(swa_dim if s else head_dim for s in sliding)
        rotated = m.get("rope.dimension_count", head_dim)
        if m.get("rope.dimension_count_swa", swa_dim) != swa_dim:
            raise NotImplementedError("partial rotary embeddings of sliding layers")
        # Gemma 4 E2B's and E4B's: an embedding of each token for each layer, and the last layers
        # attending over the keys and values of those before
        if m.get("embedding_length_per_layer_input") or m.get("attention.shared_kv_layers"):
            raise NotImplementedError("per-layer embeddings and layers that share keys and values")
        ropes = [
            _rope(m, arch, s, d if s else rotated) for s, d in zip(sliding, head_dims, strict=True)
        ]
        if arch == "gemma4":  # unit attention scale, as its q and k are normed
            scales = (1.0,) * n_layers
        elif arch == "gemma3" and n_layers == 62:  # 27B's query_pre_attn_scalar, as llama.cpp
            scales = (1 / math.sqrt(dim / n_heads),) * n_layers
        else:
            scales = tuple(1 / math.sqrt(d) for d in head_dims)
        delta_net, recurrent = None, (False,) * n_layers
        # Qwen3.5's: every n-th layer attention, the others Gated DeltaNet
        if "ssm.inner_size" in m:
            v_heads, every = m["ssm.time_step_rank"], m.get("full_attention_interval", 4)
            delta_net = DeltaNet(m["ssm.group_count"], v_heads, m["ssm.state_size"],
                                 m["ssm.inner_size"] // v_heads, m["ssm.conv_kernel"])  # fmt: skip
            recurrent = tuple((i + 1) % every != 0 for i in range(n_layers))
            if isinstance(layers := m.get("attention.recurrent_layers"), list):  # as llama.cpp
                recurrent = tuple(bool(r) for r in layers[:n_layers])
        return Config(
            arch=arch,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            vocab_size=len(metadata["tokenizer.ggml.tokens"]),
            norm_eps=m["attention.layer_norm_rms_epsilon"],
            context_length=m["context_length"],
            rope_halves=_ROPE_HALVES[arch],
            kv_heads=per_layer(m.get("attention.head_count_kv", n_heads)),
            head_dims=head_dims,
            windows=tuple(m["attention.sliding_window"] if s else 0 for s in sliding),
            ropes=tuple(ropes),
            scales=scales,
            recurrent=recurrent,
            delta_net=delta_net,
            q_gate=arch == "qwen35moe",
            experts=m.get("expert_count", 0),
            experts_used=m.get("expert_used_count", 0),
            glu={"gemma3": "gelu", "gemma4": "gelu", "gpt-oss": "oai"}.get(arch, "silu"),
            embed_scale=math.sqrt(dim) if arch in ("gemma3", "gemma4") else 1.0,
            v_norm=arch == "gemma4",
            logit_cap=m.get("final_logit_softcapping", 0.0),
            mrope=tuple(m.get("rope.dimension_sections", [])[:3]) if arch == "qwen35moe" else (),
        )

    def uses(self, name: str) -> bool:
        """Whether the model reads a tensor: all but those of layers past its own, as Qwen3.5's
        layer for predicting further tokens."""
        return not name.startswith("blk.") or int(name.split(".")[1]) < self.n_layers


def _rope(m: dict[str, Any], arch: str, sliding: bool, dims: int) -> Rope:
    # a layer's RoPE, of a sliding-window layer or else one that sees all positions, its cos and
    # sin times rope.scaling.attn_factor, as llama.cpp's every layer
    attn_factor, base = m.get("rope.scaling.attn_factor", 1.0), m.get("rope.freq_base", 10000.0)
    swa_base = m.get("rope.freq_base_swa", base if arch == "gpt-oss" else 10000.0)
    if (
        sliding and arch != "gpt-oss"
    ):  # their own base, unscaled; gpt-oss's are scaled as the others
        return Rope(swa_base, dims, mscale=attn_factor)
    # rope_freqs divides the frequencies of the layers that see all positions: all of Llama
    # 3.1's, few of Gemma 4's
    rope = Rope(swa_base if sliding else base, dims, freqs=not sliding, mscale=attn_factor)
    # as llama.cpp, a factor of no type scales positions, and of its old key too
    kind = m.get("rope.scaling.type", "linear")
    factor = m.get("rope.scaling.factor", m.get("rope.scale_linear", 0.0))
    original = m.get("rope.scaling.original_context_length", m["context_length"])
    if arch == "phi3" and original < m["context_length"]:  # LongRoPE
        return replace(rope, longrope=original)
    if kind == "linear" and factor:
        return replace(rope, scale=1 / factor)
    if kind == "yarn" and factor:
        beta = m.get("rope.scaling.yarn_beta_fast", 32.0), m.get("rope.scaling.yarn_beta_slow", 1.0)
        # llama.cpp's attention factor, which with no log multiplier leaves ggml's own
        # 1 + 0.1 ln(factor), times rope.scaling.attn_factor
        mscale = attn_factor
        if log_mul := m.get("rope.scaling.yarn_log_multiplier", 0.0):
            mscale *= _yarn_mscale(factor, 1) / _yarn_mscale(factor, log_mul)
            mscale /= 1 + 0.1 * math.log(factor)
        return replace(rope, scale=1 / factor, yarn=(original, *beta), mscale=mscale)
    return rope


def _yarn_mscale(scale: float, multiplier: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * multiplier * math.log(scale) + 1


class Transformer:
    """Weights, RoPE tables and a KV cache of `slots` sequences of up to `max_context` tokens, and
    for Gated DeltaNet's layers in place of attention, the recurrent state of each sequence.

    Optional parts are used where the GGUF has their tensors: biases of q, k, v and the attention
    output, RMSNorms of q and k, of the attention and MLP outputs, attention sinks, a shared MLP
    beside the experts, gated or not, the router's and experts' biases, and a scale per layer
    output. Fused tensors, q, k and v in one or gate and up in one, are split.
    """

    def __init__(
        self, config: Config, weights: dict[str, QTensor], max_context: int, slots: int = 1,
        saved_tokens: int = 0,
    ):  # fmt: skip
        if not 0 < max_context <= config.context_length:
            raise ValueError(
                f"max_context must be in [1, {config.context_length}], got {max_context}"
            )
        self.config, self.max_context, self.slots = config, max_context, slots
        # each layer's tensors by name, without "blk.{i}." and ".weight"
        self.layers: list[dict[str, QTensor]] = [{} for _ in range(config.n_layers)]
        for name, w in weights.items():
            if name.startswith("blk.") and config.uses(name):
                i, part = name.removeprefix("blk.").split(".", 1)
                self.layers[int(i)][part.removesuffix(".weight")] = w
        for n, layer in enumerate(self.layers):
            _unfuse(layer, config, n)
        mlp = _EXPERTS if config.experts else _MLP
        # values of their own, but in Gemma 4, whose full-attention layers' keys are their values
        attention = _LAYER + (() if config.arch == "gemma4" else ("attn_v",))
        missing = [
            f"blk.{i}.{n}"
            for i, (layer, recurrent) in enumerate(zip(self.layers, config.recurrent, strict=True))
            for n in (_DELTA_NET if recurrent else attention) + mlp
            if n not in layer
        ]
        if missing:
            raise ValueError(f"missing {len(missing)} tensors, first: {missing[0]}")
        for layer in self.layers:
            _stack(layer)
        # norm weights, biases, sinks and scales, decoded once
        self.small = [
            {n: w.dequant().realize() for n, w in layer.items() if _small(n, w)}
            for layer in self.layers
        ]
        # Gemma 4 routes from x normed with a weight of its own, over sqrt(dim)
        for s in self.small:
            if "ffn_gate_inp.scale" in s:
                s["router_norm"] = (s["ffn_gate_inp.scale"] / math.sqrt(config.dim)).realize()
        self.embed = weights["token_embd.weight"]
        self.output = weights.get("output.weight", self.embed)  # tied embeddings when absent
        self.output_norm = weights["output_norm.weight"].dequant()
        # RoPE tables by base, rotated dimensions and scaling
        tables: dict[Rope, tuple[Tensor, Tensor]] = {}
        for rope in config.ropes:
            if rope.dims and rope not in tables:
                tables[rope] = rope_table(rope, max_context, _factors(rope, weights, max_context))
        self.rope = [tables.get(rope) for rope in config.ropes]
        # of M-RoPE, each slot's tables, which shift() moves on from the base ones
        self.base = self.rope
        if config.mrope:
            own = {r: (_slots(cos, slots), _slots(sin, slots)) for r, (cos, sin) in tables.items()}
            self.rope = [own.get(rope) for rope in config.ropes]
        # whole tiles of positions, which the attention kernels need; the rest stay unused
        positions = -(-max_context // CACHE_TILE) * CACHE_TILE
        self.cache = [
            None if recurrent else _zeros(2, slots, kv_heads, positions, dim, dtype=dtypes.half)
            for kv_heads, dim, recurrent in zip(
                config.kv_heads, config.head_dims, config.recurrent, strict=True
            )
        ]
        # a Gated DeltaNet layer's state of each slot: the last inputs of its convolution, and
        # the recurrence's state of each value head, keys by values
        d = config.delta_net
        self.states = [
            (_zeros(slots, d.conv - 1, 2 * d.k_heads * d.k_dim + d.v_heads * d.v_dim),
             _zeros(slots, d.v_heads, d.k_dim, d.v_dim)) if d and recurrent else None
            for recurrent in config.recurrent
        ]  # fmt: skip
        # and a copy of them kept partway through a prompt, from which a later one may go on
        self.kept = [
            None if s is None else (_zeros(*map(int, s[0].shape)), _zeros(*map(int, s[1].shape)))
            for s in self.states
        ]
        # and those after each of a run's `saved_tokens` tokens, from which it may go back
        self.saved = [
            None if s is None or not saved_tokens
            else (_zeros(saved_tokens, *map(int, s[0].shape[1:])),
                  _zeros(saved_tokens, *map(int, s[1].shape[1:])))
            for s in self.states
        ]  # fmt: skip

    def __call__(self, tokens: Tensor, start_pos: int | UOp, slot: int | UOp = 0) -> Tensor:
        """Runs `tokens` (1, T) at positions `start_pos...` of cache slot `slot` and returns normed
        hidden states."""
        return self.run(tokens, [Span(slot, start_pos, tokens.shape[1])])

    def run(
        self, tokens: Tensor, spans: list[Span], live: int | UOp | None = None, save: bool = False
    ) -> Tensor:
        """Runs `tokens` (1, T), the spans' in turn, each in its slot and at its positions, and
        returns normed hidden states. Several sequences share the reads of every weight; each
        attends over its own slot alone. Tokens past the first `live`, if given, pad a batch:
        the mixtures of experts skip them. With `save`, the tokens keep the recurrent states
        after each of them, the t-th token's at t, from which rewind() goes back."""
        x = ops.embedding(tokens, self.embed)
        if (scale := self.config.embed_scale) != 1:
            x = x * scale
        return self.forward(x, spans, live, save)

    def forward(
        self, x: Tensor, spans: list[Span], live: int | UOp | None = None, save: bool = False,
        positions: Tensor | None = None,
    ) -> Tensor:  # fmt: skip
        """run() from the tokens' embeddings x (1, T, dim), or any inputs of the first layer;
        of M-RoPE, an image's, at `positions` (T, 3), each token's time, height and width."""
        if len(spans) > 1 and not all(isinstance(s.length, int) for s in spans):
            raise ValueError("several spans need lengths known in advance")
        for i in range(self.config.n_layers):
            if self.config.recurrent[i]:
                x = self._delta_net(i, x, spans, save)
            else:
                x = self._attention(i, x, spans, positions)
            x = self._feed_forward(i, x, live)
        return ops.rms_norm(x, self.output_norm, self.config.norm_eps)

    def logits(self, hidden: Tensor) -> Tensor:
        logits = ops.linear(hidden, self.output)
        if cap := self.config.logit_cap:
            logits = (logits / cap).tanh() * cap
        return logits

    def copy(self, source: int | UOp, slot: int | UOp) -> None:
        """Copies the cache and recurrent states of slot `source` to slot `slot`."""
        # Whole slots, as tinygrad runs a copy of a bound number of positions on few threads: for
        # Llama 3.1 8B on the 3090, 2130 positions took 12.3 ms, whole slots of 4096 3.1 ms.
        copied = []
        for cache in self.cache:
            if cache is not None:
                copied.append(cache[:, slot : slot + 1].assign(cache[:, source : source + 1]))
        for state in (state for states in self.states if states for state in states):
            copied.append(state[slot : slot + 1].assign(state[source : source + 1]))
        Tensor.realize(*copied)

    def keep(self, slot: int | UOp) -> None:
        """Copies the recurrent states of slot `slot` to its kept copy."""
        _copy_slot(self.states, self.kept, slot)

    def restore(self, slot: int | UOp) -> None:
        """Copies the kept recurrent states of slot `slot` back."""
        _copy_slot(self.kept, self.states, slot)

    def rewind(self, slot: int | UOp, token: int | UOp) -> None:
        """Sets the recurrent states of slot `slot` to those a saving run kept after its token
        `token`, of the slot's span, as if the span had stopped there."""
        writes = [
            state[slot : slot + 1].assign(after[token : token + 1])
            for states, saved in zip(self.states, self.saved, strict=True) if states and saved
            for state, after in zip(states, saved, strict=True)
        ]  # fmt: skip
        if writes:
            Tensor.realize(*writes)

    def shift(self, slot: int | UOp, offset: int | UOp) -> None:
        """Turns slot `slot`'s queries and keys from here on as RoPE turns those `offset`
        positions on from theirs: M-RoPE's text after images, whose positions take more of the
        cache than of RoPE's."""
        writes = []
        for base, own in {id(o): (b, o) for b, o in zip(self.base, self.rope, strict=True)
                          if o is not None and b is not None}.values():  # fmt: skip
            rows = (Tensor.arange(int(base[0].shape[0])) + Tensor(offset)).maximum(0)
            writes += [o[slot : slot + 1].assign(b[rows].unsqueeze(0))
                       for b, o in zip(base, own, strict=True)]  # fmt: skip
        Tensor.realize(*writes)

    def store(self, i: int, x: Tensor, spans: list[Span]) -> None:
        """Stores layer i's keys and values of x, its inputs, in its cache, as running it would."""
        Tensor.realize(self._rotated(i, x, spans)[1])

    def _attention(
        self, i: int, x: Tensor, spans: list[Span], positions: Tensor | None = None
    ) -> Tensor:
        # x + the attention block's output
        c, w, s = self.config, self.layers[i], self.small[i]
        (q, gate), cache = self._rotated(i, x, spans, positions)
        out = ops.attention(q, cache, spans, c.scales[i], c.windows[i], s.get("attn_sinks"))
        if gate is not None:
            out = out * gate.reshape(out.shape).sigmoid()
        # gpt-oss's output bias joins the residual
        residual = x + s["attn_output.bias"] if "attn_output.bias" in s else x
        if "post_attention_norm" not in s:
            return ops.linear(out, w["attn_output"], residual=residual)
        return ops.add_normed(
            residual, [(ops.linear(out, w["attn_output"]), s["post_attention_norm"])], None,
            c.norm_eps,
        )  # fmt: skip

    def _rotated(
        self, i: int, x: Tensor, spans: list[Span], positions: Tensor | None = None
    ) -> tuple[tuple[Tensor, Tensor | None], Tensor]:
        # layer i's queries of x, rotated, of M-RoPE at their positions if given, and the gates
        # of their heads' outputs if any, and its cache with x's keys and values stored
        c, w, s = self.config, self.layers[i], self.small[i]
        B, T, _ = x.shape
        kv_heads, dim, eps = c.kv_heads[i], c.head_dims[i], c.norm_eps
        gate = None
        # Gemma 4's full-attention layers have no values of their own: the keys are, before norm
        proj = [w["attn_q"], w["attn_k"]] + ([w["attn_v"]] if "attn_v" in w else [])
        q, k, *values = ops.linears(x, *proj, norm=(s["attn_norm"], eps))
        if c.q_gate:  # each head's q, then the gate of its output
            q, gate = q.reshape(B, T, c.n_heads, 2 * dim).chunk(2, dim=-1)
        q, k = q.reshape(B, T, c.n_heads, dim), k.reshape(B, T, kv_heads, dim)
        v = (values[0] if values else k).reshape(B, T, kv_heads, dim)
        biases = None
        if "attn_q.bias" in s:  # Qwen2's and gpt-oss's
            biases = (s["attn_q.bias"], s["attn_k.bias"], s["attn_v.bias"])
        norms = (s["attn_q_norm"], s["attn_k_norm"]) if "attn_q_norm" in s else None
        table = self.rope[i]
        if positions is not None and (base := self.base[i]) is not None:
            table = _mrope(base, positions, c.mrope)  # each token's own angles
        rope = None if table is None else (table, c.ropes[i].dims)
        cache = self.cache[i]
        assert cache is not None
        q, cache = ops.rotate(q, k, v, cache, spans, rope, c.rope_halves, biases, norms, c.v_norm,
                              eps, positions is not None)  # fmt: skip
        return (q, gate), cache

    def _delta_net(self, i: int, x: Tensor, spans: list[Span], save: bool = False) -> Tensor:
        # x + a Gated DeltaNet block's output, in place of attention: queries, keys and values
        # and the output's gate z projected together, the decay's alpha and beta together
        w, s, eps = self.layers[i], self.small[i], self.config.norm_eps
        states = self.states[i]
        assert states is not None
        mixed, z = ops.linears(x, w["attn_qkv"], w["attn_gate"], norm=(s["attn_norm"], eps))
        gates = ops.router(x, (s["attn_norm"], eps), w["ssm_alpha_beta"])
        decay = (s["ssm_a"], s["ssm_dt.bias"])
        saved = self.saved[i] if save else None
        out = ops.delta_net(mixed, z, gates, w["ssm_conv1d"].dequant(), decay,
                            (s["ssm_norm"], eps), states, spans, saved)  # fmt: skip
        return ops.linear(out, w["ssm_out"], residual=x)

    def _feed_forward(self, i: int, x: Tensor, live: int | UOp | None = None) -> Tensor:
        # x + the MLP block's output: an MLP, a mixture of experts, as in Gemma 4 both, each output
        # normed and then their sum, and the layer's output scaled, or as in Qwen3.5 experts beside
        # a shared one
        c, w, s = self.config, self.layers[i], self.small[i]
        eps, scale = c.norm_eps, s.get("layer_output_scale")
        norm = (s["ffn_norm"], eps)
        mlp = (w["ffn_gate"], w["ffn_up"], w["ffn_down"]) if "ffn_gate" in w else None
        if mlp and not c.experts:
            if "post_ffw_norm" not in s:
                return ops.feed_forward(x, *mlp, norm, c.glu)
            out = ops.feed_forward(x, *mlp, norm, c.glu, residual=False)
            return ops.add_normed(x, [(out, s["post_ffw_norm"])], None, eps, scale)
        router = (s.get("router_norm", s["ffn_norm"]), eps)
        scores = ops.router(x, router, w["ffn_gate_inp"], s.get("ffn_gate_inp.bias"))
        # stacked gate and up matrices, or one stack of both, the gate's rows first
        gate, up = (w["ffn_gate_up_exps"], None) if "ffn_gate_up_exps" in w else (
            w["ffn_gate_exps"], w["ffn_up_exps"])  # fmt: skip
        biases = None
        if "ffn_down_exps.bias" in s:  # gpt-oss's
            biases = (s["ffn_gate_exps.bias"], s["ffn_up_exps.bias"], s["ffn_down_exps.bias"])
        experts = (scores, gate, up, w["ffn_down_exps"], c.experts_used)
        if "ffn_gate_inp_shexp" in w:  # Qwen3.5's shared expert, scaled by its gate's sigmoid
            mixed = ops.mixture(x, *experts, norm, c.glu, residual=False, live=live)
            shexp = (w["ffn_gate_shexp"], w["ffn_up_shexp"], w["ffn_down_shexp"])
            out = ops.feed_forward(x, *shexp, norm, c.glu, residual=False)
            return x + mixed + out * ops.router(x, norm, w["ffn_gate_inp_shexp"]).sigmoid()
        if not mlp:
            return ops.mixture(x, *experts, norm, c.glu, biases=biases, live=live)
        scales = s["ffn_down_exps.scale"]
        mixed = ops.mixture(
            x, *experts, (s["pre_ffw_norm_2"], eps), c.glu, scales, False, live=live
        )
        shared = ops.feed_forward(x, *mlp, norm, c.glu, residual=False)
        parts = [(shared, s["post_ffw_norm_1"]), (mixed, s["post_ffw_norm_2"])]
        return ops.add_normed(x, parts, s["post_ffw_norm"], eps, scale)


def _mrope(
    tables: tuple[Tensor, Tensor], positions: Tensor, sections: tuple[int, ...]
) -> tuple[Tensor, Tensor]:
    # cos and sin (T, R/2) of tokens at positions (T, 3) of time, height and width, of the
    # tables of plain RoPE: frequency j turns by the token's height if j % 3 is 1, below three
    # times the height's section, by its width if 2, below three times the width's, else by time
    half = int(tables[0].shape[1])
    axes = [1 if j % 3 == 1 and j < 3 * sections[1] else 2 if j % 3 == 2 and j < 3 * sections[2]
            else 0 for j in range(half)]  # fmt: skip
    pick = Tensor.arange(3).reshape(1, 3, 1) == Tensor(axes).reshape(1, 1, half)
    return tuple(pick.where(t[positions], 0.0).sum(1) for t in tables)  # type: ignore[return-value]


def _factors(rope: Rope, weights: dict[str, QTensor], max_context: int) -> Tensor | None:
    # what divides a layer's RoPE frequencies, if anything
    name = "rope_freqs.weight" if rope.freqs else None
    if rope.longrope:
        name = f"rope_factors_{'long' if max_context > rope.longrope else 'short'}.weight"
    return weights[name].dequant() if name in weights else None


def _small(name: str, w: QTensor) -> bool:
    # tensors decoded once: norm weights, biases, sinks and scales
    return len(w.shape) == 1 or name.endswith((".bias", ".scale"))


def _rows(w: QTensor, start: int, stop: int) -> QTensor:
    # rows start..stop of a matrix, still in storage: a view
    elements, _ = BLOCK[w.type]
    per = w.shape[1] if w.type in NATIVE else w.shape[1] // elements  # values or blocks a row
    return QTensor(w.data[start * per : stop * per], w.type, (stop - start, *w.shape[1:]))


def _stack(layer: dict[str, QTensor]) -> None:
    # Qwen3.5's projections of the normed input, as F32 matrices the router's kernel takes:
    # alpha's and beta's stacked, scored together, whatever each is stored as, and the shared
    # expert's gate a row
    if "ssm_alpha" in layer:
        alpha, beta = layer.pop("ssm_alpha"), layer.pop("ssm_beta")
        data = alpha.dequant().cat(beta.dequant()).flatten().contiguous().realize()
        layer["ssm_alpha_beta"] = QTensor(data, GGMLType.F32, (2 * alpha.shape[0], alpha.shape[1]))
    if (gate := layer.get("ffn_gate_inp_shexp")) is not None:
        layer["ffn_gate_inp_shexp"] = QTensor(gate.data, gate.type, (1, *gate.shape))


def _copy_slot(
    sources: list[tuple[Tensor, Tensor] | None], targets: list[tuple[Tensor, Tensor] | None],
    slot: int | UOp,
) -> None:  # fmt: skip
    pairs = [(s, t) for ss, ts in zip(sources, targets, strict=True) if ss and ts
             for s, t in zip(ss, ts, strict=True)]  # fmt: skip
    Tensor.realize(*(t[slot : slot + 1].assign(s[slot : slot + 1]) for s, t in pairs))


def _slots(table: Tensor, slots: int) -> Tensor:
    # a table for each of the slots, each as it is
    return table.unsqueeze(0).expand(slots, *table.shape).contiguous().realize()


def _zeros(*shape: int, dtype: DType = dtypes.float32) -> Tensor:
    return Tensor.zeros(*shape, dtype=dtype).contiguous().realize()


def _unfuse(layer: dict[str, QTensor], c: Config, i: int) -> None:
    # Phi-3's q, k and v in one matrix, and its gate and up in one, gate first; the norm before
    # the experts of gpt-oss and Qwen3.5, named for after attention
    if "attn_qkv" in layer and not c.recurrent[i]:
        qkv, q, kv = (
            layer.pop("attn_qkv"),
            c.n_heads * c.head_dims[i],
            c.kv_heads[i] * c.head_dims[i],
        )
        layer["attn_q"], layer["attn_k"] = _rows(qkv, 0, q), _rows(qkv, q, q + kv)
        layer["attn_v"] = _rows(qkv, q + kv, q + 2 * kv)
    if "ffn_up" in layer and "ffn_gate" not in layer and not c.experts:
        up, hidden = layer["ffn_up"], layer["ffn_up"].shape[0] // 2
        layer["ffn_gate"], layer["ffn_up"] = _rows(up, 0, hidden), _rows(up, hidden, 2 * hidden)
    if c.arch in ("gpt-oss", "qwen35moe") and "post_attention_norm" in layer:
        layer["ffn_norm"] = layer.pop("post_attention_norm")


def rope_table(rope: Rope, length: int, factors: Tensor | None) -> tuple[Tensor, Tensor]:
    # cos and sin (length, dims / 2) of angle = pos * theta^(-2i/d) / factor_i in f32, the order
    # llama.cpp's rope kernels use; scaled, and with YaRN's ramp from the scaled angle to the
    # unscaled one, as ggml's rope_yarn
    d = rope.dims
    freqs = Tensor([rope.theta ** (-2 * i / d) for i in range(d // 2)])
    angles = Tensor.arange(length).float().unsqueeze(1) * freqs.unsqueeze(0)
    if factors is not None:
        angles = angles / factors.unsqueeze(0)
    mscale = rope.mscale
    if rope.yarn is not None:
        original, fast, slow = rope.yarn

        def corr(beta: float) -> float:
            return d * math.log(original / (beta * 2 * math.pi)) / (2 * math.log(rope.theta))

        low, high = max(0.0, math.floor(corr(fast))), min(d - 1.0, math.ceil(corr(slow)))
        ramp = [1 - min(1.0, max(0.0, (i - low) / max(0.001, high - low))) for i in range(d // 2)]
        # theta * scale * (1 - mix) + theta * mix, with mix the ramp
        blend = Tensor([rope.scale * (1 - r) + r for r in ramp])
        angles = angles * blend.unsqueeze(0)
        mscale *= 1 + 0.1 * math.log(1 / rope.scale)
    elif rope.scale != 1:
        angles = angles * rope.scale
    cos, sin = angles.cos(), angles.sin()
    if mscale != 1:
        cos, sin = cos * mscale, sin * mscale
    return cos.contiguous().realize(), sin.contiguous().realize()
