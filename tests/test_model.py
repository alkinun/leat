import math
import subprocess

import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes

from leat import bench
from leat.engine import FEW_TOKENS, KEEP_BACK, Engine, graph
from leat.gguf import GGUF
from leat.kernels import VARIABLES
from leat.model import CACHE_TILE, Config, Transformer, _stack, rope_table
from leat.quant import GGMLType, QTensor
from leat.sampler import GREEDY, Sampling
from tests.helpers import (
    CONTEXT,
    P3_MSCALE,
    P3_ORIGINAL,
    P3_ROTATED,
    random_blocks,
    reference_logits,
)

PROMPT = [5, 77, 120, 3, 299, 42, 8, 150, 61, 200, 9, 33]
ARCHS = [
    "llama",
    "qwen2",
    "qwen3",
    "qwen3moe",
    "gemma4",
    "gemma4-dense",
    "gemma3",
    "gpt-oss",
    "phi3",
    "qwen35moe",
]


def transformer(path, max_context: int = CONTEXT, slots: int = 1) -> Transformer:
    f = GGUF.open(path)
    return Transformer(Config.from_gguf(f.metadata), f.load(), max_context, slots)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_forward_matches_reference(tiny, arch):
    path, weights = tiny(arch)
    model, expected = transformer(path), reference_logits(weights, PROMPT, arch)
    logits = model.logits(model(Tensor([PROMPT], dtype=dtypes.int32), 0)).numpy()[0]
    np.testing.assert_allclose(logits, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("arch", ARCHS)
def test_kernels_match_reference(tiny, arch):
    # the kernels, where they take the tiny models' shapes, run every architecture as the f64
    # reference does but for their int8 activations' rounding, some hundredths of the largest
    # logit: a whole prompt, as prefilling, and a token at a time, as decoding
    path, weights = tiny(arch)
    model, expected = transformer(path), reference_logits(weights, PROMPT, arch)
    whole = model.logits(model(Tensor([PROMPT], dtype=dtypes.int32), 0)).numpy()[0]
    pos = UOp.variable("start_pos", 0, CONTEXT - 1)
    each = [model.logits(model(Tensor([[t]], dtype=dtypes.int32), pos.bind(i))).numpy()[0, 0]
            for i, t in enumerate(PROMPT)]  # fmt: skip
    for got in (whole, np.stack(each)):
        np.testing.assert_allclose(got, expected, atol=0.05 * np.abs(expected).max())


def test_cache_whole_tiles(tiny_model):
    # attention kernels take caches of whole tiles; the model still holds max_context positions
    model = transformer(tiny_model[0], 50)
    assert model.cache[0].shape[3] == CACHE_TILE and model.max_context == 50


def test_bad_models(tiny_model):
    path, _ = tiny_model
    with pytest.raises(ValueError, match=r"max_context must be in \[1, 64\]"):
        transformer(path, CONTEXT + 1)
    f = GGUF.open(path)
    weights = {n: w for n, w in f.load().items() if n != "blk.1.ffn_down.weight"}
    with pytest.raises(ValueError, match="missing 1 tensors, first: blk.1.ffn_down"):
        Transformer(Config.from_gguf(f.metadata), weights, CONTEXT)
    # values, which only Gemma 4's full-attention layers take from their keys
    weights = {n: w for n, w in f.load().items() if n != "blk.0.attn_v.weight"}
    with pytest.raises(ValueError, match="missing 1 tensors, first: blk.0.attn_v"):
        Transformer(Config.from_gguf(f.metadata), weights, CONTEXT)
    with pytest.raises(NotImplementedError, match="architecture 'mamba'"):
        config("mamba")


def config(arch: str, values: dict | None = None) -> Config:
    # the Config of the least metadata an architecture takes, and `values`, without the prefix
    least = {"block_count": 4, "embedding_length": 256, "attention.head_count": 4,
             "context_length": 64, "attention.layer_norm_rms_epsilon": 1e-5}  # fmt: skip
    metadata = {f"{arch}.{k}": v for k, v in (least | (values or {})).items()}
    return Config.from_gguf(metadata | {"general.architecture": arch, "tokenizer.ggml.tokens": []})


def test_config():
    # Gemma 3 27B scales its scores by dim / heads, as llama.cpp, rather than its heads' size
    gemma = {"block_count": 62, "embedding_length": 5376, "attention.head_count": 32,
             "attention.key_length": 128}  # fmt: skip
    assert config("gemma3", gemma).scales == (168**-0.5,) * 62
    # Qwen3.5's layers are Gated DeltaNet's but every full_attention_interval-th, 4 by default
    delta_net = {"ssm.inner_size": 64, "ssm.time_step_rank": 2, "ssm.group_count": 1,
                 "ssm.state_size": 32, "ssm.conv_kernel": 4, "block_count": 8}  # fmt: skip
    assert config("qwen35moe", delta_net).recurrent == (True, True, True, False) * 2
    delta_net["full_attention_interval"] = 2
    assert config("qwen35moe", delta_net).recurrent == (True, False) * 4
    # or as recurrent_layers says, where given
    layers = [True, True, False, True, False, False, True, False]
    assert config("qwen35moe", delta_net | {"attention.recurrent_layers": layers}).recurrent == (
        tuple(layers)
    )
    # a sliding_window_pattern of a period n, as llama.cpp's: every n-th layer sees all positions,
    # the others the window, and 0 all of them; with none, Gemma 3's own period of 6
    gemma = {"attention.sliding_window": 4, "block_count": 12}
    full = config("gemma3", gemma | {"attention.sliding_window_pattern": 6}).windows
    assert full == config("gemma3", gemma).windows == ((4,) * 5 + (0,)) * 2
    assert config("gemma3", gemma | {"attention.sliding_window_pattern": 0}).windows == (4,) * 12
    # sliding-window layers rotate all their dimensions
    partial = {"attention.sliding_window": 4, "attention.sliding_window_pattern": [True, False],
               "rope.dimension_count_swa": 32, "block_count": 2}  # fmt: skip
    with pytest.raises(NotImplementedError, match="partial rotary"):
        config("gemma4", partial)
    # Gemma 4 E2B's and E4B's embeddings per layer and shared keys and values are refused, not
    # left unread
    for values in ({"embedding_length_per_layer_input": 256}, {"attention.shared_kv_layers": 20}):
        with pytest.raises(NotImplementedError, match="per-layer embeddings"):
            config("gemma4", values)


YARN, YARN_SCALE = {"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0}, 1 + 0.1 * math.log(4)


@pytest.mark.parametrize(
    "scaling, scale",
    [
        ({}, 1.0),
        (YARN, 1 + 0.1 * math.log(4)),
        (YARN | {"rope.scaling.attn_factor": 0.5}, 0.5 * (1 + 0.1 * math.log(4))),
        ({"rope.scaling.attn_factor": 0.5}, 0.5),  # with no scaling too, as llama.cpp
        (
            YARN | {"rope.scaling.yarn_log_multiplier": 0.5},
            (1 + 0.1 * math.log(4)) / (1 + 0.05 * math.log(4)),
        ),
    ],
)
def test_rope_scale(scaling, scale):
    # RoPE's cos and sin times YaRN's 1 + 0.1 ln(factor) and rope.scaling.attn_factor, or with a
    # log multiplier, as llama.cpp, mscale(factor, 1) / mscale(factor, multiplier) instead
    cos, sin = rope_table(config("llama", scaling).ropes[0], 8, None)
    np.testing.assert_allclose(np.hypot(cos.numpy(), sin.numpy()), scale, rtol=1e-6)


def test_stack_alpha_beta():
    # Qwen3.5's alpha and beta projections stacked as F32, whatever each is stored as
    rng = np.random.default_rng(0)
    alpha = rng.standard_normal((4, 32)).astype(np.float32)
    beta = QTensor(Tensor(random_blocks(GGMLType.Q8_0, 4, rng)), GGMLType.Q8_0, (4, 32))
    layer = {"ssm_alpha": QTensor(Tensor(alpha.ravel()), GGMLType.F32, (4, 32)), "ssm_beta": beta}
    _stack(layer)
    stacked = layer["ssm_alpha_beta"]
    assert stacked.type == GGMLType.F32 and stacked.shape == (8, 32)
    expected = np.concatenate([alpha, beta.dequant().numpy()])
    np.testing.assert_array_equal(stacked.dequant().numpy(), expected)


def test_rope_metadata():
    # as llama.cpp: a scaling factor of no type, or of the old key, scales positions linearly;
    # attn_factor scales sliding layers' cos and sin too; gpt-oss's sliding layers take
    # freq_base_swa where given, scaled as the others
    for key in ("rope.scaling.factor", "rope.scale_linear"):
        assert config("llama", {key: 4.0}).ropes[0].scale == 0.25
    gemma = {"attention.sliding_window": 4, "rope.scaling.attn_factor": 0.5, "block_count": 6}
    assert {r.mscale for r in config("gemma3", gemma).ropes} == {0.5}
    oss = {"attention.sliding_window": 4, "rope.freq_base": 1e5, "block_count": 2} | YARN
    assert [r.theta for r in config("gpt-oss", oss).ropes] == [1e5, 1e5]
    ropes = config("gpt-oss", oss | {"rope.freq_base_swa": 5e5}).ropes
    assert [r.theta for r in ropes] == [5e5, 1e5] and ropes[0].yarn == ropes[1].yarn


def test_longrope(tiny):
    # Phi-3's frequencies are divided by LongRoPE's long factors past its original context, and
    # by its short ones within it
    path, weights = tiny("phi3")
    for context, kind in ((P3_ORIGINAL, "short"), (CONTEXT, "long")):
        cos, _ = transformer(path, context).rope[0]
        freqs = 10000.0 ** (-np.arange(0, P3_ROTATED, 2) / P3_ROTATED)
        angles = np.arange(context)[:, None] * freqs / weights[f"rope_factors_{kind}.weight"]
        np.testing.assert_allclose(cos.numpy(), np.cos(angles) * P3_MSCALE, rtol=1e-5, atol=1e-5)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_generate_matches_reference(tiny, arch):
    path, weights = tiny(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=5)  # chunks of 5, 5 and 2
    out = list(engine.generate(PROMPT, 8))
    # every generated token is the reference argmax given all tokens before it
    expected = reference_logits(weights, PROMPT + out, arch)[len(PROMPT) - 1 :].argmax(-1)
    assert out == expected[: len(out)].tolist()

    # a prompt that extends the previous one reuses the cache: only the new tokens are prefilled,
    # but for a model with recurrent state, which holds every token its slot ran
    longer = PROMPT + out[:3] + [7]
    assert engine.cached_prefix(longer) == (0 if arch == "qwen35moe" else len(PROMPT) + 3)
    again = list(engine.generate(longer, 4))
    assert again == list(Engine(path, max_context=CONTEXT, prefill_chunk=5).generate(longer, 4))


@pytest.mark.usefixtures("reference_ops")
def test_generate_fills_context(tiny_model):
    # one capture of the decode graph replays correctly at every position up to the last
    path, weights = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    out = list(engine.generate(PROMPT, 1000))
    assert len(out) == CONTEXT - len(PROMPT) + 1
    expected = reference_logits(weights, PROMPT + out[:-1])[len(PROMPT) - 1 :].argmax(-1)
    assert out == expected.tolist()
    captured = engine._decode[1].captured
    engine.reset()
    assert list(engine.generate(PROMPT, 1000)) == out and engine._decode[1].captured is captured


@pytest.mark.usefixtures("reference_ops")
def test_prefill_graphs(tiny_model):
    # chunks of more than FEW_TOKENS tokens take the graph bound to prefill_chunk, and of up to
    # FEW_TOKENS, a single token too, the one bound to FEW_TOKENS
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=FEW_TOKENS + 4)
    prompt = PROMPT * 3  # chunks of FEW_TOKENS + 4 and 36 - FEW_TOKENS - 4
    out = list(engine.generate(prompt, 4))
    assert out == generated(path, prompt, 4)
    assert engine._chunk.captured is not None and engine._few_chunk.captured is not None
    longer = prompt + out[:3] + [7]  # a chunk of one token past the cached ones
    calls = engine._chunk.cnt, engine._few_chunk.cnt
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert (engine._chunk.cnt, engine._few_chunk.cnt) == (calls[0], calls[1] + 1)


def generated(path, prompt: list[int], n: int, sampling=GREEDY, seed=None) -> list[int]:
    # what an engine with nothing cached generates
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    return list(engine.generate(prompt, n, sampling, seed))


def prefill_starts(engine: Engine, monkeypatch) -> list[int]:
    # records the first position of each prefilled chunk
    starts, prefill = [], engine._prefill

    def spy(sequence, size):
        starts.append(len(engine._cached[sequence.slot]))
        return prefill(sequence, size)

    monkeypatch.setattr(engine, "_prefill", spy)
    return starts


@pytest.mark.usefixtures("reference_ops")
def test_slots_keep_their_sequences(tiny_model, monkeypatch):
    # a sequence in one slot leaves the keys and values of the others as they were
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    out = list(engine.generate(PROMPT, 6))
    held = len(PROMPT) + 5  # the last token was never run
    before = engine.model.cache[0][:, 0, :, :held].numpy()
    other = [9, 8, 7, 6, 5]
    assert list(engine.generate(other, 6)) == generated(path, other, 6)  # in the other slot
    np.testing.assert_array_equal(engine.model.cache[0][:, 0, :, :held].numpy(), before)

    # the conversation continues in its slot, from where it was
    starts, longer = prefill_starts(engine, monkeypatch), PROMPT + out + [7]
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert starts == [held]


@pytest.mark.usefixtures("reference_ops")
def test_shared_prefix_is_copied(tiny_model, monkeypatch):
    # a prompt that leaves a slot's tokens takes another slot, starting from a copy of the prefix
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    list(engine.generate(PROMPT, 4))
    starts, branch = prefill_starts(engine, monkeypatch), PROMPT[:9] + [1, 2, 3]
    assert list(engine.generate(branch, 6)) == generated(path, branch, 6)
    assert starts == [9]
    # and the first conversation's tokens are still cached
    longer = PROMPT + [4]
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert starts == [9, len(PROMPT)]


@pytest.mark.usefixtures("reference_ops")
def test_owners_share_their_own_prefixes(tiny_model, monkeypatch):
    # a slot's tokens are its owner's: another's prompt that shares them starts over in a slot of
    # its own, and the owner's still goes on from them; a slot of another's that a prompt takes,
    # the least recently used, starts anew
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    list(engine.generate(PROMPT, 4, owner="a"))
    longer = PROMPT + [4]
    assert engine.cached_prefix(longer, "a") == len(PROMPT)
    assert engine.cached_prefix(longer, "b") == engine.cached_prefix(longer) == 0
    starts, branch = prefill_starts(engine, monkeypatch), PROMPT[:9] + [1, 2, 3]
    assert list(engine.generate(branch, 6, owner="b")) == generated(path, branch, 6)
    assert starts == [0, 8]
    assert list(engine.generate(longer, 4, owner="a")) == generated(path, longer, 4)
    assert starts == [0, 8, len(PROMPT)]
    assert list(engine.generate(branch + [5], 4, owner="c")) == generated(path, branch + [5], 4)
    assert starts == [0, 8, len(PROMPT), 0, 8]
    assert engine.cached_prefix(branch + [6], "b") == 0
    assert engine.cached_prefix(longer + [6], "a") == len(longer)


@pytest.mark.usefixtures("reference_ops")
def test_recurrent_state_resumes_for_its_owner(tiny):
    # a slot's kept recurrent state goes on only for its owner's prompts
    path, _ = tiny("qwen35moe")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    prompt = (PROMPT * 2)[:20]
    list(engine.generate(prompt, 3, owner="a"))
    turn = prompt[:18] + [7, 1, 2]
    assert engine.cached_prefix(turn, "a") == len(prompt) - KEEP_BACK
    assert engine.cached_prefix(turn, "b") == 0


@pytest.mark.usefixtures("reference_ops")
def test_recurrent_state_shares_whole_slots(tiny, monkeypatch):
    # recurrent state holds all a slot ran: a prompt that shares part of a slot's tokens starts
    # over in another, and one that shares all of them goes on from there
    path, _ = tiny("qwen35moe")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    out = list(engine.generate(PROMPT, 4))
    starts, branch = prefill_starts(engine, monkeypatch), PROMPT[:9] + [1, 2, 3]
    assert list(engine.generate(branch, 6)) == generated(path, branch, 6)
    longer = PROMPT + out + [7]
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert starts == [0, 8, len(PROMPT) + 3]


@pytest.mark.usefixtures("reference_ops")
def test_recurrent_state_resumes_where_kept(tiny, monkeypatch):
    # a prompt that shares another but for its last tokens, as a chat's next turn, goes on from
    # the state kept KEEP_BACK tokens before that one's end: here from 4 of 20, a chunk of 1
    # ending where the new prompt's state is kept, at 5 of 21, then chunks of 8
    path, _ = tiny("qwen35moe")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    prompt = (PROMPT * 2)[:20]
    list(engine.generate(prompt, 3))
    starts, turn = prefill_starts(engine, monkeypatch), prompt[:18] + [7, 1, 2]
    assert engine.cached_prefix(turn) == len(prompt) - KEEP_BACK  # as the server reports it
    assert list(engine.generate(turn, 4)) == generated(path, turn, 4)
    assert starts == [len(prompt) - KEEP_BACK, len(turn) - KEEP_BACK, 13]


@pytest.mark.usefixtures("reference_ops")
def test_copy_takes_recurrent_state(tiny):
    model = transformer(tiny("qwen35moe")[0], slots=2)
    model(Tensor([PROMPT], dtype=dtypes.int32), 0, 0)
    model.copy(0, 1)
    after = Tensor([[7]], dtype=dtypes.int32)
    first, copied = (model.logits(model(after, len(PROMPT), slot)).numpy() for slot in (0, 1))
    np.testing.assert_array_equal(first, copied)


@pytest.mark.parametrize(
    "arch, slots, batches",
    [("llama", 5, [1, 2, 4, 5]), ("qwen35moe", 2, [1, 2]), ("gemma4", 2, [1, 2])],
)
def test_warm_up(tiny, tiny_assistant, monkeypatch, arch, slots, batches):
    # compiles every graph, the copy's and every batch's too, and keeping and restoring recurrent
    # state's, and a drafter's speculative step's, and leaves nothing cached that a generation
    # could see. Few tokens are 4 here: the reference Gated DeltaNet's graphs grow with their
    # tokens.
    monkeypatch.setattr("leat.engine.FEW_TOKENS", 4)
    monkeypatch.setattr("leat.engine.KEEP_BACK", 4)
    path, _ = tiny(arch)
    draft = {"gemma4": tiny_assistant[0], "qwen35moe": path}.get(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=slots, draft=draft)
    engine.warm_up()
    graphs = [engine._chunk, engine._few_chunk, *engine._decode.values(), engine._copy]
    graphs += [engine._keep, engine._restore] if arch == "qwen35moe" else []
    if draft:  # of each number of sequences
        graphs += [*engine._speculate.values(), *engine._settle.values()]
        assert list(engine._speculate) == list(range(1, min(slots, engine.drafter.sequences) + 1))
    assert list(engine._decode) == batches
    captured = [jit.captured for jit in graphs]
    assert all(captured) and engine.cached_prefix(PROMPT) == 0
    assert list(engine.generate(PROMPT, 6)) == generated(path, PROMPT, 6)
    assert [jit.captured for jit in graphs] == captured


def test_graph_binds_kernel_variables():
    # a graph whose kernels bind a variable inside, as matmul's on RDNA, replays with its value:
    # TinyJit alone takes values from a graph's arguments
    groups = VARIABLES["groups"].unbind()[0]
    replayed = graph(lambda x: (x + Tensor.arange(8).float()[: groups.bind(4)].sum()).realize())
    assert [replayed(Tensor([1.0])).item() for _ in range(3)] == [7.0] * 3


def test_one_generation_at_a_time(tiny_model):
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)
    tokens = engine.generate(PROMPT, 6)
    next(tokens)
    with pytest.raises(RuntimeError, match="unfinished"):
        next(engine.generate(PROMPT, 6))
    tokens.close()
    next(engine.generate(PROMPT, 6))


def test_end_of_generation(tiny_model, monkeypatch):
    # a generation ends at an end-of-generation token, which it yields, unless it ignores them
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)
    out = list(engine.generate(PROMPT, 8))
    monkeypatch.setattr(engine.tokenizer, "eog_ids", {out[3]})
    assert list(engine.generate(PROMPT, 8)) == out[: out.index(out[3]) + 1]
    assert list(engine.generate(PROMPT, 8, ignore_eog=True)) == out


def test_least_recently_used_slot(tiny_model):
    # a prompt that shares no slot's tokens takes the slot that least recently started one
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)
    for prompt in (PROMPT, [9, 8, 7, 6], PROMPT + [1], [3, 3, 3]):
        list(engine.generate(prompt, 2))
    assert engine.cached_prefix(PROMPT + [5]) == len(PROMPT)
    assert engine.cached_prefix([9, 8, 7, 6, 5]) == 0


def test_seeded_sampling(tiny_model):
    # a seed repeats a sampled generation, whatever part of its prompt was cached; graphs captured
    # on the first call draw anew in every generation without one
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)
    first = list(engine.generate(PROMPT, 8, Sampling(1.0), seed=1))
    assert list(engine.generate(PROMPT, 8, Sampling(1.0), seed=1)) == first  # prefix cached
    engine.reset()
    assert list(engine.generate(PROMPT, 8, Sampling(1.0), seed=1)) == first
    assert len({tuple(engine.generate(PROMPT, 8, Sampling(1.0))) for _ in range(3)}) == 3


def test_presence_penalty(tiny_model):
    # off the logits of every token a generation has generated, the first from its prompt's last
    # chunk, a single token here, and none of another generation's
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)
    penalized = Sampling(presence_penalty=100.0)
    out = list(engine.generate(PROMPT, 12, penalized, ignore_eog=True))
    assert len(set(out)) == len(out)
    list(engine.generate(PROMPT[:11], 4, ignore_eog=True))  # 11 of its 12 tokens cached
    assert list(engine.generate(PROMPT, 12, penalized, ignore_eog=True)) == out
    # but for a format's markup, which a reply writes again and again
    special = engine.tokenizer.special_ids
    assert special and engine._text.numpy().tolist() == [
        i not in special for i in range(len(engine._text.numpy()))]  # fmt: skip


# ******** several sequences at once ********


def run_all(engine: Engine, starts: dict[int, tuple]) -> dict[int, list[int]]:
    # steps until every sequence is done, starting each of `starts` {step: start() arguments} at
    # its step; returns each one's tokens, as step() gave them, by its step
    sequences, out, n = {}, {}, 0
    while n == 0 or engine.active or n <= max(starts):
        if n in starts:
            sequences[n] = engine.start(*starts[n])
            out[n] = []
        for sequence, token in engine.step():
            out[next(k for k, s in sequences.items() if s is sequence)].append(token)
        n += 1
    assert all(sequences[k].tokens == tokens for k, tokens in out.items())
    return out


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ["llama", "qwen3moe", "gemma4", "gpt-oss", "qwen35moe"])
def test_batched_matches_alone(tiny, arch):
    # sequences that join and leave a batch, 3 padded to 4 too, generate what each would alone:
    # greedy or seeded, and with prompts as long as a chunk or shared in part
    path, _ = tiny(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=4)
    starts = {
        0: (PROMPT, 9),
        1: (PROMPT[:9] + [1, 2], 6, Sampling(1.0, top_k=3, presence_penalty=1.0), 5),
        3: ([4, 2], 4),
        12: (PROMPT[::-1] * 2, 5, Sampling(0.8, top_p=0.9), 7),
    }
    got = run_all(engine, starts)
    for n, args in starts.items():
        assert got[n] == generated(path, *args), n


@pytest.mark.usefixtures("reference_ops")
def test_batches_past_the_largest(tiny_model, monkeypatch):
    # more sequences than a decode graph takes run in batches in turn, two of the same graph
    monkeypatch.setattr("leat.engine.BATCH", 2)
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=5)
    assert list(engine._decode) == [1, 2]
    starts = {
        0: (PROMPT, 7),
        1: ([4, 2], 6),
        2: (PROMPT[3:], 5, Sampling(1.0, min_p=0.1), 3),
        3: ([7, 7, 1], 6),
        4: ([9], 4),
    }
    got = run_all(engine, starts)
    for n, args in starts.items():
        assert got[n] == generated(path, *args), n


@pytest.mark.usefixtures("reference_ops")
def test_prefill_shares_steps(tiny_model):
    # a long prompt prefills a chunk per step, while the sequences past their prompts each get a
    # token in every one of those steps
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=4, slots=2)
    first = engine.start(PROMPT, 20)
    while not first.tokens:
        engine.step()
    second = engine.start(PROMPT[::-1] + [3, 1], 2)  # 14 tokens: 4 chunks
    steps = [[s for s, _ in engine.step()] for _ in range(4)]
    assert steps == [[first]] * 3 + [[second, first]]
    assert len(first.tokens) == 5 and len(second.tokens) == 1


@pytest.mark.usefixtures("reference_ops")
def test_prefill_shares_smaller_chunks(tiny_model, monkeypatch):
    # while others decode, a prompt prefills in chunks of SHARED_CHUNK at most, and alone in
    # chunks of prefill_chunk
    monkeypatch.setattr("leat.engine.SHARED_CHUNK", 3)
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    starts = prefill_starts(engine, monkeypatch)
    first = engine.start(PROMPT, 20)  # 12 tokens: chunks of 8 and 4, alone
    while not first.tokens:
        engine.step()
    second = engine.start([9, 8, 7, 6, 5, 4, 3], 2)  # chunks of 3, 3 and 1, beside the first
    while not second.tokens:
        engine.step()
    assert starts == [0, 8, 0, 3, 6]
    assert second.tokens == generated(path, [9, 8, 7, 6, 5, 4, 3], 1)


@pytest.mark.usefixtures("reference_ops")
def test_cancel_mid_prompt(tiny_model, monkeypatch):
    # a sequence cancelled partway through its prompt frees its slot, which keeps the chunks it
    # prefilled: the same prompt continues from there, and generates as it would from scratch
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=4, slots=1)
    sequence = engine.start(PROMPT, 6)
    engine.step()
    engine.step()
    engine.cancel(sequence)
    assert not engine.active and engine.cached_prefix(PROMPT) == 8
    starts = prefill_starts(engine, monkeypatch)
    assert list(engine.generate(PROMPT, 6)) == generated(path, PROMPT, 6)
    assert starts == [8]


def test_start_needs_a_free_slot(tiny_model):
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)
    first, _ = engine.start(PROMPT, 4), engine.start([3, 1, 4], 4)
    with pytest.raises(RuntimeError, match="all 2 slots"):
        engine.start([1, 5, 9], 4)
    with pytest.raises(RuntimeError, match="unfinished"):
        next(engine.generate([1, 5, 9], 4))
    engine.cancel(first)
    assert first.done and first not in engine.active
    engine.start([1, 5, 9], 4)


# the most mean KL divergence from llama.cpp and the least agreement on the top token each
# architecture allows. Both paths score about 0.0012 and 98% on Llama 3.1 8B, int8 activations
# adding noise as in llama.cpp; for scale, dropping its rope frequency factors, a subtle bug, scored
# 0.0026 on the reference ops. Qwen3 8B, more sensitive, scores 0.0025 and 98% on those too,
# Qwen2.5 7B 0.0031 and both paths 0.0036, and Qwen3 30B A3B 0.003 to 0.0053, where noise also
# flips a token's choice of experts. The top token is the noisier measure: on the 1020 positions
# here, a change of rounding in the matrix kernels took Llama's decode path's KL from 0.00128 to
# 0.00126 and its agreement from 98.2 to 97.9%. Qwen3.6 35B A3B, of 256 experts and recurrent
# state, scores 0.0082 and 97.4% on both paths. Mistral Small 3.2 24B, of the llama
# architecture, scores 0.0021 and 98.1% on the prompt's path, and 0.0025 and 98.4% on the decode
# path, its depth adding to the noise, as the 0.0017 of 20 chunks of the README's table.
LIMITS = {
    "llama": (0.0015, 0.975), "qwen2": (0.0045, 0.97), "qwen3": (0.0035, 0.97),
    "qwen3moe": (0.007, 0.97), "qwen35moe": (0.01, 0.965), "Mistral Small": (0.003, 0.975),
}  # fmt: skip


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
def test_matches_llama_cpp(model_path, llama_cpp, wikitext, tmp_path, decode):
    metadata = GGUF.open(model_path).metadata
    arch, name = metadata["general.architecture"], metadata.get("general.name", "")
    if (limits := next((LIMITS[k] for k in LIMITS if k in name), LIMITS.get(arch))) is None:
        pytest.skip(f"{arch} is instruction-tuned only, and scores raw text badly")
    args = ["-m", model_path, "-f", wikitext, "-c", "512", "--chunks", "4"]
    args += ["--kl-divergence-base", base := tmp_path / "base.kld"]
    subprocess.run([llama_cpp / "llama-perplexity", *args], check=True, capture_output=True)
    engine = Engine(model_path, max_context=512)
    quality = bench.kl_divergence(engine, base, decode=decode)
    kl, top1 = limits
    assert quality.kl_mean is not None and quality.kl_mean < kl
    assert quality.top1 is not None and quality.top1 > top1


@pytest.mark.gpu
@pytest.mark.model
def test_chunked_prefill(model_path):
    # prefilling in chunks of a bound length, through the kernels' symbolic paths, leaves the first
    # layer's keys and values, or its recurrent states, of one pass over the whole prompt, with its
    # fixed shapes, and a next token it scores as likely. Deeper layers drift apart: kernels
    # compiled for either differ in the last bit here and there, which can move an activation to
    # the next int8 step or flip a choice of experts, and so a near tie: on tinygrad's own
    # attention, Qwen2.5 7B's chunked pass once picked a token 0.07 below the best.
    engine = Engine(model_path, max_context=512, prefill_chunk=128)
    vocab, bos = engine.config.vocab_size, engine.tokenizer.bos_id
    # as many tokens as every layer's cache holds in one run: gpt-oss's windows of 128 keep rings
    # of 256 positions, which a run of more than 129 would wrap over the keys its first tokens read
    m = engine.model
    n = min([300] + [size - window + 1 for size, window, ring
                     in zip(m.sizes, m.config.windows, m.rings, strict=True) if ring])  # fmt: skip
    prompt = [0 if bos is None else bos] + [(i * 7919) % vocab for i in range(n - 1)]

    def held() -> list[np.ndarray]:  # what the first layer holds of the prompt
        if (cache := engine.model.cache[0]) is not None:
            return [cache[:, :, :, : len(prompt)].numpy().astype(np.float32)]
        return [state[0].numpy() for state in engine.model.states[0]]

    token = next(engine.generate(prompt, 1))
    chunked = held()
    hidden = engine.model(Tensor([prompt], dtype=dtypes.int32), 0)  # the same positions again
    logits = engine.model.logits(hidden[:, -1]).numpy().reshape(-1)
    assert logits[token] > logits.max() - 0.1
    for got, whole in zip(chunked, held(), strict=True):
        np.testing.assert_allclose(got, whole, rtol=1e-2, atol=1e-2)
