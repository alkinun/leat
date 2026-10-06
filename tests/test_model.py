import subprocess

import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from leat import bench
from leat.engine import FEW_TOKENS, KEEP_BACK, Engine
from leat.gguf import GGUF
from leat.model import CACHE_TILE, Config, Transformer
from leat.sampler import GREEDY, Sampling
from tests.helpers import CONTEXT, reference_logits

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


@pytest.fixture
def reference_ops(monkeypatch):
    # exact comparisons with the f64 reference: fast kernels quantize activations to int8
    monkeypatch.setenv("LEAT_KERNELS", "ref")


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_forward_matches_reference(tiny, arch):
    path, weights = tiny(arch)
    f = GGUF.open(path)
    model = Transformer(Config.from_gguf(f.metadata), f.load(), CONTEXT)
    tokens = Tensor([PROMPT], dtype=dtypes.int32)
    logits = model.logits(model(tokens, 0)).numpy()[0]
    expected = reference_logits(weights, PROMPT, arch)
    np.testing.assert_allclose(logits, expected, rtol=2e-3, atol=2e-3)


def test_cache_whole_tiles(tiny_model):
    # attention kernels take caches of whole tiles; generation still stops at max_context
    path, _ = tiny_model
    f = GGUF.open(path)
    model = Transformer(Config.from_gguf(f.metadata), f.load(), 50)
    assert model.cache[0].shape[3] == CACHE_TILE and model.max_context == 50


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_generate_matches_reference(tiny, arch):
    path, weights = tiny(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=5)  # chunks of 5, 5 and 2
    out = list(engine.generate(PROMPT, 8))
    # every generated token is the reference argmax given all tokens before it
    expected = reference_logits(weights, PROMPT + out, arch)[len(PROMPT) - 1 :].argmax(-1)
    assert out == expected[: len(out)].tolist()

    # a prompt that extends the previous one reuses the cache: only the new tokens are prefilled
    longer = PROMPT + out[:3] + [7]
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
    # chunks of more than FEW_TOKENS tokens take the graph bound to prefill_chunk, of 2 to
    # FEW_TOKENS the one bound to FEW_TOKENS, and a single token the decode graph
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=FEW_TOKENS + 4)
    prompt = PROMPT * 3  # chunks of FEW_TOKENS + 4 and 36 - FEW_TOKENS - 4
    out = list(engine.generate(prompt, 4))
    assert out == generated(path, prompt, 4)
    assert engine._chunk.captured is not None and engine._few_chunk.captured is not None
    longer = prompt + out[:3] + [7]  # one token past the cached ones
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)


def generated(path, prompt: list[int], n: int) -> list[int]:
    # what an engine with nothing cached generates
    return list(Engine(path, max_context=CONTEXT, prefill_chunk=8).generate(prompt, n))


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
    assert list(engine.generate(turn, 4)) == generated(path, turn, 4)
    assert starts == [len(prompt) - KEEP_BACK, len(turn) - KEEP_BACK, 13]


@pytest.mark.usefixtures("reference_ops")
def test_copy_takes_recurrent_state(tiny):
    path, _ = tiny("qwen35moe")
    f = GGUF.open(path)
    model = Transformer(Config.from_gguf(f.metadata), f.load(), CONTEXT, slots=2)
    model(Tensor([PROMPT], dtype=dtypes.int32), 0, 0)
    model.copy(0, 1)
    after = Tensor([[7]], dtype=dtypes.int32)
    first, copied = (model.logits(model(after, len(PROMPT), slot)).numpy() for slot in (0, 1))
    np.testing.assert_array_equal(first, copied)


@pytest.mark.parametrize("arch", ["llama", "qwen35moe"])
def test_warm_up(tiny, arch):
    # compiles every graph, the copy's and every batch's too, and keeping and restoring recurrent
    # state's, and leaves nothing cached that a generation could see
    path, _ = tiny(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=FEW_TOKENS + 4, slots=5)
    engine.warm_up()
    graphs = [engine._chunk, engine._few_chunk, *engine._decode.values(), engine._copy]
    graphs += [engine._keep, engine._restore] if arch == "qwen35moe" else []
    assert list(engine._decode) == [1, 2, 4, 5]
    captured = [jit.captured for jit in graphs]
    assert all(captured) and engine.cached_prefix(PROMPT) == 0
    assert list(engine.generate(PROMPT, 6)) == generated(path, PROMPT, 6)
    assert [jit.captured for jit in graphs] == captured


def test_one_generation_at_a_time(tiny_model):
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)
    tokens = engine.generate(PROMPT, 6)
    next(tokens)
    with pytest.raises(RuntimeError, match="unfinished"):
        next(engine.generate(PROMPT, 6))
    tokens.close()
    next(engine.generate(PROMPT, 6))


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
    # sequences that join and leave a batch, 3 padded to 4 too, whose padding the mixtures of
    # experts skip, generate what each would alone: greedy or seeded, and with prompts as long
    # as a chunk or shared in part
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
        assert got[n] == generated_with(path, *args), n


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
        assert got[n] == generated_with(path, *args), n


def generated_with(path, prompt, n, sampling=GREEDY, seed=None) -> list[int]:
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    return list(engine.generate(prompt, n, sampling, seed))


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
# state, scores 0.0082 and 97.4% on both paths.
LIMITS = {
    "llama": (0.0015, 0.975), "qwen2": (0.0045, 0.97), "qwen3": (0.0035, 0.97),
    "qwen3moe": (0.007, 0.97), "qwen35moe": (0.01, 0.965),
}  # fmt: skip


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
def test_matches_llama_cpp(model_path, llama_cpp, wikitext, tmp_path, decode):
    if (arch := GGUF.open(model_path).metadata["general.architecture"]) not in LIMITS:
        pytest.skip(f"{arch} is instruction-tuned only, and scores raw text badly")
    args = ["-m", model_path, "-f", wikitext, "-c", "512", "--chunks", "4"]
    args += ["--kl-divergence-base", base := tmp_path / "base.kld"]
    subprocess.run([llama_cpp / "llama-perplexity", *args], check=True, capture_output=True)
    engine = Engine(model_path, max_context=512)
    quality = bench.kl_divergence(engine, base, decode=decode)
    kl, top1 = LIMITS[engine.gguf.metadata["general.architecture"]]
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
    prompt = [0 if bos is None else bos] + [(i * 7919) % vocab for i in range(299)]

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
