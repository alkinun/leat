import subprocess

import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from leat import bench
from leat.engine import Engine
from leat.gguf import GGUF
from leat.model import CACHE_TILE, Config, Transformer
from tests.helpers import CONTEXT, reference_logits

PROMPT = [5, 77, 120, 3, 299, 42, 8, 150, 61, 200, 9, 33]
ARCHS = ["llama", "qwen3", "qwen3moe", "gemma4", "gemma4-dense"]


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
    captured = engine._decode.captured
    engine.reset()
    assert list(engine.generate(PROMPT, 1000)) == out and engine._decode.captured is captured


def generated(path, prompt: list[int], n: int) -> list[int]:
    # what an engine with nothing cached generates
    return list(Engine(path, max_context=CONTEXT, prefill_chunk=8).generate(prompt, n))


def prefill_starts(engine: Engine, monkeypatch) -> list[int]:
    # records the first position of each prefilled chunk
    starts, prefill = [], engine._prefill

    def spy(tokens, slot, start_pos, *sampling):
        starts.append(start_pos.unbind()[1])
        return prefill(tokens, slot, start_pos, *sampling)

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
    first = list(engine.generate(PROMPT, 8, temperature=1.0, seed=1))
    assert list(engine.generate(PROMPT, 8, temperature=1.0, seed=1)) == first  # prefix cached
    engine.reset()
    assert list(engine.generate(PROMPT, 8, temperature=1.0, seed=1)) == first
    assert len({tuple(engine.generate(PROMPT, 8, temperature=1.0)) for _ in range(3)}) == 3


# the most mean KL divergence from llama.cpp and the least agreement on the top token each
# architecture allows. Both paths score about 0.0012 and 98% on Llama 3.1 8B, int8 activations
# adding noise as in llama.cpp; for scale, dropping its rope frequency factors, a subtle bug, scored
# 0.0026 on the reference ops. Qwen3 8B, more sensitive, scores 0.0025 and 98% on those too,
# and Qwen3 30B A3B 0.003 to 0.0053, where noise also flips a token's choice of experts. The top
# token is the noisier measure: on the 1020 positions here, a change of rounding in the matrix
# kernels took the decode path's KL from 0.00128 to 0.00126 and its agreement from 98.2 to 97.9%.
LIMITS = {"llama": (0.0015, 0.975), "qwen3": (0.0035, 0.97), "qwen3moe": (0.007, 0.97)}


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
    # layer's keys and values and the next token of one pass over the whole prompt, with its fixed
    # shapes. Deeper layers drift apart: kernels compiled for either differ in the last bit here
    # and there, which can move an activation to the next int8 step or flip a choice of experts.
    prompt = [128000] + [(i * 7919) % 128000 for i in range(299)]
    engine = Engine(model_path, max_context=512, prefill_chunk=128)
    token = next(engine.generate(prompt, 1))
    chunked = engine.model.cache[0][:, :, :, : len(prompt)].numpy()
    hidden = engine.model(Tensor([prompt], dtype=dtypes.int32), 0)  # the same positions again
    assert token == engine.model.logits(hidden[:, -1]).argmax().item()
    whole = engine.model.cache[0][:, :, :, : len(prompt)].numpy()
    np.testing.assert_allclose(chunked.astype(np.float32), whole, rtol=1e-2, atol=1e-2)
