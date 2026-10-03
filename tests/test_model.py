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


@pytest.fixture
def reference_ops(monkeypatch):
    # exact comparisons with the f64 reference: fast kernels quantize activations to int8
    monkeypatch.setenv("LEAT_KERNELS", "ref")


@pytest.mark.usefixtures("reference_ops")
def test_forward_matches_reference(tiny_model):
    path, weights = tiny_model
    f = GGUF.open(path)
    model = Transformer(Config.from_gguf(f.metadata), f.load(), CONTEXT)
    tokens = Tensor([PROMPT], dtype=dtypes.int32)
    logits = model.logits(model(tokens, 0)).numpy()[0]
    np.testing.assert_allclose(logits, reference_logits(weights, PROMPT), rtol=2e-3, atol=2e-3)


def test_cache_whole_tiles(tiny_model):
    # attention kernels take caches of whole tiles; generation still stops at max_context
    path, _ = tiny_model
    f = GGUF.open(path)
    model = Transformer(Config.from_gguf(f.metadata), f.load(), 50)
    assert model.cache[0].shape[3] == CACHE_TILE and model.max_context == 50


@pytest.mark.usefixtures("reference_ops")
def test_generate_matches_reference(tiny_model):
    path, weights = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=5)  # chunks of 5, 5 and 2
    out = list(engine.generate(PROMPT, 8))
    # every generated token is the reference argmax given all tokens before it
    expected = reference_logits(weights, PROMPT + out)[len(PROMPT) - 1 :].argmax(-1)
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


def test_sampling_varies(tiny_model):
    # each generation draws new random numbers, though the graphs are captured on their first call
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)
    assert len({tuple(engine.generate(PROMPT, 8, temperature=1.0)) for _ in range(3)}) == 3


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
def test_matches_llama_cpp(model_path, llama_cpp, wikitext, tmp_path, decode):
    args = ["-m", model_path, "-f", wikitext, "-c", "512", "--chunks", "4"]
    args += ["--kl-divergence-base", base := tmp_path / "base.kld"]
    subprocess.run([llama_cpp / "llama-perplexity", *args], check=True, capture_output=True)
    quality = bench.kl_divergence(Engine(model_path, max_context=512), base, decode=decode)
    # 0.0011 for prefill and 0.0013 for decode, whose int8 activations add noise as in llama.cpp;
    # dropping llama 3.1's rope frequency factors, a subtle bug, scores 0.0026
    assert quality.kl_mean is not None and quality.kl_mean < 0.0015
    assert quality.top1 is not None and quality.top1 > 0.98


@pytest.mark.gpu
@pytest.mark.model
def test_chunked_prefill(model_path):
    # prefilling in chunks of a bound length, through the kernels' symbolic paths, leaves the cache
    # and the next token of one pass over the whole prompt
    prompt = [128000] + [(i * 7919) % 128000 for i in range(299)]
    whole, chunked = (Engine(model_path, max_context=512, prefill_chunk=n) for n in (512, 128))
    assert next(whole.generate(prompt, 1)) == next(chunked.generate(prompt, 1))
    for i in (0, 31):
        got, want = (e.model.cache[i][:, :, :, : len(prompt)].numpy() for e in (chunked, whole))
        np.testing.assert_allclose(got.astype(np.float32), want, rtol=1e-2, atol=1e-2)
