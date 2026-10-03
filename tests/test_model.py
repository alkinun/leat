import numpy as np
from tinygrad import Tensor, dtypes

from leat.engine import Engine
from leat.gguf import GGUF
from leat.model import Config, Transformer
from tests.helpers import CONTEXT, reference_logits

PROMPT = [5, 77, 120, 3, 299, 42, 8, 150, 61, 200, 9, 33]


def test_forward_matches_reference(tiny_model):
    path, weights = tiny_model
    f = GGUF.open(path)
    model = Transformer(Config.from_gguf(f.metadata), f.load(), CONTEXT)
    tokens = Tensor([PROMPT], dtype=dtypes.int32)
    logits = model.logits(model(tokens, 0)).numpy()[0]
    np.testing.assert_allclose(logits, reference_logits(weights, PROMPT), rtol=2e-3, atol=2e-3)


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


def test_generate_stops_at_context_end(tiny_model):
    engine = Engine(tiny_model[0], max_context=16, prefill_chunk=8)
    assert len(list(engine.generate(PROMPT, 100))) == 16 - len(PROMPT) + 1
