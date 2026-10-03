import numpy as np
from tinygrad import Tensor

from leat.sampler import sample

LOGITS = [2.0, 1.0, 0.0, -1.0, 0.5]


def test_greedy():
    logits = Tensor([LOGITS, LOGITS[::-1]])
    assert sample(logits, Tensor([0.0])).numpy().ravel().tolist() == [0, 4]


def test_temperature_matches_softmax():
    Tensor.manual_seed(0)
    n, temperature = 20000, 0.7
    draws = sample(Tensor([LOGITS]).expand(n, len(LOGITS)), Tensor([temperature])).numpy().ravel()
    p = np.exp(np.array(LOGITS) / temperature)
    np.testing.assert_allclose(
        np.bincount(draws, minlength=len(LOGITS)) / n, p / p.sum(), atol=0.015
    )
