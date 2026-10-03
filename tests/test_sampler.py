import numpy as np
from tinygrad import Tensor, dtypes

from leat.sampler import sample

LOGITS = [2.0, 1.0, 0.0, -1.0, 0.5]


def draws(logits: list[list[float]], temperature: float, seed: int, position: int) -> list[int]:
    seeds = Tensor([seed], dtype=dtypes.uint32)
    return sample(Tensor(logits), Tensor([temperature]), seeds, position).numpy().ravel().tolist()


def test_greedy():
    assert draws([LOGITS, LOGITS[::-1]], 0.0, 0, 0) == [0, 4]


def test_temperature_matches_softmax():
    n, temperature = 20000, 0.7
    got = draws([LOGITS] * n, temperature, 0, 0)
    p = np.exp(np.array(LOGITS) / temperature)
    np.testing.assert_allclose(np.bincount(got, minlength=len(LOGITS)) / n, p / p.sum(), atol=0.015)


def test_seeded():
    # draws repeat for a seed and position, and change with either
    flat = [[0.0] * 1000] * 16
    first = draws(flat, 1.0, 7, 3)
    assert draws(flat, 1.0, 7, 3) == first
    assert draws(flat, 1.0, 8, 3) != first and draws(flat, 1.0, 7, 4) != first
