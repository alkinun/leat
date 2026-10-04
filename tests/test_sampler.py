import numpy as np
from tinygrad import Tensor, dtypes

from leat.sampler import sample

LOGITS = [2.0, 1.0, 0.0, -1.0, 0.5]


def draws(
    logits: list[list[float]], temperature: float | list[float], seed: int | list[int],
    position: int | list[int],
) -> list[int]:  # fmt: skip
    # one value for every row, or one per row
    temperatures = Tensor(temperature if isinstance(temperature, list) else [temperature])
    seeds = Tensor(seed if isinstance(seed, list) else [seed], dtype=dtypes.uint32)
    positions = Tensor(position, dtype=dtypes.int32) if isinstance(position, list) else position
    return sample(Tensor(logits), temperatures, seeds, positions).numpy().ravel().tolist()


def test_greedy():
    assert draws([LOGITS, LOGITS[::-1]], 0.0, 0, 0) == [0, 4]


def test_temperature_matches_softmax():
    # each row a sequence of its own seed
    n, temperature = 20000, 0.7
    got = draws([LOGITS] * n, temperature, list(range(n)), 0)
    p = np.exp(np.array(LOGITS) / temperature)
    np.testing.assert_allclose(np.bincount(got, minlength=len(LOGITS)) / n, p / p.sum(), atol=0.015)


def test_seeded():
    # draws repeat for a seed and position, and change with either
    flat = [[0.0] * 1000]
    first = draws(flat, 1.0, 7, 3)
    assert draws(flat, 1.0, 7, 3) == first
    assert draws(flat, 1.0, 8, 3) != first and draws(flat, 1.0, 7, 4) != first


def test_rows_are_their_own():
    # a row draws as it would alone, whatever the other rows of its batch: greedy or not, and
    # whatever their seeds and positions
    flat, peaked = [0.0] * 1000, LOGITS + [-9.0] * 995
    alone = [draws([flat], 1.0, 7, 3)[0], draws([peaked], 0.0, 0, 9)[0]]
    batched = draws([flat, peaked, flat], [1.0, 0.0, 1.0], [7, 0, 8], [3, 9, 5])
    assert batched[:2] == alone and batched[2] == draws([flat], 1.0, 8, 5)[0]
