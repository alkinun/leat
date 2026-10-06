"""Token sampling on the device: only the chosen ids ever leave the GPU."""

import math
from dataclasses import dataclass

from tinygrad import Tensor, UOp, dtypes

from leat import ops


@dataclass(frozen=True)
class Sampling:
    """How a sequence picks its tokens: greedily at temperature 0, else by drawing from
    softmax(logits / temperature) over those that top_k, top_p and min_p keep (see ops.cutoff()).
    presence_penalty comes off the logits of the tokens it has generated, greedy or not."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0

    def values(self) -> list[float]:
        # a row of the options sample() takes
        return [self.temperature, self.top_k, self.top_p, self.min_p, self.presence_penalty]


GREEDY = Sampling()


def sample(
    logits: Tensor, options: Tensor, seed: Tensor, position: Tensor | int | UOp,
    seen: Tensor | None = None,
) -> Tensor:  # fmt: skip
    """Picks one token per row of `logits` (B, V), each row a sequence of its own, as its
    Sampling.values() in `options` float32 (B, 5) say; seen (B, V) bool marks the tokens each has
    generated, those presence_penalty lowers.

    `options` and `seed` uint32 hold a row per row, or one for all, (1, 5) and (1,); they are
    tensors so one compiled graph serves every value. A row's draw depends only on its logits,
    its options, its seed and its `position`, the sampled token's position, one per row or one for
    all: a seeded generation repeats however much of its prompt was cached, whatever the other
    rows.
    """
    temperature, top_k, top_p, min_p, presence = (options[:, i : i + 1] for i in range(5))
    if seen is not None:
        logits = logits - seen.where(presence, 0.0)
    scores = logits / temperature.maximum(1e-6)
    cut = ops.cutoff(scores, top_k, top_p, min_p)
    # Gumbel-max: argmax(scores - log(-log(u))) is a draw from softmax(scores)
    gumbel = -(-_uniform(logits, seed, position).log()).log()
    drawn = (scores >= cut).where(scores + gumbel, -math.inf)
    return ops.argmax((temperature > 0).where(drawn, logits))


def _uniform(like: Tensor, seed: Tensor, position: Tensor | int | UOp) -> Tensor:
    # in (0, 1) for each value of `like` (B, V), hashed from its row's seed and position and its
    # index in the row, so draws keep no state
    position = position if isinstance(position, Tensor) else Tensor(position)
    key = _hash(_hash(seed.reshape(-1, 1)) + position.cast(dtypes.uint32).reshape(-1, 1))
    bits = _hash(Tensor.arange(like.shape[-1], dtype=dtypes.uint32).reshape(1, -1) ^ key)
    return ((bits >> 9).cast(dtypes.float32) + 0.5) * 2**-23  # 23 bits: 1 - 2^-24 is a float32


def _hash(x: Tensor) -> Tensor:
    # lowbias32 from Chris Wellons' hash prospector: any input bit flips each output bit with
    # probability close to 1/2. Every hash here mixes in the seed's buffer: tinygrad folds this one
    # over constants without wrapping the products to 32 bits.
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    return x ^ (x >> 16)
