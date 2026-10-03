"""Token sampling on the device: only the chosen id ever leaves the GPU."""

from tinygrad import Tensor, UOp, dtypes

from leat import ops


def sample(logits: Tensor, temperature: Tensor, seed: Tensor, position: int | UOp) -> Tensor:
    """Picks one token per row of `logits` (B, V): argmax when `temperature` is 0, else a draw from
    softmax(logits / temperature).

    `temperature` (1,) float32 and `seed` (1,) uint32 are tensors so one compiled graph serves
    every value. Draws depend only on the seed and `position`, the sampled token's position, so a
    seeded generation repeats however much of its prompt was cached.
    """
    # Gumbel-max: argmax(logits / t - log(-log(u))) is a draw from softmax(logits / t)
    gumbel = -(-_uniform(logits, seed, position).log()).log()
    scores = (temperature > 0).where(logits / temperature.maximum(1e-6) + gumbel, logits)
    return ops.argmax(scores)


def _uniform(like: Tensor, seed: Tensor, position: int | UOp) -> Tensor:
    # in (0, 1) for each value of `like`, hashed from the seed, the position and the value's index,
    # so draws keep no state
    key = _hash(_hash(seed) + Tensor(position).cast(dtypes.uint32))
    bits = _hash(Tensor.arange(like.numel(), dtype=dtypes.uint32).reshape(like.shape) ^ key)
    return ((bits >> 9).cast(dtypes.float32) + 0.5) * 2**-23  # 23 bits: 1 - 2^-24 is a float32


def _hash(x: Tensor) -> Tensor:
    # lowbias32 from Chris Wellons' hash prospector: any input bit flips each output bit with
    # probability close to 1/2. Every hash here mixes in the seed's buffer: tinygrad folds this one
    # over constants without wrapping the products to 32 bits.
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    return x ^ (x >> 16)
