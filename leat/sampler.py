"""Token sampling on the device: only the chosen ids ever leave the GPU."""

from tinygrad import Tensor, UOp, dtypes

from leat import ops


def sample(
    logits: Tensor, temperature: Tensor, seed: Tensor, position: Tensor | int | UOp
) -> Tensor:
    """Picks one token per row of `logits` (B, V), each row a sequence of its own: argmax where its
    temperature is 0, else a draw from softmax(logits / temperature).

    `temperature` float32 and `seed` uint32 hold a value per row, (B,), or one for all, (1,); they
    are tensors so one compiled graph serves every value. A row's draw depends only on its logits,
    its seed and its `position`, the sampled token's position, one per row or one for all: a
    seeded generation repeats however much of its prompt was cached, whatever the other rows.
    """
    temperature = temperature.reshape(-1, 1)
    # Gumbel-max: argmax(logits / t - log(-log(u))) is a draw from softmax(logits / t)
    gumbel = -(-_uniform(logits, seed, position).log()).log()
    scores = (temperature > 0).where(logits / temperature.maximum(1e-6) + gumbel, logits)
    return ops.argmax(scores)


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
