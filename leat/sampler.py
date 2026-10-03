"""Token sampling on the device: only the chosen id ever leaves the GPU."""

from tinygrad import Tensor

from leat import ops


def sample(logits: Tensor, temperature: Tensor) -> Tensor:
    """Picks one token per row of `logits` (B, V): argmax when `temperature` is 0, else a sample.

    `temperature` is a (1,) tensor so one compiled graph serves every value.
    """
    # Gumbel-max: argmax(logits / t - log(-log(u))) is a draw from softmax(logits / t)
    gumbel = -(-Tensor.rand_like(logits).maximum(1e-12).log()).log()
    scores = (temperature > 0).where(logits / temperature.maximum(1e-6) + gumbel, logits)
    return ops.argmax(scores)
