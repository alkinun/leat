"""Hand-written kernels, in tinygrad's UOp DSL, rendered as CUDA C or HIP C.

The warp-level kernels, which carry decoding, run on NVIDIA GPUs (DEV=NV or CUDA) and on AMD's RDNA
GPUs (DEV=AMD), as Strix Halo's; those on tensor cores, matmul's and FlashAttention's, on NVIDIA's
mma.sync and RDNA 3's WMMA. Each family the model's ops dispatch to has a `supports_*` check beside
the op itself: leat.ops chooses between them and the reference ops.
"""

from leat.kernels.argmax import argmax, supports_argmax
from leat.kernels.attention import (
    attention,
    flash_attention,
    rotate,
    supports_attention,
    supports_flash_attention,
    supports_rotate,
)
from leat.kernels.common import GROUP
from leat.kernels.cutoff import cutoff, supports_cutoff
from leat.kernels.delta import delta_net, supports_delta_net
from leat.kernels.experts import mixture, route, scores, supports_mixture, supports_scores
from leat.kernels.matmul import feed_forward, matmuls, supports_matmul
from leat.kernels.matvec import MATVEC_TOKENS, matvecs, supports_matvec, swiglu
from leat.kernels.norms import add_normed, supports_add_normed
from leat.kernels.quantize import quantize_q8

__all__ = [
    "GROUP", "MATVEC_TOKENS", "add_normed", "argmax", "attention", "cutoff", "delta_net",
    "feed_forward", "flash_attention", "matmuls", "matvecs", "mixture", "quantize_q8", "rotate",
    "route", "scores", "supports_add_normed", "supports_argmax", "supports_attention",
    "supports_cutoff", "supports_delta_net", "supports_flash_attention", "supports_matmul",
    "supports_matvec", "supports_mixture", "supports_rotate", "supports_scores", "swiglu",
]  # fmt: skip
