"""NVIDIA kernels for DEV=NV and DEV=CUDA, written in tinygrad's UOp DSL and rendered as CUDA C.

Each kernel family has a `supports_*` check and the op itself; leat.ops chooses between them and
the reference ops.
"""

from leat.nv.argmax import argmax, supports_argmax
from leat.nv.attention import (
    attention,
    flash_attention,
    supports_attention,
    supports_flash_attention,
)
from leat.nv.common import GROUP
from leat.nv.experts import mixture, route, scores, supports_mixture, supports_scores
from leat.nv.matmul import feed_forward, matmuls, supports_matmul
from leat.nv.matvec import matvecs, supports_matvec, swiglu
from leat.nv.quantize import quantize_q8

__all__ = [
    "GROUP", "argmax", "attention", "feed_forward", "flash_attention", "matmuls", "matvecs",
    "mixture", "quantize_q8", "route", "scores", "supports_argmax", "supports_attention",
    "supports_flash_attention", "supports_matmul", "supports_matvec", "supports_mixture",
    "supports_scores", "swiglu",
]  # fmt: skip
