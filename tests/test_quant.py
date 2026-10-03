import numpy as np
import pytest
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from tinygrad import Tensor

from leat.quant import BLOCK, GGMLType, QTensor

# byte offsets of each block's f16 scales; random bytes there would be inf/nan
F16_FIELDS = {
    GGMLType.Q8_0: (0,),
    GGMLType.Q4_K: (0, 2),
    GGMLType.Q5_K: (0, 2),
    GGMLType.Q6_K: (208,),
}


def random_blocks(
    ggml_type: GGMLType, n: int, rng: np.random.Generator, scale: float = 1.0
) -> np.ndarray:
    blocks = rng.integers(0, 256, (n, BLOCK[ggml_type][1]), dtype=np.uint8)
    for offset in F16_FIELDS[ggml_type]:
        d = rng.uniform(-scale, scale, (n, 1)).astype(np.float16)
        blocks[:, offset : offset + 2] = d.view(np.uint8)
    return blocks


@pytest.mark.parametrize("ggml_type", list(F16_FIELDS))
def test_dequant_matches_ggml(ggml_type):
    rng = np.random.default_rng(int(ggml_type))
    blocks = random_blocks(ggml_type, 64, rng)
    shape = (8, 64 * BLOCK[ggml_type][0] // 8)
    out = QTensor(Tensor(blocks), ggml_type, shape).dequant().numpy()
    expected = dequantize(blocks, GGMLQuantizationType(ggml_type)).reshape(shape)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("ggml_type", [GGMLType.F32, GGMLType.F16, GGMLType.BF16])
def test_dequant_native(ggml_type):
    values = np.random.default_rng(0).standard_normal((4, 32)).astype(np.float32)
    data = Tensor(values).cast(
        {GGMLType.F32: "float32", GGMLType.F16: "half", GGMLType.BF16: "bfloat16"}[ggml_type]
    )
    out = QTensor(data.flatten(), ggml_type, (4, 32)).dequant().numpy()
    np.testing.assert_array_equal(out, data.float().numpy())


def test_dequant_unsupported():
    with pytest.raises(NotImplementedError, match="IQ2_XXS"):
        QTensor(Tensor.zeros(1, 66), GGMLType.IQ2_XXS, (256,)).dequant()
