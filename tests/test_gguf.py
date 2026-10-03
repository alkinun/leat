import gguf
import numpy as np
import pytest
from gguf.quants import dequantize

from leat.gguf import GGUF
from leat.quant import GGMLType
from tests.test_quant import random_blocks


@pytest.fixture
def tiny_gguf(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "tiny.gguf"
    tensors = {
        "f32": rng.standard_normal((3, 5)).astype(np.float32),
        "f16": rng.standard_normal((2, 64)).astype(np.float16),
        # quantized tensors are written as (rows, bytes per row)
        "q8_0": random_blocks(GGMLType.Q8_0, 8, rng).reshape(4, 2 * 34),
        "q6_k": random_blocks(GGMLType.Q6_K, 4, rng).reshape(2, 2 * 210),
    }
    w = gguf.GGUFWriter(path, arch="llama")
    w.add_string("general.name", "tiny")
    w.add_uint32("llama.block_count", 2)
    w.add_float32("llama.rope.freq_base", 500000.0)
    w.add_bool("tokenizer.ggml.add_bos_token", True)
    w.add_array("tokenizer.ggml.tokens", ["<s>", "a", "ab"])
    w.add_array("tokenizer.ggml.token_type", [3, 1, 1])
    w.add_tensor("f32", tensors["f32"])
    w.add_tensor("f16", tensors["f16"])
    w.add_tensor("q8_0", tensors["q8_0"], raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("q6_k", tensors["q6_k"], raw_dtype=gguf.GGMLQuantizationType.Q6_K)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, tensors


def test_metadata(tiny_gguf):
    f = GGUF.open(tiny_gguf[0])
    assert f.metadata["general.architecture"] == "llama"
    assert f.metadata["general.name"] == "tiny"
    assert f.metadata["llama.block_count"] == 2
    assert f.metadata["llama.rope.freq_base"] == 500000.0
    assert f.metadata["tokenizer.ggml.add_bos_token"] is True
    assert f.metadata["tokenizer.ggml.tokens"] == ["<s>", "a", "ab"]
    assert f.metadata["tokenizer.ggml.token_type"] == [3, 1, 1]


def test_tensor_index(tiny_gguf):
    f = GGUF.open(tiny_gguf[0])
    index = {name: (t.type, t.shape, t.nbytes) for name, t in f.tensors.items()}
    assert index == {
        "f32": (GGMLType.F32, (3, 5), 60),
        "f16": (GGMLType.F16, (2, 64), 256),
        "q8_0": (GGMLType.Q8_0, (4, 64), 8 * 34),
        "q6_k": (GGMLType.Q6_K, (2, 512), 4 * 210),
    }


def test_load(tiny_gguf):
    path, tensors = tiny_gguf
    loaded = GGUF.open(path).load()
    np.testing.assert_array_equal(loaded["f32"].dequant().numpy(), tensors["f32"])
    np.testing.assert_array_equal(
        loaded["f16"].dequant().numpy(), tensors["f16"].astype(np.float32)
    )
    for name, qtype in (
        ("q8_0", gguf.GGMLQuantizationType.Q8_0),
        ("q6_k", gguf.GGMLQuantizationType.Q6_K),
    ):
        expected = dequantize(tensors[name], qtype).reshape(loaded[name].shape)
        np.testing.assert_array_equal(loaded[name].dequant().numpy(), expected)


def test_not_gguf(tmp_path):
    (path := tmp_path / "x.gguf").write_bytes(b"GGML" + bytes(60))
    with pytest.raises(ValueError, match="not a GGUF file"):
        GGUF.open(path)


def test_truncated(tiny_gguf, tmp_path):
    data = tiny_gguf[0].read_bytes()
    (path := tmp_path / "cut.gguf").write_bytes(data[:-100])
    with pytest.raises(ValueError, match="truncated"):
        GGUF.open(path)
