import json
import os
import random
import subprocess

import pytest

from leat.gguf import GGUF
from leat.tokenizer import _BYTE_CHAR, BYTE, CONTROL, NORMAL, UNKNOWN, Tokenizer
from tests.helpers import ids, tiny_metadata


def tiny_tokenizer(**overrides) -> Tokenizer:
    return Tokenizer(tiny_metadata(**overrides))


def test_merges_apply_in_rank_order():
    tok = tiny_tokenizer()
    assert tok.encode("abc", bos=False) == ids(tok, "a", "bc")  # "b c" outranks "a b"
    assert tok.encode("hellx", bos=False) == ids(tok, "hell", "x")


def test_bos():
    tok = tiny_tokenizer()
    assert tok.add_bos and tok.encode("ab")[0] == tok.bos_id
    assert tok.encode("ab", bos=False) == ids(tok, "ab")
    assert tiny_tokenizer(**{"tokenizer.ggml.add_bos_token": False}).encode("ab") == ids(tok, "ab")


def test_special_tokens():
    tok = tiny_tokenizer()
    text = "ab<|eot|>ab<user>"
    assert tok.encode(text, bos=False, special=True) == ids(tok, "ab", "<|eot|>", "ab", "<user>")
    # without special parsing, control text is spelled out, but user-defined tokens still match
    plain = tok.encode(text, bos=False)
    assert ids(tok, "<|eot|>")[0] not in plain and plain[-1] == ids(tok, "<user>")[0]
    assert tok.eog_ids == {tok.eos_id}


def test_decode_skips_control_tokens():
    tok = tiny_tokenizer()
    assert tok.decode(tok.encode("ab<|eot|>héllo 🚀", special=True)) == "abhéllo 🚀"


def test_stream_holds_partial_utf8():
    tok = tiny_tokenizer()
    step = tok.stream()
    assert [step(i) for i in tok.encode("é🚀", bos=False)] == ["", "é", "", "", "", "🚀"]
    cut = tok.encode("🚀", bos=False)[:2]  # an incomplete character ends as decode() ends it
    assert [step(i) for i in cut] + [step(None)] == ["", "", tok.decode(cut)]


def test_sentencepiece_style():
    # Gemma 4: BPE over characters with spaces as U+2581, whole lines as words, and byte tokens
    # for characters the vocab lacks
    tokens = [f"<0x{b:02X}>" for b in range(256)] + [
        "\u2581",
        "a",
        "b",
        "\u2581a",
        "ab",
        "\n",
        "\n\n",
    ]
    tokens += ["<bos>", "<turn|>"]
    types = [BYTE] * 256 + [NORMAL] * 7 + [CONTROL] * 2
    tok = Tokenizer(
        {"tokenizer.ggml.model": "gemma4", "tokenizer.ggml.tokens": tokens,
         "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": ["\u2581 a", "a b"],
         "tokenizer.ggml.bos_token_id": tokens.index("<bos>")}
    )  # fmt: skip
    text = "ab a\n\né"
    assert tok.encode(text, bos=False) == [tokens.index(t) for t in ["ab", "\u2581a", "\n\n"]] + [
        0xC3,
        0xA9,
    ]
    assert tok.decode(tok.encode(text)) == text and tok.eog_ids == {tokens.index("<turn|>")}


def test_sentencepiece():
    # Mistral: merges the pair whose token scores highest, the leftmost on ties, from characters
    # with spaces as U+2581 and a space before text at the start or after a special token
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    tokens += ["\u2581", "a", "b", "\u2581a", "ab", "\u2581ab"]
    scores = [0.0] * 259 + [-9.0, -9.0, -9.0, -1.0, -2.0, -0.5]
    types = [UNKNOWN, CONTROL, CONTROL] + [BYTE] * 256 + [NORMAL] * 6
    tok = Tokenizer(
        {"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": tokens,
         "tokenizer.ggml.scores": scores, "tokenizer.ggml.token_type": types,
         "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2}
    )  # fmt: skip
    # "\u2581a" (-1) beats "ab" (-2), and then "\u2581ab" (-0.5) takes its b
    assert tok.encode("ab ab") == [1] + [tokens.index("\u2581ab")] * 2
    pieces = ["<s>", "\u2581", "b", "</s>", "\u2581", "<0xC3>", "<0xA9>"]  # é is not in the vocab
    assert tok.encode("b</s>é", special=True) == [tokens.index(t) for t in pieces]
    # decoding drops the space it puts before text, as llama.cpp, unless the text starts with BOS
    assert tok.decode(tok.encode("ab ab", bos=False)) == "ab ab"
    assert tok.decode(tok.encode("ab ab")) == " ab ab"


def test_unsupported():
    with pytest.raises(NotImplementedError, match="spm"):
        tiny_tokenizer(**{"tokenizer.ggml.model": "spm"})
    with pytest.raises(NotImplementedError, match="qwen9"):
        tiny_tokenizer(**{"tokenizer.ggml.pre": "qwen9"})


CORPUS = [
    "Hello world! It's 2026.",
    "I'm sure you'RE right, they'll've DONE it'S 'ſ 'K",  # Python's (?i) would fold U+017F into s
    "   leading spaces\tand\ttabs\r\nwindows\r\n\r\nlines\n\n\n  trailing   ",
    " nbsp　ideo thin\x1cfs\x1d\x1e\x1f\x85nel ls",  # Python's \s differs here
    "1234567 3.14159 1,000,000 ٣٤٥٦ ²³ ½ Ⅻ ① 0x1F 007",  # Python's \d differs here
    "İstanbul'da ışık, Straße, Ærøskøbing, ДОБРЫЙ день",
    "你好世界，こんにちは、안녕하세요。नमस्ते مرحبا שלום สวัสดี",
    "é ä 👨‍👩‍👧‍👦 🚀🔥 ❤️ 🇹🇷 ​‍﻿",
    "<|eot_id|> plain <|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
    'def f(x):\n    return x**2  # comment\n\n\tif x: pass\n{"a": [1, {"b": null}]}',
    "",
    " ",
    "''s x' a  b   c    d\n",
]


def random_text(rng: random.Random) -> str:
    blocks = [(0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F),
              (0x2000, 0x218F), (0x2460, 0x24FF), (0x3000, 0x30FF), (0x4E00, 0x4FFF),
              (0xAC00, 0xACFF), (0x1D400, 0x1D7FF), (0x1F300, 0x1F64F)]  # fmt: skip
    return "".join(
        chr(rng.randint(*rng.choice(blocks))) if rng.random() < 0.8 else rng.choice(" \n\t'09")
        for _ in range(300)
    )


@pytest.mark.model
def test_matches_llama_cpp(model_path, llama_cpp, tmp_path):
    matches_llama_cpp(model_path, llama_cpp, tmp_path)


# the vocab-only GGUFs llama.cpp tests its tokenizers with, of each pre-tokenizer leat supports
VOCABS = ["gpt-2", "mpt", "starcoder", "refact", "command-r", "qwen2", "qwen35", "llama-bpe",
          "llama-spm", "phi-3", "gemma-4"]  # fmt: skip


@pytest.mark.parametrize("name", VOCABS)
def test_vocab_matches_llama_cpp(llama_cpp, tmp_path, name):
    path = llama_cpp.parents[1] / "models" / f"ggml-vocab-{name}.gguf"
    if not path.exists():
        pytest.skip(f"{path} is not in llama.cpp's checkout")
    matches_llama_cpp(path, llama_cpp, tmp_path)


def matches_llama_cpp(model_path, llama_cpp, tmp_path):
    tok = Tokenizer(GGUF.open(model_path).metadata)
    rng = random.Random(0)
    for i, text in enumerate(CORPUS + [random_text(rng) for _ in range(10)]):
        (path := tmp_path / f"{i}.txt").write_bytes(text.encode())
        for special in (False, True):
            args = [llama_cpp / "llama-tokenize", "-m", model_path, "-f", path, "--ids"]
            args += ["--no-bos", "--no-escape", "--log-disable"]
            args += [] if special else ["--no-parse-special"]
            # on the CPU: ggml's CUDA backend, which tokenizing does not need, now and then aborts
            # starting up beside tinygrad's hold of the GPU
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            out = subprocess.run(args, capture_output=True, text=True, check=True, env=env).stdout
            got = tok.encode(text, bos=False, special=special)
            assert got == json.loads(out.splitlines()[-1]), (text, special)
        if all(c in tok._vocab for c in _BYTE_CHAR.values()) or not tok._byte_level:
            assert tok.decode(tok.encode(text, bos=False)) == text
