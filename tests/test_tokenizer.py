import json
import random
import subprocess

import pytest

from leat.gguf import GGUF
from leat.tokenizer import BYTE, CONTROL, NORMAL, Tokenizer
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
    tok = Tokenizer(GGUF.open(model_path).metadata)
    rng = random.Random(0)
    for i, text in enumerate(CORPUS + [random_text(rng) for _ in range(10)]):
        (path := tmp_path / f"{i}.txt").write_bytes(text.encode())
        for special in (False, True):
            args = [llama_cpp / "llama-tokenize", "-m", model_path, "-f", path, "--ids"]
            args += ["--no-bos", "--no-escape", "--log-disable"]
            args += [] if special else ["--no-parse-special"]
            out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
            assert tok.encode(text, bos=False, special=special) == json.loads(out.splitlines()[-1])
        assert tok.decode(tok.encode(text, bos=False)) == text
