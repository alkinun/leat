import json
import struct

import numpy as np
import pytest

from leat.cli import main
from leat.gguf import GGUF
from leat.tokenizer import Tokenizer
from tests.helpers import reference_logits

CTX, FIRST = 16, 8  # llama-perplexity scores positions FIRST .. CTX-2 of each chunk
TEXT = ("the quick brown fox jumps over the lazy dog " * 2)[:64]  # 4 chunks of byte tokens


def chunks(path) -> list[list[int]]:
    tokens = Tokenizer(GGUF.open(path).metadata).encode(TEXT)  # the tiny model has no BOS
    return [tokens[i : i + CTX] for i in range(0, len(tokens), CTX)]


def reference_logprobs(weights, chunk: list[int]) -> np.ndarray:
    logits = reference_logits(weights, chunk)[FIRST : CTX - 1]
    logits = logits - logits.max(-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(-1, keepdims=True))


def write_kl_base(path, weights, chunks: list[list[int]], vocab: int) -> None:
    # llama-perplexity --kl-divergence-base: log-probs stored as uint16 over [max - 16, max]
    row = 2 * ((vocab + 1) // 2) + 4
    with open(path, "wb") as f:
        f.write(b"_logits_" + struct.pack("<3i", CTX, vocab, len(chunks)))
        f.write(np.array(chunks, dtype=np.int32).tobytes())
        for chunk in chunks:
            for lp in reference_logprobs(weights, chunk).astype(np.float32):
                low = max(lp.min(), lp.max() - 16)
                scale = (lp.max() - low) / 65535
                out = np.zeros(row, dtype=np.uint16)
                out[:4] = np.array([scale, low], dtype=np.float32).view(np.uint16)
                out[4 : 4 + vocab] = np.where(lp > low, np.rint((lp - low) / scale), 0)
                f.write(out.tobytes())


def run_json(capsys, *args) -> dict:
    main([*map(str, args), "--json"])
    return json.loads(capsys.readouterr().out)


def test_bench(tiny_model, capsys):
    result = run_json(capsys, "bench", tiny_model[0], "-p", 8, "-n", 4, "-r", 1)
    assert result["prefill"] > 0 and result["decode"] > 0 and result["weight_gbs"] > 0


def test_perplexity(tiny_model, tmp_path, capsys):
    path, weights = tiny_model
    (text := tmp_path / "text.txt").write_text(TEXT)
    result = run_json(capsys, "perplexity", path, "--text", text, "--ctx", CTX)
    scored = np.concatenate(
        [
            reference_logprobs(weights, c)[np.arange(CTX - 1 - FIRST), c[FIRST + 1 :]]
            for c in chunks(path)
        ]
    )
    assert result["perplexity"] == pytest.approx(np.exp(-scored.mean()), rel=1e-3)


def test_kl_divergence(tiny_model, tmp_path, capsys):
    path, weights = tiny_model
    vocab = weights["output.weight"].shape[0]
    write_kl_base(base := tmp_path / "base.kld", weights, chunks(path), vocab)
    result = run_json(capsys, "perplexity", path, "--kl-base", base, "--ctx", CTX)
    assert result["kl_mean"] < 1e-4 and result["top1"] == 1.0


def test_run(tiny_model, monkeypatch, capsys):
    replies = iter(["hello", "again"])

    def fake_input(prompt):
        if (reply := next(replies, None)) is None:
            raise EOFError
        return reply

    monkeypatch.setattr("builtins.input", fake_input)
    main(["run", str(tiny_model[0]), "--max-context", "64"])
    assert capsys.readouterr().out.count("tok/s]") == 2
