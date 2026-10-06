import json
import struct

import jinja2
import numpy as np
import pytest

from leat.chat import ChatTemplate
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


@pytest.mark.parametrize("sequences", [1, 3])
def test_bench(tiny_model, capsys, sequences):
    args = ("-p", 8, "-n", 6, "-r", 1, "-s", sequences)
    result = run_json(capsys, "bench", tiny_model[0], *args)
    assert result["prefill"] > 0 and result["decode"] > 0 and result["weight_gbs"] > 0
    assert result["sequences"] == sequences


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("mode", [[], ["--decode"]])
def test_perplexity(tiny_model, tmp_path, capsys, mode):
    path, weights = tiny_model
    (text := tmp_path / "text.txt").write_text(TEXT)
    result = run_json(capsys, "perplexity", path, "--text", text, "--ctx", CTX, *mode)
    scored = np.concatenate(
        [
            reference_logprobs(weights, c)[np.arange(CTX - 1 - FIRST), c[FIRST + 1 :]]
            for c in chunks(path)
        ]
    )
    assert result["perplexity"] == pytest.approx(np.exp(-scored.mean()), rel=1e-3)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("mode", [[], ["--decode"]])
def test_kl_divergence(tiny_model, tmp_path, capsys, mode):
    path, weights = tiny_model
    vocab = weights["output.weight"].shape[0]
    write_kl_base(base := tmp_path / "base.kld", weights, chunks(path), vocab)
    # the file's chunks set the context: the default --ctx is over the tiny model's
    result = run_json(capsys, "perplexity", path, "--kl-base", base, *mode)
    assert result["kl_mean"] < 1e-4 and result["top1"] == 1.0


def test_run(tiny_model, monkeypatch, capsys):
    # a chat in the terminal: each reply printed, and kept in the chat for the next turn
    replies, chats, render = iter(["hello", "again"]), [], ChatTemplate.render

    def fake_input(prompt):
        if (reply := next(replies, None)) is None:
            raise EOFError
        return reply

    def rendered(self, messages, **kwargs):
        chats.append(list(messages))
        return render(self, messages, **kwargs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("leat.cli.ChatTemplate.render", rendered)
    sampling = ["--temperature", "0.5", "--top-k", "5", "--top-p", "0.9", "--min-p", "0.05",
                "--presence-penalty", "0.5"]  # fmt: skip
    main(["run", str(tiny_model[0]), "--max-context", "64", *sampling])
    out = capsys.readouterr().out
    assert [m["role"] for m in chats[1]] == ["user", "assistant", "user"]
    assert chats[1][1]["content"] in out and out.count("tok/s]") == 2


def test_run_refused_chat(tiny_model, monkeypatch):
    # a chat the model's template refuses, as Mistral 7B v0.3's does a system prompt, ends the run
    # with the template's reason rather than a traceback
    def refuse(self, messages, **kwargs):
        raise jinja2.TemplateError("Only user and assistant roles are supported!")

    monkeypatch.setattr("leat.cli.ChatTemplate.render", refuse)
    monkeypatch.setattr("builtins.input", lambda prompt: "hello")
    with pytest.raises(SystemExit, match="refuses this chat: Only user and assistant"):
        main(["run", str(tiny_model[0]), "--max-context", "64", "--system", "Be brief."])
