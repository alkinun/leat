import json
import struct
import threading

import jinja2
import numpy as np
import pytest

from leat import bench
from leat.chat import ChatTemplate
from leat.cli import main
from leat.engine import Engine
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


@pytest.mark.parametrize("prompt, sequences", [(8, 1), (8, 3), (2, 3)])  # more than its tokens
def test_bench(tiny_model, capsys, prompt, sequences):
    args = ("-p", prompt, "-n", 6, "-r", 1, "-s", sequences)
    result = run_json(capsys, "bench", tiny_model[0], *args)
    assert result["prefill"] > 0 and result["decode"] > 0 and result["weight_gbs"] > 0
    assert result["sequences"] == sequences


def test_bench_draft(tiny, tiny_assistant, capsys):
    result = run_json(capsys, "bench", tiny("gemma4")[0], "-p", 8, "-n", 6, "-r", 1, "--draft",
                      tiny_assistant[0])  # fmt: skip
    assert result["decode"] > 0


@pytest.mark.parametrize("sequences", [1, 2])
def test_bench_chat(tiny, tiny_assistant, capsys, sequences):
    # replies to chat prompts, which a drafter that guesses them all takes 4 tokens a step of
    path = tiny("gemma4")[0]
    args = ("-n", 9, "-r", 1, "-s", sequences, "--draft", tiny_assistant[0], "--chat")
    result = run_json(capsys, "bench", path, *args)
    assert result["decode"] > 0 and 1 <= result["per_step"] <= 4
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


def test_perplexity_arguments(tiny_model, tmp_path, monkeypatch, capsys):
    # counts of one at least, chunks that score a position, and the text's bytes as they are
    path = tiny_model[0]
    (text := tmp_path / "text.txt").write_bytes(TEXT.replace(" ", "\r\n").encode())
    for bad in (["--ctx", "0"], ["--chunks", "0"]):
        with pytest.raises(SystemExit):
            main(["perplexity", str(path), "--text", str(text), *bad])
    assert "must be at least 1, got 0" in capsys.readouterr().err
    with pytest.raises(ValueError, match="chunks of 2 tokens score none"):
        main(["perplexity", str(path), "--text", str(text), "--ctx", "2"])
    with pytest.raises(ValueError, match="chunks of 16 tokens are over max_context 8"):
        bench.perplexity(Engine(path, max_context=8), TEXT, ctx=16)
    read = []
    monkeypatch.setattr(bench, "perplexity", lambda e, t, *_: read.append(t) or bench.Quality(1))
    main(["perplexity", str(path), "--text", str(text), "--ctx", str(CTX)])
    assert read == [TEXT.replace(" ", "\r\n")]


def test_percentile():
    # between the two values about it, as llama-perplexity's
    assert bench._percentile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5
    assert bench._percentile([float(i) for i in range(101)], 0.99) == pytest.approx(99.0)


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
    monkeypatch.setattr("leat.chat.ChatTemplate.render", rendered)
    sampling = ["--temperature", "0.5", "--top-k", "5", "--top-p", "0.9", "--min-p", "0.05",
                "--presence-penalty", "0.5"]  # fmt: skip
    main(["run", str(tiny_model[0]), "--max-context", "64", *sampling])
    out = capsys.readouterr().out
    assert [m["role"] for m in chats[1]] == ["user", "assistant", "user"]
    assert chats[1][1]["content"] in out and out.count("tok/s]") == 2


def test_run_past_the_context(tiny_model, monkeypatch, capsys):
    # past --max-context, the chat starts over from the message just written, and a message too
    # long alone is dropped, each said so
    # a template of every message's text, a token a character: two of 32 and 33 do not fit 64
    replies, chats = iter(["a" * 32, "a" * 33, "b" * 80]), []

    def fake_input(prompt):
        if (reply := next(replies, None)) is None:
            raise EOFError
        return reply

    def rendered(self, messages, **kwargs):
        chats.append(list(messages))
        return "".join(m["content"] for m in messages)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("leat.chat.ChatTemplate.render", rendered)
    main(["run", str(tiny_model[0]), "--max-context", "64"])
    out = capsys.readouterr().out
    assert "starting over" in out and "the message makes" in out
    assert [{"role": "user", "content": "a" * 33}] in chats


def test_serve_directories(tiny_model, tmp_path, monkeypatch, capsys):
    # of directories alone, the first file found loads at start, served meanwhile; a server on
    # every address is browsed at this machine's, and needs keys; a model that fails to load stops
    # the server
    (models := tmp_path / "models").mkdir()
    (models / "tiny.gguf").symlink_to(tiny_model[0])
    loaded, failing, keys, samplings = [], [], [], []

    class Fake:
        server_port = 8080

        def __init__(self, files, host, port, given, sampling, **options):
            self.files, self.stopped = files, threading.Event()
            keys.append(given)
            samplings.append(sampling)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def load(self, name):
            if failing:
                raise RuntimeError(f"loading {name} failed")
            loaded.append(name)
            self.stopped.set()  # as Ctrl-C would, once it is ready

        def serve_forever(self):
            assert self.stopped.wait(5)

        def shutdown(self):
            self.stopped.set()

    monkeypatch.setattr("leat.server.Server", Fake)
    with pytest.raises(SystemExit, match="to serve beyond this machine, at 0.0.0.0, give --keys"):
        main(["serve", str(models), "--host", "0.0.0.0"])
    with pytest.raises(SystemExit, match="there are no keys at"):
        main(["serve", str(models), "--host", "0.0.0.0", "--keys", str(tmp_path / "keys.json")])
    main(["keys", "add", "alkin", "--file", str(tmp_path / "keys.json")])
    main(["serve", str(models), "--host", "0.0.0.0", "--keys", str(tmp_path / "keys.json")])
    out = capsys.readouterr().out
    assert loaded == ["tiny"] and "the API at http://127.0.0.1:8080/v1, compiling..." in out
    assert keys[-1].path == tmp_path / "keys.json"
    main(["serve", str(models), "--host", "::1"])  # this machine's alone: no keys needed
    assert keys[-1] is None and samplings[-1] is None
    # an operator's sampling, read as it starts, which a file of none or a broken one stops
    sampling = tmp_path / "sampling.json"
    with pytest.raises(SystemExit, match="cannot be read"):
        main(["serve", str(models), "--sampling", str(sampling)])
    sampling.write_text('{"tiny": {"temperature": 9}}')
    with pytest.raises(SystemExit, match="temperature must be a number from 0 to 2"):
        main(["serve", str(models), "--sampling", str(sampling)])
    sampling.write_text('{"tiny": {"temperature": 0.5}}')
    main(["serve", str(models), "--sampling", str(sampling)])
    assert samplings[-1].of("tiny") == {"temperature": 0.5}
    failing.append(True)
    with pytest.raises(RuntimeError, match="loading tiny failed"):
        main(["serve", str(models)])
    (empty := tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no GGUF files"):
        main(["serve", str(empty)])
    with pytest.raises(SystemExit, match="--draft is of one model"):
        main(["serve", str(models), str(tiny_model[0]), "--draft", str(tiny_model[0])])


def test_keys(tmp_path, capsys):
    # a key made is shown once, listed by its name, and removed by it
    file = str(tmp_path / "keys.json")
    main(["keys", "add", "alkin", "--file", file])
    key = capsys.readouterr().out.split("\n")[0]
    assert key.startswith("leat-") and key not in (tmp_path / "keys.json").read_text()
    main(["keys", "list", "--file", file])
    assert capsys.readouterr().out.startswith("alkin\t20")
    for args, error in ((["add", "alkin"], "already"), (["remove", "bo"], "no key named bo"),
                        (["add"], "needs the key's name")):  # fmt: skip
        with pytest.raises(SystemExit, match=error):
            main(["keys", *args, "--file", file])
    main(["keys", "remove", "alkin", "--file", file])
    main(["keys", "list", "--file", file])
    assert capsys.readouterr().out.endswith("from the next request on.\n")
    (tmp_path / "keys.json").write_text("[]")  # no file of keys: said so, as of every action
    for args in (["list"], ["add", "bo"], ["remove", "bo"]):
        with pytest.raises(SystemExit, match="is no file of leat's keys"):
            main(["keys", *args, "--file", file])


def test_agent(tmp_path, monkeypatch, capsys):
    # leat agent keeps its state in --data, made if it is not there, and serves the app
    agents = []

    class Fake:
        server_port = 8000

        def __init__(self, agent, host, port):
            agents.append(agent)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def serve_forever(self):
            pass

    monkeypatch.setattr("leat.cli.AgentServer", Fake)
    main(["agent", "--data", str(tmp_path / "leat"), "--engine", "http://127.0.0.1:9999"])
    assert (tmp_path / "leat" / "leat.db").exists()
    assert agents[0].engine.url == "http://127.0.0.1:9999"
    assert "leat agent at http://127.0.0.1:8000" in capsys.readouterr().out


def test_run_refused_chat(tiny_model, monkeypatch):
    # a chat the model's template refuses, as Mistral 7B v0.3's does a system prompt, ends the run
    # with the template's reason rather than a traceback
    def refuse(self, messages, **kwargs):
        raise jinja2.TemplateError("Only user and assistant roles are supported!")

    monkeypatch.setattr("leat.chat.ChatTemplate.render", refuse)
    monkeypatch.setattr("builtins.input", lambda prompt: "hello")
    with pytest.raises(SystemExit, match="refuses this chat: Only user and assistant"):
        main(["run", str(tiny_model[0]), "--max-context", "64", "--system", "Be brief."])
