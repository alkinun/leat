import json
import os
import struct

import pytest

from leat.defaults import OPENAI, Overrides, checked, recommended


def test_recommended():
    # each family's, as its makers recommend it, every option said: Qwen's apart as it thinks or
    # not, and what none says OpenAI's
    gpt_oss = recommended({"general.architecture": "gpt-oss"})
    assert gpt_oss[0] == gpt_oss[1] == OPENAI | {"temperature": 1.0, "top_p": 1.0}
    thinks, plain = recommended({"general.architecture": "qwen3", "general.name": "Qwen3 8B"})
    assert (thinks["temperature"], thinks["top_p"], thinks["top_k"]) == (0.6, 0.95, 20)
    assert (plain["temperature"], plain["top_p"], plain["top_k"]) == (0.7, 0.8, 20)
    for arch in ("qwen35", "qwen35moe"):  # Qwen3.6 27B's, and 35B A3B's
        assert recommended({"general.architecture": arch})[0]["presence_penalty"] == 1.5
    # of one architecture, by the maker its name says
    llama = {"general.architecture": "llama"}
    assert recommended(llama | {"general.name": "Meta Llama 3.1 8B Instruct"})[0]["top_p"] == 0.9
    small = llama | {"general.name": "Mistral-Small-3.2-24B-Instruct-2506"}
    assert recommended(small)[0]["temperature"] == 0.15
    assert recommended(llama | {"general.name": "Mistral 7B Instruct v0.3"}) == (OPENAI, OPENAI)
    assert recommended({}) == (OPENAI, OPENAI)


def test_recommended_of_the_file():
    # a GGUF's general.sampling keys, of its generation_config.json, over its family's, of the set
    # it reasons with alone where the family has another without
    gemma = {"general.architecture": "gemma4", "general.sampling.top_k": 40}
    assert recommended(gemma)[0] == recommended(gemma)[1] == OPENAI | {
        "temperature": 1.0, "top_p": 0.95, "top_k": 40}  # fmt: skip
    qwen = {"general.architecture": "qwen3", "general.sampling.temp": 0.5}
    thinks, plain = recommended(qwen)
    assert (thinks["temperature"], plain["temperature"]) == (0.5, 0.7)
    # its float32s as their shortest decimals, as written, the same float32s
    top_p = struct.unpack("<f", struct.pack("<f", 0.95))[0]  # 0.949999988079071, as read
    assert recommended(gemma | {"general.sampling.top_p": top_p})[0]["top_p"] == 0.95


def test_checked():
    assert checked({"temperature": 1, "top_k": 40.0}, "x") == {"temperature": 1.0, "top_k": 40}
    for bad, error in [([], "x must be an object"), ({"seed": 1}, "x has 'seed', none of"),
                       ({"top_k": 1.5}, "top_k must be an integer"),
                       ({"temperature": 3}, "temperature must be a number from 0 to 2"),
                       ({"top_p": True}, "top_p must be a number")]:  # fmt: skip
        with pytest.raises(ValueError, match=error):
            checked(bad, "x")


def test_overrides(tmp_path):
    # an operator's, of every model and of one, the one's over every model's; read again as the
    # file changes
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps({"*": {"temperature": 0.5, "top_k": 10}, "gpt": {"top_k": 0}}))
    overrides = Overrides(path)
    assert overrides.of("gpt") == {"temperature": 0.5, "top_k": 0}
    assert overrides.of("other") == {"temperature": 0.5, "top_k": 10}
    path.write_text(json.dumps({"gpt": {"min_p": 0.05}}))
    os.utime(path, ns=(1, 1))  # as a change a second after would have it, the same size or not
    assert overrides.of("gpt") == {"min_p": 0.05} and overrides.of("other") == {}
    path.write_text(json.dumps({"gpt": {"min_p": 2}}))
    with pytest.raises(ValueError, match="'s gpt's min_p must be a number from 0 to 1"):
        overrides.of("gpt")
