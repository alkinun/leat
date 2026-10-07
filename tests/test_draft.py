import pytest
from tinygrad import Tensor

from leat.draft import Gemma4Assistant
from leat.engine import DRAFT_TOKENS, Engine
from leat.sampler import GREEDY, Sampling
from tests.helpers import CONTEXT, Oracle, reference_drafts, reference_mtp_drafts

PROMPT = [5, 77, 120, 3, 299, 42, 8, 150, 61, 200, 9, 33]
# Gemma 4 with its assistant, and Qwen3.5 with its MTP layer, of recurrent state that a step
# goes back on where it keeps fewer tokens than it ran
ARCHS = ["gemma4", "qwen35moe"]


def models(tiny, tiny_assistant, arch: str):
    # a tiny model's path and its drafter's
    path = tiny(arch)[0]
    return path, tiny_assistant[0] if arch == "gemma4" else path


@pytest.mark.usefixtures("reference_ops")
def test_drafts_match_reference(tiny, tiny_assistant):
    path, target = tiny("gemma4")
    engine = Engine(path, max_context=CONTEXT, draft=tiny_assistant[0])
    assert engine.drafter is not None
    sequence = engine.start(PROMPT, 4)
    ((_, token),) = engine.step()  # the prompt's run, which keeps its last hidden state
    drafts = engine.drafter.draft(Tensor([[token]]), engine._hidden[:1], [0], [len(PROMPT)], 3)
    expected = reference_drafts(target, tiny_assistant[1], PROMPT + [token], 3)
    assert drafts.tolist() == [expected]
    engine.cancel(sequence)


@pytest.mark.usefixtures("reference_ops")
def test_mtp_drafts_match_reference(tiny):
    # after a prompt of two chunks, whose second takes the first's last hidden state
    path, weights = tiny("qwen35moe")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, draft=path)
    assert engine.drafter is not None
    sequence = engine.start(PROMPT, 4)
    engine.step()
    ((_, token),) = engine.step()
    drafts = engine.drafter.draft(Tensor([[token]]), engine._hidden[:1], [0], [len(PROMPT)], 3)
    assert drafts.tolist() == [reference_mtp_drafts(weights, PROMPT + [token], 3)]
    engine.cancel(sequence)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("sampling", [GREEDY, Sampling(temperature=0.9, top_k=50)])
@pytest.mark.parametrize("guessed", ["none", "all", "some"])
def test_speculative_generates_as_plain(tiny, tiny_assistant, arch, sampling, guessed):
    # whatever the drafter guesses, the tokens are those plain decoding generates; a drafter that
    # guesses them all has each step keep all its drafts
    path, draft = models(tiny, tiny_assistant, arch)
    plain = list(Engine(path, max_context=CONTEXT).generate(PROMPT, 20, sampling, seed=3))
    engine = Engine(path, max_context=CONTEXT, draft=draft)
    tokens = PROMPT + plain
    if guessed == "some":
        tokens = wrong(tokens)
    if guessed != "none":
        engine.drafter = Oracle(tokens)  # type: ignore[assignment]
    steps = 0
    speculative = engine._speculative

    def counted(sequence):
        nonlocal steps
        steps += 1
        return speculative(sequence)

    engine._speculative = counted  # type: ignore[method-assign]
    assert list(engine.generate(PROMPT, 20, sampling, seed=3)) == plain
    if guessed == "all":  # the first token from the prompt's run, then all drafts and one more
        assert steps == -(-19 // (DRAFT_TOKENS + 1))
    assert steps > 0


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_speculative_stops_where_plain_does(tiny, tiny_assistant, arch):
    # at max_tokens, though the step generated more, and at the end of the context
    path, draft = models(tiny, tiny_assistant, arch)
    plain = list(Engine(path, max_context=CONTEXT).generate(PROMPT, 7))
    engine = Engine(path, max_context=CONTEXT, draft=draft)
    engine.drafter = Oracle(PROMPT + plain)  # type: ignore[assignment]
    assert list(engine.generate(PROMPT, 7)) == plain
    full = list(Engine(path, max_context=CONTEXT).generate(PROMPT, CONTEXT))
    engine = Engine(path, max_context=CONTEXT, draft=draft)
    engine.drafter = Oracle(PROMPT + full)  # type: ignore[assignment]
    assert list(engine.generate(PROMPT, CONTEXT)) == full


def wrong(tokens: list[int]) -> list[int]:
    # every third token wrong, as a drafter's guesses of them might be
    return [t if i % 3 else (t + 1) % 300 for i, t in enumerate(tokens)]


def several(engine: Engine, prompts: list[list[int]], sampling: Sampling) -> list[list[int]]:
    # the prompts' tokens, of the first two decoding together, and the third starting while they
    # do, prefilled in steps that also decode them
    sequences = [engine.start(p, 20, sampling, seed=i) for i, p in enumerate(prompts[:2])]
    for _ in range(3):
        engine.step()
    sequences += [engine.start(p, 20, sampling, seed=i + 2) for i, p in enumerate(prompts[2:])]
    while engine.active:
        engine.step()
    return [s.tokens for s in sequences]


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("guessed", ["none", "some"])
def test_speculative_several(tiny, tiny_assistant, monkeypatch, arch, guessed):
    # sequences decoding at once speculate together, each generating the tokens it would alone,
    # also in steps that prefill another: 2 drafting 3 tokens each, 3 drafting 1, and more than
    # the drafter takes, 3 here, in plain steps
    monkeypatch.setattr(Gemma4Assistant, "sequences", 3)
    path, draft = models(tiny, tiny_assistant, arch)
    prompts = [PROMPT, PROMPT[::-1], PROMPT[3:], PROMPT[5:] + PROMPT[:2]]
    sampling = Sampling(temperature=0.9, top_k=50)
    alone = Engine(path, max_context=CONTEXT)
    plain = [list(alone.generate(p, 20, sampling, seed=i)) for i, p in enumerate(prompts)]
    engine = Engine(path, max_context=CONTEXT, slots=len(prompts), draft=draft)
    assert engine._speculated == [1, 2, 3]
    if guessed == "some":  # slot i takes prompt i
        guesses = [wrong(p + t) for p, t in zip(prompts, plain, strict=True)]
        engine.drafter = Oracle(*guesses)  # type: ignore[assignment]
    sizes = []
    speculative = engine._speculative

    def counted(sequences):
        sizes.append(len(sequences))
        return speculative(sequences)

    engine._speculative = counted  # type: ignore[method-assign]
    assert several(engine, prompts, sampling) == plain
    assert {2, 3} <= set(sizes) and max(sizes) == 3


def test_drafter_needs_its_target(tiny_model, tiny_assistant):
    with pytest.raises(ValueError, match="drafts for no llama model"):
        Engine(tiny_model[0], max_context=CONTEXT, draft=tiny_assistant[0])


@pytest.mark.gpu
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("guessed", ["none", "some"])
def test_speculative_kernels(tiny, tiny_assistant, arch, guessed):
    # through the kernels, the drafts' step on the matrix-vector kernels for several tokens,
    # FlashAttention and Gated DeltaNet's saving kernels: the tokens of plain decoding, which
    # rounds differently only where these tiny models have no near ties
    path, draft = models(tiny, tiny_assistant, arch)
    plain = list(Engine(path, max_context=CONTEXT).generate(PROMPT, 16))
    engine = Engine(path, max_context=CONTEXT, draft=draft)
    if guessed == "some":
        engine.drafter = Oracle(wrong(PROMPT + plain))  # type: ignore[assignment]
    assert list(engine.generate(PROMPT, 16)) == plain


@pytest.mark.gpu
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("guessed", ["none", "some"])
def test_speculative_several_kernels(tiny, tiny_assistant, monkeypatch, arch, guessed):
    # through the kernels, of several sequences' rows of several tokens each: the attention and
    # rotate kernels' rows of a token, and Gated DeltaNet's rows of several, saving each state
    monkeypatch.setattr(Gemma4Assistant, "sequences", 3)
    path, draft = models(tiny, tiny_assistant, arch)
    prompts = [PROMPT, PROMPT[::-1], PROMPT[3:]]
    alone = Engine(path, max_context=CONTEXT)
    plain = [list(alone.generate(p, 20, GREEDY)) for p in prompts]
    engine = Engine(path, max_context=CONTEXT, slots=3, draft=draft)
    if guessed == "some":
        guesses = [wrong(p + t) for p, t in zip(prompts, plain, strict=True)]
        engine.drafter = Oracle(*guesses)  # type: ignore[assignment]
    assert several(engine, prompts, GREEDY) == plain
