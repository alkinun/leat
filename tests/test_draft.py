import pytest
from tinygrad import Tensor

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
    drafts = engine.drafter.draft(Tensor([[token]]), engine._hidden[:1], 0, len(PROMPT), 3)
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
    drafts = engine.drafter.draft(Tensor([[token]]), engine._hidden[:1], 0, len(PROMPT), 3)
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
    if guessed == "some":  # every third position wrong
        tokens = [t if i % 3 else (t + 1) % 300 for i, t in enumerate(tokens)]
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
        tokens = [t if i % 3 else (t + 1) % 300 for i, t in enumerate(PROMPT + plain)]
        engine.drafter = Oracle(tokens)  # type: ignore[assignment]
    assert list(engine.generate(PROMPT, 16)) == plain
