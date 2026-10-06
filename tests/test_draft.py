import pytest
from tinygrad import Tensor, UOp, dtypes

from leat.engine import DRAFT_TOKENS, Engine
from leat.sampler import GREEDY, Sampling
from tests.helpers import CONTEXT, reference_drafts

PROMPT = [5, 77, 120, 3, 299, 42, 8, 150, 61, 200, 9, 33]


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


class _Oracle:
    """Drafts the tokens a list holds at the positions after a draft's, as a drafter that guessed
    them would."""

    def __init__(self, tokens: list[int]):
        self.tokens = Tensor(tokens + [0] * CONTEXT, dtype=dtypes.int32).realize()

    def draft(self, token: Tensor, hidden: Tensor, slot: UOp, pos: UOp, count: int) -> Tensor:
        return self.tokens[pos + 1 : pos + 1 + count].reshape(1, count)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("sampling", [GREEDY, Sampling(temperature=0.9, top_k=50)])
@pytest.mark.parametrize("guessed", ["none", "all", "some"])
def test_speculative_generates_as_plain(tiny, tiny_assistant, sampling, guessed):
    # whatever the drafter guesses, the tokens are those plain decoding generates; a drafter that
    # guesses them all has each step keep all its drafts
    path, _ = tiny("gemma4")
    plain = list(Engine(path, max_context=CONTEXT).generate(PROMPT, 20, sampling, seed=3))
    engine = Engine(path, max_context=CONTEXT, draft=tiny_assistant[0])
    tokens = PROMPT + plain
    if guessed == "some":  # every third position wrong
        tokens = [t if i % 3 else (t + 1) % 300 for i, t in enumerate(tokens)]
    if guessed != "none":
        engine.drafter = _Oracle(tokens)  # type: ignore[assignment]
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
def test_speculative_stops_where_plain_does(tiny, tiny_assistant):
    # at max_tokens, though the step generated more, and at the end of the context
    path, _ = tiny("gemma4")
    plain = list(Engine(path, max_context=CONTEXT).generate(PROMPT, 7))
    engine = Engine(path, max_context=CONTEXT, draft=tiny_assistant[0])
    engine.drafter = _Oracle(PROMPT + plain)  # type: ignore[assignment]
    assert list(engine.generate(PROMPT, 7)) == plain
    full = list(Engine(path, max_context=CONTEXT).generate(PROMPT, CONTEXT))
    engine = Engine(path, max_context=CONTEXT, draft=tiny_assistant[0])
    engine.drafter = _Oracle(PROMPT + full)  # type: ignore[assignment]
    assert list(engine.generate(PROMPT, CONTEXT)) == full


def test_drafter_needs_its_target(tiny_model, tiny_assistant):
    with pytest.raises(ValueError, match="drafts for no llama model"):
        Engine(tiny_model[0], max_context=CONTEXT, draft=tiny_assistant[0])
