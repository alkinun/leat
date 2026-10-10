# Sliding-window layers' rings: caches of their window and a run more, rather than the context.

import contextlib

import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes

import tests.helpers
from leat import ops, vision
from leat.engine import Engine
from leat.gguf import GGUF
from leat.model import Config, Transformer
from leat.sampler import Sampling
from tests.helpers import Oracle, reference_drafts, write_tiny_mmproj, write_tiny_model
from tests.test_draft import wrong
from tests.test_model import prefill_starts

LONG = 512  # the context of the tiny models here, past their rings of 64
CHUNK = 16  # a chunk of prompt: rings of 64 for windows of 4
SAMPLING = Sampling(temperature=0.9, top_k=50)
ARCHS = ["gemma4", "gemma3", "gpt-oss"]  # each of a sliding window
PROMPT = np.random.default_rng(0).integers(0, 256, 150).tolist()


@pytest.fixture(scope="session")
def long(tmp_path_factory):
    # long(arch): the tiny model of an architecture, of a context of LONG
    made: dict[str, tuple] = {}

    def model(arch: str) -> tuple:
        if arch not in made:
            path = tmp_path_factory.mktemp(f"long-{arch}") / "tiny.gguf"
            context, tests.helpers.CONTEXT = tests.helpers.CONTEXT, LONG
            try:
                made[arch] = path, write_tiny_model(path, arch)
            finally:
                tests.helpers.CONTEXT = context
        return made[arch]

    return model


@pytest.fixture(scope="session")
def tiny_mmproj(tmp_path_factory) -> tuple:
    # Gemma 4's tiny vision encoder, for long("gemma4")
    path = tmp_path_factory.mktemp("gemma4v") / "mmproj.gguf"
    return path, write_tiny_mmproj(path)


def engine(path, whole: bool = False, monkeypatch=None, **options) -> Engine:
    # an engine of rings, or with `whole` of caches of the whole context, as before rings
    with monkeypatch.context() if whole else contextlib.nullcontext() as patch:
        if patch is not None:
            patch.setattr("leat.engine.Transformer", lambda *args: Transformer(*args[:5]))
        e = Engine(path, max_context=LONG, prefill_chunk=CHUNK, **options)
    assert any(e.model.rings) != whole
    return e


def test_ring_sizes(long):
    # a window and a run more, as a power of 2 of 64 positions or more, but not past the whole
    # cache; layers that see all positions hold them all
    f = GGUF.open(long("gemma4")[0])
    config, weights = Config.from_gguf(f.metadata), f.load()
    for run, size in [(CHUNK, 64), (61, 64), (62, 128), (509, 512), (600, 512)]:
        model = Transformer(config, weights, LONG, run=run)
        assert model.sizes == (size, LONG) and model.rings == (size < LONG, False)
        assert [c.shape[3] for c in model.cache if c is not None] == [size, LONG]
    assert Transformer(config, weights, LONG).sizes == (LONG, LONG)
    # Gemma 4 26B A4B's window of 1024, of chunks of 512, at 16384 positions
    gemma = Config.from_gguf(f.metadata | {"gemma4.attention.sliding_window": 1024,
                                           "gemma4.context_length": 16384})  # fmt: skip
    assert Transformer(gemma, weights, 16384, run=512).sizes == (2048, 16384)


def test_runs_fit_their_rings(long):
    # a ring of 64 for a window of 4 takes runs of 61 tokens at most: a longer one would overwrite
    # the keys its first tokens read, and is refused
    f = GGUF.open(long("gpt-oss")[0])
    model = Transformer(Config.from_gguf(f.metadata), f.load(), LONG, run=CHUNK)
    model(Tensor([[1] * 61], dtype=dtypes.int32), 0).realize()
    with pytest.raises(ValueError, match="longer than the 61 the cache's sliding windows hold"):
        model(Tensor([[1] * 62], dtype=dtypes.int32), 0)


def test_holds(long):
    # a ring of 64 for a window of 4 holds the last 64 positions written: a token sees the 3
    # positions before it
    f = GGUF.open(long("gemma4")[0])
    model = Transformer(Config.from_gguf(f.metadata), f.load(), LONG, run=CHUNK)
    assert model.holds(100, 0) and model.holds(100, 100) and model.holds(100, 39)
    assert not model.holds(100, 38) and model.holds(50, 1) and not model.holds(65, 1)
    assert Transformer(model.config, f.load(), LONG).holds(500, 1)  # without rings


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_ring_generates_as_whole(long, monkeypatch, arch):
    # a prompt that wraps its rings, in chunks, then tokens that wrap them again
    path = long(arch)[0]
    ring = engine(path)
    got = list(ring.generate(PROMPT, 80, SAMPLING, seed=1, ignore_eog=True))
    whole = engine(path, True, monkeypatch)
    assert got == list(whole.generate(PROMPT, 80, SAMPLING, seed=1, ignore_eog=True))


@pytest.mark.usefixtures("reference_ops")
def test_ring_shares_held_prefixes(long, monkeypatch):
    # a prompt shares a slot's tokens where its rings still hold the window before: one that
    # leaves them near their end goes on from there, copied to another slot, and one that leaves
    # them long before their end, past what the rings hold, starts anew
    path = long("gemma3")[0]
    e = engine(path, slots=2)
    out = list(e.generate(PROMPT, 10, SAMPLING, seed=1, ignore_eog=True))
    held = len(PROMPT) + 9  # the rings hold positions from held - 64 on
    near, far = PROMPT[:140] + [1, 2, 3], PROMPT[:80] + [1, 2, 3]
    assert e.cached_prefix(near) == 140 and e.cached_prefix(far) == 0
    assert e.cached_prefix(PROMPT + out) == held
    starts = prefill_starts(e, monkeypatch)
    fresh = engine(path, True, monkeypatch)

    def generated(prompt: list[int]) -> list[int]:
        return list(fresh.generate(prompt, 20, SAMPLING, seed=2, ignore_eog=True))

    branched = []
    for prompt in (near, far):
        branched.append(list(e.generate(prompt, 20, SAMPLING, seed=2, ignore_eog=True)))
        assert branched[-1] == generated(prompt)
    assert starts == [140, 0, 16, 32, 48, 64, 80]  # the far one in the least recently used slot
    # the near one goes on in the slot it was copied to, which holds what it ran since
    longer = near + branched[0] + [7]
    starts.clear()
    assert list(e.generate(longer, 20, SAMPLING, seed=2, ignore_eog=True)) == generated(longer)
    assert starts == [len(near) + 19]


@pytest.mark.usefixtures("reference_ops")
def test_ring_shares_its_own_slot(long, monkeypatch):
    # a prompt that leaves a slot's tokens goes on in that slot where its rings hold the window
    # before, overwriting what the slot held past there
    path = long("gpt-oss")[0]
    e = engine(path)
    list(e.generate(PROMPT, 3))
    starts, branch = prefill_starts(e, monkeypatch), PROMPT[:120] + [1, 2]
    got = list(e.generate(branch, 30, SAMPLING, seed=3, ignore_eog=True))
    assert starts == [120]
    whole = engine(path, True, monkeypatch)
    assert got == list(whole.generate(branch, 30, SAMPLING, seed=3, ignore_eog=True))


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("guessed", ["drafter", "some"])
def test_ring_speculative(long, tiny_assistant, monkeypatch, guessed):
    # speculative steps, whose drafts the rings hold past the window before and which the next
    # step overwrites where it keeps fewer, generate as plain steps over whole caches; the
    # drafter reads the target's rings
    path, weights = long("gemma4")
    e = engine(path, draft=tiny_assistant[0])
    whole = engine(path, True, monkeypatch)
    plain = list(whole.generate(PROMPT, 80, SAMPLING, seed=4, ignore_eog=True))
    if guessed == "some":
        e.drafter = Oracle(wrong(PROMPT + plain))  # type: ignore[assignment]
    steps, speculative = 0, e._speculative

    def counted(sequences):
        nonlocal steps
        steps += 1
        return speculative(sequences)

    monkeypatch.setattr(e, "_speculative", counted)
    assert list(e.generate(PROMPT, 80, SAMPLING, seed=4, ignore_eog=True)) == plain
    assert steps > 0
    if guessed == "drafter":  # its drafts after a prompt past its rings, as the reference's
        e.reset()
        sequence = e.start(PROMPT, 4)
        ((_, token),) = [x for _ in range(-(-len(PROMPT) // CHUNK)) for x in e.step()]
        assert e.drafter is not None
        drafts = e.drafter.draft(Tensor([[token]]), e._hidden[:1], [0], [len(PROMPT)], 3)
        expected = reference_drafts(weights, tiny_assistant[1], PROMPT + [token], 3)
        assert drafts.tolist() == [expected]
        e.cancel(sequence)


@pytest.mark.usefixtures("reference_ops")
def test_ring_image(long, tiny_mmproj, monkeypatch):
    # an image's chunk, which runs whole and whose tokens see each other, wrapping the ring, and
    # past it; its ring holds the window before the image and the image more
    from tests.test_vision import png

    monkeypatch.setattr(vision, "IMAGE_TOKENS", 30)
    path = long("gemma4")[0]
    e = engine(path, vision=tiny_mmproj[0])
    assert e.model.sizes[0] == 64  # 4 + 30 - 1 within a tile
    image = e.image(png(60, 40))
    assert image.size > 4
    prompt = PROMPT[:60] + image.tokens + PROMPT[60:100] + image.tokens + PROMPT[100:]
    got = list(e.generate(prompt, 30, SAMPLING, seed=5, ignore_eog=True, images=[image]))
    whole = engine(path, True, monkeypatch, vision=tiny_mmproj[0])
    expected = whole.generate(prompt, 30, SAMPLING, seed=5, ignore_eog=True, images=[image])
    assert got == list(expected)


@pytest.mark.gpu
@pytest.mark.parametrize("arch", ["gemma3", "gpt-oss"])
def test_ring_kernels(long, arch, monkeypatch):
    # the kernels, where they take the tiny models' heads, store in and read rings as whole
    # caches: chunks of a prompt that wraps the rings, then tokens one at a time, at positions
    # bound as the engine's graphs bind them. Of attention alone: the tiny gpt-oss's 4 experts,
    # too few for the experts' kernels, take the reference ops' every expert for a chunk's pairs,
    # whose kernels took 9 minutes a chunk on tinygrad's emulated GPU, past CI's time for all
    monkeypatch.setattr(ops, "mixture", lambda x, *args, residual=True, **kwargs: (
        x if residual else x.zeros_like()))  # fmt: skip
    f = GGUF.open(long(arch)[0])
    config, weights = Config.from_gguf(f.metadata), f.load()
    ring, whole = Transformer(config, weights, LONG, run=CHUNK), Transformer(config, weights, LONG)
    assert any(ring.rings) and ring.sizes != whole.sizes
    start, count = UOp.variable("start", 0, LONG - 1), UOp.variable("count", 1, CHUNK)
    runs = [(at, PROMPT[at : at + CHUNK]) for at in range(0, len(PROMPT), CHUNK)]
    runs += [(len(PROMPT) + i, [t]) for i, t in enumerate(PROMPT[:40])]
    for at, chunk in runs:
        x = Tensor([chunk + [0] * (CHUNK - len(chunk))], dtype=dtypes.int32)
        x = x.shrink(((0, 1), (0, count.bind(len(chunk)))))
        got, expected = (m.logits(m(x, start.bind(at))[:, -1]).numpy() for m in (ring, whole))
        np.testing.assert_allclose(got, expected, atol=1e-3 * np.abs(expected).max())
