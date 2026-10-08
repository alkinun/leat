import io

import numpy as np
import pytest
from PIL import Image as Picture

from leat import vision
from leat.engine import Engine
from leat.gguf import GGUF
from leat.tokenizer import Tokenizer
from leat.vision import Vision, _fit
from tests.helpers import (
    CONTEXT,
    G_IMAGE,
    V_PATCH,
    D,
    ids,
    reference_image,
    reference_logits,
    write_tiny_mmproj,
)
from tests.test_model import prefill_starts

TOKENS = 4  # the embeddings an image takes at most, a budget the tiny model's context holds


@pytest.fixture(scope="session")
def tiny_mmproj(tmp_path_factory):
    path = tmp_path_factory.mktemp("mmproj") / "mmproj.gguf"
    return path, write_tiny_mmproj(path)


@pytest.fixture(autouse=True)
def _budget(monkeypatch):
    monkeypatch.setattr(vision, "IMAGE_TOKENS", TOKENS)


def png(width: int, height: int, seed: int = 0, mode: str = "RGB") -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, len(mode)), dtype=np.uint8)
    out = io.BytesIO()
    Picture.fromarray(pixels, mode).save(out, "PNG")
    return out.getvalue()


def embeddings(weights: dict, data: bytes) -> np.ndarray:
    # the f64 reference's embeddings of an image, scaled as Vision scales it
    picture = Picture.open(io.BytesIO(data)).convert("RGB")
    size = _fit(*picture.size, V_PATCH, 3, TOKENS)
    return reference_image(weights, np.asarray(picture.resize(size, Picture.Resampling.BICUBIC)))


def engine(tiny, mmproj, **options) -> Engine:
    return Engine(tiny("gemma4")[0], max_context=CONTEXT, vision=mmproj[0], **options)


# transformers' sizes for Gemma 4's patches and pools, at budgets of 280 and 70 embeddings: images
# scaled up and down to fill them, and sides too thin for a cell
@pytest.mark.parametrize(
    "size, tokens, expected",
    [
        ((1280, 960), 280, (912, 672)), ((640, 480), 70, (432, 336)), ((1, 1), 280, (768, 768)),
        ((3000, 2000), 280, (960, 624)), ((4000, 30), 280, (9264, 48)),
        ((30, 4000), 70, (48, 3360)), ((800, 10), 70, (3360, 48)),
    ],
)  # fmt: skip
def test_fit(size, tokens, expected):
    assert _fit(*size, 16, 3, tokens) == expected


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("size", [(30, 30), (50, 20), (9, 70)])
def test_encode_matches_reference(tiny, tiny_mmproj, size):
    # an image scaled to whole cells of 3 by 3 patches: its embeddings those the f64 reference
    # gives, padding after them, and its tokens those that open and close an image around its key
    tokenizer = Tokenizer(GGUF.open(tiny("gemma4")[0]).metadata)
    v = Vision(GGUF.open(tiny_mmproj[0]), tokenizer, D)
    data = png(*size)
    image, expected = v.image(data), embeddings(tiny_mmproj[1], data)
    opened, closed = ids(tokenizer, G_IMAGE[0], G_IMAGE[2])
    assert image.key < 0 and image.tokens == [opened, *[image.key] * len(expected), closed]
    got = v.encode(image.pixels, image.positions).numpy()
    assert got.shape == (TOKENS, D)
    np.testing.assert_allclose(got[: image.size], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_encode_on_matrix_cores(tiny, tiny_mmproj):
    # in f16 on the matrix cores, the embeddings point where the f64 reference's do
    tokenizer = Tokenizer(GGUF.open(tiny("gemma4")[0]).metadata)
    v = Vision(GGUF.open(tiny_mmproj[0]), tokenizer, D)
    data = png(50, 20)
    image, expected = v.image(data), embeddings(tiny_mmproj[1], data)
    got = v.encode(image.pixels, image.positions).numpy()[: image.size]
    cosine = (got * expected).sum(-1) / np.linalg.norm(got, axis=-1)
    assert (cosine / np.linalg.norm(expected, axis=-1) > 0.999).all()


def test_image_bytes(tiny, tiny_mmproj):
    # an image's key is its bytes'; transparency lies over white, and EXIF turns it upright
    v = engine(tiny, tiny_mmproj)
    assert v.image(png(30, 30)).key == v.image(png(30, 30)).key != v.image(png(30, 30, 1)).key
    clear = io.BytesIO()
    Picture.new("RGBA", (24, 24), (0, 0, 0, 0)).save(clear, "PNG")
    image = v.image(clear.getvalue())
    pixels = image.pixels.numpy()[(image.positions.numpy()[:, 0] >= 0)]
    assert (pixels == 255).all()
    turned, exif = io.BytesIO(), Picture.Exif()
    exif[0x0112] = 6  # rotated a quarter turn
    Picture.new("RGB", (48, 12)).save(turned, "JPEG", exif=exif)
    positions = v.image(turned.getvalue()).positions.numpy()
    assert positions[:, 0].max() < positions[:, 1].max()  # taller than wide


@pytest.mark.usefixtures("reference_ops")
def test_generate_with_images(tiny, tiny_mmproj, monkeypatch):
    # an image between tokens, its embeddings in their place, which see each other, generates as
    # the f64 reference does; a text chunk stops short of it, and it runs whole
    path, weights = tiny("gemma4")
    e = engine(tiny, tiny_mmproj, prefill_chunk=3, slots=2)
    data = png(30, 30)
    image, shown = e.image(data), {}
    shown[image.key] = embeddings(tiny_mmproj[1], data)
    starts = prefill_starts(e, monkeypatch)
    prompt = [5, 77, *image.tokens, 120, 3, *image.tokens, 9]  # the same image twice
    out = list(e.generate(prompt, 6, images=[image]))
    expected = reference_logits(weights, prompt + out, "gemma4", shown)
    assert out == expected[len(prompt) - 1 :].argmax(-1)[: len(out)].tolist()
    assert starts == [0, 3, 7, 10, 11, 15]  # [5, 77, open], image, [close, 120, 3], [open], ...

    # a prompt that goes on from it runs only its new tokens; one of another image after the
    # same tokens runs from that image on, the first image's keys and values copied
    longer = prompt + out[:2] + [7]
    starts.clear()
    assert list(e.generate(longer, 4, images=[image])) == fresh(tiny, tiny_mmproj, longer, image)
    assert starts == [len(prompt) + 2]
    other = e.image(png(30, 30, 1))
    branch = prompt[:10] + other.tokens + [9]
    starts.clear()
    got = list(e.generate(branch, 4, images=[image, other]))
    assert got == fresh(tiny, tiny_mmproj, branch, image, other)
    assert starts[0] == 11


def fresh(tiny, mmproj, prompt: list[int], *images) -> list[int]:
    # what an engine with nothing cached generates
    return list(engine(tiny, mmproj, prefill_chunk=8).generate(prompt, 4, images=images))


@pytest.mark.usefixtures("reference_ops")
def test_drafter_takes_images(tiny, tiny_assistant, tiny_mmproj):
    # speculative decoding past an image generates what plain decoding does
    plain, drafting = engine(tiny, tiny_mmproj), engine(tiny, tiny_mmproj, draft=tiny_assistant[0])
    data = png(50, 20)
    prompt = [5, 77, *plain.image(data).tokens, 120]
    expected = list(plain.generate(prompt, 8, images=[plain.image(data)]))
    assert list(drafting.generate(prompt, 8, images=[drafting.image(data)])) == expected


def test_start_checks_images(tiny, tiny_mmproj):
    e = engine(tiny, tiny_mmproj)
    image = e.image(png(30, 30))
    with pytest.raises(ValueError, match="none of those given"):
        e.start([5, *image.tokens], 4)
    with pytest.raises(ValueError, match="none of those given"):
        e.start([5, *image.tokens[:-2], image.tokens[-1]], 4, images=[image])
    with pytest.raises(ValueError, match="no vision encoder"):
        Engine(tiny("gemma4")[0], max_context=CONTEXT).image(png(30, 30))


@pytest.mark.usefixtures("reference_ops")
def test_warm_up_compiles_images(tiny, tiny_mmproj):
    e = engine(tiny, tiny_mmproj)
    e.warm_up()
    assert e._encode.captured is not None and e._image_chunk.captured is not None
    image = e.image(png(30, 30))
    prompt = [5, *image.tokens, 7]
    assert list(e.generate(prompt, 4, images=[image])) == fresh(tiny, tiny_mmproj, prompt, image)
