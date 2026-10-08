import base64
import io
from pathlib import Path

import gguf
import numpy as np
import openai
import pytest
from PIL import Image as Picture

from leat import vision
from leat.engine import Engine
from leat.gguf import GGUF
from leat.tokenizer import Tokenizer
from leat.vision import _fit, beside, projector
from tests.helpers import (
    CONTEXT,
    G3_IMAGE,
    G_IMAGE,
    V_PATCH,
    D,
    ids,
    reference_image,
    reference_logits,
    reference_siglip,
    write_tiny_mmproj,
)
from tests.helpers import (
    _finish as finish,
)
from tests.test_model import prefill_starts

TOKENS = 4  # the embeddings an image takes at most, a budget the tiny model's context holds
G3_TOKENS = ("<start_of_image>", "<end_of_image>")


# each projector's tiny model
FAMILIES = {"gemma4v": "gemma4", "gemma3": "gemma3"}


@pytest.fixture(scope="session")
def projectors(tmp_path_factory):
    # projectors(kind): a random projector of a kind and its weights, written once per session
    made: dict[str, tuple[Path, dict]] = {}

    def projector(kind: str) -> tuple[Path, dict]:
        if kind not in made:
            path = tmp_path_factory.mktemp(kind) / "mmproj.gguf"
            made[kind] = path, write_tiny_mmproj(path, kind)
        return made[kind]

    return projector


@pytest.fixture(scope="session")
def tiny_mmproj(projectors):
    return projectors("gemma4v")


@pytest.fixture(autouse=True)
def _budget(monkeypatch):
    monkeypatch.setattr(vision, "IMAGE_TOKENS", TOKENS)


def png(width: int, height: int, seed: int = 0, mode: str = "RGB") -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, len(mode)), dtype=np.uint8)
    out = io.BytesIO()
    Picture.fromarray(pixels, mode).save(out, "PNG")
    return out.getvalue()


def embeddings(weights: dict, data: bytes, kind: str = "gemma4v") -> np.ndarray:
    # the f64 reference's embeddings of an image, scaled as its kind's encoder scales it
    picture = Picture.open(io.BytesIO(data)).convert("RGB")
    if kind == "gemma3":
        square = picture.resize((G3_IMAGE, G3_IMAGE), Picture.Resampling.BILINEAR)
        return reference_siglip(weights, np.asarray(square))
    size = _fit(*picture.size, V_PATCH, 3, TOKENS)
    return reference_image(weights, np.asarray(picture.resize(size, Picture.Resampling.BICUBIC)))


def engine(tiny, mmproj, kind: str = "gemma4v", **options) -> Engine:
    return Engine(tiny(FAMILIES[kind])[0], max_context=CONTEXT, vision=mmproj[0], **options)


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
def test_siglip_matches_reference(tiny, projectors):
    # Gemma 3's: the image squashed square, its embeddings those the f64 reference gives, shown
    # between newlines
    path, weights = projectors("gemma3")
    tokenizer = Tokenizer(GGUF.open(tiny("gemma3")[0]).metadata)
    v = vision.load(path, tokenizer, D)
    data = png(40, 25)
    image, expected = v.image(data), embeddings(weights, data, "gemma3")
    lines, (opened, closed) = tokenizer.encode("\n\n", bos=False), ids(tokenizer, *G3_TOKENS)
    assert image.tokens == [*lines, opened, *[image.key] * len(expected), closed, *lines]
    got = v.encode(*v.inputs(image)).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("size", [(30, 30), (50, 20), (9, 70)])
def test_encode_matches_reference(tiny, tiny_mmproj, size):
    # an image scaled to whole cells of 3 by 3 patches: its embeddings those the f64 reference
    # gives, padding after them, and its tokens those that open and close an image around its key
    tokenizer = Tokenizer(GGUF.open(tiny("gemma4")[0]).metadata)
    v = vision.load(tiny_mmproj[0], tokenizer, D)
    data = png(*size)
    image, expected = v.image(data), embeddings(tiny_mmproj[1], data)
    opened, closed = ids(tokenizer, G_IMAGE[0], G_IMAGE[2])
    assert image.key < 0 and image.tokens == [opened, *[image.key] * len(expected), closed]
    got = v.encode(*v.inputs(image)).numpy()
    assert got.shape == (TOKENS, D)
    np.testing.assert_allclose(got[: image.size], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_encode_on_matrix_cores(tiny, tiny_mmproj):
    # in f16 on the matrix cores, the embeddings point where the f64 reference's do
    tokenizer = Tokenizer(GGUF.open(tiny("gemma4")[0]).metadata)
    v = vision.load(tiny_mmproj[0], tokenizer, D)
    data = png(50, 20)
    image, expected = v.image(data), embeddings(tiny_mmproj[1], data)
    got = v.encode(*v.inputs(image)).numpy()[: image.size]
    cosine = (got * expected).sum(-1) / np.linalg.norm(got, axis=-1)
    assert (cosine / np.linalg.norm(expected, axis=-1) > 0.999).all()


def test_image_bytes(tiny, tiny_mmproj):
    # an image's key is its bytes'; transparency lies over white, and EXIF turns it upright
    v = engine(tiny, tiny_mmproj)
    assert v.image(png(30, 30)).key == v.image(png(30, 30)).key != v.image(png(30, 30, 1)).key
    clear = io.BytesIO()
    Picture.new("RGBA", (24, 24), (0, 0, 0, 0)).save(clear, "PNG")
    pixels, positions = (t.numpy() for t in v.vision.inputs(v.image(clear.getvalue())))
    assert (pixels[positions[:, 0] >= 0] == 255).all()
    turned, exif = io.BytesIO(), Picture.Exif()
    exif[0x0112] = 6  # rotated a quarter turn
    Picture.new("RGB", (48, 12)).save(turned, "JPEG", exif=exif)
    _, positions = (t.numpy() for t in v.vision.inputs(v.image(turned.getvalue())))
    assert positions[:, 0].max() < positions[:, 1].max()  # taller than wide
    with pytest.raises(ValueError, match="not an image"):
        v.image(b"GIF89a, but not")


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("kind", FAMILIES)
def test_generate_with_images(tiny, projectors, monkeypatch, kind):
    # an image between tokens, its embeddings in their place, which see each other, generates as
    # the f64 reference does; a text chunk stops short of it, and it runs whole
    arch, mmproj = FAMILIES[kind], projectors(kind)
    path, weights = tiny(arch)
    e = engine(tiny, mmproj, kind, prefill_chunk=3, slots=2)
    data = png(30, 30)
    image, shown = e.image(data), {}
    shown[image.key] = embeddings(mmproj[1], data, kind)
    starts = prefill_starts(e, monkeypatch)
    prompt = [5, 77, *image.tokens, 120, 3, *image.tokens, 9]  # the same image twice
    out = list(e.generate(prompt, 6, images=[image]))
    expected = reference_logits(weights, prompt + out, arch, shown)
    assert out == expected[len(prompt) - 1 :].argmax(-1)[: len(out)].tolist()
    first = [i for i, t in enumerate(prompt) if t == image.key and prompt[i - 1] != t]
    within = [i for i in starts if prompt[i] == image.key and prompt[i - 1] == image.key]
    assert set(first) <= set(starts) and not within

    # a prompt that goes on from it runs only its new tokens; one of another image after the
    # same tokens runs from that image on, the first image's keys and values copied
    longer = prompt + out[:2] + [7]
    starts.clear()
    got = list(e.generate(longer, 4, images=[image]))
    assert got == fresh(tiny, mmproj, kind, longer, image)
    assert starts == [len(prompt) + 2]
    other = e.image(png(30, 30, 1))
    branch = prompt[: first[1]] + other.tokens[other.tokens.index(other.key) :] + [9]
    starts.clear()
    got = list(e.generate(branch, 4, images=[image, other]))
    assert got == fresh(tiny, mmproj, kind, branch, image, other)
    assert starts[0] == first[1]


def fresh(tiny, mmproj, kind: str, prompt: list[int], *images) -> list[int]:
    # what an engine with nothing cached generates
    return list(engine(tiny, mmproj, kind, prefill_chunk=8).generate(prompt, 4, images=images))


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
    assert list(e.generate(prompt, 4, images=[image])) == fresh(
        tiny, tiny_mmproj, "gemma4v", prompt, image
    )


def test_server_takes_images(tiny, tiny_mmproj):
    # an image_url part of a data: URL generates as the engine does given its image; the model
    # says it takes images, and one without a vision encoder refuses them
    from tests.test_server import connect, serving

    path, data = tiny("gemma4")[0], png(30, 30)
    url = f"data:image/png;base64,{base64.b64encode(data).decode()}"
    parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": url}}]
    e = engine(tiny, tiny_mmproj, prefill_chunk=8)
    image = e.image(data)
    prompt = e.tokenizer.encode("a") + image.tokens  # the tiny model's template is the content
    expected = e.tokenizer.decode(list(e.generate(prompt, 4, images=[image])))
    with serving(path, max_context=CONTEXT, prefill_chunk=8, vision=tiny_mmproj[0]) as server:
        server.load("tiny")
        client = connect(server)
        assert client.models.list().data[0].vision is True
        messages = [{"role": "user", "content": parts}]
        reply = client.chat.completions.create(
            model="tiny", messages=messages, max_tokens=4, temperature=0
        )
        assert reply.choices[0].message.content == expected
    with serving(path, max_context=CONTEXT) as server:
        server.load("tiny")
        with pytest.raises(openai.BadRequestError, match="takes no images"):
            connect(server).chat.completions.create(model="tiny", messages=messages)


def test_projector_beside(tmp_path):
    # the projector for the model's embeddings whose name, or file name without one, holds the
    # model's name but for case and punctuation, is its own
    def write(name: str, general: str | None, dim: int = 8) -> Path:
        kind = "mmproj" if projector(Path(name)) else "model"
        w = gguf.GGUFWriter(tmp_path / name, arch="clip" if kind == "mmproj" else "llama")
        if general:
            w.add_name(general)
        if kind == "mmproj":
            w.add_string("clip.vision.projector_type", "gemma4v")
            w.add_uint32("clip.vision.projection_dim", dim)
        else:
            w.add_uint32("llama.embedding_length", 8)
        finish(w)
        return tmp_path / name

    model = write("Gemma-3-4b-it-Q4_K_M.gguf", "Gemma 3 4b It")
    write("mmproj-a.gguf", "Qwen3.6 35B A3B")
    write("mmproj-b.gguf", "gemma-3-4b-it", dim=16)
    assert beside(model) is None
    own = write("mmproj-google_gemma-3-4b-it-f16.gguf", None)
    assert beside(model) == own and not projector(model)
