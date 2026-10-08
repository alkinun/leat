"""Vision encoders: images as embeddings a model reads in place of tokens, of an mmproj GGUF as
llama.cpp's.

Gemma 4's, as transformers' has it: a ViT over patches of 16 by 16 pixels with 2D RoPE, whose
outputs pool 3 by 3 into the embeddings. An image is scaled, its aspect kept, to as many patches as
fit the budget of IMAGE_TOKENS embeddings; the encoder takes every image padded to that many, so
that one compiled graph runs them all.
"""

import array
import hashlib
import io
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as Picture
from PIL import ImageOps
from tinygrad import Tensor, dtypes

from leat import ops
from leat.gguf import GGUF
from leat.quant import GGMLType, QTensor
from leat.tokenizer import Tokenizer

# the embeddings an image takes at most: of Gemma 4's budgets, 70, 140, 280, 560 or 1120, its
# processor's default
IMAGE_TOKENS = 280
KEPT = 16  # images kept, made of their bytes, to be given again
PROJECTORS = {"gemma4v"}  # the vision encoders leat runs, as llama.cpp's projector types name them
_LAYER = ("ln1", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_out",
          "attn_post_norm", "ln2", "ffn_gate", "ffn_up", "ffn_down", "ffn_post_norm")  # fmt: skip


@dataclass(frozen=True, eq=False)
class Image:
    """An image ready for a prompt. `tokens` show it there: those that open and close it, around
    a position for each of its embeddings, which hold `key`, a negative id of a digest of the
    image by which the cache tells images apart. `pixels` and `positions` are the encoder's, as
    bytes the engine uploads when it encodes the image: each patch's pixels, rows of RGB, and its
    column and row as int32, -1 for padding."""

    key: int
    tokens: list[int]
    pixels: bytes
    positions: bytes

    @property
    def size(self) -> int:
        """The image's embeddings, as many as the positions it takes."""
        return self.tokens.count(self.key)


class Vision:
    """Gemma 4's vision encoder, of an mmproj GGUF, for a model whose embeddings are `dim` wide
    and whose tokenizer has the tokens that open, fill and close an image."""

    def __init__(self, gguf: GGUF, tokenizer: Tokenizer, dim: int):
        m = {k.removeprefix("clip.vision."): v for k, v in gguf.metadata.items()}
        if (kind := m.get("projector_type")) not in PROJECTORS:
            raise NotImplementedError(f"vision projector {kind!r} is not supported")
        if m["projection_dim"] != dim:
            raise ValueError(f"vision projects to {m['projection_dim']} dims, the model has {dim}")
        self.patch, self.width = m["patch_size"], m["embedding_length"]
        self.heads, self.eps = m["attention.head_count"], m["attention.layer_norm_epsilon"]
        self.pool = m.get("projector.scale_factor", 3)  # patches a side of each embedding pools
        self.tokens = IMAGE_TOKENS
        self._kept: dict[int, Image] = {}  # the latest images, by key, the latest used last
        # the patches an image takes at most, padded to whole tiles of the matrix cores
        self.patches = -(-self.tokens * self.pool**2 // 64) * 64
        w = gguf.load(names=[n for n in gguf.tensors if n.startswith(("v.", "mm."))])
        self.layers = [
            {n: w[f"v.blk.{i}.{n}.weight"] for n in _LAYER} for i in range(m["block_count"])
        ]
        self.small = [{n: t.dequant().realize() for n, t in layer.items() if len(t.shape) == 1}
                      for layer in self.layers]  # fmt: skip
        # the patches' projection, a convolution's (width, 3, patch, patch), as a matrix of
        # patches of rows of RGB
        embed = w["v.patch_embd.weight"].dequant().permute(0, 2, 3, 1).reshape(self.width, -1)
        shape = (self.width, 3 * self.patch**2)
        self.embed = QTensor(embed.flatten().contiguous().realize(), GGMLType.F32, shape)
        # learned embeddings of each column, then of each row
        table = w["v.position_embd.weight"]
        self.columns = int(table.shape[1])
        self.table = QTensor(table.data, table.type, (2 * self.columns, self.width))
        self.std = tuple(w[f"v.std_{n}"].dequant().realize() for n in ("bias", "scale") if
                         f"v.std_{n}" in w)  # fmt: skip
        self.projection = w["mm.input_projection.weight"]
        # the theta of the RoPE of columns and of rows, each over half of a head's dimensions
        head = self.width // self.heads
        self.freqs = Tensor([100.0 ** (-2 * i / (head // 2)) for i in range(head // 4)])
        self.open, self.fill, self.close = (
            _special(tokenizer, t) for t in ("<|image>", "<|image|>", "<image|>")
        )

    def image(self, data: bytes) -> Image:
        """An image of its file's bytes, in any format Pillow reads, upright as its EXIF has it,
        and transparency over white; a ValueError for bytes of no image, or of one so large it
        may be a decompression bomb. Of the host alone, so that any thread may call it. The
        latest are kept, as a conversation sends its images again with every message."""
        key = -1 - int.from_bytes(hashlib.sha256(data).digest()[:7], "little")
        if (kept := self._kept.pop(key, None)) is not None:
            self._kept[key] = kept  # the latest used, last
            return kept
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Picture.DecompressionBombWarning)
                with Picture.open(io.BytesIO(data)) as opened:
                    picture = ImageOps.exif_transpose(opened)
        except (OSError, Picture.DecompressionBombError, Picture.DecompressionBombWarning) as e:
            raise ValueError(f"not an image Pillow reads: {e}") from None
        if picture.has_transparency_data:
            rgba = picture.convert("RGBA")
            picture = Picture.alpha_composite(Picture.new("RGBA", rgba.size, "white"), rgba)
        picture = picture.convert("RGB")
        width, height = _fit(*picture.size, self.patch, self.pool, self.tokens)
        raw = picture.resize((width, height), Picture.Resampling.BICUBIC).tobytes()
        lines = [raw[3 * width * y : 3 * width * (y + 1)] for y in range(height)]
        p, columns, rows = self.patch, width // self.patch, height // self.patch
        pixels = b"".join(  # each patch's lines of RGB, the patches row by row
            lines[y * p + i][3 * p * x : 3 * p * (x + 1)]
            for y in range(rows) for x in range(columns) for i in range(p)
        )  # fmt: skip
        pad = self.patches - columns * rows
        pixels += bytes(pad * 3 * p * p)
        places = [c for y in range(rows) for x in range(columns) for c in (x, y)] + [-1] * 2 * pad
        size = columns * rows // self.pool**2
        tokens = [self.open, *[key] * size, self.close]
        image = self._kept[key] = Image(key, tokens, pixels, array.array("i", places).tobytes())
        while len(self._kept) > KEPT:
            self._kept.pop(next(iter(self._kept)), None)
        return image

    def inputs(self, image: Image) -> tuple[Tensor, Tensor]:
        """An image's pixels and positions on the device, as encode() takes them."""
        pixels = Tensor(image.pixels, dtype=dtypes.uint8).reshape(self.patches, -1)
        return pixels, Tensor(image.positions, dtype=dtypes.int32).reshape(self.patches, 2)

    def encode(self, pixels: Tensor, positions: Tensor) -> Tensor:
        """The embeddings (tokens, dim) of an image's pixels and positions, as Image holds them:
        the image's first, as many as it has, then padding."""
        n, eps, half = self.patches, self.eps, ops.halved(pixels)
        valid = positions[:, 0] >= 0
        column, row = (positions[:, i].maximum(0) for i in (0, 1))
        # in f32 whatever half is, as rounding the pixels to f16 skews every layer after them
        x = _linear((pixels.float() * (2 / 255) - 1).reshape(1, n, -1), self.embed, False)
        x = x + ops.embedding(column.stack(row + self.columns), self.table).sum(0)
        rope = [self._rope(at) for at in (column, row)]
        mask = valid.where(0.0, -math.inf).reshape(1, 1, 1, n)
        for w, s in zip(self.layers, self.small, strict=True):
            out = self._attention(ops.rms_norm(x, s["ln1"], eps), w, s, rope, mask, half)
            x = ops.add_normed(x, [(out, s["attn_post_norm"])], None, eps)
            h = ops.rms_norm(x, s["ln2"], eps)
            # each product its own kernel, on the matrix cores, rather than one of both and GELU
            gate, up = (_linear(h, w[f"ffn_{p}"], half).contiguous() for p in ("gate", "up"))
            out = _linear(ops.glu("gelu", gate, up), w["ffn_down"], half)
            x = ops.add_normed(x, [(out, s["ffn_post_norm"])], None, eps)
        # each embedding the mean of its patches, padding none's, scaled by sqrt(width)
        k = self.pool
        cell = (column // k + (column.max() + 1) // k * (row // k)).reshape(n, 1)
        pools = (cell == Tensor.arange(self.tokens).reshape(1, -1)) & valid.reshape(n, 1)
        x = pools.float().T @ x.reshape(n, -1) * (math.sqrt(self.width) / k**2)
        if self.std:  # standardized
            x = (x - self.std[0]) * self.std[1]
        x = ops.rms_norm(x, None, eps).reshape(1, self.tokens, self.width)
        return _linear(x, self.projection, half).reshape(self.tokens, -1)

    def _attention(
        self, h: Tensor, w: dict[str, QTensor], s: dict[str, Tensor],
        rope: list[tuple[Tensor, Tensor]], mask: Tensor, half: bool,
    ) -> Tensor:  # fmt: skip
        # the patches' attention over each other, all but padding, of their normed h: q and k
        # normed per head and rotated, the first half of each head's dimensions by column and the
        # other by row, v normed without a weight, and scores unscaled
        _, n, _ = h.shape
        eps, head = self.eps, self.width // self.heads
        q, k, v = (_linear(h, w[f"attn_{c}"], half).reshape(1, n, self.heads, head) for c in "qkv")
        q, k = (ops.rms_norm(t, s[f"attn_{c}_norm"], eps) for t, c in ((q, "q"), (k, "k")))
        q, k = (self._rotated(t.transpose(1, 2), rope) for t in (q, k))
        v = ops.rms_norm(v, None, eps).transpose(1, 2)
        # products of f16 if half, the scores and their softmax f32: unscaled, they run to tens;
        # v padded to whole tiles of the matrix cores, its product whole before the padding goes
        v = v.pad_to((*v.shape[:-1], -(-head // 16) * 16))
        q, k, v = (t.half() if half else t for t in (q, k, v))
        weights = (q.dot(k.transpose(-1, -2), dtype=dtypes.float32) + mask).softmax(-1)
        out = (weights.half() if half else weights).dot(v, dtype=dtypes.float32)
        out = out.contiguous()[..., :head].transpose(1, 2).reshape(1, n, self.width)
        return _linear(out.contiguous(), w["attn_out"], half)  # a copy the matrix cores read

    def _rope(self, at: Tensor) -> tuple[Tensor, Tensor]:
        # cos and sin (patches, head / 4) of the angles of each patch's column or row
        angles = at.float().reshape(-1, 1) * self.freqs.reshape(1, -1)
        return angles.cos(), angles.sin()

    def _rotated(self, t: Tensor, rope: list[tuple[Tensor, Tensor]]) -> Tensor:
        half = int(t.shape[-1]) // 2
        parts = (t[..., :half], t[..., half:])
        return Tensor.cat(*(ops.rotary(p, *r, True) for p, r in zip(parts, rope, strict=True)),
                          dim=-1)  # fmt: skip


def _linear(x: Tensor, w: QTensor, half: bool) -> Tensor:
    # x @ w.T, its inputs f16 and its sums f32 if half
    if half:
        return x.half().dot(w.dequant(dtypes.half).T, dtype=dtypes.float32)
    return x @ w.dequant().T


def _fit(width: int, height: int, patch: int, pool: int, tokens: int) -> tuple[int, int]:
    # the largest width and height of whole cells of pool by pool patches, aspect kept, of at
    # most `tokens` cells, as transformers' Gemma 4 processor has it: a side too short for a cell
    # one cell, the other as many as its aspect gives, `tokens` at most
    cell = pool * patch
    factor = math.sqrt(tokens * cell**2 / (width * height))
    w, h = (math.floor(side * factor / cell) * cell for side in (width, height))
    if h == 0:
        h, w = cell, min(width // height * cell, tokens * cell)
    elif w == 0:
        w, h = cell, min(height // width * cell, tokens * cell)
    return w, h


def _special(tokenizer: Tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, bos=False, special=True)
    if len(ids) != 1:
        raise ValueError(f"the model's vocab lacks {text}")
    return ids[0]


def projector(path: Path) -> bool:
    """Whether a GGUF is a projector, as llama.cpp's converters name them: mmproj-*.gguf."""
    return "mmproj" in path.name.lower()


def beside(model: Path) -> Path | None:
    """The projector beside a model, of a kind leat runs, for its embeddings and of its name, as
    its converter writes both: the first in the model's directory whose general.name, or file
    name without one, holds the model's general.name, but for case and punctuation."""

    def name(metadata: dict, path: Path) -> str:
        return re.sub(r"[^a-z0-9]", "", str(metadata.get("general.name", path.stem)).lower())

    m = GGUF.open(model).metadata
    own, dim = name(m, model), m.get(f"{m.get('general.architecture')}.embedding_length")
    for path in sorted(model.parent.glob("*.gguf")):
        p = GGUF.open(path).metadata if projector(path) else {}
        kind, width = p.get("clip.vision.projector_type"), p.get("clip.vision.projection_dim")
        if kind in PROJECTORS and width == dim and own and own in name(p, path):
            return path
    return None


def blank() -> bytes:
    """A PNG of a black square: an image to compile the graphs with."""
    out = io.BytesIO()
    Picture.new("RGB", (64, 64)).save(out, "PNG")
    return out.getvalue()
