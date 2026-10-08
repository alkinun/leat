"""Vision encoders: images as embeddings a model reads in place of tokens, of llama.cpp's mmproj
GGUFs, each as transformers has it.

An encoder is a ViT over an image's patches, then a projector into the model's embeddings. It takes
every image padded to as many patches as its largest, so that one compiled graph runs them all:
- Gemma 3's, SigLIP over the image scaled to 896 by 896 pixels, its patches pooled 4 by 4 into 256
  embeddings;
- Gemma 4's, of 2D RoPE over the image scaled, its aspect kept, to as many patches of 16 by 16
  pixels as fit IMAGE_TOKENS embeddings, each of 3 by 3 patches pooled.
"""

import array
import hashlib
import io
import math
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image as Picture
from PIL import ImageOps
from tinygrad import Tensor, dtypes

from leat import ops
from leat.gguf import GGUF
from leat.quant import GGMLType, QTensor
from leat.tokenizer import Tokenizer

# the embeddings an image of Gemma 4 takes at most: of its budgets, 70, 140, 280, 560 or 1120, its
# processor's default
IMAGE_TOKENS = 280
KEPT = 16  # images kept, made of their bytes, to be given again


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
    """A vision encoder of an mmproj GGUF's metadata `m` and weights `w`, for a model whose
    tokenizer has the tokens that show an image. Its layers' parts are used where the GGUF has
    them, as the text model's are: biases, norms of q and k and after attention and the MLP, which
    is gated or not, and LayerNorms of their biases or else RMSNorms. A family's own scales and
    cuts an image into patches, and projects the layers' outputs."""

    causal = False  # whether the model reads an image's embeddings causally, or each sees all
    v_norm = False  # values RMSNormed without a weight, as Gemma 4's
    scale: float | None = None  # attention's, 1 / sqrt(head) if None
    tokens: int  # embeddings an image takes at most
    fill: int  # the token of an image's positions, which a drafter takes in
    patches: int  # patches it takes at most, padded to whole tiles of the matrix cores

    def __init__(self, m: dict[str, Any], w: dict[str, QTensor], tokenizer: Tokenizer):
        self.patch, self.width = m["patch_size"], m["embedding_length"]
        self.heads, self.eps = m["attention.head_count"], m["attention.layer_norm_epsilon"]
        self.act = "silu" if m.get("use_silu") else "gelu"
        self.mean, self.std = m.get("image_mean", [0.0] * 3), m.get("image_std", [1.0] * 3)
        self.tokenizer, self.w = tokenizer, w
        self.layers: list[dict[str, QTensor]] = [{} for _ in range(m["block_count"])]
        for name, t in w.items():
            if name.startswith("v.blk."):
                i, part = name.removeprefix("v.blk.").split(".", 1)
                self.layers[int(i)][part.removesuffix(".weight")] = t
        self.small = [{n: t.dequant().realize() for n, t in layer.items() if len(t.shape) == 1}
                      for layer in self.layers]  # fmt: skip
        # the patches' projection, a convolution's (width, 3, patch, patch), as a matrix of
        # patches of rows of RGB; of two frames of the same image summed, as Qwen's video's
        kernels = [
            w[n].dequant() for n in ("v.patch_embd.weight", "v.patch_embd.weight.1") if n in w
        ]
        embed = sum(kernels[1:], kernels[0]).permute(0, 2, 3, 1).flatten().contiguous().realize()
        self.embed = QTensor(embed, GGMLType.F32, (self.width, 3 * self.patch**2))
        self._kept: dict[int, Image] = {}  # the latest images, by key, the latest used last

    def image(self, data: bytes) -> Image:
        """An image of its file's bytes, in any format Pillow reads, upright as its EXIF has it,
        and transparency over white; a ValueError for bytes of no image, or of one so large it
        may be a decompression bomb. Of the host alone, so that any thread may call it. The
        latest are kept, as a conversation sends its images again with every message."""
        key = -1 - int.from_bytes(hashlib.sha256(data).digest()[:7], "little")
        if (kept := self._kept.pop(key, None)) is not None:
            self._kept[key] = kept  # the latest used, last
            return kept
        image = self._kept[key] = self._image(_picture(data), key)
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
        raise NotImplementedError

    def _image(self, picture: Picture.Image, key: int) -> Image:
        # an image of its RGB picture, of the key of its bytes
        raise NotImplementedError

    def _patches(
        self, picture: Picture.Image, width: int, height: int, resample: Picture.Resampling
    ) -> tuple[bytes, bytes]:
        # the picture scaled to width by height, as its patches' pixels and positions, padded
        raw = picture.resize((width, height), resample).tobytes()
        lines = [raw[3 * width * y : 3 * width * (y + 1)] for y in range(height)]
        p, columns, rows = self.patch, width // self.patch, height // self.patch
        pixels = b"".join(  # each patch's lines of RGB, the patches row by row
            lines[y * p + i][3 * p * x : 3 * p * (x + 1)]
            for y in range(rows) for x in range(columns) for i in range(p)
        )  # fmt: skip
        pad = self.patches - columns * rows
        places = [c for y in range(rows) for x in range(columns) for c in (x, y)] + [-1] * 2 * pad
        return pixels + bytes(pad * 3 * p * p), array.array("i", places).tobytes()

    def _embedded(self, pixels: Tensor) -> Tensor:
        # the patches' pixels normalized and projected, in f32 whatever the rest is, as rounding
        # the pixels to f16 skews every layer after them: (1, patches, width)
        n, channels = int(pixels.shape[0]), 3
        mean, std = (Tensor(v).reshape(1, 1, channels) for v in (self.mean, self.std))
        x = ((pixels.float().reshape(n, -1, channels) / 255 - mean) / std).reshape(1, n, -1)
        x = _linear(x, self.embed, False)
        return x + self.w["v.patch_embd.bias"].dequant() if "v.patch_embd.bias" in self.w else x

    def _encoder(
        self, x: Tensor, mask: Tensor, half: bool,
        rope: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:  # fmt: skip
        # the layers over the patches x (1, n, width), each attending over all but those the
        # additive mask (1, 1, 1, n) hides, rotated by rope if given; then the norm after, if any
        if "v.pre_ln.weight" in self.w:
            x = self._norm(x, "v.pre_ln")
        for w, s in zip(self.layers, self.small, strict=True):
            out = self._attention(self._norm(x, "ln1", s), w, s, mask, half, rope)
            x = self._added(x, out, s, "attn_post_norm")
            h = self._norm(x, "ln2", s)
            # each product its own kernel, on the matrix cores, rather than one of both and GELU
            up, down = ("ffn_up", "ffn_down")
            if w[up].shape[1] != self.width:  # named the other way round, as SigLIP's are
                up, down = down, up
            hidden = self._linear(h, w, s, up, half)
            if "ffn_gate" in w:
                gate = self._linear(h, w, s, "ffn_gate", half).contiguous()
                hidden = ops.glu(self.act, gate, hidden.contiguous())
            else:
                hidden = hidden.gelu() if self.act == "gelu" else hidden.silu()
            x = self._added(x, self._linear(hidden, w, s, down, half), s, "ffn_post_norm")
        return self._norm(x, "v.post_ln") if "v.post_ln.weight" in self.w else x

    def _attention(
        self, h: Tensor, w: dict[str, QTensor], s: dict[str, Tensor], mask: Tensor, half: bool,
        rope: Callable[[Tensor], Tensor] | None,
    ) -> Tensor:  # fmt: skip
        # the patches' attention over each other, of their normed h: q, k and v of one matrix or
        # three, q and k normed per head if they have norms, and rotated by rope if given
        _, n, _ = h.shape
        head = self.width // self.heads
        if "attn_qkv" in w:
            q, k, v = self._linear(h, w, s, "attn_qkv", half).chunk(3, dim=-1)
        else:
            q, k, v = (self._linear(h, w, s, f"attn_{c}", half) for c in "qkv")
        q, k, v = (t.reshape(1, n, self.heads, head) for t in (q, k, v))
        if "attn_q_norm" in s:
            q, k = (ops.rms_norm(t, s[f"attn_{c}_norm"], self.eps) for t, c in ((q, "q"), (k, "k")))
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            q, k = rope(q), rope(k)
        if self.v_norm:
            v = ops.rms_norm(v, None, self.eps)
        if (scale := self.scale or head**-0.5) != 1:
            q = q * scale
        # products of f16 if half, the scores and their softmax f32: unscaled, they run to tens;
        # v padded to whole tiles of the matrix cores, its product whole before the padding goes
        v = v.pad_to((*v.shape[:-1], -(-head // 16) * 16))
        q, k, v = (t.half() if half else t for t in (q, k, v))
        weights = (q.dot(k.transpose(-1, -2), dtype=dtypes.float32) + mask).softmax(-1)
        out = (weights.half() if half else weights).dot(v, dtype=dtypes.float32)
        out = out.contiguous()[..., :head].transpose(1, 2).reshape(1, n, self.width)
        return self._linear(out.contiguous(), w, s, "attn_out", half)  # a copy the cores read

    def _linear(
        self, x: Tensor, w: dict[str, QTensor], s: dict[str, Tensor], name: str, half: bool
    ) -> Tensor:
        # a layer's matrix, plus its bias if it has one
        out = _linear(x, w[name], half)
        return out + s[f"{name}.bias"] if f"{name}.bias" in s else out

    def _norm(self, x: Tensor, name: str, s: dict[str, Tensor] | None = None) -> Tensor:
        # a layer's norm, or the encoder's: a LayerNorm if it has a bias, else an RMSNorm
        if s is None:
            s = {name: self.w[f"{name}.weight"].dequant()}
            if f"{name}.bias" in self.w:
                s[f"{name}.bias"] = self.w[f"{name}.bias"].dequant()
        if f"{name}.bias" in s:
            return x.layernorm(eps=self.eps) * s[name] + s[f"{name}.bias"]
        return ops.rms_norm(x, s[name], self.eps)

    def _added(self, x: Tensor, out: Tensor, s: dict[str, Tensor], norm: str) -> Tensor:
        # x plus a block's output, normed first if the layer has the norm
        return ops.add_normed(x, [(out, s[norm])], None, self.eps) if norm in s else x + out

    def _special(self, text: str) -> int:
        ids = self.tokenizer.encode(text, bos=False, special=True)
        if len(ids) != 1:
            raise ValueError(f"the model's vocab lacks {text}")
        return ids[0]


class Gemma3(Vision):
    """Gemma 3's SigLIP: the image scaled to image_size square, its patches each with a learned
    position, pooled pool by pool into embeddings, 256 of 896 pixels, each RMSNormed and projected;
    shown between newlines, as transformers' processor writes it."""

    def __init__(self, m: dict[str, Any], w: dict[str, QTensor], tokenizer: Tokenizer):
        super().__init__(m, w, tokenizer)
        self.square = m["image_size"]  # the side it scales an image to
        self.patches = (self.square // self.patch) ** 2
        self.pool = m.get("projector.scale_factor", 4)  # patches a side of each embedding pools
        self.tokens = self.patches // self.pool**2
        self.positions = w["v.position_embd.weight"].dequant().realize()
        self.norm = w["mm.soft_emb_norm.weight"].dequant().realize()
        lines = tokenizer.encode("\n\n", bos=False)
        self.open = [*lines, self._special("<start_of_image>")]
        self.close = [self._special("<end_of_image>"), *lines]
        self.fill = self.open[-1]  # as GGUFs' vocabs end before its <image_soft_token>

    def encode(self, pixels: Tensor, positions: Tensor) -> Tensor:
        half, side = ops.halved(pixels), self.square // self.patch
        x = self._embedded(pixels) + self.positions
        x = self._encoder(x, Tensor.zeros(1, 1, 1, self.patches), half)
        k = self.pool
        x = x.reshape(side // k, k, side // k, k, self.width).mean((1, 3))
        x = ops.rms_norm(x.reshape(1, -1, self.width), self.norm, self.eps)
        projection = self.w["mm.input_projection.weight"]  # (width, dim): multiplied as it is
        if half:
            return x.half().dot(projection.dequant(dtypes.half), dtype=dtypes.float32)[0]
        return (x @ projection.dequant())[0]

    def _image(self, picture: Picture.Image, key: int) -> Image:
        pixels, positions = self._patches(
            picture, self.square, self.square, Picture.Resampling.BILINEAR
        )
        return Image(key, [*self.open, *[key] * self.tokens, *self.close], pixels, positions)


class Gemma4(Vision):
    """Gemma 4's: the image scaled, its aspect kept, to as many patches as fit IMAGE_TOKENS
    embeddings, each of pool by pool patches; its patches with learned embeddings of their column
    and row, and rotated, the first half of each head's dimensions by column and the other by
    row; q and k normed, v too, and scores unscaled; pooled, scaled by sqrt(width), standardized,
    normed and projected."""

    v_norm, scale = True, 1.0

    def __init__(self, m: dict[str, Any], w: dict[str, QTensor], tokenizer: Tokenizer):
        super().__init__(m, w, tokenizer)
        self.mean = self.std = [0.5] * 3  # as its model scales the pixels to [-1, 1]
        self.pool = m.get("projector.scale_factor", 3)  # patches a side of each embedding pools
        self.tokens = IMAGE_TOKENS
        self.patches = -(-self.tokens * self.pool**2 // 64) * 64
        # learned embeddings of each column, then of each row
        table = w["v.position_embd.weight"]
        self.columns = int(table.shape[1])
        self.table = QTensor(table.data, table.type, (2 * self.columns, self.width))
        self.standard = tuple(w[f"v.std_{n}"].dequant().realize() for n in ("bias", "scale")
                              if f"v.std_{n}" in w)  # fmt: skip
        # the RoPE of columns and of rows, theta 100, each over half of a head's dimensions
        head = self.width // self.heads
        self.freqs = Tensor([100.0 ** (-2 * i / (head // 2)) for i in range(head // 4)])
        self.open, self.fill, self.close = (
            self._special(t) for t in ("<|image>", "<|image|>", "<image|>")
        )

    def encode(self, pixels: Tensor, positions: Tensor) -> Tensor:
        n, half = self.patches, ops.halved(pixels)
        valid = positions[:, 0] >= 0
        column, row = (positions[:, i].maximum(0) for i in (0, 1))
        x = self._embedded(pixels)
        x = x + ops.embedding(column.stack(row + self.columns), self.table).sum(0)
        tables = [self._rope(at) for at in (column, row)]
        mask = valid.where(0.0, -math.inf).reshape(1, 1, 1, n)
        x = self._encoder(x, mask, half, lambda t: self._rotated(t, tables))
        # each embedding the mean of its patches, padding none's, scaled by sqrt(width)
        k = self.pool
        cell = (column // k + (column.max() + 1) // k * (row // k)).reshape(n, 1)
        pools = (cell == Tensor.arange(self.tokens).reshape(1, -1)) & valid.reshape(n, 1)
        x = pools.float().T @ x.reshape(n, -1) * (math.sqrt(self.width) / k**2)
        if self.standard:
            x = (x - self.standard[0]) * self.standard[1]
        x = ops.rms_norm(x, None, self.eps).reshape(1, self.tokens, self.width)
        return _linear(x, self.w["mm.input_projection.weight"], half).reshape(self.tokens, -1)

    def _image(self, picture: Picture.Image, key: int) -> Image:
        width, height = _fit(*picture.size, self.patch, self.pool, self.tokens)
        pixels, positions = self._patches(picture, width, height, Picture.Resampling.BICUBIC)
        size = width * height // (self.patch * self.pool) ** 2
        return Image(key, [self.open, *[key] * size, self.close], pixels, positions)

    def _rope(self, at: Tensor) -> tuple[Tensor, Tensor]:
        # cos and sin (patches, head / 4) of the angles of each patch's column or row
        angles = at.float().reshape(-1, 1) * self.freqs.reshape(1, -1)
        return angles.cos(), angles.sin()

    def _rotated(self, t: Tensor, tables: list[tuple[Tensor, Tensor]]) -> Tensor:
        half = int(t.shape[-1]) // 2
        parts = (t[..., :half], t[..., half:])
        return Tensor.cat(*(ops.rotary(p, *r, True) for p, r in zip(parts, tables, strict=True)),
                          dim=-1)  # fmt: skip


# the encoders leat runs, by the projector types of llama.cpp's GGUFs
PROJECTORS: dict[str, type[Vision]] = {"gemma3": Gemma3, "gemma4v": Gemma4}


def load(path: str | Path, tokenizer: Tokenizer, dim: int) -> Vision:
    """The vision encoder of an mmproj GGUF, for a model whose embeddings are `dim` wide."""
    gguf = GGUF.open(path)
    m = _metadata(gguf.metadata)
    if (kind := m.get("projector_type")) not in PROJECTORS:
        raise NotImplementedError(f"vision projector {kind!r} is not supported")
    if m["projection_dim"] != dim:
        raise ValueError(f"vision projects to {m['projection_dim']} dims, the model has {dim}")
    weights = gguf.load(names=[n for n in gguf.tensors if n.startswith(("v.", "mm."))])
    return PROJECTORS[kind](m, weights, tokenizer)


def _metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # a projector's keys without clip.vision. or clip., which older GGUFs write some under
    return {k.removeprefix("clip.").removeprefix("vision."): v for k, v in metadata.items()}


def _picture(data: bytes) -> Picture.Image:
    # an image file's RGB picture, upright and its transparency over white
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
    return picture.convert("RGB")


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
        kind, width = _metadata(p).get("projector_type"), _metadata(p).get("projection_dim")
        if kind in PROJECTORS and width == dim and own and own in name(p, path):
            return path
    return None


def blank() -> bytes:
    """A PNG of a black square: an image to compile the graphs with."""
    out = io.BytesIO()
    Picture.new("RGB", (64, 64)).save(out, "PNG")
    return out.getvalue()
