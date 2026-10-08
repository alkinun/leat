"""Vision encoders: images as embeddings a model reads in place of tokens, of llama.cpp's mmproj
GGUFs, each as transformers has it.

An encoder is a ViT over an image's patches, then a projector into the model's embeddings. It takes
every image padded to as many patches as its largest, so that one compiled graph runs them all:
- Gemma 3's, SigLIP over the image scaled to 896 by 896 pixels, its patches pooled 4 by 4 into 256
  embeddings;
- Gemma 4's, of 2D RoPE over the image scaled, its aspect kept, to as many patches of 16 by 16
  pixels as fit IMAGE_TOKENS embeddings, each of 3 by 3 patches pooled;
- Mistral Small 3's, Pixtral, of 2D RoPE over the image scaled, its aspect kept, to whole cells
  of 2 by 2 patches of 14 pixels, each merged into an embedding, PIXTRAL_TOKENS at most;
- Qwen3.5's, of 2D RoPE too, over whole cells of 2 by 2 patches of 16 pixels, QWEN_TOKENS at most.
"""

import array
import hashlib
import io
import math
import re
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image as Picture
from PIL import ImageOps
from tinygrad import Tensor, dtypes

from leat import kernels, ops
from leat.gguf import GGUF
from leat.quant import GGMLType, QTensor
from leat.tokenizer import Tokenizer

# the embeddings an image of Gemma 4 takes at most: of its budgets, 70, 140, 280, 560 or 1120, its
# processor's default
IMAGE_TOKENS = 280
# those of Mistral Small 3's, but for its breaks, as llama.cpp's: an image of 896 by 896 pixels
PIXTRAL_TOKENS = 1024
# those of Qwen3.5's, of 1024 by 1024 pixels, a quarter of llama.cpp's, whose patches' attention
# would not fit the GPU; and the pixels it takes at least, as transformers'
QWEN_TOKENS, QWEN_PIXELS = 1024, 65536
KEPT = 16  # images kept, made of their bytes, to be given again


@dataclass(frozen=True, eq=False)
class Image:
    """An image ready for a prompt. `tokens` show it there: those that open and close it, around
    a position for each of its embeddings, which hold `key`, a negative id of a digest of the
    image by which the cache tells images apart; `grid`, the rows and columns of its embeddings.
    `pixels` and `positions` are the encoder's, as bytes the engine uploads when it encodes the
    image: each patch's pixels, rows of RGB, and its column and row as int32, -1 for padding."""

    key: int
    tokens: list[int]
    grid: tuple[int, int]
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
        self._lock = threading.Lock()  # of _kept, which handlers' threads share

    def image(self, data: bytes) -> Image:
        """An image of its file's bytes, in any format Pillow reads, upright as its EXIF has it,
        and transparency over white; a ValueError for bytes of no image, or of one so large it
        may be a decompression bomb. Of the host alone, so that any thread may call it. The
        latest are kept, as a conversation sends its images again with every message."""
        key = -1 - int.from_bytes(hashlib.sha256(data).digest()[:7], "little")
        with self._lock:
            if (kept := self._kept.pop(key, None)) is not None:
                self._kept[key] = kept  # the latest used, last
                return kept
        image = self._image(_picture(data), key)
        with self._lock:
            self._kept[key] = image
            while len(self._kept) > KEPT:
                del self._kept[next(iter(self._kept))]
        return image

    @property
    def typical(self) -> int:
        """The embeddings a square image takes, as most images are: as many as any may."""
        return self.tokens

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
        self, picture: Picture.Image, width: int, height: int, resample: Picture.Resampling,
        merge: int = 1,
    ) -> tuple[bytes, bytes]:  # fmt: skip
        # the picture scaled to width by height, as its patches' pixels and positions, padded:
        # row by row, or of windows of merge by merge patches row by row, each's row by row
        raw = picture.resize((width, height), resample).tobytes()
        lines = [raw[3 * width * y : 3 * width * (y + 1)] for y in range(height)]
        p, columns, rows = self.patch, width // self.patch, height // self.patch
        order = [
            (y * merge + i, x * merge + j)
            for y in range(rows // merge)
            for x in range(columns // merge)
            for i in range(merge)
            for j in range(merge)
        ]
        pixels = b"".join(  # each patch's lines of RGB
            lines[y * p + i][3 * p * x : 3 * p * (x + 1)] for y, x in order for i in range(p)
        )  # fmt: skip
        pad = self.patches - columns * rows
        places = [c for y, x in order for c in (x, y)] + [-1] * 2 * pad
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
        scale = self.scale or head**-0.5
        if (out := self._flashed(q, k, v, mask, scale) if half else None) is None:
            # products of f16 if half, the scores and their softmax f32, as they run to tens; v
            # padded to whole tiles of the matrix cores, its product whole before the padding
            # goes; q, k and v made first, as fused into the scores' product, the rotation runs
            # once for each pair
            v = v.pad_to((*v.shape[:-1], -(-head // 16) * 16))
            q, k, v = ((t.half() if half else t).contiguous() for t in (q * scale, k, v))
            weights = (q.dot(k.transpose(-1, -2), dtype=dtypes.float32) + mask).softmax(-1)
            out = (weights.half() if half else weights).dot(v, dtype=dtypes.float32)
            out = out.contiguous()[..., :head].transpose(1, 2).reshape(1, n, self.width)
        return self._linear(out.contiguous(), w, s, "attn_out", half)  # a copy the cores read

    def _flashed(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: float
    ) -> Tensor | None:
        # attention (1, n, width) of q, k and v (1, heads, n, head) by FlashAttention's kernel,
        # which makes no matrix of scores, if it takes them: each head widened to whole tiles,
        # the dimension past its own 1 in q, and in k 0 for patches, and for padding, as the
        # additive mask (1, 1, 1, n) has it, its only one, far below, so that it takes no weight
        _, heads, n, head = (int(d) for d in q.shape)
        width = -(-(head + 1) // 16) * 16
        spare = (Tensor.arange(width) == head).float()
        # made first: fused into the stack of keys and values, their products run off the cores
        q, k, v = (t.contiguous().pad_to((1, heads, n, width)) for t in (q, k, v))
        padding = mask.reshape(1, 1, n, 1) < 0  # whose keys take only the dimension past
        q, k = q + spare, padding.where(spare * -30000.0, k)
        cache = Tensor.stack(k, v).half().contiguous()  # as a slot's keys and values
        if not kernels.supports_flash_attention(q, cache):
            return None
        out = kernels.flash_attention(q, cache, 0, 0, scale, causal=False)
        return out.reshape(1, n, heads, width)[..., :head].reshape(1, n, heads * head)

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
        side = math.isqrt(self.tokens)
        tokens = [*self.open, *[key] * self.tokens, *self.close]
        return Image(key, tokens, (side, side), pixels, positions)


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
        tables = [_angles(at, self.freqs) for at in (column, row)]
        mask = valid.where(0.0, -math.inf).reshape(1, 1, 1, n)
        x = self._encoder(x, mask, half, lambda t: _rotated(t, tables, True))
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
        cell = self.patch * self.pool
        grid = height // cell, width // cell
        tokens = [self.open, *[key] * (grid[0] * grid[1]), self.close]
        return Image(key, tokens, grid, pixels, positions)


class Pixtral(Vision):
    """Mistral Small 3's: the image scaled, its aspect kept, to at most image_size a side and
    PIXTRAL_TOKENS embeddings, each side to whole cells of merge by merge patches, up; its
    patches rotated, the first half of each head's dimensions by row and the other by column, of
    RoPE's even frequencies and its odd ones, as q and k are of adjacent pairs in the GGUF; each
    cell's patches merged into an embedding, projected, and each row of them but the last ended by
    [IMG_BREAK]'s embedding; read causally, the image closed by [IMG_END]."""

    causal = True

    def __init__(self, m: dict[str, Any], w: dict[str, QTensor], tokenizer: Tokenizer):
        super().__init__(m, w, tokenizer)
        self.side, self.merge = m["image_size"], m.get("spatial_merge_size", 1)
        self.cells = PIXTRAL_TOKENS  # embeddings of patches at most
        self.patches = -(-self.cells * self.merge**2 // 64) * 64
        self.tokens = 2 * self.cells - 1  # and of breaks, as a column of cells has most
        head = self.width // self.heads
        freqs = [m.get("rope.freq_base", 10000.0) ** (-2 * i / head) for i in range(head // 2)]
        self.freqs = Tensor(freqs[0::2]), Tensor(freqs[1::2])  # of rows, and of columns
        self.close = self._special("[IMG_END]")
        self.fill = self._special("[IMG]")

    def encode(self, pixels: Tensor, positions: Tensor) -> Tensor:
        n, half, k = self.patches, ops.halved(pixels), self.merge
        valid = positions[:, 0] >= 0
        column, row = (positions[:, i].maximum(0) for i in (0, 1))
        tables = [_angles(at, f) for at, f in zip((row, column), self.freqs, strict=True)]
        mask = valid.where(0.0, -math.inf).reshape(1, 1, 1, n)
        x = self._encoder(self._embedded(pixels), mask, half, lambda t: _rotated(t, tables, False))
        x = ops.rms_norm(x, self.w["mm.input_norm.weight"].dequant(), self.eps)
        # each cell's patches, which come together, as one row of their channels' patches
        x = x.reshape(n // k**2, k**2, self.width).permute(0, 2, 1).reshape(1, n // k**2, -1)
        x = _linear(x, self.w["mm.patch_merger.weight"], half)
        x = self._projected(x, "mm.1", half)
        x = self._projected(0.5 * x * (1 + (x / math.sqrt(2)).erf()), "mm.2", half)
        x = x[0].contiguous()  # a product of its own, not fused into the rows that follow
        # the embeddings row by row, each row ended by a break but the last: of the cells' and
        # the break's, after them
        columns = (column.max() + 1) // k
        at, cells = Tensor.arange(self.tokens), n // k**2
        line, place = at // (columns + 1), at % (columns + 1)
        source = (place == columns).where(cells, (line * columns + place).minimum(cells - 1))
        breaks = self.w["v.token_embd.img_break"].dequant().reshape(1, -1)
        return x.cat(breaks)[source]

    def _image(self, picture: Picture.Image, key: int) -> Image:
        cell = self.patch * self.merge
        width, height = _within(*picture.size, self.side, cell, self.cells)
        bicubic = Picture.Resampling.BICUBIC
        pixels, positions = self._patches(picture, width, height, bicubic, self.merge)
        rows, columns = height // cell, width // cell
        tokens = [*[key] * (rows * (columns + 1) - 1), self.close]
        return Image(key, tokens, (rows, columns + 1), pixels, positions)

    @property
    def typical(self) -> int:
        return self.cells + math.isqrt(self.cells) - 1  # with a break ending each row but the last

    def _projected(self, x: Tensor, name: str, half: bool) -> Tensor:
        out = _linear(x, self.w[f"{name}.weight"], half)
        return out + self.w[f"{name}.bias"].dequant() if f"{name}.bias" in self.w else out


class Qwen3(Vision):
    """Qwen3.5's and Qwen3.6's, Qwen3-VL's merger: the image scaled, its aspect kept, to whole
    cells of merge by merge patches, QWEN_PIXELS to QWEN_TOKENS cells, as transformers'
    smart_resize; its patches of a video's two frames, the image's twice, each with the learned
    embedding of its place, of a grid side by side interpolated to the image's, and rotated by row
    over the first quarter of each head's frequencies and by column over the second; each cell's
    patches, which come together, normed and joined, then projected. Read causally, its embeddings
    at the image's M-RoPE positions."""

    causal = True

    def __init__(self, m: dict[str, Any], w: dict[str, QTensor], tokenizer: Tokenizer):
        super().__init__(m, w, tokenizer)
        self.merge, self.tokens = m.get("spatial_merge_size", 2), QWEN_TOKENS
        self.patches = -(-self.tokens * self.merge**2 // 64) * 64
        table = w["v.position_embd.weight"]
        self.side = math.isqrt(int(table.shape[0]))  # of the learned grid
        head = self.width // self.heads
        self.freqs = Tensor([10000.0 ** (-2 * i / (head // 2)) for i in range(head // 4)])
        self.open, self.fill, self.close = (
            self._special(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
        )

    def encode(self, pixels: Tensor, positions: Tensor) -> Tensor:
        n, half, k = self.patches, ops.halved(pixels), self.merge
        valid = positions[:, 0] >= 0
        column, row = (positions[:, i].maximum(0) for i in (0, 1))
        x = self._embedded(pixels) + self._places(column, row)
        angles = (row.float().reshape(-1, 1) * self.freqs).cat(
            column.float().reshape(-1, 1) * self.freqs, dim=-1)  # fmt: skip
        cos, sin = angles.cos(), angles.sin()
        mask = valid.where(0.0, -math.inf).reshape(1, 1, 1, n)
        x = self._encoder(x, mask, half, lambda t: ops.rotary(t, cos, sin, True))
        x = x.reshape(1, n // k**2, -1)  # each cell's patches, which come together, joined
        x = _linear(x, self.w["mm.0.weight"], half) + self.w["mm.0.bias"].dequant()
        x = 0.5 * x * (1 + (x / math.sqrt(2)).erf())
        return (_linear(x, self.w["mm.2.weight"], half) + self.w["mm.2.bias"].dequant())[0]

    def _image(self, picture: Picture.Image, key: int) -> Image:
        cell = self.patch * self.merge
        width, height = _resized(*picture.size, cell, QWEN_PIXELS, self.tokens * cell**2)
        bicubic = Picture.Resampling.BICUBIC
        pixels, positions = self._patches(picture, width, height, bicubic, self.merge)
        grid = height // cell, width // cell
        tokens = [self.open, *[key] * (grid[0] * grid[1]), self.close]
        return Image(key, tokens, grid, pixels, positions)

    def _places(self, column: Tensor, row: Tensor) -> Tensor:
        # each patch's learned embedding of its place: of the four of the learned grid around
        # it, bilinearly, as the image's grid spreads over the learned one, corner to corner
        (r0, r1, fr), (c0, c1, fc) = (self._spread(at) for at in (row, column))
        corners = [(r0, c0, (1 - fr) * (1 - fc)), (r0, c1, (1 - fr) * fc),
                   (r1, c0, fr * (1 - fc)), (r1, c1, fr * fc)]  # fmt: skip
        rows = Tensor.stack(*(r * self.side + c for r, c, _ in corners))
        weights = Tensor.stack(*(wt for _, _, wt in corners)).unsqueeze(-1)
        places = (ops.embedding(rows, self.w["v.position_embd.weight"]) * weights).sum(0)
        return places.reshape(1, -1, self.width)

    def _spread(self, at: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # the learned grid's rows or columns before and after each of the image's, and how far
        # between, as transformers' linspace over the learned side
        spot = at.float() * ((self.side - 1) / at.max().maximum(1).float())
        low = spot.floor().cast(dtypes.int32)
        return low, (low + 1).minimum(self.side - 1), spot - low


# the encoders leat runs, by the projector types of llama.cpp's GGUFs
PROJECTORS: dict[str, type[Vision]] = {
    "gemma3": Gemma3, "gemma4v": Gemma4, "pixtral": Pixtral, "qwen3vl_merger": Qwen3,
}  # fmt: skip


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


def _angles(at: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    # cos and sin (patches, frequencies) of the angles of each patch's position on an axis
    angles = at.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return angles.cos(), angles.sin()


def _rotated(t: Tensor, tables: list[tuple[Tensor, Tensor]], halves: bool) -> Tensor:
    # each half of each head's dimensions of t rotated by its table, of i with i + R/2 if halves,
    # else of adjacent pairs, as ops.rotary rotates them
    half = int(t.shape[-1]) // 2
    parts = (t[..., :half], t[..., half:])
    rotated = (ops.rotary(p, *r, halves) for p, r in zip(parts, tables, strict=True))
    return Tensor.cat(*rotated, dim=-1)


def _within(width: int, height: int, side: int, cell: int, cells: int) -> tuple[int, int]:
    # as transformers' Pixtral: scaled down, its aspect kept, to `side` at most a side, then each
    # side up to whole cells; and down again, as llama.cpp, to `cells` at most
    ratio = max(width / side, height / side, math.sqrt(width * height / (cells * cell**2)))
    if ratio > 1:
        width, height = max(int(width / ratio), 1), max(int(height / ratio), 1)
    w, h = -(-width // cell) * cell, -(-height // cell) * cell
    while (w // cell) * (h // cell) > cells:  # rounded up past the budget: a cell less
        w, h = (w - cell, h) if w >= h else (w, h - cell)
    return w, h


def _resized(width: int, height: int, cell: int, least: int, most: int) -> tuple[int, int]:
    # as transformers' smart_resize: each side rounded to whole cells, then scaled, its aspect
    # kept, to `least` pixels at least and `most` at most; a ValueError for a side 200 times the
    # other, which it refuses too, as the cells would not fit
    if max(width, height) > 200 * min(width, height):
        raise ValueError(f"an image of {width} by {height} pixels is too thin: 200 to 1 at most")
    w, h = round(width / cell) * cell, round(height / cell) * cell
    if w * h > most:
        beta = math.sqrt(width * height / most)
        w, h = (max(cell, math.floor(side / beta / cell) * cell) for side in (width, height))
    elif w * h < least:
        beta = math.sqrt(least / (width * height))
        w, h = (math.ceil(side * beta / cell) * cell for side in (width, height))
    return w, h


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
