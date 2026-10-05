"""GGUF reader: metadata, tensor index and tensor data.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import math
import mmap
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinygrad import Tensor

from leat.quant import BLOCK, NATIVE, GGMLType, QTensor

# metadata value type -> struct format; 8 is a string, 9 an array
_SCALAR = dict(enumerate("BbHhIif?")) | {10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


@dataclass(frozen=True)
class TensorInfo:
    name: str
    type: GGMLType
    shape: tuple[int, ...]  # row-major; GGUF itself lists dimensions innermost first
    offset: int  # absolute byte offset in the file
    nbytes: int


class _Reader:
    def __init__(self, buf: memoryview):
        self.buf, self.pos = buf, 0

    def scalar(self, fmt: str) -> Any:
        (value,) = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize(fmt)
        return value

    def string(self) -> str:
        n = self.scalar("Q")
        self.pos += n
        return str(self.buf[self.pos - n : self.pos], "utf-8")

    def value(self, vtype: int) -> Any:
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, count = self.scalar("I"), self.scalar("Q")
            if etype in _SCALAR:
                fmt = f"<{count}{_SCALAR[etype]}"
                values = list(struct.unpack_from(fmt, self.buf, self.pos))
                self.pos += struct.calcsize(fmt)
                return values
            return [self.value(etype) for _ in range(count)]
        if vtype not in _SCALAR:
            raise ValueError(f"unknown GGUF metadata value type {vtype}")
        return self.scalar(_SCALAR[vtype])


@dataclass(frozen=True)
class GGUF:
    """An opened GGUF file: its metadata and tensor index. Tensor data is read by `load`."""

    path: Path
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]

    @staticmethod
    def open(path: str | Path) -> "GGUF":
        path = Path(path)
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with memoryview(mm) as buf:
                metadata, tensors = _parse(buf)
            size = len(mm)
        for t in tensors.values():
            if t.offset + t.nbytes > size:
                raise ValueError(f"{path}: truncated, tensor {t.name!r} ends past the end of file")
        return GGUF(path, metadata, tensors)

    def load(
        self, device: str | None = None, names: Iterable[str] | None = None
    ) -> dict[str, QTensor]:
        """Copies the data of the tensors named, or of all, to `device` in one transfer and returns
        a view per tensor."""
        tensors = list(self.tensors.values()) if names is None else [self.tensors[n] for n in names]
        if not tensors:
            return {}
        start = min(t.offset for t in tensors)
        end = max(t.offset + t.nbytes for t in tensors)
        data = Tensor(self.path)[start:end].to(device).realize()
        out = {}
        for t in tensors:
            raw = data[t.offset - start : t.offset - start + t.nbytes]
            if t.type in NATIVE:
                out[t.name] = QTensor(raw.bitcast(NATIVE[t.type]), t.type, t.shape)
            else:
                out[t.name] = QTensor(raw.reshape(-1, BLOCK[t.type][1]), t.type, t.shape)
        return out


def _parse(buf: memoryview) -> tuple[dict[str, Any], dict[str, TensorInfo]]:
    r = _Reader(buf)
    if bytes(buf[:4]) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.pos = 4
    if (version := r.scalar("I")) not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors, n_kv = r.scalar("Q"), r.scalar("Q")
    metadata = {}
    for _ in range(n_kv):
        key = r.string()
        metadata[key] = r.value(r.scalar("I"))

    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.scalar("Q") for _ in range(r.scalar("I"))]
        type_id, offset = r.scalar("I"), r.scalar("Q")
        infos.append((name, dims, type_id, offset))

    alignment = metadata.get("general.alignment", 32)
    data_start = -(-r.pos // alignment) * alignment
    tensors = {}
    for name, dims, type_id, offset in infos:
        try:
            ggml_type = GGMLType(type_id)
        except ValueError:
            raise ValueError(f"tensor {name!r} has unknown ggml type {type_id}") from None
        elements, block_bytes = BLOCK[ggml_type]
        if (numel := math.prod(dims)) % elements:
            raise ValueError(f"tensor {name!r} has {numel} elements, not a multiple of {elements}")
        nbytes = numel // elements * block_bytes
        shape = tuple(reversed(dims))
        tensors[name] = TensorInfo(name, ggml_type, shape, data_start + offset, nbytes)
    return metadata, tensors
