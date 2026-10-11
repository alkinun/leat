"""Lists the files of the initrd in a unified kernel image, as the box boots it: the cpio archives,
plain or compressed by zstd, one after another, of its .initrd section.

    uv run --with pefile --with zstandard python box/image/initrd.py UKI.efi [PATTERN]
"""

import re
import sys

import pefile
import zstandard


def section(path: str, name: bytes) -> bytes:
    """A section of a PE file, as its .initrd."""
    pe = pefile.PE(path, fast_load=True)
    for s in pe.sections:
        if s.Name.rstrip(b"\0") == name:
            return s.get_data()[: s.Misc_VirtualSize]
    raise LookupError(f"{path} has no {name.decode()} section")


def files(data: bytes) -> list[str]:
    """The names of the files of concatenated cpio archives, plain or zstd's."""
    names: list[str] = []
    while data := data.lstrip(b"\0"):
        if data.startswith(b"\x28\xb5\x2f\xfd"):  # a zstd frame: an archive, compressed
            reader = zstandard.ZstdDecompressor().decompressobj()
            names += files(reader.decompress(data))
            data = reader.unused_data
        elif data.startswith(b"070701"):
            at = 0
            while True:  # newc headers: 110 bytes, then the name, then the file, each to 4 bytes
                header = data[at : at + 110]
                size, length = int(header[54:62], 16), int(header[94:102], 16)
                name = data[at + 110 : at + 110 + length - 1].decode()
                at = (at + 110 + length + 3) & ~3
                at = (at + size + 3) & ~3
                if name == "TRAILER!!!":
                    break
                names.append(name)
            data = data[at:]
        else:
            raise ValueError(f"neither cpio nor zstd: {data[:8]!r}")
    return names


if __name__ == "__main__":
    found = files(section(sys.argv[1], b".initrd"))
    pattern = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
    for name in found:
        if pattern is None or pattern.search(name):
            print(name)
    print(f"({len(found)} files)", file=sys.stderr)
