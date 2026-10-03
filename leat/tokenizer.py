"""Byte-level BPE tokenizer built from GGUF metadata, matching llama.cpp's llama-vocab.cpp."""

import codecs
import itertools
import re
import unicodedata
from collections.abc import Callable
from typing import Any

# GPT-2's reversible byte <-> character mapping; vocab entries are spelled in these characters.
_PRINTABLE = [*range(33, 127), *range(161, 173), *range(174, 256)]
_UNPRINTABLE = [b for b in range(256) if b not in _PRINTABLE]
_BYTE_CHAR = {b: chr(b) for b in _PRINTABLE} | {b: chr(256 + i) for i, b in enumerate(_UNPRINTABLE)}
_CHAR_BYTE = {c: b for b, c in _BYTE_CHAR.items()}

NORMAL, UNKNOWN, CONTROL, USER_DEFINED = 1, 2, 3, 4
# llama.cpp treats these as end of generation in addition to the eos/eot/eom ids
_EOG_TEXT = {"<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"}


def _category_classes() -> dict[str, str]:
    # regex class bodies for the Unicode letter, number and separator categories, as ranges so `re`
    # stays fast on long inputs. Python's own \s, \d and \w differ from llama.cpp's definitions.
    cps: dict[str, list[int]] = {"L": [], "N": [], "Z": []}
    for cp in range(0x323B0):  # one past the last letter; numbers and separators end earlier
        if (cat := unicodedata.category(chr(cp))[0]) in cps:
            cps[cat].append(cp)
    return {cat: "".join(_range(run) for run in _runs(v)) for cat, v in cps.items()}


def _runs(cps: list[int]) -> list[list[int]]:
    return [
        [cp for _, cp in g] for _, g in itertools.groupby(enumerate(cps), lambda e: e[1] - e[0])
    ]


def _range(run: list[int]) -> str:
    lo, hi = re.escape(chr(run[0])), re.escape(chr(run[-1]))
    return lo if len(run) == 1 else f"{lo}-{hi}"


def _llama3_pattern() -> re.Pattern[str]:
    # llama.cpp LLAMA_VOCAB_PRE_TYPE_LLAMA3, with contractions spelled out in ASCII: Python's (?i)
    # would also fold characters like U+017F into 's'.
    # (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|
    # \s*[\r\n]+|\s+(?!\S)|\s+
    c = _category_classes()
    L, N, S = c["L"], c["N"], r"\t\n\x0b\x0c\r\x85" + c["Z"]
    contractions = "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]"
    return re.compile(
        rf"{contractions}|[^\r\n{L}{N}]?[{L}]+|[{N}]{{1,3}}| ?[^{S}{L}{N}]+[\r\n]*"
        rf"|[{S}]*[\r\n]+|[{S}]+(?![^{S}])|[{S}]+"
    )


_PRE_TOKENIZERS = {"llama-bpe": _llama3_pattern}


class Tokenizer:
    """Encodes text to token ids and back.

    `encode(special=True)` turns control-token text such as `<|eot_id|>` into its token; leave it
    off for untrusted text. User-defined tokens are always matched, as in llama.cpp.
    """

    def __init__(self, metadata: dict[str, Any]):
        if (model := metadata.get("tokenizer.ggml.model")) != "gpt2":
            raise NotImplementedError(f"tokenizer model {model!r} is not supported")
        if (pre := metadata.get("tokenizer.ggml.pre")) not in _PRE_TOKENIZERS:
            raise NotImplementedError(f"pre-tokenizer {pre!r} is not supported")
        tokens: list[str] = metadata["tokenizer.ggml.tokens"]
        types: list[int] = metadata.get("tokenizer.ggml.token_type", [NORMAL] * len(tokens))
        self._pattern = _PRE_TOKENIZERS[pre]()
        self._vocab = {
            t: i for i, t in enumerate(tokens) if types[i] not in (CONTROL, USER_DEFINED)
        }
        if missing := [c for c in _BYTE_CHAR.values() if c not in self._vocab]:
            raise ValueError(f"vocab lacks {len(missing)} byte tokens, it is not byte-level BPE")
        merges = metadata.get("tokenizer.ggml.merges", [])
        self._ranks = {tuple(m.split(" ", 1)): r for r, m in enumerate(merges)}
        self._ignore_merges = pre == "llama-bpe"  # whole-word vocab hits skip BPE, as in tiktoken
        self._cache: dict[str, tuple[int, ...]] = {}

        self._bytes = [
            b"" if ty == CONTROL else _spell(t, ty) for t, ty in zip(tokens, types, strict=True)
        ]
        special = {t: i for i, t in enumerate(tokens) if types[i] in (CONTROL, USER_DEFINED)}
        user = {t: i for t, i in special.items() if types[i] == USER_DEFINED}
        self._special, self._user = special, user
        self._split_special, self._split_user = _alternation(special), _alternation(user)

        self.bos_id: int | None = metadata.get("tokenizer.ggml.bos_token_id")
        self.eos_id: int | None = metadata.get("tokenizer.ggml.eos_token_id")
        self.add_bos: bool = metadata.get("tokenizer.ggml.add_bos_token", pre == "llama-bpe")
        ids = (metadata.get(f"tokenizer.ggml.{k}_token_id") for k in ("eos", "eot", "eom"))
        eog_text = {special[t] for t in _EOG_TEXT & special.keys()}
        self.eog_ids: set[int] = {i for i in ids if i is not None} | eog_text

    @property
    def vocab_size(self) -> int:
        return len(self._bytes)

    def encode(self, text: str, bos: bool | None = None, special: bool = False) -> list[int]:
        ids = (
            [self.bos_id]
            if (self.add_bos if bos is None else bos) and self.bos_id is not None
            else []
        )
        split, table = (
            (self._split_special, self._special) if special else (self._split_user, self._user)
        )
        pos = 0
        for m in split.finditer(text) if split else ():
            ids += self._encode_ordinary(text[pos : m.start()])
            ids.append(table[m.group()])
            pos = m.end()
        return ids + self._encode_ordinary(text[pos:])

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._bytes[i] for i in ids).decode("utf-8", errors="replace")

    def stream(self) -> Callable[[int | None], str]:
        """Returns a decoder that maps one id at a time to the text it completes. None ends the
        text, flushing an incomplete character as decode() would."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        return lambda i: decoder.decode(b"" if i is None else self._bytes[i], final=i is None)

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for word in self._pattern.findall(text):
            out += self._bpe("".join(_BYTE_CHAR[b] for b in word.encode()))
        return out

    def _bpe(self, word: str) -> tuple[int, ...]:
        if (cached := self._cache.get(word)) is not None:
            return cached
        if self._ignore_merges and word in self._vocab:
            return (self._vocab[word],)
        parts, never = list(word), len(self._ranks)
        while len(parts) > 1:
            pairs = enumerate(itertools.pairwise(parts))
            rank, i = min((self._ranks.get(pair, never), i) for i, pair in pairs)
            if rank == never:
                break
            parts[i : i + 2] = [parts[i] + parts[i + 1]]
        if len(self._cache) > 1 << 16:
            self._cache.clear()
        ids = self._cache[word] = tuple(self._vocab[p] for p in parts)
        return ids


def _spell(token: str, ttype: int) -> bytes:
    if ttype == USER_DEFINED:
        return token.encode()
    try:
        return bytes(_CHAR_BYTE[c] for c in token)
    except KeyError:
        return token.encode()


def _alternation(tokens: dict[str, int]) -> re.Pattern[str] | None:
    # longest first, so a special token never matches a prefix of a longer one
    return (
        re.compile("|".join(map(re.escape, sorted(tokens, key=len, reverse=True))))
        if tokens
        else None
    )
