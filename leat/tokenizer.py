"""Tokenizers built from GGUF metadata, matching llama.cpp's llama-vocab.cpp: BPE, byte-level as
GPT-2's or SentencePiece-style over characters as Gemma 4's, and SentencePiece's own, as Mistral's,
which merges the pair whose token scores highest."""

import codecs
import functools
import heapq
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

NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
# llama.cpp treats these as end of generation in addition to the eos/eot/eom ids, and as control
# tokens whatever their type
_EOG_TEXT = {"<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>",
             "<|end|>", "<|return|>", "<|call|>", "<end_of_turn>", "<eos>", "<turn|>",
             "<|tool_response>", "<EOT>", "_<EOT>"}  # fmt: skip
# gpt-oss's harmony format: tokens that decode as text, as llama.cpp has them, and its ends of a
# message that do not end a generation, as llama.cpp's workaround has it
_HARMONY = {"<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"}
SPACE = "\u2581"  # how SentencePiece spells a space


def _category_classes() -> dict[str, str]:
    # regex class bodies for the Unicode letter, number and separator categories, as ranges so `re`
    # stays fast on long inputs. Python's own \s, \d and \w differ from llama.cpp's definitions.
    # U and W are the letters but a-z and A-Z: those GPT-4o's pre-tokenizer takes for upper and
    # lower case, as llama.cpp rewrites its classes
    cps: dict[str, list[int]] = {"L": [], "N": [], "Z": []}
    for cp in range(0x323B0):  # one past the last letter; numbers and separators end earlier
        if (cat := unicodedata.category(chr(cp))[0]) in cps:
            cps[cat].append(cp)
    cps["U"] = [cp for cp in cps["L"] if not 0x61 <= cp <= 0x7A]
    cps["W"] = [cp for cp in cps["L"] if not 0x41 <= cp <= 0x5A]
    return {cat: "".join(_range(run) for run in _runs(v)) for cat, v in cps.items()}


def _runs(cps: list[int]) -> list[list[int]]:
    return [
        [cp for _, cp in g] for _, g in itertools.groupby(enumerate(cps), lambda e: e[1] - e[0])
    ]


def _range(run: list[int]) -> str:
    lo, hi = re.escape(chr(run[0])), re.escape(chr(run[-1]))
    return lo if len(run) == 1 else f"{lo}-{hi}"


@functools.cache
def _patterns(kind: str) -> tuple[re.Pattern[str], ...]:
    # llama.cpp's pre-tokenizer regexes for a kind of pre-tokenizer, each applied to the pieces of
    # the last. Contractions are spelled out in ASCII, as Python's (?i) would also fold characters
    # like U+017F into 's'.
    c = _category_classes()
    L, N, U, W, S = c["L"], c["N"], c["U"], c["W"], r"\t\n\x0b\x0c\r\x85" + c["Z"]
    contractions = "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]"
    # GPT-2's: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
    gpt2 = rf"'s|'t|'re|'ve|'m|'ll|'d| ?[{L}]+| ?[{N}]+| ?[^{S}{L}{N}]+|[{S}]+(?![^{S}])"
    if kind in ("llama3", "qwen2"):  # QWEN2's splits numbers into digits, not runs of up to 3
        # (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
        # ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
        digits = "{1,3}" if kind == "llama3" else ""
        regexes = [
            rf"{contractions}|[^\r\n{L}{N}]?[{L}]+|[{N}]{digits}| ?[^{S}{L}{N}]+[\r\n]*"
            rf"|[{S}]*[\r\n]+|[{S}]+(?![^{S}])|[{S}]+"
        ]
    elif kind == "gpt2":
        regexes = [gpt2]
    elif kind == "digits-gpt2":  # StarCoder's and others': single digits first
        regexes = [rf"[{N}]", gpt2]
    elif kind in ("gpt-4o", "tekken"):  # words that run upper then lower case; Tekken's without
        # contractions, and with single digits
        tail = f"(?:{contractions})?" if kind == "gpt-4o" else ""
        digits = "{1,3}" if kind == "gpt-4o" else ""
        regexes = [
            rf"[^\r\n{L}{N}]?[{U}]*[{W}]+{tail}|[^\r\n{L}{N}]?[{U}]+[{W}]*{tail}"
            rf"|[{N}]{digits}| ?[^{S}{L}{N}]+[\r\n/]*|[{S}]*[\r\n]+|[{S}]+(?![^{S}])|[{S}]+"
        ]
    else:
        raise ValueError(kind)
    return tuple(re.compile(r) for r in regexes)


# tokenizer.ggml.pre of byte-level BPE vocabs: the kind of pre-tokenizer, whether a word the vocab
# holds skips merging, as in tiktoken, and whether a BOS starts text by default, as llama.cpp has
# them in llama-vocab.cpp
_PRE_TOKENIZERS = {
    **dict.fromkeys(("llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral",
                     "midm-2.0", "lfm2"), ("llama3", True, True)),
    **dict.fromkeys(("dbrx", "smaug-bpe", "chatglm-bpe"), ("llama3", False, False)),
    **dict.fromkeys(("glm4", "glm5"), ("llama3", True, False)),
    **dict.fromkeys(("qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "megrez", "stablelm2",
                     "hunyuan", "solar-open", "grok-2"), ("qwen2", False, False)),
    **dict.fromkeys(("gpt-2", "phi-2", "jina-es", "jina-de", "jina-v2-es", "jina-v2-de",
                     "gigachat", "a.x-4.0", "mellum", "exaone4", "mpt", "olmo", "jais",
                     "trillion", "granite-docling"), ("gpt2", False, False)),
    **dict.fromkeys(("starcoder", "refact", "command-r", "smollm", "codeshell", "exaone",
                     "minerva-7b", "mellum2"), ("digits-gpt2", False, False)),
    **dict.fromkeys(("gpt-4o", "llama4", "kanana2", "talkie", "minimax-m2"),
                    ("gpt-4o", False, False)),
    "tekken": ("tekken", True, True),
}  # fmt: skip


def _split(patterns: tuple[re.Pattern[str], ...], text: str) -> list[str]:
    # text into words: each regex splits each piece of the last into its matches and the text
    # between them, which llama.cpp keeps as words of their own
    pieces = [text]
    for pattern in patterns:
        out = []
        for piece in pieces:
            at = 0
            for m in pattern.finditer(piece):
                if m.start() > at:
                    out.append(piece[at : m.start()])
                out.append(m.group())
                at = m.end()
            if at < len(piece):
                out.append(piece[at:])
        pieces = out
    return pieces


class Tokenizer:
    """Encodes text to token ids and back.

    `encode(special=True)` turns control-token text such as `<|eot_id|>` into its token; leave it
    off for untrusted text. User-defined tokens are always matched, as in llama.cpp.
    """

    def __init__(self, metadata: dict[str, Any]):
        model, pre = metadata.get("tokenizer.ggml.model"), metadata.get("tokenizer.ggml.pre")
        tokens: list[str] = metadata["tokenizer.ggml.tokens"]
        types = list(metadata.get("tokenizer.ggml.token_type", [NORMAL] * len(tokens)))
        # SentencePiece's: a fragment between special tokens is one word, spaces spelled SPACE
        self._scores: dict[str, float] | None = None
        self._space_prefix, ignore_merges, default_bos = False, False, model == "llama"
        self._patterns: tuple[re.Pattern[str], ...]
        if model == "llama":
            self._scores = dict(zip(tokens, metadata["tokenizer.ggml.scores"], strict=True))
            self._space_prefix = metadata.get("tokenizer.ggml.add_space_prefix", True)
            self._patterns, self._byte_level = (re.compile(r".+", re.S),), False
        elif model == "gemma4":  # spaces as SPACE, whole lines as words, and no byte-level mapping
            self._patterns, self._byte_level = (re.compile(r"[^\n]+|\n+"),), False
        elif model == "gpt2" and pre in _PRE_TOKENIZERS:
            kind, ignore_merges, default_bos = _PRE_TOKENIZERS[pre]
            self._patterns, self._byte_level = _patterns(kind), True
        else:
            raise NotImplementedError(
                f"tokenizer {model!r} with pre-tokenizer {pre!r} is not supported"
            )
        self._ignore_merges = ignore_merges  # whole-word vocab hits skip BPE
        eog = {i for i, t in enumerate(tokens) if t in _EOG_TEXT}
        harmony = {"<|return|>", "<|call|>", "<|end|>"} <= set(tokens)
        for i, t in enumerate(tokens):  # as llama.cpp overrides them
            if t in _HARMONY or (harmony and t == "<|end|>"):
                types[i] = USER_DEFINED
                eog.discard(i)
            elif i in eog:
                types[i] = CONTROL
        # tokens that text spells out, all of them for SentencePiece as in llama.cpp
        hidden = () if self._scores is not None else (CONTROL, USER_DEFINED)
        self._vocab = {t: i for i, t in enumerate(tokens) if types[i] not in hidden}
        # some byte-level vocabs lack a few bytes' tokens, as MPT's and StarCoder's control
        # characters: llama.cpp drops those bytes
        missing = [c for c in _BYTE_CHAR.values() if c not in self._vocab and self._byte_level]
        if len(missing) > 128:
            raise ValueError(f"vocab lacks {len(missing)} byte tokens, it is not byte-level BPE")
        merges = metadata.get("tokenizer.ggml.merges", [])
        self._ranks = {_pair(m): r for r, m in enumerate(merges)}
        self._cache: dict[str, tuple[int, ...]] = {}

        self._bytes = [_spell(t, ty, self._byte_level) for t, ty in zip(tokens, types, strict=True)]
        special = {t: i for i, t in enumerate(tokens) if types[i] in (CONTROL, USER_DEFINED)}
        user = {t: i for t, i in special.items() if types[i] == USER_DEFINED}
        self._special, self._user = special, user
        self._split_special, self._split_user = _alternation(special), _alternation(user)

        self.bos_id: int | None = metadata.get("tokenizer.ggml.bos_token_id")
        self.eos_id: int | None = metadata.get("tokenizer.ggml.eos_token_id")
        if pre in ("chatglm-bpe", "glm4", "glm5"):  # llama.cpp drops their BOS
            self.bos_id = None
        self.add_bos: bool = metadata.get("tokenizer.ggml.add_bos_token", default_bos)
        ids = (metadata.get(f"tokenizer.ggml.{k}_token_id") for k in ("eos", "eot", "eom"))
        self.eog_ids: set[int] = {i for i in ids if i is not None} | eog

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
        pieces = [self._bytes[i] for i in ids]
        if pieces and ids[0] != self.bos_id:
            pieces[0] = self._lstrip(pieces[0])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def stream(self) -> Callable[[int | None], str]:
        """Returns a decoder that maps one id at a time to the text it completes. None ends the
        text, flushing an incomplete character as decode() would."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pieces = itertools.count()

        def step(i: int | None) -> str:
            if i is None:
                return decoder.decode(b"", final=True)
            piece = self._bytes[i]
            return decoder.decode(self._lstrip(piece) if next(pieces) == 0 else piece)

        return step

    def _lstrip(self, piece: bytes) -> bytes:
        # the space SentencePiece puts before text, off its first piece, as llama.cpp does unless
        # the text starts with BOS
        return piece.removeprefix(b" ") if self._space_prefix else piece

    def _encode_ordinary(self, text: str) -> list[int]:
        # text between special tokens, which SentencePiece starts with a space, as llama.cpp
        if self._space_prefix and text:
            text = " " + text
        out: list[int] = []
        for word in _split(self._patterns, text if self._byte_level else text.replace(" ", SPACE)):
            if self._byte_level:
                word = "".join(_BYTE_CHAR[b] for b in word.encode())
            out += self._bpe(word)
        return out

    def _bpe(self, word: str) -> tuple[int, ...]:
        if (cached := self._cache.get(word)) is not None:
            return cached
        # whole-word vocab hits skip merging with llama-bpe, and runs of newlines with gemma4
        if word in self._vocab and (self._ignore_merges or not word.strip("\n")):
            return (self._vocab[word],)
        ids = tuple(i for part in _merge(word, self._priority) for i in self._ids(part))
        if len(self._cache) > 1 << 16:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def _priority(self, left: str, right: str) -> float | None:
        # which pair merges first, the lowest; None for pairs that never merge
        if self._scores is not None:
            score = self._scores.get(left + right)
            return None if score is None else -score
        return self._ranks.get((left, right))

    def _ids(self, part: str) -> list[int]:
        # a merged part's token, or else its bytes': as <0xXX> tokens where they are not byte-level
        if part in self._vocab:
            return [self._vocab[part]]
        if self._byte_level:
            return [self._vocab[c] for c in part if c in self._vocab]
        return [self._vocab[f"<0x{b:02X}>"] for b in part.encode()]


def _pair(merge: str) -> tuple[str, str]:
    # the two parts of a merge, split at its first space after the first character, as llama.cpp
    i = merge.index(" ", 1)
    return merge[:i], merge[i + 1 :]


def _merge(word: str, priority: Callable[[str, str], float | None]) -> list[str]:
    # merges the adjacent pair of lowest priority, the leftmost on ties, until no pair has one. A
    # heap of candidate pairs, as in llama.cpp, keeps long words fast: Gemma's are whole lines.
    parts, after = list(word), [*range(1, len(word)), -1]
    before, heap = [*range(-1, len(word) - 1)], list[tuple[float, int, int, str]]()

    def push(i: int) -> None:  # the pair i begins, if it merges
        if i < 0 or (j := after[i]) < 0:
            return
        if (rank := priority(parts[i], parts[j])) is not None:
            heapq.heappush(heap, (rank, i, j, parts[i] + parts[j]))

    for i in range(len(parts) - 1):
        push(i)
    while heap:
        _, i, j, merged = heapq.heappop(heap)
        if after[i] != j or parts[i] + parts[j] != merged:  # a stale pair
            continue
        parts[i], parts[j], after[i] = merged, "", after[j]
        if after[j] >= 0:
            before[after[j]] = i
        push(before[i])
        push(i)
    return [p for p in parts if p]


def _spell(token: str, ttype: int, byte_level: bool) -> bytes:
    # the bytes a token decodes to: none for control and unused tokens, as in llama.cpp
    if ttype in (CONTROL, UNUSED):
        return b""
    if ttype == USER_DEFINED:
        return token.encode()
    if ttype == BYTE:
        return bytes([int(token[3:5], 16)])  # <0xXX>
    if not byte_level:
        return token.replace(SPACE, " ").encode()
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
