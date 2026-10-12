"""A reply's quotes, each checked by code against the sources it cites, as people fear an answer
made up: one found in its source word for word, but for case, spacing and the forms of quotation
marks and dashes, is marked found, and one that is not is flagged, for its reader to check.

A quote is text in quotation marks, of QUOTED words at least, as a clause or a sentence quoted is,
cited at once after it, as `“Die Kündigungsfrist beträgt drei Monate.” [1]`, by the numbers before
the sentence's end; or a block quoted, Markdown's `> …`, cited in it or right after it, or by none,
as models quote so, which is checked against every source the conversation read, as is one in
quotation marks of no citation of UNCITED words at least. One left out of
with an ellipsis, as `“Die Miete … ist monatlich fällig”`, is found if its parts are, in their
order.
"""

import re
import unicodedata

QUOTED = 4  # words of a quote at least, as a term in quotation marks is none
UNCITED = 8  # of one in quotation marks cited by none, as a sentence quoted is, checked against all
AFTER = 80  # characters after a quote its citations may be in, before its sentence ends
# a quote, between quotation marks as English, German and French write them
# a quote, between quotation marks as English, German and French write them, and as models close
# a German one, of a plain mark
_QUOTE = re.compile(r'"([^"\n]+)"|“([^”"\n]+)[”"]|„([^“”"\n]+)[“”"]|«([^»\n]+)»|»([^«\n]+)«')
_BLOCK = re.compile(r"(?:^[ \t]*>[^\n]*(?:\n|$))+", re.M)  # Markdown's quoted block, its lines
_CITES = re.compile(r"\[(\d{1,3})\]|【(\d{1,3})")
_CITE = r"(?:\[\d{1,3}\]|【\d{1,3}[^】\n]*】)"  # [1], or as gpt-oss writes one, 【1†L1-L3】
_CLUSTER = re.compile(rf"{_CITE}(?:[\s,]*{_CITE})*")
_ELLIPSIS = re.compile(r"\s*(?:\[…\]|\[\.\.\.\]|\(…\)|\(\.\.\.\)|…|\.\.\.)\s*")
_MARKS = str.maketrans({"’": "'", "‘": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "«": '"',
                        "»": '"', "–": "-", "—": "-", "\u00ad": None})  # fmt: skip


def quoted(text: str) -> list[tuple[str, list[int]]]:
    """A reply's quotes, each of QUOTED words at least, and the numbers of the sources cited right
    after it: of a block quoted, its own or those right after it, or none, as it is checked
    against every source; of one in quotation marks, those right after it, or none of one of
    UNCITED words at least, as a sentence quoted is."""
    found = []
    for block in _BLOCK.finditer(text):
        lines = [re.sub(r"^[ \t]*>[ \t]?", "", line) for line in block[0].splitlines()]
        inside = " ".join(lines)
        following = text[block.end() : block.end() + AFTER].lstrip()
        cluster = _CLUSTER.search(inside) or _CLUSTER.match(following)
        cited = [int(a or b) for a, b in _CITES.findall(cluster[0])] if cluster else []
        quote = _CLUSTER.sub("", inside).strip(" *_\"“”„«»'")
        if len(quote.split()) >= QUOTED:
            found.append((quote, cited))
    text = _BLOCK.sub("\n", text)  # and its quotation marks counted no more
    for match in _QUOTE.finditer(text):
        quote = next(g for g in match.groups() if g is not None).strip()
        after = text[match.end() : match.end() + AFTER]
        after = re.sub(r"^[.,;:]\s*(?=\[\d|【)", "", after)  # as `“…”. [1]`
        sentence = re.split(r"(?<=[.!?])\s|\n|[\"“„«»]", after, maxsplit=1)[0]
        cluster = _CLUSTER.search(sentence)  # the first citations, as [1][2] or [1], [2]
        cited = [int(a or b) for a, b in _CITES.findall(cluster[0])] if cluster else []
        if len(quote.split()) >= (QUOTED if cited else UNCITED):
            found.append((quote, cited))
    return found


def found(quote: str, source: str) -> bool:
    """Whether a quote is in a source word for word, but for case, spacing and the forms of
    quotation marks and dashes, its parts in order, of one left out of with an ellipsis."""
    text, at = _plain(source), 0
    for part in _ELLIPSIS.split(quote):
        if not (part := _plain(part).strip(" .,;:")):
            continue
        if (at := text.find(part, at)) < 0:
            return False
        at += len(part)
    return True


def _plain(text: str) -> str:
    # text as quotes are matched in it: its letters as Unicode composes them, in lowercase, its
    # dashes plain, a word broken over lines whole, spaces one, emphasis and quotation marks gone
    text = unicodedata.normalize("NFKC", text).translate(_MARKS).casefold()
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r'[*_`"]', "", text)  # and quotation marks, as a block quotes its own
    return re.sub(r"\s+", " ", text)
