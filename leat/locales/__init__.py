"""The words of Leat's app, its agent's answers and its box's screen in each language but English,
in which they are written: a table of each, `de.json`, of each English text to its translation,
a value in it written {name} in both. The app reads its language's table as the agent serves it,
and the agent and the screen theirs here, so that all three say the same of the same.

A text the agent makes of values, as an error or a warning, is translated whole: as the table has
it, or as a text of values it has, its values put in the translation's places.
"""

import functools
import json
import re
from pathlib import Path

LANGUAGES = ("en", "de")  # English first, as every text is written in it
HERE = Path(__file__).parent


def t(text: str, language: str | None, **values: object) -> str:
    """A text in a language, of a table's, its values put in its places; in English, or of a text
    the table has not, as it is."""
    said = table(language).get(text, text)
    return said.format(**values) if values else said


def translated(text: str, language: str | None) -> str:
    """A text the agent made, of values or not, in a language: as the table has it, or as a text
    of values it has, as "1 GB are left" of "{free} GB are left"; else as it is."""
    words = table(language)
    if text in words:
        return words[text]
    for pattern, translation in _patterns(language or "en"):
        if found := pattern.fullmatch(text):
            return translation.format(**found.groupdict())
    return text


@functools.cache
def table(language: str | None) -> dict[str, str]:
    """A language's table, of each English text to its translation; English's, and none's, empty."""
    if language not in LANGUAGES or language == "en":
        return {}
    return json.loads((HERE / f"{language}.json").read_text(encoding="utf-8"))


@functools.cache
def _patterns(language: str) -> list[tuple[re.Pattern[str], str]]:
    # each text of values a language's table has, as a pattern that finds its values, the longest
    # first, as a text's more of its words are surer; but of one of a word alone besides its
    # values, as "{some} and {last}", which would find a text of any "and"
    found = []
    for text, translation in table(language).items():
        parts = re.split(r"\{(\w+)\}", text)
        if len(parts) > 1 and len(re.findall(r"[^\W\d]{2,}", "".join(parts[::2]))) >= 2:
            pattern = "".join(re.escape(p) if i % 2 == 0 else f"(?P<{p}>.+?)"
                              for i, p in enumerate(parts))  # fmt: skip
            found.append((re.compile(pattern, re.S), translation))
    return sorted(found, key=lambda f: -len(f[0].pattern))
