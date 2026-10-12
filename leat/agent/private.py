"""What of an office's work must not leave the box in what Leat asks of the internet, as a web
search's query, into which the model may write what it read: the names of its clients and of the
people of their matters, its own people's, and any e-mail address or IBAN.

A name is found as offices write one: after a title or a role, as "Herr Hofmann", "Frau Dr. Anna
Weber" or "Mandantin Yılmaz", and before a company's form, as "Hofmann Textil GmbH" or "Müller &
Co. KG", of the conversation's files and what its person wrote; before a company's form, but for a
word written after an article elsewhere, as "die Kündigung", a noun's, which German capitalizes as a
name, and which a heading, as "Kündigung Müller GmbH", puts where a name would be; after a title,
always, as "Herr Richter" is one, whatever "der Richter" is. And whole, as the box's people's names
and its office's. A query is checked by its words: each the files hold is looked for there, in their
passages that hold it, as the index finds them, those of a title or a company's form first.
"""

import re

from leat.agent.index import COMMON, fold

# the words a name follows, as they are written in lowercase, a dot after them or not
_TITLES = """
herr herrn frau dr prof mandant mandantin mandanten kläger klägerin beklagte beklagter beklagten
antragsteller antragstellerin erblasser erblasserin firma kanzlei notar notarin rechtsanwalt
rechtsanwältin mr mrs ms miss client messrs
"""
TITLES = frozenset(_TITLES.split())
# the forms of companies a name comes before
_FORMS = "gmbh mbh ag kg ohg ug gbr se kgaa partg ltd llc llp inc plc corp co"
FORMS = frozenset(_FORMS.split())
# the words a noun follows, and a name, but for a company's, does not
_ARTICLES = """
der die das den dem des ein eine einen einem einer eines kein keine keinen keinem keiner jede jeder
jedes jeden diese dieser dieses diesen zur zum im am vom beim ins the a an this that these those
every each no
"""
ARTICLES = frozenset(_ARTICLES.split())
NAMED = 2  # words of a name at most, as "Anna Weber" or "Hofmann Textil"
_WORD = re.compile(r"\w+(?:[-']\w+)*")
_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")


def names(text: str) -> set[str]:
    """The words of the names a text gives, as the index folds them: those after a title or a
    role; and those before a company's form, but for those it writes after an article, as nouns
    are, elsewhere than before a company's form."""
    words = _WORD.findall(text)
    folded = [fold(w) for w in words]
    titled, formed, nouns = set[str](), set[str](), set[str]()
    for i, word in enumerate(folded):
        if word in TITLES:
            after = [w for w in words[i + 1 : i + 1 + NAMED + 1] if fold(w) not in TITLES]
            titled |= _named(after[:NAMED])
        elif word in FORMS and words[i][0].isupper():
            before = [w for w in reversed(words[max(0, i - NAMED - 1) : i]) if fold(w) not in FORMS]
            formed |= _named(before[:NAMED])
        elif i and folded[i - 1] in ARTICLES and not FORMS & set(folded[i + 1 : i + 1 + NAMED]):
            nouns.add(word)
    return titled | (formed - nouns)


def held(query: str, named: set[str], wholes: list[str]) -> list[str]:
    """What of a query must not leave the box, as it is written there: its words of the names
    `named`, the whole names `wholes` it holds, and its e-mail addresses and IBANs."""
    addresses = _ADDRESS.findall(query) + _IBAN.findall(query)
    for address in addresses:  # and none of their words again
        query = query.replace(address, " ")
    words, kept = _WORD.findall(query), []
    folded = [fold(w) for w in words]
    for whole in wholes:
        parts = [fold(w) for w in _WORD.findall(whole)]
        n = len(parts)
        if n and any(folded[i : i + n] == parts for i in range(len(folded) - n + 1)):
            kept.append(whole)
    kept += [w for w, f in zip(words, folded, strict=True) if f in named and w not in kept]
    return kept + addresses


def asked(query: str) -> list[str]:
    """The words of a query that may be names, to look for in the files: of three letters at
    least, and none of every sentence's."""
    return [w for w in dict.fromkeys(fold(w) for w in _WORD.findall(query))
            if len(w) >= 3 and w not in COMMON and not w.isdigit()]  # fmt: skip


def _named(words: list[str]) -> set[str]:
    # the words of a name, of those that may be one: each capitalized, as a name is, until one
    # is not, and none of every sentence's
    found = set()
    for word in words:
        if not word[0].isupper() or fold(word) in COMMON:
            break
        if len(word) >= 3:
            found.add(fold(word))
    return found
