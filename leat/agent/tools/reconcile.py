"""Reconciling a bank statement with the books, as a tool: reconcile reads both tables, a CSV or a
workbook's, finds in each its dates, amounts and what each line says, by their columns' headings,
in German or English, as a German bank's export and a DATEV ledger's head them, or else by what
they hold, and matches each of the bank's lines with one of the books': first by a document's
number both name, as "RE-2026-0342", whose amounts may then differ,
and else by the same amount within DAYS days, the nearest first. What matched, what matched but in
its amount, and what is in one alone, it answers and saves as a workbook, a sheet of each.

The matching is the tool's, not the model's, so that it is the same each time, as an accountant
needs it to be; an amount is matched as it is, its sign aside, as a bank writes a payment as less
and the books as a debt.
"""

import datetime
import re
from dataclasses import dataclass
from typing import Any

from leat.agent import sheets
from leat.agent.tools import Context, Result, Tool

DAYS = 10  # days a payment may come after, or before, its entry, at most
SHOWN = 40  # lines of each part the model reads; the workbook holds every one
# the headings of each column's role, folded, in German and English, as their words begin
_ROLES = {
    "date": ("buchungstag", "buchungsdatum", "belegdatum", "datum", "valuta", "wertstellung",
             "date", "booking"),
    "debit": ("soll", "belastung", "ausgang", "debit"),
    "credit": ("haben", "gutschrift", "eingang", "credit"),
    "amount": ("betrag", "umsatz", "summe", "amount", "sum", "total"),
}  # fmt: skip
# a currency, as an amount may be written with it
_CURRENCY = re.compile(r"(?i)\s|eur\b|€|chf\b|£|gbp\b|\$|usd\b")
# a document's number: letters, then a separator or none, then digits, as "RE-2026-0342" or
# "INV2026/0042"; or digits with a separator in them, as "2026-0311"
_NUMBER = re.compile(r"\b(?=[\w/-]*\d)[A-Z]{1,6}[-/]?\d[\w/-]*\d\b|\b\d{2,}[-/]\d[\w/-]*\b")


@dataclass
class Line:
    """A line of a table: its row, its date, its amount, less if paid out, what it says, and its
    date as the table writes it."""

    row: int
    date: datetime.date | None
    amount: float
    said: str
    day: str = ""

    def numbers(self) -> set[str]:
        return {n.upper() for n in _NUMBER.findall(self.said.upper())}


def tools() -> list[Tool]:
    """reconcile, in its call's conversation's workspace."""
    return [
        Tool(
            "reconcile",
            "Reconcile a bank statement with the books, the ledger's entries, as a Kontoabstimmung "
            "or Bankabstimmung does, each a table, a CSV or a spreadsheet: which of the bank's "
            "lines match an entry, by a document's number both name or by the same amount a few "
            "days apart, which match but in their amount, and which are in one alone; saved as a "
            "spreadsheet too",
            {
                "type": "object",
                "properties": {
                    "bank": {"type": "string", "description": "the bank statement's file"},
                    "books": {"type": "string", "description": "the ledger's file"},
                    "name": {"type": "string", "description": "the spreadsheet's name"},
                },
                "required": ["bank", "books"],
            },
            lambda context, bank, books, name=None: reconcile(context, bank, books, name),
        )
    ]


def reconcile(context: Context, bank: str, books: str, name: str | None = None) -> Result:
    space = context.space()
    paid, entered = lines(sheets.read(space, bank), bank=True), lines(sheets.read(space, books))
    matched, differ, unpaid = match(paid, entered)
    banked = {id(b) for b, _ in matched + differ}
    unentered = [b for b in paid if id(b) not in banked]
    n, m = context.cite(f"file:{bank}", bank), context.cite(f"file:{books}", books)
    parts = [
        ("Amounts differ", ["Date", "Bank", "Books", "Difference", "Bank says", "Books say"],
         [[b.day, b.amount, e.amount, round(abs(b.amount) - abs(e.amount), 2), b.said,
           e.said] for b, e in differ]),
        ("In the bank alone", ["Date", "Amount", "Bank says"],
         [[b.day, b.amount, b.said] for b in unentered]),
        ("In the books alone", ["Date", "Amount", "Books say"],
         [[e.day, e.amount, e.said] for e in unpaid]),
        ("Matched", ["Bank date", "Books date", "Amount", "Bank says", "Books say"],
         [[b.day, e.day, b.amount, b.said, e.said] for b, e in matched]),
    ]  # fmt: skip
    said = [f"Of the bank's {len(paid)} lines [{n}] and the books' {len(entered)} [{m}]: "
            f"{len(matched)} matched, {len(differ)} matched but in their amount, {len(unentered)} "
            f"in the bank alone, and {len(unpaid)} in the books alone."]  # fmt: skip
    for title, heading, rows in parts[:3]:
        if rows:
            said.append(f"\n{title}:\n{_markdown([heading, *rows[:SHOWN]])}")
            if len(rows) > SHOWN:
                said.append(f"(The first {SHOWN} of {len(rows)}.)")
    saved = None
    try:
        saved = sheets.write(space, [(t, [h, *r]) for t, h, r in parts], name or "Reconciliation")
        said.append(f"\nSaved as {saved}, a sheet of each, the matched too.")
    except ValueError as e:
        said.append(f"\nIt could not be saved: {e}")
    sources = [{"n": n, "url": f"file:{bank}", "title": bank, "file": bank},
               {"n": m, "url": f"file:{books}", "title": books, "file": books}]  # fmt: skip
    info = {"matched": len(matched), "differ": len(differ), "bank alone": len(unentered),
            "books alone": len(unpaid), "results": sources}  # fmt: skip
    return Result("\n".join(said), info | ({"files": [saved]} if saved else {}))


def lines(rows: list[list[Any]], bank: bool = False) -> list[Line]:
    """A table's lines, of its rows after its heading, by the roles its columns have: those
    without a date or an amount left out, as a total's or a balance's carried. Of a debit and a
    credit, a `bank`'s debit is paid out, as its statement says, and the books' paid in, as their
    bank account's is."""
    if not rows:
        return []
    heading, body = [_fold(str(h or "")) for h in rows[0]], rows[1:]
    roles = _roles(heading, body)
    found = []
    for i, row in enumerate(body, 2):
        written = str(_cell(row, roles, "date") or "").strip()
        date = _date(written)
        if "amount" in roles:
            amount = _amount(_cell(row, roles, "amount"))
        else:
            debit, credit = (
                _amount(_cell(row, roles, "debit")),
                _amount(_cell(row, roles, "credit")),
            )
            amount = (debit or 0.0) - (credit or 0.0) if debit or credit else None
            amount = -amount if bank and amount else amount
        if date is None or not amount:
            continue
        said = " · ".join(str(row[c]) for c in roles.get("texts", []) if c < len(row) and row[c])
        day = written[:10] if re.match(r"\d{4}-\d\d-\d\dT", written) else written  # a workbook's
        found.append(Line(i, date, amount, said, day))
    return found


Pair = tuple[Line, Line]  # a bank's line, and the books' it matched


def match(paid: list[Line], entered: list[Line]) -> tuple[list[Pair], list[Pair], list[Line]]:
    """The bank's lines matched with the books': those that match, those that match by a number
    both name but differ in their amount, as pairs, and the books' lines left."""
    left, matched, differ = list(entered), list[Pair](), list[Pair]()
    for b in paid:  # by a number both name
        numbers = b.numbers()
        if numbers and (e := next((e for e in left if numbers & e.numbers()), None)):
            left.remove(e)
            (matched if _same(b, e) else differ).append((b, e))
    paired = {id(b) for b, _ in matched + differ}
    for b in (b for b in paid if id(b) not in paired):  # by amount, the nearest in days
        near = [e for e in left if _same(b, e) and _apart(b, e) <= DAYS]
        if near:
            e = min(near, key=lambda e: _apart(b, e))
            left.remove(e)
            matched.append((b, e))
    order = {id(b): i for i, b in enumerate(paid)}
    matched.sort(key=lambda pair: order[id(pair[0])])
    return matched, differ, left


def _cell(row: list[Any], roles: dict[str, Any], role: str) -> Any:
    # a row's cell of a role, or None if the table has no such column, or the row no such cell
    return row[roles[role]] if role in roles and roles[role] < len(row) else None


def _roles(heading: list[str], body: list[list[Any]]) -> dict[str, Any]:
    # each role's column, by its heading's words, or a date's and an amount's by what the column
    # holds; and "texts", the columns of what a line says, which hold neither dates nor amounts,
    # nor the same in every line, as an account's number or a currency
    roles: dict[str, Any] = {}
    for role, words in _ROLES.items():
        free = (c for c, h in enumerate(heading) if h.startswith(words) and c not in roles.values())
        if (column := next(free, None)) is not None:
            roles[role] = column
    if "date" not in roles:
        roles["date"] = _most(body, lambda v: _date(v) is not None, set(roles.values()))
    if "amount" not in roles and "debit" not in roles:
        roles["amount"] = _most(body, lambda v: _amount(v) is not None, set(roles.values()))
    taken = set(roles.values())
    roles["texts"] = [c for c in range(len(heading)) if c not in taken
                      and _most(body, _text, set(), c) > len(body) // 2
                      and _varies(body, c)]  # fmt: skip
    return roles


def _varies(body: list[list[Any]], column: int) -> bool:
    # whether a column's cells differ, of a table of a few lines at least
    return len(body) < 3 or len({str(row[column]) for row in body if column < len(row)}) > 1


def _text(value: Any) -> bool:
    # whether a cell is text: neither a date nor an amount
    return isinstance(value, str) and bool(value.strip()) and not (_date(value) or _amount(value))


def _most(body: list[list[Any]], fits: Any, taken: set[int], column: int | None = None) -> int:
    # the column most of whose cells fit, of those not taken; or of `column`, how many of its
    # cells fit
    width = max((len(row) for row in body), default=0)
    scores = [(sum(1 for r in body if c < len(r) and fits(r[c])), c) for c in range(width)]
    if column is not None:
        return scores[column][0] if column < width else 0
    return max((s for s in scores if s[1] not in taken), default=(0, 0))[1]


def _date(value: Any) -> datetime.date | None:
    # a cell's date: of ISO, as a workbook's, or a day, a month and a year, as German banks write
    text = str(value or "").strip()[:10]
    for form in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%y"):
        try:
            return datetime.datetime.strptime(text, form).date()
        except ValueError:
            pass
    return None


def _amount(value: Any) -> float | None:
    # a cell's amount: a number as it is, or text as German, much of Europe or English writes it,
    # 1.234,56 or 1,234.56 or (1.234,56), its currency aside
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = _CURRENCY.sub("", str(value)).replace("−", "-")
    if not re.fullmatch(r"[-(]?[\d.,]*\d[\d.,]*\)?", text):
        return None
    negative, digits = text.startswith(("-", "(")), text.strip("-()")
    if "," in digits and "." in digits:  # the last of them is the decimal one
        decimal = "," if digits.rindex(",") > digits.rindex(".") else "."
    elif "," in digits:
        decimal = "," if re.search(r",\d{1,2}$", digits) else ""
    else:
        decimal = "." if re.search(r"\.\d{1,2}$", digits) and digits.count(".") == 1 else ""
    whole, _, part = digits.rpartition(decimal) if decimal else (digits, "", "")
    number = float(re.sub(r"[.,]", "", whole) + (f".{part}" if decimal else ""))
    return -number if negative else number


def _same(b: Line, e: Line) -> bool:
    return abs(abs(b.amount) - abs(e.amount)) < 0.005


def _apart(b: Line, e: Line) -> int:
    return abs((b.date - e.date).days) if b.date and e.date else DAYS + 1


def _fold(text: str) -> str:
    # a heading as its role's words are written: lowercase, umlauts and ß spelled out plainly
    folded = text.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    return " ".join(folded.split())


def _markdown(rows: list[list[Any]]) -> str:
    # rows as a markdown table, the first its heading
    def cell(value: Any) -> str:
        text = f"{value:,.2f}" if isinstance(value, float) else str(value)
        return " ".join(text.split()).replace("|", "\\|") or " "

    lines = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join([lines[0], "|" + " --- |" * len(rows[0]), *lines[1:]])
