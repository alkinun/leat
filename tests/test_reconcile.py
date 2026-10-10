"""Reconciling a bank statement with the books: amounts and dates as German banks write them,
columns found by their headings or what they hold, lines matched by a number both name or by their
amount a few days apart, and the whole, of a bank's CSV and a ledger's workbook, read and saved in
the sandbox, which needs LEAT_SANDBOX, as tests/test_files.py's documents do."""

import datetime
import os
from pathlib import Path

import pytest

from leat.agent.tools import Context, files, reconcile
from leat.agent.tools.reconcile import _amount, _date, lines, match
from leat.agent.workspace import Workspace
from tests.test_files import documents as needs_sandbox


def test_amounts():
    # as German, much of Europe and English write them, a currency aside
    for said, amount in [("1.234,56", 1234.56), ("-46.200,00 EUR", -46200.0), ("1,234.56", 1234.56),
                         ("(1.234,56)", -1234.56), ("12.400", 12400.0), ("3,5", 3.5), (900, 900.0),
                         ("19,99 €", 19.99), ("−7,50", -7.5)]:  # fmt: skip
        assert _amount(said) == amount, said
    for said in ("320 Lieferanten", "RE-2026-0342", "", None, True, "DE89370400440532013000"):
        assert _amount(said) is None, said
    assert _date("05.03.2026") == _date("2026-03-05T00:00:00") == datetime.date(2026, 3, 5)
    assert _date("Miete") is None


def test_columns():
    # each role's column by its German heading; an amount of debit and credit, as a ledger's
    rows = [["Belegdatum", "Belegnr.", "Konto", "Buchungstext", "Soll", "Haben"],
            ["02.03.2026", "RE-2026-0342", "1600", "Müller Bürobedarf", None, "1.190,00"],
            ["05.03.2026", "MIETE-03", "4210", "Miete März", "2.500,00", None],
            ["", "", "", "Summe", "2.500,00", "1.190,00"]]  # fmt: skip
    found = lines(rows)
    march = [datetime.date(2026, 3, day) for day in (2, 5)]
    assert [(line.row, line.date, line.amount) for line in found] == [
        (2, march[0], -1190.0), (3, march[1], 2500.0)]  # fmt: skip
    assert found[0].said == "RE-2026-0342 · Müller Bürobedarf" and found[0].numbers() == {
        "RE-2026-0342"}  # fmt: skip


def test_match():
    # by a number both name first, whose amounts may differ; then by amount, the nearest within
    # DAYS days; the rest in one alone
    bank = lines([["Buchungstag", "Verwendungszweck", "Betrag"],
                  ["05.03.2026", "Müller Bürobedarf RE-2026-0342", "-1.190,00"],
                  ["09.03.2026", "Schmidt IT RE-2026-0350", "-2.000,00"],
                  ["06.03.2026", "Miete März", "-2.500,00"],
                  ["30.03.2026", "Gebühren", "-12,90"],
                  ["28.03.2026", "Zahlung Kunde Weber", "4.760,00"]])  # fmt: skip
    books = lines([["Datum", "Text", "Betrag"],
                   ["02.03.2026", "RE-2026-0342 Müller", "1.190,00"],
                   ["04.03.2026", "RE-2026-0350 Schmidt IT", "2.380,00"],
                   ["05.03.2026", "Miete", "2.500,00"],
                   ["01.03.2026", "Ausgangsrechnung Weber AR-17", "4.760,00"],
                   ["20.03.2026", "Telefon", "59,00"]])  # fmt: skip
    matched, differ, left = match(bank, books)
    assert [(b.said, e.said) for b, e in matched] == [
        ("Müller Bürobedarf RE-2026-0342", "RE-2026-0342 Müller"),
        ("Miete März", "Miete"),
    ]
    assert [(b.amount, e.amount) for b, e in differ] == [(-2000.0, 2380.0)]
    assert [e.said for e in left] == ["Ausgangsrechnung Weber AR-17", "Telefon"]  # 27 days apart


SANDBOX = os.environ.get("LEAT_SANDBOX")


@needs_sandbox
def test_reconcile(tmp_path):
    # a German bank's CSV, of Windows' Western European, and a ledger's workbook: what matched,
    # what did not, said and saved, a sheet of each
    space = Workspace(tmp_path / "workspace", Path(SANDBOX) if SANDBOX else None)
    iban = "DE89370400440532013000"
    statement = ("Auftragskonto;Buchungstag;Verwendungszweck;Begünstigter/Zahlungspflichtiger;"
                 "Betrag;Waehrung\n"
                 f"{iban};05.03.2026;RE-2026-0342;Müller Bürobedarf;-1.190,00;EUR\n"
                 f"{iban};06.03.2026;Miete März;Hausverwaltung Kühn;-2.500,00;EUR\n"
                 f"{iban};30.03.2026;Kontoführung;Sparkasse;-12,90;EUR\n")  # fmt: skip
    space.path("Kontoauszug.csv").write_bytes(statement.encode("cp1252"))
    ran = files.run(space, """
from openpyxl import Workbook
book = Workbook()
sheet = book.active
sheet.append(["Belegdatum", "Belegnr.", "Buchungstext", "Soll", "Haben"])
sheet.append(["02.03.2026", "RE-2026-0342", "Müller Bürobedarf", None, 1190.0])
sheet.append(["05.03.2026", "MIETE-03", "Miete März", 2500.0, None])
sheet.append(["20.03.2026", "TEL-03", "Telefon", 59.0, None])
book.save("Buchungen.xlsx")
""")  # fmt: skip
    assert ran.info["status"] == 0, ran.content
    numbers: dict[str, int] = {}
    context = Context("c", lambda url, title: numbers.setdefault(url, len(numbers) + 1),
                      workspace=space)  # fmt: skip
    result = reconcile.reconcile(context, "Kontoauszug.csv", "Buchungen.xlsx", "Abstimmung März")
    assert result.content.startswith(
        "Of the bank's 3 lines [1] and the books' 3 [2]: 2 matched, 0 matched but in their "
        "amount, 1 in the bank alone, and 1 in the books alone.")  # fmt: skip
    assert "| 30.03.2026 | -12.90 | DE89370400440532013000 · Kontoführung · Sparkasse · EUR |" in (
        result.content)  # fmt: skip
    assert result.info["files"] == ["Abstimmung März.xlsx"]
    read = files.read(space, "Abstimmung März.xlsx").content
    assert [line for line in read.splitlines() if line.startswith("## ")] == [
        "## Amounts differ",
        "## In the bank alone",
        "## In the books alone",
        "## Matched",
    ]
    with pytest.raises(FileNotFoundError):
        reconcile.reconcile(context, "nothing.csv", "Buchungen.xlsx")
