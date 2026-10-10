"""Makes a demo of Leat for a small German law and tax office, in a new state of its own: a client's
books, its month's invoices, two of them scanned, its bank statement and ledger, which differ where
a reconciliation should find them out, its lease and a fee letter's template; and an employment
case, its contract, with clauses an employee's lawyer would change. Each project has workflows of
what the office asks of it. Then `leat agent --data DIR` serves it, and whoever first opens the app
owns it, the demo theirs.

    uv run python scripts/demo.py DIR [--sandbox ~/.local/share/leat/sandbox]

The documents are made in the sandbox, with its libraries, as the agent's are, in a font of the
box's. Asking the scanned invoices of their text needs a model that sees images.
"""

import argparse
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leat.agent.agent import Agent  # noqa: E402
from leat.agent.client import Client  # noqa: E402
from leat.agent.store import Store  # noqa: E402
from leat.agent.workspace import Workspace  # noqa: E402

# the client whose books the office keeps, and the instructions of its project
CLIENT = "Hofmann Textil GmbH"
BOOKS = """\
Mandantin ist die Hofmann Textil GmbH, ein Textilgroßhandel in München, USt-IdNr. DE284719350; \
das Geschäftsjahr endet am 30. Juni. Antworte auf Deutsch, schreibe Beträge wie 1.234,56 EUR, \
nenne zu jeder Zahl den Beleg, aus dem sie stammt, und sage, wo du unsicher bist."""
# the month's invoices: number, date, supplier, net amount, VAT rate in percent, and whether it is
# scanned, an image of a page rather than its text
INVOICES = [
    ("RE-2026-0342", "02.03.2026", "Müller Bürobedarf GmbH", 1_000.00, 19, False),
    ("2026-118342", "03.03.2026", "Weberei Lindner KG", 14_250.00, 19, False),
    ("LG-77310", "05.03.2026", "Weber Logistik KG", 2_380.00, 19, False),
    ("SWM-5530182", "06.03.2026", "Stadtwerke München", 1_424.50, 19, False),
    ("KA-00932", "09.03.2026", "Kaya Verpackungen", 712.50, 19, True),
    ("2026-118690", "12.03.2026", "Weberei Lindner KG", 9_800.00, 19, False),
    ("FB-2026-215", "14.03.2026", "Färberei Brandt GmbH", 3_960.00, 19, False),
    ("RE-2026-0350", "18.03.2026", "Schmidt IT-Service", 2_000.00, 19, False),
    ("TS-44102", "20.03.2026", "Technik Service Ost", 315.00, 19, True),
    ("LG-77391", "24.03.2026", "Weber Logistik KG", 1_130.00, 19, False),
    ("BK-7781", "26.03.2026", "Bäckerei Kraus", 448.00, 7, False),
    ("SWM-5531447", "30.03.2026", "Stadtwerke München", 1_387.25, 19, False),
]
# where the books differ from the bank, for the reconciliation to find: the invoice paid but not
# booked, the one booked but not paid, and the one paid other than it was booked
UNBOOKED, UNPAID, MISPAID = "FB-2026-215", "TS-44102", "2026-118690"
LEASE = [
    ("Gewerbemietvertrag", None),
    ("§ 1 Mietsache", "Vermietet wird die Lagerhalle Am Gewerbering 12, 85748 Garching, von der "
     "Kühn Immobilien GmbH an die Hofmann Textil GmbH."),
    ("§ 2 Mietzeit", "Das Mietverhältnis beginnt am 1. März 2026 und läuft auf unbestimmte Zeit."),
    ("§ 3 Miete", "Die monatliche Miete beträgt 4.200 EUR zuzüglich Umsatzsteuer und ist bis zum "
     "dritten Werktag eines jeden Monats im Voraus zu zahlen."),
    ("§ 4 Kaution", "Die Mieterin leistet eine Kaution in Höhe von drei Monatsmieten."),
    ("§ 5 Kündigung", "Die Kündigungsfrist beträgt sechs Monate zum Ende eines "
     "Kalendervierteljahres."),
    ("§ 6 Gerichtsstand", "Gerichtsstand ist München."),
]  # fmt: skip
LETTER = [
    "Hofmann Textil GmbH",
    "z. Hd. {{Ansprechpartner}}",
    "Honorarabrechnung {{Monat}}",
    "Sehr geehrte Frau Hofmann,",
    "für die laufende Finanzbuchhaltung und Lohnabrechnung im {{Monat}} berechnen wir Ihnen "
    "{{Honorar}} zuzüglich Umsatzsteuer, zahlbar bis zum {{Fälligkeit}}.",
    "Mit freundlichen Grüßen",
    "{{Partner}}",
]
BOOKS_WORKFLOWS = [
    ("Rechnungsübersicht", "Erstelle eine Tabelle aller Rechnungen im Ordner Rechnungen mit "
     "Rechnungsnummer, Datum, Lieferant, Umsatzsteuer und Bruttobetrag, und nenne die drei "
     "Lieferanten, an die wir am meisten gezahlt haben."),
    ("Kontoabstimmung", "Stimme den Kontoauszug März 2026 mit den Buchungen März 2026 ab und "
     "erkläre jede Abweichung."),
    ("Honorarschreiben", "Fülle die Vorlage für das Honorarschreiben für März 2026 aus: Honorar "
     "1.850 EUR, fällig am 10. April 2026, Ansprechpartnerin Anna Hofmann, gezeichnet von "
     "Dr. Clara Becker."),
]  # fmt: skip
# the employment case, its instructions, its contract and its workflows
CASE = "Schulz ./. Bauer Logistik GmbH"
CASE_INSTRUCTIONS = """\
Wir vertreten den Arbeitnehmer Markus Schulz gegen seine Arbeitgeberin, die Bauer Logistik GmbH. \
Prüfe Klauseln nach deutschem Arbeitsrecht, nenne die Vorschrift, auf die du dich stützt, und \
antworte auf Deutsch, außer man bittet dich um eine andere Sprache."""
CONTRACT = [
    ("Arbeitsvertrag", None),
    ("§ 1 Parteien", "Zwischen der Bauer Logistik GmbH, Hamburg (Arbeitgeberin), und Herrn "
     "Markus Schulz (Arbeitnehmer) wird folgender Arbeitsvertrag geschlossen."),
    ("§ 2 Tätigkeit", "Der Arbeitnehmer wird als Disponent eingestellt. Die Arbeitgeberin kann ihm "
     "jederzeit jede andere Tätigkeit an jedem Ort zuweisen."),
    ("§ 3 Arbeitszeit", "Die regelmäßige Arbeitszeit beträgt 40 Stunden in der Woche. Überstunden "
     "sind mit dem Gehalt abgegolten."),
    ("§ 4 Vergütung", "Der Arbeitnehmer erhält ein Bruttomonatsgehalt von 3.900 EUR."),
    ("§ 5 Urlaub", "Der Arbeitnehmer hat Anspruch auf 18 Arbeitstage Urlaub im Kalenderjahr."),
    ("§ 6 Kündigung", "Das Arbeitsverhältnis kann von beiden Seiten mit einer Frist von zwei "
     "Wochen gekündigt werden."),
    ("§ 7 Wettbewerbsverbot", "Der Arbeitnehmer darf zwei Jahre nach Ende des Arbeitsverhältnisses "
     "für kein Unternehmen der Logistikbranche tätig werden."),
    ("§ 8 Ausschlussfrist", "Ansprüche aus dem Arbeitsverhältnis verfallen, wenn sie nicht "
     "innerhalb von vier Wochen nach Fälligkeit schriftlich geltend gemacht werden."),
]  # fmt: skip
CASE_WORKFLOWS = [
    ("Vertragsprüfung", "Prüfe den Arbeitsvertrag auf Klauseln, die unwirksam oder für unseren "
     "Mandanten nachteilig sind, und schlage Änderungen als Nachverfolgung in Word vor, jede mit "
     "einer Begründung und der Vorschrift."),
    ("Übersetzung", "Übersetze den Arbeitsvertrag ins Englische, für den Konzernanwalt der "
     "Arbeitgeberin."),
]  # fmt: skip
# makes the documents of the JSON at sys.argv[1], in the font it names, in the sandbox: PDFs,
# of lines, each of its text or, scanned, an image of it; workbooks, of a sheet's rows, the first
# its heading; CSVs, of rows, in an encoding; and Word documents, of headings and paragraphs
_MAKE = """
import json, sys
from pathlib import Path
from docx import Document
from fpdf import FPDF
from openpyxl import Workbook
from openpyxl.styles import Font
from PIL import Image, ImageDraw, ImageFont

spec = json.load(open(sys.argv[1], encoding="utf-8"))
font = spec["font"]
for name, lines, scanned in spec["pdfs"]:
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.add_font("Sans", fname=font)
    pdf.add_page()
    if scanned:  # an image of the page, slightly askew, as a scanner's
        page = Image.new("L", (1240, 1754), 248)
        draw, face = ImageDraw.Draw(page), ImageFont.truetype(font, 34)
        for i, line in enumerate(lines):
            draw.text((110, 140 + i * 62), line, fill=30, font=face)
        page.rotate(0.6, fillcolor=248).save(".scan.png")
        pdf.image(".scan.png", x=0, y=0, w=210)
        Path(".scan.png").unlink()
    else:
        pdf.set_font("Sans", size=16)
        pdf.cell(text=lines[0], new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Sans", size=11)
        for line in lines[1:]:
            if line:
                pdf.cell(text=line, new_x="LMARGIN", new_y="NEXT", h=8)
            else:
                pdf.ln(8)
    pdf.output(name)
for name, title, rows in spec["workbooks"]:
    book = Workbook()
    sheet = book.active
    sheet.title = title
    for row in rows:
        sheet.append(row)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for column in sheet.columns:
        width = max(len(str(cell.value or "")) for cell in column)
        sheet.column_dimensions[column[0].column_letter].width = min(50, width + 2)
    book.save(name)
for name, encoding, rows in spec["csvs"]:
    with open(name, "w", encoding=encoding, newline="") as f:
        f.writelines(";".join(row) + "\\r\\n" for row in rows)
for name, blocks in spec["documents"]:
    document = Document()
    for heading, text in blocks:
        if heading is not None:
            document.add_heading(heading, level=0 if text is None else 1)
        if text is not None:
            document.add_paragraph(text)
    document.save(name)
print("{}")
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="a new folder for the demo's state")
    parser.add_argument(
        "--sandbox", type=Path, default=Path.home() / ".local/share/leat/sandbox",
        help="the sandbox's environment, as leat/agent/sandbox.txt makes it",
    )  # fmt: skip
    args = parser.parse_args()
    if args.data.exists() and any(args.data.iterdir()):
        raise SystemExit(f"{args.data} is not empty: give a new folder for the demo")
    if not args.sandbox.exists():
        raise SystemExit(f"there is no sandbox environment at {args.sandbox}: make it as the "
                         "README says")  # fmt: skip
    font = _font()
    args.data.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(args.data / "workspace", args.sandbox)
    agent = Agent(Store(args.data / "leat.db"), Client("http://127.0.0.1:9"), [], workspace)
    books = agent.add_project(CLIENT, None, BOOKS, shared=True)["id"]
    case = agent.add_project(CASE, None, CASE_INSTRUCTIONS, shared=True)["id"]
    made = {
        books: {
            "pdfs": [[f"Rechnungen/{i[0]}.pdf", _invoice(*i[:5]), i[5]] for i in INVOICES],
            "workbooks": [["Buchungen März 2026.xlsx", "Buchungen", _ledger()]],
            "csvs": [["Kontoauszug März 2026.csv", "cp1252", _statement()]],
            "documents": [
                ["Gewerbemietvertrag.docx", LEASE],
                ["Honorarschreiben Vorlage.docx", [(None, line) for line in LETTER]],
            ],
        },  # fmt: skip
        case: {
            "pdfs": [],
            "workbooks": [],
            "csvs": [],
            "documents": [["Arbeitsvertrag.docx", CONTRACT]],
        },  # fmt: skip
    }
    try:
        for project, spec in made.items():
            agent.space(project).given(_MAKE, spec | {"font": font}, timeout=300)
    except ValueError as e:
        shutil.rmtree(args.data)  # rather than a demo half made
        raise SystemExit(f"the documents could not be made: {e}") from e
    for project, workflows in ((books, BOOKS_WORKFLOWS), (case, CASE_WORKFLOWS)):
        for name, prompt in workflows:
            agent.add_workflow(name, prompt, None, project)
    for project in (books, case):
        names = sorted(f["name"] for f in agent.space(project).files())
        counted = f"{len(names)} {'file' if len(names) == 1 else 'files'}"
        print(f"{(agent.store.project(project) or {})['name']}, {counted}:")
        print("\n".join(f"  {name}" for name in names))
    print(f"\nServe it with: uv run leat agent --data {args.data}")


def _font() -> str:
    # a font of the box's with German's letters, its file's path, under /usr, which the sandbox sees
    found = subprocess.run(["fc-match", "-f", "%{file}", "sans-serif:lang=de"],
                           capture_output=True, text=True, check=False).stdout  # fmt: skip
    if not found.startswith("/usr/") or not Path(found).is_file():
        raise SystemExit("no font was found: install one, as Noto Sans or DejaVu Sans")
    return found


def _invoice(number: str, date: str, supplier: str, net: float, rate: int) -> list[str]:
    # an invoice's lines, as its page shows them
    vat = round(net * rate / 100, 2)
    return [
        f"RECHNUNG  {number}", f"Rechnungsdatum: {date}", f"Lieferant: {supplier}",
        "Rechnungsempfänger: Hofmann Textil GmbH, Lindwurmstraße 88, 80337 München", "",
        f"Nettobetrag: {_eur(net)}", f"Umsatzsteuer {rate} %: {_eur(vat)}",
        f"Rechnungsbetrag: {_eur(net + vat)}", "", "Zahlbar innerhalb von 14 Tagen ohne Abzug.",
    ]  # fmt: skip


def _ledger() -> list[list]:
    # the month's bookings, each invoice's but one, the rent and the salaries, as a ledger's
    # export heads them
    rows: list[list] = []
    for number, date, supplier, net, rate, _ in INVOICES:
        if number != UNBOOKED:
            gross = round(net * (1 + rate / 100), 2)
            rows.append([date, number, "1600 Verbindlichkeiten", supplier, None, gross])
    rows.append(["03.03.2026", "MIETE-03", "4210 Miete", "Miete März Lagerhalle", 4_998.00, None])
    rows.append(["31.03.2026", "LOHN-03", "1740 Löhne", "Löhne März", None, 38_640.00])
    rows.sort(key=lambda row: datetime.datetime.strptime(row[0], "%d.%m.%Y"))
    return [["Belegdatum", "Belegnr.", "Konto", "Buchungstext", "Soll", "Haben"], *rows]


def _statement() -> list[list[str]]:
    # the month's payments, as a Sparkasse's CSV export has them: each invoice's but the unpaid
    # one, one less than its invoice says, the rent and the salaries
    iban, payments = "DE89701500000012345678", []
    for number, date, supplier, net, rate, _ in INVOICES:
        if number == UNPAID:
            continue
        gross = round(net * (1 + rate / 100), 2) - (500.0 if number == MISPAID else 0.0)
        day = min(
            datetime.datetime.strptime(date, "%d.%m.%Y") + datetime.timedelta(days=4),
            datetime.datetime(2026, 3, 31),
        )  # within the statement's month
        payments.append((day, f"{number} {supplier}", supplier, -gross))
    payments.append((datetime.datetime(2026, 3, 3), "Miete März", "Kühn Immobilien GmbH", -4998.0))
    payments.append((datetime.datetime(2026, 3, 31), "Lohn März", "Lohnzahlungen", -38_640.00))
    heading = ["Auftragskonto", "Buchungstag", "Verwendungszweck",
               "Begünstigter/Zahlungspflichtiger", "Betrag", "Währung"]  # fmt: skip
    return [heading] + [[iban, f"{day:%d.%m.%Y}", said, who, _eur(amount, unit=False), "EUR"]
                        for day, said, who, amount in sorted(payments)]  # fmt: skip


def _eur(amount: float, unit: bool = True) -> str:
    # an amount as German writes it, 1.234,56, with its currency if `unit`
    said = f"{amount:,.2f}".replace(",", "x").replace(".", ",").replace("x", ".")
    return f"{said} EUR" if unit else said


if __name__ == "__main__":
    main()
