"""Makes a demo of Leat for an accounting firm, in a new state of its own: a client's project, its
month's invoices, two of them scanned, its bank statement and ledger, which differ where a
reconciliation should find them out, a lease, a fee letter's template, and workflows of what an
accountant asks of them each month. Then `leat agent --data DIR` serves it, and whoever first opens
the app owns it, the demo theirs.

    uv run python scripts/demo.py DIR [--sandbox ~/.local/share/leat/sandbox]

The documents are made in the sandbox, with its libraries, as the agent's are, in Turkish, as a
Turkish firm's would be, written in a font of the box's that has Turkish's letters. Asking the
scanned invoices of their text needs a model that sees images.
"""

import argparse
import json
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

CLIENT = "Yılmaz Tekstil"
INSTRUCTIONS = """\
Müşterimiz Yılmaz Tekstil Ltd. Şti., Bursa'da bir tekstil toptancısı; vergi numarası 8340021957, \
mali yılı Haziran'da biter. Türkçe yanıt ver, tutarları Türk biçiminde yaz (1.234,56 TL), her \
rakamın hangi belgeden geldiğini göster ve emin olmadığın yeri söyle."""
# the month's invoices: number, date, seller, net amount, VAT rate in percent, and whether it is
# scanned, an image of a page rather than its text
INVOICES = [
    ("A-2026-0311", "02.03.2026", "Akın İplik A.Ş.", 38_500.00, 20, False),
    ("BK-118342", "03.03.2026", "Bursa Kumaş San. ve Tic.", 61_250.00, 20, False),
    ("EL-2026-077", "05.03.2026", "Ege Lojistik", 9_800.00, 20, False),
    ("ME-5530182", "06.03.2026", "Marmara Enerji", 14_240.50, 20, False),
    ("KA-00932", "09.03.2026", "Kaya Ambalaj", 7_125.00, 20, True),
    ("A-2026-0347", "12.03.2026", "Akın İplik A.Ş.", 42_000.00, 20, False),
    ("DB-2026-215", "14.03.2026", "Deniz Boya Kimya", 18_960.00, 20, False),
    ("BK-118690", "18.03.2026", "Bursa Kumaş San. ve Tic.", 27_400.00, 20, False),
    ("TS-44102", "20.03.2026", "Toros Servis", 3_150.00, 20, True),
    ("EL-2026-091", "24.03.2026", "Ege Lojistik", 11_300.00, 20, False),
    ("YB-7781", "26.03.2026", "Yeşil Bahçe Gıda", 4_480.00, 10, False),
    ("ME-5531447", "30.03.2026", "Marmara Enerji", 13_870.25, 20, False),
]
# where the books differ from the bank, for the reconciliation to find: the invoice paid but not
# entered in the ledger, the one entered but not paid, and the one paid other than it was entered
UNENTERED, UNPAID, MISPAID = "DB-2026-215", "TS-44102", "BK-118690"
LEASE = [
    ("Kira Sözleşmesi", None),
    ("Taraflar", "Kiraya veren Bursa Gayrimenkul A.Ş. ile kiracı Yılmaz Tekstil Ltd. Şti. "
     "arasında, Nilüfer, Bursa'daki depo için yapılmıştır."),
    ("Süre", "Sözleşme 1 Mart 2026'da başlar ve üç yıl sürer."),
    ("Kira", "Aylık kira 40.000 TL olup her ayın 5'inde ödenir. Kira her yıl TÜFE oranında "
     "artırılır."),
    ("Depozito", "Kiracı, iki aylık kira tutarında, 80.000 TL depozito öder."),
    ("Fesih", "Taraflardan her biri doksan gün önceden yazılı bildirimle sözleşmeyi "
     "feshedebilir."),
    ("Uyuşmazlık", "Uyuşmazlıklarda Bursa mahkemeleri ve icra daireleri yetkilidir."),
]  # fmt: skip
LETTER = [
    "Sayın {{Müşteri}},",
    "{{Ay}} ayına ait muhasebe ve beyanname hizmetlerimizin ücreti {{Ücret}} olup son ödeme "
    "tarihi {{Son ödeme tarihi}}'dir.",
    "Ekte bu ayın beyannamelerinin ve mutabakatın bir özetini bulabilirsiniz.",
    "Saygılarımızla,",
    "{{Ortak}}",
]
WORKFLOWS = [
    ("Fatura tablosu", "Faturalar klasöründeki her faturanın numarasını, tarihini, satıcısını, "
     "KDV'sini ve toplamını bir tabloya çıkar, ve en çok ödediğimiz üç satıcıyı söyle."),
    ("Banka mutabakatı", "Mart banka ekstresini muhasebe kayıtlarıyla karşılaştır: eşleşmeyen her "
     "hareketi, tarihi, tutarı ve olası nedeniyle bir tabloda listele."),
    ("Aylık ücret mektubu", "Ücret mektubu şablonunu Mart ayı için doldur: ücret 6.000 TL, son "
     "ödeme tarihi 10 Nisan 2026, imzalayan Ayşe Kaya."),
]  # fmt: skip
# makes the documents of the JSON at sys.argv[1], in the font at sys.argv[2], in the sandbox
_MAKE = """
import json, sys
from pathlib import Path
from docx import Document
from fpdf import FPDF
from openpyxl import Workbook
from openpyxl.styles import Font
from PIL import Image, ImageDraw, ImageFont

spec, font = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2]
Path("Faturalar").mkdir(exist_ok=True)
for invoice in spec["invoices"]:
    lines = invoice["lines"]
    path = f"Faturalar/{invoice['number']}.pdf"
    pdf = FPDF()
    pdf.add_font("Noto", fname=font)
    pdf.add_page()
    if invoice["scanned"]:  # an image of the page, slightly askew, as a scanner's
        page = Image.new("L", (1240, 1754), 248)
        draw, face = ImageDraw.Draw(page), ImageFont.truetype(font, 34)
        for i, line in enumerate(lines):
            draw.text((110, 140 + i * 62), line, fill=30, font=face)
        page = page.rotate(0.6, fillcolor=248)
        page.save(f".{invoice['number']}.png")
        pdf.image(f".{invoice['number']}.png", x=0, y=0, w=210)
        Path(f".{invoice['number']}.png").unlink()
    else:
        pdf.set_font("Noto", size=16)
        pdf.cell(text=lines[0], new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Noto", size=11)
        for line in lines[1:]:
            if line:
                pdf.cell(text=line, new_x="LMARGIN", new_y="NEXT", h=8)
            else:
                pdf.ln(8)
    pdf.output(path)

book = Workbook()
sheet = book.active
sheet.title = "Mart 2026"
sheet.append(["Tarih", "Belge no", "Hesap", "Açıklama", "Borç", "Alacak"])
for cell in sheet[1]:
    cell.font = Font(bold=True)
for row in spec["ledger"]:
    sheet.append(row)
for column, width in zip("ABCDEF", (12, 16, 22, 40, 14, 14)):
    sheet.column_dimensions[column].width = width
book.save("Muhasebe kayıtları Mart 2026.xlsx")

with open("Banka ekstresi Mart 2026.csv", "w", encoding="utf-8-sig") as f:
    f.write("Tarih;Açıklama;Tutar;Bakiye\\n")
    for row in spec["bank"]:
        f.write(";".join(row) + "\\n")

document = Document()
for heading, text in spec["lease"]:
    if text is None:
        document.add_heading(heading, level=0)
    else:
        document.add_heading(heading, level=1)
        document.add_paragraph(text)
document.save("Kira Sözleşmesi.docx")

document = Document()
for line in spec["letter"]:
    document.add_paragraph(line)
document.save("Ücret mektubu şablonu.docx")
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
    project = agent.add_project(CLIENT, None, INSTRUCTIONS, shared=True)["id"]
    space = agent.space(project)
    spec = {"invoices": [_invoice(*i) for i in INVOICES], "ledger": _ledger(), "bank": _bank(),
            "lease": LEASE, "letter": LETTER}  # fmt: skip
    space.path(".spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    try:
        ran = space.run(_MAKE, "/workspace/.spec.json", font, timeout=300)
    finally:
        space.path(".spec.json").unlink(missing_ok=True)
    if ran.status != 0:
        shutil.rmtree(args.data)  # rather than a demo half made
        raise SystemExit(f"the documents could not be made:\n{ran.output}")
    for name, prompt in WORKFLOWS:
        agent.add_workflow(name, prompt, None, project)
    made = sorted(f["name"] for f in space.files())
    print(f"A demo of {len(made)} files in the project {CLIENT}, at {args.data}:")
    print("\n".join(f"  {name}" for name in made))
    print(f"\nServe it with: uv run leat agent --data {args.data}")


def _font() -> str:
    # a font of the box's with Turkish's letters, its file's path, under /usr, which the sandbox
    # sees
    found = subprocess.run(["fc-match", "-f", "%{file}", "sans-serif:lang=tr"],
                           capture_output=True, text=True, check=False).stdout  # fmt: skip
    if not found.startswith("/usr/") or not Path(found).is_file():
        raise SystemExit("no font with Turkish letters was found: install one, as Noto Sans")
    return found


def _invoice(number: str, date: str, seller: str, net: float, rate: int, scanned: bool) -> dict:
    # an invoice's lines, as its page shows them
    vat = round(net * rate / 100, 2)
    return {
        "number": number, "scanned": scanned,
        "lines": [
            f"FATURA  {number}", f"Tarih: {date}", f"Satıcı: {seller}",
            "Alıcı: Yılmaz Tekstil Ltd. Şti., Nilüfer, Bursa (VKN 8340021957)", "",
            f"Mal ve hizmet toplamı: {_tl(net)}", f"KDV (%{rate}): {_tl(vat)}",
            f"Genel toplam: {_tl(net + vat)}", "", "Ödeme: fatura tarihinden itibaren 15 gün.",
        ],
    }  # fmt: skip


def _ledger() -> list[list]:
    # the month's entries, each invoice's but one, the rent and the salaries
    rows: list[list] = []
    for number, date, seller, net, rate, _ in INVOICES:
        if number == UNENTERED:
            continue
        total = round(net * (1 + rate / 100), 2)
        rows.append([date, number, "320 Satıcılar", f"{seller} faturası", None, total])
    rows.append(["05.03.2026", "KİRA-03", "770 Genel yönetim gid.", "Mart kirası", 40_000.00, None])
    rows.append(["31.03.2026", "BORDRO-03", "335 Personele borçlar", "Mart maaşları", None,
                 186_400.00])  # fmt: skip
    return sorted(rows, key=lambda row: row[0].split(".")[::-1])


def _bank() -> list[list[str]]:
    # the month's payments, each invoice's but the unpaid one, one other than its invoice says
    balance, rows = 1_250_000.00, []
    payments = []
    for number, date, seller, net, rate, _ in INVOICES:
        if number == UNPAID:
            continue
        total = round(net * (1 + rate / 100), 2) - (500.0 if number == MISPAID else 0.0)
        day = f"{min(int(date[:2]) + 3, 31):02d}{date[2:]}"
        payments.append((day, f"EFT {seller} {number}", -total))
    payments.append(("05.03.2026", "Bursa Gayrimenkul A.Ş. Mart kirası", -40_000.00))
    payments.append(("31.03.2026", "Maaş ödemeleri Mart 2026", -186_400.00))
    for day, said, amount in sorted(payments, key=lambda p: p[0].split(".")[::-1]):
        balance += amount
        rows.append([day, said, _tl(amount, unit=False), _tl(balance, unit=False)])
    return rows


def _tl(amount: float, unit: bool = True) -> str:
    # an amount as Turkish writes it, 1.234,56, with its currency if `unit`
    said = f"{amount:,.2f}".replace(",", "x").replace(".", ",").replace("x", ".")
    return f"{said} TL" if unit else said


if __name__ == "__main__":
    main()
