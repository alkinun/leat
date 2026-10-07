---
name: pdf
description: PDF documents, to print or send as they are
---
# PDF documents

Make them with fpdf2, in one run, and save them in the workspace. For a document the user may want
to change, make a Word document instead.

```python
from pathlib import Path

import matplotlib
from fpdf import FPDF

fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu, which has every letter
pdf = FPDF()  # A4, in millimetres
pdf.set_margins(20, 20, 20)
pdf.add_page()
pdf.add_font("Sans", "", fonts / "DejaVuSans.ttf")
pdf.add_font("Sans", "B", fonts / "DejaVuSans-Bold.ttf")

pdf.set_font("Sans", "B", 20)
pdf.cell(0, 12, "Packing list", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Sans", "", 11)
pdf.multi_cell(0, 6, "For the trip to Lisbon, 14 to 21 May. Pack the day before.")
pdf.ln(4)

with pdf.table(col_widths=(60, 30), text_align=("LEFT", "RIGHT")) as table:
    for row in [("Item", "How many"), ("Shirts", "5"), ("Chargers", "2")]:
        table.row(row)

# pdf.image("chart.png", w=170)  # a picture, as a chart matplotlib drew
pdf.output("Packing list.pdf")
```

- Keep to the DejaVu fonts: the built-in ones, as "Helvetica", have no letters past Latin-1.
- Then read the file back to check it, and tell the user its name.
