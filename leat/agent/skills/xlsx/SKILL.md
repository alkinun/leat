---
name: xlsx
description: Excel spreadsheets, such as budgets, lists and tables of numbers
---
# Excel spreadsheets

Make them with openpyxl, in one run, and save them in the workspace. Let the spreadsheet compute:
write formulas, not the numbers you worked out.

```python
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill

wb = Workbook()
ws = wb.active
ws.title = "Budget"
ws.append(["Item", "Monthly", "Yearly"])
for item, monthly in [("Rent", 900), ("Food", 350), ("Transport", 80)]:
    ws.append([item, monthly])
for row in range(2, ws.max_row + 1):
    ws[f"C{row}"] = f"=B{row}*12"
total = ws.max_row + 1
ws[f"A{total}"], ws[f"B{total}"], ws[f"C{total}"] = (
    "Total",
    f"=SUM(B2:B{total - 1})",
    f"=SUM(C2:C{total - 1})",
)

for cell in ws[1]:  # the header
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="C96442")
    cell.alignment = Alignment(horizontal="center")
for cell in ws[total]:
    cell.font = Font(bold=True)
for column in "BC":
    for cell in ws[column][1:]:
        cell.number_format = "#,##0.00"
ws.column_dimensions["A"].width = 20
ws.freeze_panes = "A2"

chart = BarChart()
chart.title = "Monthly costs"
chart.add_data(Reference(ws, min_col=2, min_row=1, max_row=total - 1), titles_from_data=True)
chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=total - 1))
ws.add_chart(chart, "E2")
wb.save("Budget.xlsx")
```

- One table to a sheet, its header in the first row, frozen; units in the header, as "Cost (€)".
- Dates as Python dates, with a number_format such as "yyyy-mm-dd".
- Then read the file back to check it, and tell the user its name.
