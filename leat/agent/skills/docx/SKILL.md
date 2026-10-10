---
name: docx
description: Word documents, such as letters, CVs and reports
---
# Word documents

A letter, memo or report is made by make_document, of its text in markdown, in the office's
letterhead if they have one, rather than by writing code. Make with python-docx, in one run, only
what markdown cannot hold, as pictures, columns or a layout of its own, and save it in the
workspace under a name that says what it is. Fill in the user's own details; leave none of the
template's.

To fill a Word template the user has, whose fields are written {{Client name}} or «Client name»,
call fill_template rather than writing code: it keeps the template's formatting, and says which
fields had no value. To suggest changes to a document, as a contract's, read it, then call
suggest_edits: each edit a passage as it is written there, its replacement, and a comment that
says why, which the user accepts or rejects in Word as tracked changes.

Write of the user only what they told you: never invent their experience, duties, results,
courses or skill levels. Where the document wants more, leave a gap
marked as "[your duties at Springfield General]", and say in your answer what they may add.

```python
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

doc = Document()
style = doc.styles["Normal"]
style.font.name, style.font.size = "Calibri", Pt(11)
for section in doc.sections:  # margins of 2 cm
    section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = Pt(57)

title = doc.add_heading("Ada Lovelace", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
line = doc.add_paragraph("London · ada@example.com · +44 20 7946 0000")
line.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading("Experience", level=1)
role = doc.add_paragraph()
role.add_run("Analyst, Analytical Engine").bold = True
role.add_run("\t1842 – 1843").italic = True
doc.add_paragraph("Wrote the first published algorithm for a machine.", style="List Bullet")

table = doc.add_table(rows=1, cols=2, style="Light Grid Accent 1")
table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Skill", "Level"
row = table.add_row().cells
row[0].text, row[1].text = "Mathematics", "Expert"

doc.add_page_break()  # only between parts that need their own pages
doc.save("Ada Lovelace CV.docx")
```

- Headings structure it: level 0 the title, 1 its sections, 2 their parts.
- "List Bullet" and "List Number" make lists; a table suits rows of like things.
- A letter: the sender's address, the date, the recipient's, then the greeting, the body in short
  paragraphs, and the closing, each a paragraph of its own.
- Then read the file back to check it, and tell the user its name.
