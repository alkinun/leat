---
name: pptx
description: PowerPoint slides, as presentations and decks
---
# PowerPoint slides

Make them with python-pptx, in one run, and save them in the workspace. A slide says one thing:
a title that states it, and a few short points, or a picture or chart.

```python
from pptx import Presentation
from pptx.util import Inches, Pt

deck = Presentation()
deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)  # 16:9

slide = deck.slides.add_slide(deck.slide_layouts[0])  # the title slide
slide.shapes.title.text = "Our year in numbers"
slide.placeholders[1].text = "Budget review · October 2026"

slide = deck.slides.add_slide(deck.slide_layouts[1])  # a title, and points
slide.shapes.title.text = "We saved more than we planned"
points = slide.placeholders[1].text_frame
points.text = "Saved €4,200, against €3,000 planned"
for text in ["Food cost 12% less", "Holidays cost as planned"]:
    paragraph = points.add_paragraph()
    paragraph.text, paragraph.level = text, 0
for paragraph in points.paragraphs:
    paragraph.font.size = Pt(28)

slide = deck.slides.add_slide(deck.slide_layouts[5])  # a title alone, then a picture
slide.shapes.title.text = "Where the money went"
# a chart drawn by matplotlib, saved as chart.png first
# slide.shapes.add_picture("chart.png", Inches(1.5), Inches(1.6), height=Inches(5.5))

deck.save("Year in numbers.pptx")
```

- Five to ten slides; three to five points to a slide, each a line.
- Charts: draw them with matplotlib, plt.savefig("chart.png", dpi=200, bbox_inches="tight").
- Then read the file back to check it, and tell the user its name.
