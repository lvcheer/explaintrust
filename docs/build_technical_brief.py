"""Render the two-page technical brief from its tracked Markdown source."""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "docs/explaintrust-technical-brief.md"
OUTPUT = ROOT / "output/pdf/explaintrust-technical-brief.pdf"
PAGE_BREAK_MARKER = "<!-- PAGE BREAK -->"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#246BCE")
TEAL = colors.HexColor("#167D7F")
PALE_BLUE = colors.HexColor("#EEF4FF")
PALE_GRAY = colors.HexColor("#F5F7FA")
TEXT = colors.HexColor("#243342")
MUTED = colors.HexColor("#5C6B78")


def _inline_markup(value: str) -> str:
    value = html.escape(value, quote=False)
    value = re.sub(
        r"\[([^]]+)]\(([^)]+)\)",
        r'<link href="\2" color="#246BCE">\1</link>',
        value,
    )
    value = re.sub(
        r"`([^`]+)`",
        r'<font name="Helvetica-Bold" color="#246BCE">\1</font>',
        value,
    )
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", value)
    return value


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "BriefTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=24, leading=27, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "BriefSubtitle", parent=base["Heading2"], fontName="Helvetica",
            fontSize=12.5, leading=15, textColor=TEAL, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "BriefH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12.5, leading=14.5, textColor=NAVY, spaceBefore=7,
            spaceAfter=3, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "BriefH3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=9.8, leading=11.4, textColor=TEAL, spaceBefore=5,
            spaceAfter=2, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BriefBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.8, leading=10.8, textColor=TEXT, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "BriefBullet", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.4, leading=10.1, textColor=TEXT,
        ),
        "table": ParagraphStyle(
            "BriefTable", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.5, leading=8.8, textColor=TEXT,
        ),
        "code": ParagraphStyle(
            "BriefCode", parent=base["Code"], fontName="Courier",
            fontSize=7.2, leading=8.7, textColor=TEXT, leftIndent=5,
            rightIndent=5, spaceBefore=2, spaceAfter=5,
        ),
    }


def _table(lines: list[str], style: ParagraphStyle) -> Table:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
    ]
    rows = [rows[0], *rows[2:]]
    cells = []
    for row_index, row in enumerate(rows):
        rendered = []
        for cell in row:
            markup = _inline_markup(cell)
            if row_index == 0:
                markup = f'<font color="#FFFFFF"><b>{markup}</b></font>'
            rendered.append(Paragraph(markup, style))
        cells.append(rendered)
    available = A4[0] - 30 * mm
    columns = len(cells[0])
    if columns == 3:
        widths = [available * 0.28, available * 0.52, available * 0.20]
    else:
        widths = [available / columns] * columns
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), PALE_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D3E0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _render_page(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = markdown.strip().splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            block = Preformatted("\n".join(code), styles["code"])
            block.backColor = PALE_GRAY
            story.append(block)
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(_table(table_lines, styles["table"]))
            story.append(Spacer(1, 3))
            continue
        if line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                item_lines = [lines[index][2:]]
                index += 1
                while index < len(lines):
                    continuation = lines[index].rstrip()
                    if not continuation or continuation.startswith(("#", "- ", "|", "```")):
                        break
                    item_lines.append(continuation.strip())
                    index += 1
                items.append(
                    ListItem(
                        Paragraph(_inline_markup(" ".join(item_lines)), styles["bullet"]),
                        leftIndent=10,
                    )
                )
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=12, bulletFontSize=5))
            story.append(Spacer(1, 3))
            continue
        if line.startswith("# "):
            story.append(Paragraph(_inline_markup(line[2:]), styles["title"]))
            index += 1
            continue
        if line.startswith("## "):
            heading = line[3:]
            key = "subtitle" if not story else "h2"
            story.append(Paragraph(_inline_markup(heading), styles[key]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(_inline_markup(line[4:]), styles["h3"]))
            index += 1
            continue

        paragraph = [line]
        index += 1
        while index < len(lines):
            following = lines[index].rstrip()
            if not following or following.startswith(("#", "- ", "|", "```")):
                break
            paragraph.append(following)
            index += 1
        story.append(Paragraph(_inline_markup(" ".join(paragraph)), styles["body"]))
    return story


def build() -> Path:
    markdown = SOURCE.read_text()
    digest = hashlib.sha256(markdown.encode()).hexdigest()
    pages = markdown.split(PAGE_BREAK_MARKER)
    if len(pages) != 2:
        raise ValueError("technical brief source must contain exactly one page break")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=14 * mm, title="explaintrust technical brief",
        author="lvcheer", subject=f"source-sha256:{digest}",
    )
    styles = _styles()
    story = _render_page(pages[0], styles) + [PageBreak()] + _render_page(pages[1], styles)

    def decorate(canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setTitle("explaintrust technical brief")
        canvas.setAuthor("lvcheer")
        canvas.setSubject(f"source-sha256:{digest}")
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 7 * mm, width, 7 * mm, fill=1, stroke=0)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(15 * mm, 7 * mm, "explaintrust | technical brief")
        canvas.drawRightString(width - 15 * mm, 7 * mm, f"{doc.page} / 2")
        canvas.restoreState()

    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(f"wrote {OUTPUT.relative_to(ROOT)} from {SOURCE.relative_to(ROOT)}")
    return OUTPUT


if __name__ == "__main__":
    build()
