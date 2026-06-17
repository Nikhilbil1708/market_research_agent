import os
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# JPMC brand colors
JPMC_BLUE   = HexColor("#003087")
JPMC_GOLD   = HexColor("#B5A147")
DARK_GRAY   = HexColor("#333333")
LIGHT_GRAY  = HexColor("#F5F5F5")
MID_GRAY    = HexColor("#888888")

def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=26,
        textColor=JPMC_BLUE,
        spaceAfter=8,
        alignment=TA_LEFT
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        fontName="Helvetica",
        fontSize=13,
        textColor=DARK_GRAY,
        spaceAfter=4,
        alignment=TA_LEFT
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta",
        fontName="Helvetica",
        fontSize=10,
        textColor=MID_GRAY,
        spaceAfter=2,
        alignment=TA_LEFT
    )
    styles["section_heading"] = ParagraphStyle(
        "section_heading",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=JPMC_BLUE,
        spaceBefore=18,
        spaceAfter=6,
        alignment=TA_LEFT
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10,
        textColor=DARK_GRAY,
        leading=16,
        spaceAfter=6,
        alignment=TA_LEFT
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=10,
        textColor=DARK_GRAY,
        leading=16,
        leftIndent=16,
        spaceAfter=4,
        bulletIndent=6,
        alignment=TA_LEFT
    )
    styles["confidence"] = ParagraphStyle(
        "confidence",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=JPMC_GOLD,
        spaceAfter=4
    )
    styles["footer"] = ParagraphStyle(
        "footer",
        fontName="Helvetica",
        fontSize=8,
        textColor=MID_GRAY,
        alignment=TA_CENTER
    )

    return styles


def parse_report_sections(report_text: str) -> dict:
    """
    Splits the synthesis agent's plain text output into named sections.
    Looks for numbered headings like '1.', '2.' etc. or ALL CAPS headings.
    Falls back to returning the full text as one block if no sections found.
    """
    sections = {}

    # Try to split on numbered sections (1. HEADING or 1. Heading)
    pattern = r'(\d+\.\s+[A-Z][^\n]+)'
    parts = re.split(pattern, report_text)

    if len(parts) > 1:
        # parts alternates: [preamble, heading, content, heading, content ...]
        i = 1
        while i < len(parts) - 1:
            heading = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections[heading] = content
            i += 2
        if parts[0].strip():
            sections["__preamble__"] = parts[0].strip()
    else:
        # No numbered sections found — treat as one block
        sections["__full__"] = report_text.strip()

    return sections


def format_content_block(text: str, styles: dict) -> list:
    """
    Converts a block of plain text into a list of ReportLab flowables.
    Lines starting with - or * become bullet points.
    Everything else becomes body paragraphs.
    """
    flowables = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 4))
            continue
        if line.startswith(("-", "*", "•")):
            clean = line.lstrip("-*• ").strip()
            flowables.append(Paragraph(f"• {clean}", styles["bullet"]))
        else:
            flowables.append(Paragraph(line, styles["body"]))
    return flowables


def add_header_footer(canvas, doc):
    """Adds JPMC branding header and page footer to every page."""
    canvas.saveState()
    width, height = A4

    # Header bar
    canvas.setFillColor(JPMC_BLUE)
    canvas.rect(0, height - 1.2 * cm, width, 1.2 * cm, fill=1, stroke=0)

    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(HexColor("#FFFFFF"))
    canvas.drawString(1.5 * cm, height - 0.85 * cm, "JPMC Market Research")

    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(
        width - 1.5 * cm,
        height - 0.85 * cm,
        "CONFIDENTIAL — Internal Use Only"
    )

    # Gold accent line under header
    canvas.setStrokeColor(JPMC_GOLD)
    canvas.setLineWidth(1.5)
    canvas.line(0, height - 1.25 * cm, width, height - 1.25 * cm)

    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GRAY)
    canvas.drawCentredString(
        width / 2,
        0.6 * cm,
        f"Page {doc.page} | Generated {datetime.now().strftime('%B %d, %Y')} | JPMC Internal Research"
    )

    canvas.restoreState()


def generate_pdf(
    final_report: str,
    query: str,
    ticker: str,
    confidence_score: float,
    sources: list,
    output_dir: str = "reports"
) -> str:
    """
    Main function. Takes synthesis agent output and generates a branded PDF.
    Returns the path to the generated file.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # File name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}\\{ticker}_research_report_{timestamp}.pdf"

    # Page setup
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.2 * cm,
        bottomMargin=1.8 * cm
    )

    styles = build_styles()
    story  = []

    # ── COVER SECTION ──────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))

    story.append(Paragraph("JPMorgan Chase &amp; Co.", styles["cover_title"]))
    story.append(Paragraph("Market Research Report", styles["cover_sub"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(HRFlowable(
        width="100%", thickness=2,
        color=JPMC_GOLD, spaceAfter=10
    ))

    story.append(Paragraph(f"<b>Research Query:</b> {query}", styles["body"]))
    story.append(Paragraph(f"<b>Ticker:</b> {ticker}", styles["cover_meta"]))
    story.append(Paragraph(
        f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        styles["cover_meta"]
    ))
    story.append(Paragraph(
        f"<b>Confidence Score:</b> {confidence_score:.2f} / 1.00",
        styles["confidence"]
    ))

    story.append(HRFlowable(
        width="100%", thickness=1,
        color=LIGHT_GRAY, spaceAfter=16
    ))

    # ── REPORT SECTIONS ────────────────────────────────────────────
    sections = parse_report_sections(final_report)

    if "__preamble__" in sections:
        story.extend(format_content_block(sections.pop("__preamble__"), styles))

    if "__full__" in sections:
        # No structured sections detected — dump full text
        story.append(Paragraph("Research Findings", styles["section_heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=JPMC_BLUE, spaceAfter=8))
        story.extend(format_content_block(sections["__full__"], styles))
    else:
        for heading, content in sections.items():
            story.append(Paragraph(heading, styles["section_heading"]))
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=JPMC_BLUE, spaceAfter=8
            ))
            story.extend(format_content_block(content, styles))

    # ── SOURCES SECTION ────────────────────────────────────────────
    if sources:
        story.append(Paragraph("Sources", styles["section_heading"]))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=JPMC_BLUE, spaceAfter=8
        ))
        seen = set()
        for src in sources:
            if src not in seen:
                seen.add(src)
                story.append(Paragraph(f"• {src}", styles["bullet"]))

    # ── DISCLAIMER ─────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=LIGHT_GRAY, spaceAfter=6
    ))
    story.append(Paragraph(
        "This report was generated by an AI-assisted research system for internal "
        "use only. All figures should be verified against primary sources before "
        "use in investment decisions. This document does not constitute financial advice.",
        styles["footer"]
    ))

    # ── BUILD PDF ──────────────────────────────────────────────────
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)

    return filename