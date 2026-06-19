import os
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Table, TableStyle, Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# Brand colors
JPMC_BLUE  = HexColor("#003087")
JPMC_GOLD  = HexColor("#B5A147")
DARK_GRAY  = HexColor("#333333")
LIGHT_GRAY = HexColor("#F5F5F5")
MID_GRAY   = HexColor("#888888")

# Ticker to company name mapping
TICKER_TO_COMPANY = {
    "JPM":   "JPMorgan Chase",
    "GS":    "Goldman Sachs",
    "MS":    "Morgan Stanley",
    "BAC":   "Bank of America",
    "WFC":   "Wells Fargo",
    "C":     "Citigroup",
    "BLK":   "BlackRock",
    "AXP":   "American Express",
    "SCHW":  "Charles Schwab",
    "USB":   "US Bancorp",
    "HSBC":  "HSBC",
    "BCS":   "Barclays",
    "DB":    "Deutsche Bank",
    "UBS":   "UBS",
    "PRU":   "Prudential",
    "MET":   "MetLife",
    "AIG":   "AIG",
    "BRK.B": "Berkshire Hathaway",
    "V":     "Visa",
    "MA":    "Mastercard",
    "PYPL":  "PayPal",
}

# Which synthesised section heading signals the Financial Analysis section
FINANCIAL_SECTION_PATTERN = re.compile(r"financial\s+summary", re.IGNORECASE)


def build_styles():
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
    styles["chart_caption"] = ParagraphStyle(
        "chart_caption",
        fontName="Helvetica",
        fontSize=8,
        textColor=MID_GRAY,
        spaceAfter=10,
        alignment=TA_CENTER
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
    Splits the synthesis agent output into named sections.
    Looks for numbered headings like 1. HEADING or 1. Heading.
    Falls back to returning the full text as one block if none found.
    """
    sections = {}
    pattern  = r'(\d+\.\s+[A-Z][^\n]+)'
    parts    = re.split(pattern, report_text)

    if len(parts) > 1:
        i = 1
        while i < len(parts) - 1:
            heading            = parts[i].strip()
            content            = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections[heading]  = content
            i += 2
        if parts[0].strip():
            sections["__preamble__"] = parts[0].strip()
    else:
        sections["__full__"] = report_text.strip()

    return sections


def format_content_block(text: str, styles: dict) -> list:
    """
    Converts a plain text block into ReportLab flowables.
    Lines starting with - * or bullet become bullet points.
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


def _chart_flowables(chart_paths: list, styles: dict, page_width: float) -> list:
    """
    Converts a list of PNG file paths into ReportLab Image flowables.
    Charts are laid out two-per-row where possible; the P/E chart (last
    one if it ends in 'pe_comparison') is always full-width.
    """
    flowables = []
    if not chart_paths:
        return flowables

    available_w = page_width - 3.6 * cm  # account for margins
    col_w = (available_w - 0.4 * cm) / 2  # two columns with a gap

    # Separate P/E chart from metric charts
    metric_charts = [p for p in chart_paths if "pe_comparison" not in p]
    pe_charts     = [p for p in chart_paths if "pe_comparison" in p]

    # Pair metric charts side-by-side
    i = 0
    while i < len(metric_charts):
        if i + 1 < len(metric_charts):
            left  = metric_charts[i]
            right = metric_charts[i + 1]
            try:
                img_l = Image(left,  width=col_w, height=col_w * 0.5)
                img_r = Image(right, width=col_w, height=col_w * 0.5)
                tbl = Table([[img_l, img_r]], colWidths=[col_w, col_w])
                tbl.setStyle(TableStyle([
                    ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING",   (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ]))
                flowables.append(KeepTogether([tbl, Spacer(1, 6)]))
            except Exception:
                pass
            i += 2
        else:
            try:
                img = Image(metric_charts[i], width=available_w, height=available_w * 0.28)
                flowables.append(KeepTogether([img, Spacer(1, 6)]))
            except Exception:
                pass
            i += 1

    # P/E chart full-width
    for pe_path in pe_charts:
        flowables.append(Spacer(1, 8))
        flowables.append(Paragraph("Competitor P/E Ratio Comparison", styles["section_heading"]))
        flowables.append(HRFlowable(width="100%", thickness=0.5, color=JPMC_BLUE, spaceAfter=8))
        try:
            img = Image(pe_path, width=available_w, height=available_w * 0.32)
            flowables.append(KeepTogether([img, Spacer(1, 6)]))
            flowables.append(Paragraph(
                "Gold bar = subject company. Blue bars = peers. Source: yfinance / market data.",
                styles["chart_caption"]
            ))
        except Exception:
            pass

    return flowables


def generate_pdf(
    final_report: str,
    query: str,
    ticker: str,
    confidence_score: float,
    sources: list,
    financial_charts: list = None,
    output_dir: str = "reports"
) -> str:
    """
    Generates a branded research report PDF.
    Company name is resolved dynamically from the ticker.
    Returns the path to the saved file.
    """
    financial_charts = financial_charts or []

    # ── Resolve company name from ticker ─────────────────────────
    company_name = TICKER_TO_COMPANY.get(ticker.upper(), ticker)

    # ── Output file setup ─────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(output_dir, f"{ticker}_research_report_{timestamp}.pdf")

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.2 * cm,
        bottomMargin=1.8 * cm
    )
    page_width = A4[0]

    styles = build_styles()
    story  = []

    # ── Cover section ─────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(company_name, styles["cover_title"]))
    story.append(Paragraph("Market Research Report", styles["cover_sub"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=JPMC_GOLD, spaceAfter=10))
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
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceAfter=16))

    # ── Report sections ───────────────────────────────────────────
    sections = parse_report_sections(final_report)

    if "__preamble__" in sections:
        story.extend(format_content_block(sections.pop("__preamble__"), styles))

    if "__full__" in sections:
        story.append(Paragraph("Research Findings", styles["section_heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=JPMC_BLUE, spaceAfter=8))
        story.extend(format_content_block(sections["__full__"], styles))
        # Append all charts at the end if there's no structured section to attach them to
        story.extend(_chart_flowables(financial_charts, styles, page_width))
    else:
        for heading, content in sections.items():
            story.append(Paragraph(heading, styles["section_heading"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=JPMC_BLUE, spaceAfter=8))
            story.extend(format_content_block(content, styles))

            # Inject charts immediately after the Financial Analysis section
            if FINANCIAL_SECTION_PATTERN.search(heading) and financial_charts:
                metric_paths = [p for p in financial_charts if "pe_comparison" not in p]
                story.extend(_chart_flowables(metric_paths, styles, page_width))

        # P/E chart always goes at the end of the financial block (handled inside _chart_flowables)
        if financial_charts and any("pe_comparison" in p for p in financial_charts):
            story.extend(_chart_flowables(
                [p for p in financial_charts if "pe_comparison" in p],
                styles, page_width
            ))

    # ── Sources section ───────────────────────────────────────────
    if sources:
        story.append(Paragraph("Sources", styles["section_heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=JPMC_BLUE, spaceAfter=8))
        seen = set()
        for src in sources:
            if src not in seen:
                seen.add(src)
                story.append(Paragraph(f"• {src}", styles["bullet"]))

    # ── Disclaimer ────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRAY, spaceAfter=6))
    story.append(Paragraph(
        f"This report was generated by an AI-assisted research system for internal "
        f"use only. Analysis covers {company_name} ({ticker}) based on publicly "
        f"available data and RAG-retrieved documents. All figures should be verified "
        f"against primary sources before use in investment decisions. "
        f"This document does not constitute financial advice.",
        styles["footer"]
    ))

    # ── Header and footer ─────────────────────────────────────────
    def header_footer(canvas, doc):
        canvas.saveState()
        width, height = A4

        canvas.setFillColor(JPMC_BLUE)
        canvas.rect(0, height - 1.2 * cm, width, 1.2 * cm, fill=1, stroke=0)

        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(HexColor("#FFFFFF"))
        canvas.drawString(1.5 * cm, height - 0.85 * cm, f"{company_name} — Market Research")

        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(
            width - 1.5 * cm,
            height - 0.85 * cm,
            "CONFIDENTIAL — Internal Use Only"
        )

        canvas.setStrokeColor(JPMC_GOLD)
        canvas.setLineWidth(1.5)
        canvas.line(0, height - 1.25 * cm, width, height - 1.25 * cm)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MID_GRAY)
        canvas.drawCentredString(
            width / 2,
            0.6 * cm,
            f"Page {doc.page} | Generated {datetime.now().strftime('%B %d, %Y')} | Internal Research"
        )
        canvas.restoreState()

    # ── Build PDF ─────────────────────────────────────────────────
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

    return filename
