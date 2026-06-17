import os
import re
from datetime import datetime
from scraper_config import (
    DOCUMENT_TYPES, OLDEST_YEAR_ALLOWED, MAX_QUARTERS_BACK,
    MAX_ANNUAL_REPORTS, MAX_PRESENTATION_AGE_YEARS,
    MAX_FILE_SIZE_MB, SKIP_ALREADY_DOWNLOADED,
    FORCE_REFRESH_DAYS, MAX_FILES_PER_COMPANY, CURRENT_YEAR
)

DOWNLOAD_DIR = "data"


def extract_year_from_text(text: str) -> int:
    """Extracts a 4-digit year from a URL or label."""
    matches = re.findall(r"20\d{2}", text)
    if matches:
        years = [int(y) for y in matches if 2010 <= int(y) <= CURRENT_YEAR + 1]
        if years:
            return max(years)
    return CURRENT_YEAR  # assume current year if not found


def extract_quarter_from_text(text: str) -> int:
    """Extracts quarter number (1-4) from a label or URL."""
    text = text.lower()
    if "q1" in text or "first quarter" in text or "march" in text:
        return 1
    if "q2" in text or "second quarter" in text or "june" in text:
        return 2
    if "q3" in text or "third quarter" in text or "september" in text:
        return 3
    if "q4" in text or "fourth quarter" in text or "december" in text:
        return 4
    return 0  # unknown quarter


def classify_document(label: str, url: str) -> str:
    """
    Classifies a document link into one of the DOCUMENT_TYPES keys.
    Returns the type string or 'unknown' if it doesn't match anything.
    """
    text = (label + " " + url).lower()

    if any(k in text for k in ["10-k", "10k", "annual report"]):
        return "10-K"
    if any(k in text for k in ["10-q", "10q", "quarterly report"]):
        return "10-Q"
    if any(k in text for k in ["investor day", "investor presentation"]):
        return "investor_day"
    if any(k in text for k in ["earnings release", "press release", "financial results"]):
        return "earnings_press_release"
    if any(k in text for k in ["earnings presentation", "earnings slides", "supplement"]):
        return "earnings_presentation"
    if any(k in text for k in ["proxy", "def 14a"]):
        return "proxy_statement"
    if any(k in text for k in ["8-k", "8k", "current report"]):
        return "8-K"

    return "unknown"


def is_doc_type_allowed(doc_type: str) -> bool:
    """Checks if this document type is enabled in config."""
    return DOCUMENT_TYPES.get(doc_type, False)


def is_year_allowed(year: int, doc_type: str) -> bool:
    """
    Checks if the document year falls within the allowed age range
    based on document type specific rules.
    """
    if year < OLDEST_YEAR_ALLOWED:
        return False

    if doc_type == "10-K":
        return year >= CURRENT_YEAR - MAX_ANNUAL_REPORTS

    if doc_type == "10-Q":
        # Allow quarters within the last MAX_QUARTERS_BACK quarters
        quarters_ago = (CURRENT_YEAR - year) * 4
        return quarters_ago <= MAX_QUARTERS_BACK

    if doc_type in ("investor_day", "earnings_presentation"):
        return year >= CURRENT_YEAR - MAX_PRESENTATION_AGE_YEARS

    return year >= OLDEST_YEAR_ALLOWED


def is_already_downloaded(filename: str) -> bool:
    """
    Checks if a file already exists in the data\ folder.
    Respects the FORCE_REFRESH_DAYS setting.
    """
    if not SKIP_ALREADY_DOWNLOADED:
        return False

    filepath = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(filepath):
        return False

    if FORCE_REFRESH_DAYS > 0:
        age_days = (
            datetime.now() -
            datetime.fromtimestamp(os.path.getmtime(filepath))
        ).days
        if age_days > FORCE_REFRESH_DAYS:
            print(f"  [Filter] {filename} is {age_days} days old — will refresh")
            return False

    return True


def filter_documents(raw_links: list) -> list:
    """
    Main filter function. Takes raw scraped links and returns
    only the ones that pass all config rules.

    Each link is a dict with 'label' and 'url' keys.
    Returns filtered list with added metadata.
    """
    approved    = []
    rejected    = []
    type_counts = {}

    for link in raw_links:
        label = link.get("label", "")
        url   = link.get("url", "")

        # Classify the document
        doc_type = classify_document(label, url)
        year     = extract_year_from_text(label + " " + url)

        rejection_reason = None

        # Rule 1 — must be a known, enabled document type
        if doc_type == "unknown":
            rejection_reason = "unknown document type"
        elif not is_doc_type_allowed(doc_type):
            rejection_reason = f"{doc_type} disabled in config"

        # Rule 2 — must be within allowed age range
        elif not is_year_allowed(year, doc_type):
            rejection_reason = f"year {year} outside allowed range for {doc_type}"

        # Rule 3 — volume cap per type
        elif type_counts.get(doc_type, 0) >= get_type_limit(doc_type):
            rejection_reason = f"already have enough {doc_type} files"

        # Rule 4 — already downloaded
        else:
            safe_label = re.sub(r"[^a-zA-Z0-9]", "_", label.lower())[:40]
            filename   = f"{safe_label}_{year}.pdf"
            if is_already_downloaded(filename):
                rejection_reason = "already downloaded"

        if rejection_reason:
            rejected.append({**link, "reason": rejection_reason})
        else:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            approved.append({
                **link,
                "doc_type": doc_type,
                "year":     year,
            })

        # Hard cap on total downloads per company
        if len(approved) >= MAX_FILES_PER_COMPANY:
            print(f"  [Filter] Reached max {MAX_FILES_PER_COMPANY} files per company")
            break

    # Print summary
    print(f"\n  [Filter] Approved : {len(approved)} documents")
    print(f"  [Filter] Rejected : {len(rejected)} documents")
    if rejected:
        print("  [Filter] Rejection reasons:")
        for r in rejected[:5]:  # show first 5
            print(f"           - {r['label'][:50]} → {r['reason']}")

    return approved


def get_type_limit(doc_type: str) -> int:
    """Returns the max number of files allowed per document type."""
    limits = {
        "10-K":                  MAX_ANNUAL_REPORTS,
        "10-Q":                  MAX_QUARTERS_BACK,
        "earnings_presentation": 4,
        "earnings_press_release":4,
        "investor_day":          2,
        "proxy_statement":       1,
        "8-K":                   3,
    }
    return limits.get(doc_type, 2)