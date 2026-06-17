from datetime import datetime

# ── Document Type Controls ───────────────────────────────────────────
# Set to True to download, False to skip

DOCUMENT_TYPES = {
    "10-K":                    True,   # Annual reports
    "10-Q":                    True,   # Quarterly reports
    "earnings_presentation":   True,   # Earnings slides
    "earnings_press_release":  False,  # Press releases
    "investor_day":            True,   # Investor day presentations
    "proxy_statement":         False,  # DEF 14A — governance docs
    "8-K":                     False,  # Current reports
    "supplement":              False,  # Financial supplements
}

# ── Age Controls ─────────────────────────────────────────────────────
CURRENT_YEAR = datetime.now().year

# Oldest year allowed — documents older than this are skipped
OLDEST_YEAR_ALLOWED = CURRENT_YEAR - 2      # e.g. 2023 if current year is 2025

# For quarterly reports specifically — how many quarters back
MAX_QUARTERS_BACK = 4                        # last 4 quarters only

# For annual reports — how many years back
MAX_ANNUAL_REPORTS = 2                       # last 2 annual reports only

# For presentations — how many years back
MAX_PRESENTATION_AGE_YEARS = 1              # only last year's presentations

# ── Volume Controls ───────────────────────────────────────────────────
MAX_FILES_PER_COMPANY = 10                  # hard cap on total downloads
MAX_FILE_SIZE_MB = 50                       # skip files larger than this

# ── Freshness Controls ────────────────────────────────────────────────
SKIP_ALREADY_DOWNLOADED = True             # don't re-download existing files
FORCE_REFRESH_DAYS = 30                    # re-download if file is older than
                                           # this many days (0 = never refresh)