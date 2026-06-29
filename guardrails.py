import os
import re
from datetime import datetime, timedelta


# ── 1. Ticker validation ────────────────────────────────────────────────

def validate_ticker(ticker: str) -> tuple[bool, str]:
    try:
        import yfinance as yf
        price = yf.Ticker(ticker).fast_info.last_price
        if price is None:
            return False, f"Ticker '{ticker}' has no market data. Check the symbol and try again."
        return True, ""
    except Exception as e:
        return False, f"Could not validate ticker '{ticker}': {e}"


# ── 2. Query relevance check ────────────────────────────────────────────

_IRRELEVANT_TOPICS = [
    "recipe", "cook", "bake", "food", "weather", "forecast rain",
    "poem", "poetry", "song", "lyrics", "joke", "story", "fiction",
    "movie", "tv show", "celebrity", "sports score", "horoscope",
    "translate", "grammar", "homework", "math problem",
]

_MIN_QUERY_WORDS = 3

def check_query_relevance(query: str) -> tuple[bool, str]:
    q = query.lower().strip()
    if len(q.split()) < _MIN_QUERY_WORDS:
        return False, "Query is too short — please provide a meaningful question about the company."
    if any(topic in q for topic in _IRRELEVANT_TOPICS):
        return False, "Query is unrelated to the company."
    return True, ""


# ── 3. Confidence score threshold ───────────────────────────────────────

_CONFIDENCE_THRESHOLD = 0.65

def check_confidence_threshold(score: float) -> tuple[bool, str]:
    if score >= _CONFIDENCE_THRESHOLD:
        return True, ""
    return False, (
        f"Confidence score is low ({score:.2f} < {_CONFIDENCE_THRESHOLD}). "
        "Report may be incomplete — limited sources or missing financial data."
    )


# ── 4. Empty section detection ──────────────────────────────────────────

_FILLER = "Not researched for this query."
_MAX_EMPTY = 2

def check_empty_sections(report: str) -> tuple[bool, str]:
    count = report.count(_FILLER)
    if count <= _MAX_EMPTY:
        return True, ""
    return False, (
        f"{count} report sections contain '{_FILLER}'. "
        "Consider broadening the query or checking agent routing."
    )


# ── 5. Source minimum ────────────────────────────────────────────────────

_MIN_SOURCES = 2

def check_source_minimum(sources: list) -> tuple[bool, str]:
    if len(sources) >= _MIN_SOURCES:
        return True, ""
    return False, (
        f"Only {len(sources)} source(s) cited (minimum {_MIN_SOURCES}). "
        "Report reliability may be limited."
    )


# ── 6. RAG staleness ────────────────────────────────────────────────────

_MAX_STALENESS_DAYS = 7

def check_rag_staleness(persist_dir: str) -> tuple[bool, str]:
    if not os.path.exists(persist_dir):
        return False, "Vector store does not exist — RAG context will be empty."
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(persist_dir))
    if age <= timedelta(days=_MAX_STALENESS_DAYS):
        return True, ""
    return False, (
        f"Vector store last updated {age.days} day(s) ago "
        f"(threshold: {_MAX_STALENESS_DAYS} days). "
        "Consider re-indexing documents for fresher context."
    )


# ── 7. LLM output length check ───────────────────────────────────────────

_SECTION_MAX_CHARS = 600

def check_output_length(report: str) -> tuple[bool, str]:
    headings  = re.findall(r'(\d+\.\s+[A-Z][A-Z &]+)', report)
    sections  = re.split(r'\d+\.\s+[A-Z][A-Z &]+', report)
    overlong  = []
    for i, section in enumerate(sections[1:]):
        if len(section.strip()) > _SECTION_MAX_CHARS:
            label = headings[i] if i < len(headings) else f"Section {i + 1}"
            overlong.append(label.strip())
    if not overlong:
        return True, ""
    return False, (
        f"Section(s) exceeded {_SECTION_MAX_CHARS}-char limit: {', '.join(overlong)}. "
        "Model may have ignored output constraints."
    )


# ── Print helper ─────────────────────────────────────────────────────────

def _print(name: str, passed: bool, message: str) -> bool:
    status = "OK  " if passed else "WARN"
    print(f"  [{status}] {name}" + (f": {message}" if not passed else ""))
    return passed


def run_input_guardrails(ticker: str, query: str) -> tuple[bool, list[str]]:
    """
    Runs guardrails that apply before the pipeline starts.
    Returns (all_blocking_passed, list_of_warnings).
    Ticker validation is blocking; query relevance is a warning only.
    """
    print("\n[Guardrails] Validating input...")
    warnings = []
    blocking_ok = True

    ok, msg = validate_ticker(ticker)
    if not _print("Ticker validation", ok, msg):
        blocking_ok = False   # hard stop — nothing useful can run

    ok, msg = check_query_relevance(query)
    if not _print("Query relevance", ok, msg):
        blocking_ok = False   # hard stop — irrelevant query

    return blocking_ok, warnings


def run_rag_guardrail(persist_dir: str) -> None:
    """Runs the RAG staleness check and prints result (warning only)."""
    print("\n[Guardrails] Checking RAG freshness...")
    ok, msg = check_rag_staleness(persist_dir)
    _print("RAG staleness", ok, msg)


def run_output_guardrails(result: dict) -> list[str]:
    """
    Runs all post-research guardrails.
    Returns a list of warning messages (non-empty means something flagged).
    All are warnings — the PDF is still generated.
    """
    print("\n[Guardrails] Checking research output...")
    report  = result.get("final_report", "")
    score   = result.get("confidence_score", 0.0)
    sources = result.get("sources", [])
    warnings = []

    for name, fn, args in [
        ("Confidence score",   check_confidence_threshold, (score,)),
        ("Empty sections",     check_empty_sections,       (report,)),
        ("Source minimum",     check_source_minimum,       (sources,)),
        ("Output length",      check_output_length,        (report,)),
    ]:
        ok, msg = fn(*args)
        _print(name, ok, msg)
        if not ok:
            warnings.append(msg)

    return warnings
