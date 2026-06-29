import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import MarketResearchState
from llm_factory import get_llm
import requests
import json
import re
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

llm = get_llm("fast")

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ── Period label helpers ──────────────────────────────────────────────

def get_period_labels() -> dict:
    """Pure Python — generates period labels for last 3 years and quarters."""
    current_year  = datetime.now().year
    current_month = datetime.now().month
    quarter       = (current_month - 1) // 3 + 1

    years = [str(current_year - i) for i in range(3)]

    quarters = []
    q = quarter
    y = current_year
    for _ in range(3):
        quarters.append(f"Q{q} {y}")
        q -= 1
        if q == 0:
            q = 4
            y -= 1

    return {
        "year_1": years[0], "year_2": years[1], "year_3": years[2],
        "quarter_1": quarters[0], "quarter_2": quarters[1], "quarter_3": quarters[2]
    }


# ── Step 1: yfinance — primary data source, zero LLM tokens ──────────

def fetch_financial_data_yfinance(ticker: str) -> dict:
    """
    Fetches real financial figures directly from yfinance.
    Returns annual and quarterly data for Revenue, EBITDA, COGS, SG&A.
    This is the primary data source - no LLM involved at all.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [Financial Agent] yfinance not installed - pip install yfinance")
        return {"annual": [], "quarterly": []}

    try:
        stock = yf.Ticker(ticker)

        annual_income    = stock.income_stmt
        quarterly_income = stock.quarterly_income_stmt

        if annual_income is None or annual_income.empty:
            print(f"  [Financial Agent] yfinance returned no annual data for {ticker}")
            return {"annual": [], "quarterly": []}

        def extract_periods(df, max_periods: int) -> list:
            """Converts a yfinance income statement DataFrame into our record format."""
            records = []
            columns = list(df.columns)[:max_periods]

            for col in columns:
                period_label = col.strftime("%Y") if hasattr(col, "strftime") else str(col)

                def get_value(row_names):
                    for name in row_names:
                        if name in df.index:
                            val = df.loc[name, col]
                            if val is not None and str(val) != "nan":
                                return round(float(val) / 1_000_000, 2)  # convert to millions
                    return None

                revenue = get_value(["Total Revenue", "Operating Revenue"])

                ebitda = get_value(["EBITDA", "Normalized EBITDA"])
                if ebitda is None:
                    # Banks have no direct EBITDA row — derive from Pretax Income + D&A
                    pretax = get_value(["Pretax Income", "Net Income Before Taxes"])
                    depn   = get_value(["Reconciled Depreciation", "Depreciation And Amortization", "Depreciation"])
                    if pretax is not None:
                        ebitda = round(pretax + (depn or 0), 2)

                # Banks have no COGS; Interest Expense is the cost-of-funding proxy
                cogs = get_value([
                    "Cost Of Revenue", "Reconciled Cost Of Revenue",
                    "Cost Of Goods Sold", "Interest Expense",
                ])

                sga = get_value(["Selling General And Administration", "SG&A Expense"])

                records.append({
                    "period":  period_label,
                    "revenue": revenue,
                    "ebitda":  ebitda,
                    "cogs":    cogs,
                    "sga":     sga
                })

            return records

        annual_data = extract_periods(annual_income, 3)

        quarterly_data = []
        if quarterly_income is not None and not quarterly_income.empty:
            quarterly_data = extract_periods(quarterly_income, 3)
            # Relabel quarterly periods properly since strftime gives year only
            for i, col in enumerate(list(quarterly_income.columns)[:3]):
                if hasattr(col, "strftime"):
                    quarter_num = (col.month - 1) // 3 + 1
                    quarterly_data[i]["period"] = f"Q{quarter_num} {col.year}"

        return {"annual": annual_data, "quarterly": quarterly_data}

    except Exception as e:
        print(f"  [Financial Agent] yfinance error for {ticker}: {e}")
        return {"annual": [], "quarterly": []}


def fetch_pe_ratio(ticker: str) -> float:
    """Fetches current P/E ratio via yfinance. Zero LLM tokens."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        pe = stock.info.get("trailingPE")
        return round(pe, 2) if pe else None
    except Exception:
        return None


COMPETITOR_MAP = {
    "GS":  ["MS", "JPM", "BAC"],
    "MS":  ["GS", "JPM", "BAC"],
    "JPM": ["GS", "MS", "BAC", "WFC"],
    "BAC": ["JPM", "WFC", "C"],
    "WFC": ["BAC", "JPM", "C"],
    "C":   ["BAC", "WFC", "JPM"],
}


# ── Step 2: HTML table extraction (pandas, zero LLM tokens) ──────────

def extract_numbers_from_row(row) -> list:
    """Pulls numeric values from a pandas row, ignoring text cells."""
    numbers = []
    for val in row.values:
        cleaned = (
            str(val)
            .replace(",", "")
            .replace("$", "")
            .replace("(", "-")
            .replace(")", "")
            .strip()
        )
        try:
            num = float(cleaned)
            if abs(num) > 1:
                numbers.append(num)
        except ValueError:
            continue
    return numbers


def guess_period_from_filename(filepath: str) -> str:
    """Extracts year from filename for period labeling."""
    match = re.search(r"(20\d{2})", filepath)
    return match.group(1) if match else None


def extract_financials_from_html_tables(ticker: str) -> dict:
    """
    Second data source. Attempts to parse Revenue, EBITDA, COGS, SG&A
    directly from indexed HTML SEC filings using pandas table parsing.
    Zero LLM tokens. Only called if yfinance found nothing.
    """
    try:
        import pandas as pd
    except ImportError:
        return {"annual": [], "quarterly": []}

    html_files = []
    for pattern in [ticker.lower(), ticker.upper()]:
        html_files.extend(glob.glob(f"data/*{pattern}*.html"))
        html_files.extend(glob.glob(f"data/*{pattern}*.htm"))

    if not html_files:
        return {"annual": [], "quarterly": []}

    metric_keywords = {
        "revenue": ["total revenue", "net revenue", "total net revenue", "net sales"],
        "ebitda":  ["ebitda", "earnings before interest", "operating income"],
        "cogs":    ["cost of goods sold", "cost of revenue", "cost of sales", "cost of products"],
        "sga":     ["selling, general and administrative", "sg&a", "general and administrative"],
    }

    found_data    = {key: None for key in metric_keywords}
    period_labels = []

    for html_file in html_files:
        period = guess_period_from_filename(html_file)
        if period and period not in period_labels:
            period_labels.append(period)

        try:
            tables = pd.read_html(html_file)
        except Exception:
            continue

        for table in tables:
            try:
                table_str = table.astype(str).apply(lambda col: col.str.lower()).to_string()
            except Exception:
                continue

            for metric_key, keywords in metric_keywords.items():
                if found_data[metric_key] is not None:
                    continue
                for keyword in keywords:
                    if keyword in table_str:
                        for _, row in table.iterrows():
                            row_str = " ".join(str(v).lower() for v in row.values)
                            if keyword in row_str:
                                numbers = extract_numbers_from_row(row)
                                if numbers:
                                    found_data[metric_key] = numbers[0]
                                break
                        break

    if any(v is not None for v in found_data.values()):
        period = period_labels[0] if period_labels else "Latest"
        record = {
            "period":  period,
            "revenue": found_data["revenue"],
            "ebitda":  found_data["ebitda"],
            "cogs":    found_data["cogs"],
            "sga":     found_data["sga"],
        }
        return {"annual": [record], "quarterly": []}

    return {"annual": [], "quarterly": []}


# ── Step 3: LLM fallback — last resort only ───────────────────────────

def extract_financial_metrics_llm(rag_context: str, ticker: str) -> dict:
    """
    True last resort. Only called when both yfinance and pandas table
    parsing found nothing. Parses unstructured prose from RAG-retrieved
    PDF text where regex/pandas cannot reliably find the numbers.
    Uses Haiku since this is extraction, not reasoning.
    """
    if not rag_context.strip():
        print(f"  [Financial Agent] No RAG context available for LLM fallback")
        return {"annual": [], "quarterly": []}

    periods = get_period_labels()

    prompt = f"""Extract financial figures for {ticker} from the text below.
Return ONLY valid JSON, no markdown, no explanation, no extra text.

Text:
{rag_context}

JSON structure:
{{
    "annual": [
        {{"period": "{periods['year_1']}", "revenue": null, "ebitda": null, "cogs": null, "sga": null}},
        {{"period": "{periods['year_2']}", "revenue": null, "ebitda": null, "cogs": null, "sga": null}},
        {{"period": "{periods['year_3']}", "revenue": null, "ebitda": null, "cogs": null, "sga": null}}
    ],
    "quarterly": [
        {{"period": "{periods['quarter_1']}", "revenue": null, "ebitda": null, "cogs": null, "sga": null}},
        {{"period": "{periods['quarter_2']}", "revenue": null, "ebitda": null, "cogs": null, "sga": null}},
        {{"period": "{periods['quarter_3']}", "revenue": null, "ebitda": null, "cogs": null, "sga": null}}
    ]
}}

Rules:
- All figures in USD millions
- Use null for any figure not found in the text
- Do not estimate or fabricate - null only if uncertain
"""
    result  = llm.invoke(prompt)
    cleaned = re.sub(r"```json|```", "", result.content).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  [Financial Agent] LLM returned invalid JSON, using empty result")
        return {"annual": [], "quarterly": []}


# ── Step 4: SG&A fallback calculation ──────────────────────────────────

def apply_sga_fallback(records: list) -> list:
    """
    If SG&A is None for a period but revenue and COGS are both
    available, estimates SG&A as 10% of operating expense
    (revenue minus COGS). Pure Python, zero tokens.
    """
    for record in records:
        if record.get("sga") is None:
            revenue = record.get("revenue")
            cogs    = record.get("cogs")

            if revenue is not None and cogs is not None:
                operating_expense      = revenue - cogs
                record["sga"]          = round(operating_expense * 0.10, 2)
                record["sga_estimated"] = True
            else:
                record["sga_estimated"] = False
        else:
            record["sga_estimated"] = False

    return records


# ── Step 5: Chart generation ───────────────────────────────────────────

def generate_metric_charts(annual_data: list, quarterly_data: list, ticker: str) -> list:
    """
    Generates one chart per metric (Revenue, EBITDA, COGS, SG&A).
    Skips a metric entirely if all values are None across both
    annual and quarterly periods. Pure matplotlib, zero LLM tokens.
    """
    metrics = {
        "revenue": "Revenue (USD millions)",
        "ebitda":  "EBITDA (USD millions)",
        "cogs":    "COGS (USD millions)",
        "sga":     "SG&A (USD millions)",
    }

    chart_paths = []

    for metric_key, metric_label in metrics.items():
        annual_labels    = [r["period"] for r in annual_data]
        annual_values    = [r.get(metric_key) for r in annual_data]
        annual_estimated = [r.get("sga_estimated", False) for r in annual_data]

        quarterly_labels    = [r["period"] for r in quarterly_data]
        quarterly_values    = [r.get(metric_key) for r in quarterly_data]
        quarterly_estimated = [r.get("sga_estimated", False) for r in quarterly_data]

        all_values = annual_values + quarterly_values
        if all(v is None for v in all_values):
            print(f"  [Charts] Skipping {metric_label} - no data available")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.35})
        fig.suptitle(f"{ticker} — {metric_label}", fontsize=12, fontweight="bold", y=1.02)

        clean_annual_labels, clean_annual_values, clean_annual_est = [], [], []
        for label, val, est in zip(annual_labels, annual_values, annual_estimated):
            if val is not None:
                clean_annual_labels.append(label)
                clean_annual_values.append(val)
                clean_annual_est.append(est)

        if clean_annual_values:
            bar_colors = ["#B5A147" if est else "#003087" for est in clean_annual_est]
            bars = ax1.bar(clean_annual_labels, clean_annual_values, color=bar_colors)
            ax1.set_title("Annual (last 3 years)", fontsize=10)
            ax1.set_ylabel("USD millions", fontsize=8)
            ax1.tick_params(axis="x", labelsize=8)
            ax1.tick_params(axis="y", labelsize=8)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            for bar, val, est in zip(bars, clean_annual_values, clean_annual_est):
                label_text = f"{val:,.0f}" + ("*" if est else "")
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label_text,
                         ha="center", va="bottom", fontsize=7)
        else:
            ax1.set_visible(False)

        clean_quarterly_labels, clean_quarterly_values, clean_quarterly_est = [], [], []
        for label, val, est in zip(quarterly_labels, quarterly_values, quarterly_estimated):
            if val is not None:
                clean_quarterly_labels.append(label)
                clean_quarterly_values.append(val)
                clean_quarterly_est.append(est)

        if clean_quarterly_values:
            bar_colors = ["#B5A147" if est else "#003087" for est in clean_quarterly_est]
            bars = ax2.bar(clean_quarterly_labels, clean_quarterly_values, color=bar_colors)
            ax2.set_title("Quarterly (last 3 quarters)", fontsize=10)
            ax2.set_ylabel("USD millions", fontsize=8)
            ax2.tick_params(axis="x", labelsize=8)
            ax2.tick_params(axis="y", labelsize=8)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            for bar, val, est in zip(bars, clean_quarterly_values, clean_quarterly_est):
                label_text = f"{val:,.0f}" + ("*" if est else "")
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label_text,
                         ha="center", va="bottom", fontsize=7)
        else:
            ax2.set_visible(False)

        all_estimated = clean_annual_est + clean_quarterly_est
        if any(all_estimated):
            fig.text(0.01, -0.02,
                     "* SG&A estimated as 10% of operating expense "
                     "(revenue minus COGS) - not explicitly disclosed",
                     fontsize=7, color="#888888")

        plt.tight_layout()
        chart_path = os.path.join(CHART_DIR, f"{ticker}_{metric_key}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        chart_paths.append(chart_path)
        print(f"  [Charts] Saved: {chart_path}")

    return chart_paths


# ── Helper: fill in missing periods with skeleton data ────────────────

def ensure_period_skeleton(annual_data: list, quarterly_data: list) -> tuple:
    """
    If a data source returned fewer than 3 periods, fills in the
    remaining period labels with None values so charts still show
    the correct period labels even when data is missing.
    """
    periods = get_period_labels()

    if not annual_data:
        annual_data = [
            {"period": periods["year_1"], "revenue": None, "ebitda": None, "cogs": None, "sga": None},
            {"period": periods["year_2"], "revenue": None, "ebitda": None, "cogs": None, "sga": None},
            {"period": periods["year_3"], "revenue": None, "ebitda": None, "cogs": None, "sga": None},
        ]

    if not quarterly_data:
        quarterly_data = [
            {"period": periods["quarter_1"], "revenue": None, "ebitda": None, "cogs": None, "sga": None},
            {"period": periods["quarter_2"], "revenue": None, "ebitda": None, "cogs": None, "sga": None},
            {"period": periods["quarter_3"], "revenue": None, "ebitda": None, "cogs": None, "sga": None},
        ]

    return annual_data, quarterly_data


# ── Main node ─────────────────────────────────────────────────────────

def financial_analysis_node(state: MarketResearchState) -> dict:
    ticker  = state["ticker"]
    rag_ctx = state.get("rag_context", "") or ""

    # ── Step 1: Try yfinance first - real numbers, zero LLM tokens ──
    print(f"  [Financial Agent] Fetching data from yfinance for {ticker}...")
    yfinance_result = fetch_financial_data_yfinance(ticker)

    annual_data    = yfinance_result.get("annual", [])
    quarterly_data = yfinance_result.get("quarterly", [])

    has_data = any(
        r.get("revenue") is not None for r in annual_data + quarterly_data
    )

    # ── Step 2: Fall back to pandas HTML table parsing ───────────────
    if not has_data:
        print(f"  [Financial Agent] yfinance found nothing usable, trying HTML tables...")
        html_result = extract_financials_from_html_tables(ticker)
        annual_data    = html_result.get("annual", []) or annual_data
        quarterly_data = html_result.get("quarterly", []) or quarterly_data
        has_data = any(
            r.get("revenue") is not None for r in annual_data + quarterly_data
        )

    # ── Step 3: True last resort - LLM extraction from RAG context ──
    if not has_data:
        print(f"  [Financial Agent] HTML tables found nothing, falling back to LLM extraction...")
        llm_result = extract_financial_metrics_llm(rag_ctx, ticker)
        annual_data    = llm_result.get("annual", []) or annual_data
        quarterly_data = llm_result.get("quarterly", []) or quarterly_data
    else:
        print(f"  [Financial Agent] Real financial data found - no LLM call needed")

    # ── Step 4: Ensure period labels exist even if all data is None ──
    annual_data, quarterly_data = ensure_period_skeleton(annual_data, quarterly_data)

    # ── Step 5: Apply SG&A fallback calculation ───────────────────────
    annual_data    = apply_sga_fallback(annual_data)
    quarterly_data = apply_sga_fallback(quarterly_data)

    # ── Step 6: Fetch P/E for company and competitors ─────────────────
    company_pe   = fetch_pe_ratio(ticker)
    competitors  = COMPETITOR_MAP.get(ticker, [])
    competitor_pe = {c: fetch_pe_ratio(c) for c in competitors}

    # ── Step 7: Generate charts ────────────────────────────────────────
    chart_paths = generate_metric_charts(annual_data, quarterly_data, ticker)

    return {
        "financial_data": "",  # no prose generated - charts speak for themselves
        "financial_metrics": {
            "annual":    annual_data,
            "quarterly": quarterly_data
        },
        "company_pe": company_pe,
        "competitor_pe": competitor_pe,
        "chart_paths": chart_paths,
        "sources": state["sources"] + [f"yfinance/{ticker}", f"SEC/{ticker}"]
    }