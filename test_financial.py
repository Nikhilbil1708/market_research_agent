"""
Test the financial agent without any LLM calls.
The LLM is patched to raise immediately if triggered,
so the test fails loudly if the yfinance path doesn't cover everything.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from unittest.mock import patch, MagicMock

# ── Patch LLM to blow up if called ───────────────────────────────────────
def _no_llm_allowed(*args, **kwargs):
    raise RuntimeError("LLM was called — yfinance data should have been sufficient.")

fake_llm = MagicMock()
fake_llm.invoke.side_effect = _no_llm_allowed

with patch("agents.financial_agent.llm", fake_llm):
    from agents.financial_agent import (
        fetch_financial_data_yfinance,
        fetch_pe_ratio,
        apply_sga_fallback,
        generate_metric_charts,
        ensure_period_skeleton,
        financial_analysis_node,
    )

    TICKER = "JPM"

    # ── 1. yfinance data fetch ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"1. Fetching yfinance data for {TICKER}...")
    result = fetch_financial_data_yfinance(TICKER)
    annual    = result.get("annual", [])
    quarterly = result.get("quarterly", [])

    print(f"   Annual periods   : {[r['period'] for r in annual]}")
    print(f"   Quarterly periods: {[r['period'] for r in quarterly]}")
    for r in annual:
        print(f"   {r['period']}: revenue={r['revenue']}M  ebitda={r['ebitda']}M  "
              f"cogs={r['cogs']}M  sga={r['sga']}M")

    # ── 2. P/E ratio ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"2. Fetching P/E ratio for {TICKER}...")
    pe = fetch_pe_ratio(TICKER)
    print(f"   P/E: {pe}")

    # ── 3. SGA fallback calculation ───────────────────────────────────────
    print(f"\n{'='*50}")
    print("3. Testing SGA fallback on records missing SGA...")
    test_records = [
        {"period": "2024", "revenue": 1000.0, "ebitda": 200.0, "cogs": 600.0, "sga": None},
        {"period": "2023", "revenue": 900.0,  "ebitda": 180.0, "cogs": 550.0, "sga": 80.0},
        {"period": "2022", "revenue": None,   "ebitda": None,  "cogs": None,   "sga": None},
    ]
    patched = apply_sga_fallback(test_records)
    for r in patched:
        estimated = r.get("sga_estimated", False)
        print(f"   {r['period']}: sga={r['sga']}  estimated={estimated}")

    # ── 4. Chart generation ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("4. Generating charts from yfinance data...")
    annual_with_sga    = apply_sga_fallback(annual)
    quarterly_with_sga = apply_sga_fallback(quarterly)
    annual_f, quarterly_f = ensure_period_skeleton(annual_with_sga, quarterly_with_sga)
    chart_paths = generate_metric_charts(annual_f, quarterly_f, TICKER)
    print(f"   Charts saved: {chart_paths}")

    # ── 5. Full node (end-to-end, no LLM) ────────────────────────────────
    print(f"\n{'='*50}")
    print("5. Running full financial_analysis_node...")
    state = {
        "ticker": TICKER,
        "query": "earnings and revenue",
        "rag_context": "",
        "sources": [],
    }
    node_result = financial_analysis_node(state)
    print(f"   company_pe   : {node_result['company_pe']}")
    print(f"   competitor_pe: {node_result['competitor_pe']}")
    print(f"   chart_paths  : {node_result['chart_paths']}")
    print(f"   annual rows  : {len(node_result['financial_metrics']['annual'])}")
    print(f"   quarterly rows: {len(node_result['financial_metrics']['quarterly'])}")

print(f"\n{'='*50}")
print("All tests passed — no LLM calls were made.")
