import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from state import MarketResearchState
import requests

llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)

def fetch_sec_summary(ticker: str) -> str:
    url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "jpmc-research@example.com"}
        )
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])[:5]
        dates = filings.get("filingDate", [])[:5]
        return "\n".join(f"{f} filed {d}" for f, d in zip(forms, dates))
    except Exception:
        return "SEC data unavailable"

def financial_analysis_node(state: MarketResearchState) -> dict:
    ticker = state["ticker"]
    sec_summary = fetch_sec_summary(ticker)

    prompt = f"""You are a JPMC financial analyst.
Analyze the financial position of {ticker}.

Recent SEC filings:
{sec_summary}

Cover:
1. Revenue and earnings trends
2. Key ratios (P/E, debt-to-equity, free cash flow)
3. Balance sheet strengths and risks
4. Year-over-year changes
"""
    result = llm.invoke(prompt)
    return {
        "financial_data": result.content,
        "sources": state["sources"] + [f"SEC/{ticker}"]
    }