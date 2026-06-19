import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from state import MarketResearchState

llm = ChatAnthropic(
    model="claude-opus-4-8",
    temperature=0
)


# ── Confidence score — pure Python, zero LLM tokens ───────────────────

def calculate_confidence_score(state: dict) -> float:
    """
    Computes confidence from concrete signals already present in state.
    No LLM call, no hallucinated number - this is a deterministic
    score based on data availability and source count.
    """
    score = 0.4   # baseline

    news_analysis = state.get("news_analysis", "") or ""
    if news_analysis and "no relevant news" not in news_analysis.lower():
        score += 0.10

    if len(state.get("sources", [])) >= 5:
        score += 0.15
    elif len(state.get("sources", [])) >= 2:
        score += 0.08

    if state.get("rag_context"):
        score += 0.15

    if state.get("company_pe") is not None:
        score += 0.10

    metrics = state.get("financial_metrics", {})
    annual_data = metrics.get("annual", [])
    has_real_financials = any(
        r.get("revenue") is not None and not r.get("revenue_estimated", False)
        for r in annual_data
    )
    if has_real_financials:
        score += 0.10

    chart_paths = state.get("chart_paths", [])
    if chart_paths:
        score += 0.05

    return min(round(score, 2), 1.0)


# ── Synthesis node ──────────────────────────────────────────────────────

def synthesis_node(state: MarketResearchState) -> dict:
    query  = state.get("query", "")
    ticker = state.get("ticker", "")

    prompt = f"""Research question: {query}
Company: {ticker}

Synthesize the sections below into a final report.
Use ONLY what is in the sections provided.
Do not add general knowledge, do not add caveats or disclaimers.
Do not write transitions between sections.
Go directly from one section heading to its content.

SOURCE MATERIAL:

COMPANY OVERVIEW AND NEWS:
{state.get('news_analysis') or 'Not researched for this query.'}

TECHNOLOGY AND IT STRATEGY:
{state.get('tech_strategy_context') or 'Not researched for this query.'}

FINANCIAL DATA:
{state.get('financial_data') or 'See financial charts in this report for Revenue, EBITDA, COGS, and SG&A figures across the last 3 years and 3 quarters.'}

MACRO AND SECTOR CONTEXT:
{state.get('macro_context') or 'Not researched for this query.'}

DOCUMENT CONTEXT (RAG):
{state.get('rag_context') or 'No documents retrieved.'}

Output in exactly this order, with these exact headings.
Hard limits apply to every section - do not exceed them:

1. COMPANY OVERVIEW
[copy directly from source material, 2 sentences maximum]

2. TECHNOLOGY AND IT STRATEGY
[copy directly from source material, bullet list only, 5 bullets maximum]

3. FINANCIAL SUMMARY
[1 sentence noting that detailed figures are shown in the accompanying charts]
[if FINANCIAL DATA above contains actual figures instead of the default message, copy them directly]

4. MACRO AND SECTOR CONTEXT
[copy directly from source material, 3 bullets maximum]

5. KEY FINDINGS
[your synthesis across all sections, 4 bullets maximum, 20 words each]

6. RISK FACTORS
[3 bullets maximum, named specific risks only, no generic statements]

7. SOURCES
[list only, no descriptions, one per line]
"""

    result = llm.invoke(prompt)

    return {
        "final_report": result.content,
        "confidence_score": calculate_confidence_score(state)   # ← Python, not LLM
    }