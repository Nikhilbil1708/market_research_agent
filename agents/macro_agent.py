import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.tools.tavily_search import TavilySearchResults
from state import MarketResearchState
from llm_factory import get_llm

llm = get_llm("fast")
search_tool = TavilySearchResults(max_results=6)

def macro_node(state: MarketResearchState) -> dict:
    ticker = state["ticker"]

    search_query = state.get("sub_queries", {}).get("macro") or f"macroeconomic outlook interest rates sector trends {ticker} 2025"
    results = search_tool.invoke(search_query)
    context = "\n\n".join(r["content"] for r in results)

    prompt = f"""Company: {ticker}

From the articles below, extract only explicitly stated macro factors.
No general economic commentary. No caveats or hedging.

ARTICLES:
{context}

Output in exactly this format:

INTEREST RATES AND FED POLICY:
[1 bullet, current rate + direct impact on {ticker} only]
[write "No specific data found" if not mentioned]

SECTOR CONDITIONS:
[maximum 2 bullets, named sector trends only, 15 words each]
[write "No specific data found" if not mentioned]

RELEVANT ECONOMIC DATA:
[maximum 2 bullets, specific named figures only, e.g. GDP 2.1%, CPI 3.2%]
[write "No specific data found" if not mentioned]
"""
    result = llm.invoke(prompt)
    return {
        "macro_context": result.content,
        "sources": state["sources"] + [r["url"] for r in results]
    }