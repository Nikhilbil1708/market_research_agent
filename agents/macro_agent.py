import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from state import MarketResearchState

llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
search_tool = TavilySearchResults(max_results=6)

def macro_node(state: MarketResearchState) -> dict:
    ticker = state["ticker"]

    results = search_tool.invoke(
        f"macroeconomic outlook interest rates sector trends {ticker} 2025"
    )
    context = "\n\n".join(r["content"] for r in results)

    prompt = f"""You are a JPMC macro strategist.
Summarize the macroeconomic and sector environment relevant to {ticker}.

Context:
{context}

Cover:
1. Interest rate and Fed policy impact
2. Sector-level tailwinds and headwinds
3. GDP, inflation, or employment data relevant to this company
4. Competitive landscape shifts
"""
    result = llm.invoke(prompt)
    return {
        "macro_context": result.content,
        "sources": state["sources"] + [r["url"] for r in results]
    }