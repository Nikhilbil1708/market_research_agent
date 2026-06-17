import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from state import MarketResearchState

llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
search_tool = TavilySearchResults(max_results=8)

def news_node(state: MarketResearchState) -> dict:
    ticker = state["ticker"]
    query = state["query"]

    results = search_tool.invoke(f"{ticker} {query} latest news 2025")
    articles = "\n\n".join(
        f"Source: {r['url']}\n{r['content']}" for r in results
    )

    prompt = f"""You are a JPMC market research analyst.
Summarize recent news about {ticker} for this question: "{query}"

Articles:
{articles}

Provide:
1. Key developments (facts only, no sentiment labels)
2. Notable management or analyst statements
3. Regulatory or macro events mentioned
4. Areas where news coverage is thin
"""
    result = llm.invoke(prompt)
    return {
        "news_analysis": result.content,
        "sources": state["sources"] + [r["url"] for r in results]
    }