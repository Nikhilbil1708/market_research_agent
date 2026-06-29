from langchain_community.tools.tavily_search import TavilySearchResults
from state import MarketResearchState
from llm_factory import get_llm

llm = get_llm("fast")
search_tool = TavilySearchResults(max_results=8)

def news_node(state: MarketResearchState) -> dict:
    ticker = state["ticker"]
    query = state["query"]

    search_query = state.get("sub_queries", {}).get("news") or f"{ticker} {query}"
    results = search_tool.invoke(f"{search_query} latest news 2025")
    articles = "\n\n".join(
        f"Source: {r['url']}\n{r['content']}" for r in results
    )

    prompt = f"""Company: {ticker}
Research question: {query}

From the articles below, extract only what is explicitly stated.
No commentary, no context, no caveats.

NEWS ARTICLES:
{articles}

Output in exactly this format, nothing else:

COMPANY OVERVIEW:
[2 sentences maximum: what the company does and its main business segments]

RECENT NEWS:
[maximum 4 bullets, one sentence each, 20 words max per bullet]
[skip this section entirely if no relevant news found]

IMPORTANT EVENTS:
[maximum 2 bullets: regulatory actions or management changes only]
[skip if none found]
"""
    result = llm.invoke(prompt)
    return {
        "news_analysis": result.content,
        "sources": state["sources"] + [r["url"] for r in results]
    }