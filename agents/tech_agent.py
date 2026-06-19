import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from state import MarketResearchState

llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
search_tool = TavilySearchResults(max_results=8)


def _search(query: str) -> tuple[list, str]:
    results = search_tool.invoke(query)
    text = "\n\n".join(f"Source: {r['url']}\n{r['content']}" for r in results)
    return results, text


def tech_strategy_node(state: MarketResearchState) -> dict:
    ticker = state["ticker"]

    # Four targeted searches covering the full tech initiative landscape
    r_cloud,   t_cloud   = _search(f"{ticker} cloud strategy AWS Azure Google Cloud hyperscaler migration 2024 2025")
    r_ai,      t_ai      = _search(f"{ticker} artificial intelligence generative AI machine learning data analytics LLM 2024 2025")
    r_gcc,     t_gcc     = _search(f"{ticker} GCC global capability center automation RPA digital transformation IT modernization 2024 2025")
    r_isv,     t_isv     = _search(f"{ticker} ISV fintech partnership cybersecurity IT services system integrator vendor 2024 2025")

    prompt = f"""Company: {ticker}

From the search results below, list ONLY named technology partnerships
and initiatives that are explicitly mentioned. Do not write a general
technology overview or add context.

SEARCH RESULTS:
{tech_articles}

Output in exactly this format, nothing else:

CLOUD PARTNERSHIPS:
[bullet per named cloud provider only, e.g. AWS, Azure, Google Cloud]
[write "None found" if no named providers appear in the results]

ISV AND TECH VENDOR PARTNERSHIPS:
[bullet per named vendor only, e.g. Oracle, SAP, Salesforce, IBM, ServiceNow, etc.]
[write "None found" if no named vendors appear]

TECHNOLOGY INITIATIVES:
[maximum 5 bullets, named initiatives only, 25 words max each]
[write "None found" if no named initiatives appear]

Do not add explanatory sentences, do not add "it is worth noting",
do not summarize the company's general technology approach.
Only named, specific items from the search results above.
"""

    result = llm.invoke(prompt)
    all_sources = (
        [r["url"] for r in r_cloud] +
        [r["url"] for r in r_ai]    +
        [r["url"] for r in r_gcc]   +
        [r["url"] for r in r_isv]
    )

    return {
        "tech_strategy_context": result.content,
        "sources": state["sources"] + all_sources,
    }
