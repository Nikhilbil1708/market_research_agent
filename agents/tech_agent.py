import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from state import MarketResearchState

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
search_tool = TavilySearchResults(max_results=8)


def _search(query: str) -> tuple[list, str]:
    results = search_tool.invoke(query)
    text = "\n\n".join(f"Source: {r['url']}\n{r['content']}" for r in results)
    return results, text


def tech_strategy_node(state: MarketResearchState) -> dict:
    ticker = state["ticker"]

    r_ai,   t_ai   = _search(f"{ticker} artificial intelligence generative AI machine learning data analytics LLM 2024 2025")
    r_tech, t_tech = _search(f"{ticker} cloud ISV vendor partnership cybersecurity GCC digital transformation technology 2024 2025")

    prompt = f"""Company: {ticker}

From the search results below, list ONLY named technology initiatives
and partnerships that are explicitly mentioned. No general commentary.

AI SEARCH RESULTS:
{t_ai}

OTHER TECH SEARCH RESULTS:
{t_tech}

Output in exactly this format, nothing else:

AI INITIATIVES:
[maximum 5 bullets, named initiatives only, 25 words max each]
[write "None found" if no named initiatives appear]

TECHNOLOGY AND PARTNERSHIPS:
[maximum 5 bullets, named cloud providers / ISV vendors / partnerships only, 25 words max each]
[write "None found" if no named items appear]

Only named, specific items from the search results above.
Do not add explanatory sentences or general technology summaries.
"""

    result = llm.invoke(prompt)
    all_sources = [r["url"] for r in r_ai] + [r["url"] for r in r_tech]

    return {
        "tech_strategy_context": result.content,
        "sources": state["sources"] + all_sources,
    }
