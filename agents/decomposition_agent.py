import sys, os, re, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import MarketResearchState
from llm_factory import get_llm

llm = get_llm("fast")

_AGENTS = ("news", "financial", "tech", "macro")


def decomposition_node(state: MarketResearchState) -> dict:
    query  = state["query"]
    ticker = state["ticker"]

    prompt = f"""You are a financial research assistant.
A user wants research on company ticker {ticker} with this query:
"{query}"

Break this into focused sub-queries — one per research agent below.
Return ONLY valid JSON, no markdown, no explanation.

{{
    "news":      "<focused sub-query for: company overview, recent news, leadership events>",
    "financial": "<focused sub-query for: earnings, revenue, EBITDA, financial metrics>",
    "tech":      "<focused sub-query for: AI strategy, cloud, digital transformation, partnerships>",
    "macro":     "<focused sub-query for: interest rates, inflation, GDP, sector trends>"
}}

Rules:
- Always include the ticker symbol {ticker} in each sub-query
- Keep each sub-query under 15 words
- Make each sub-query specific to that agent's domain
- Do not use the original query verbatim — sharpen it for each domain
"""

    result  = llm.invoke(prompt)
    cleaned = re.sub(r"```json|```", "", result.content).strip()

    try:
        sub_queries = json.loads(cleaned)
        for key in _AGENTS:
            if key not in sub_queries or not sub_queries[key].strip():
                sub_queries[key] = f"{ticker} {query}"
        print(f"  [Decomposition] Sub-queries:")
        for k, v in sub_queries.items():
            print(f"    {k:10}: {v}")
        return {"sub_queries": sub_queries}
    except json.JSONDecodeError:
        print("  [Decomposition] JSON parse failed — using original query for all agents")
        return {"sub_queries": {k: f"{ticker} {query}" for k in _AGENTS}}
