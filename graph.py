from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from state import MarketResearchState
from agents.decomposition_agent import decomposition_node
from agents.news_agent import news_node
from agents.tech_agent import tech_strategy_node
from agents.financial_agent import financial_analysis_node
from agents.macro_agent import macro_node
from agents.rag_node import rag_retrieval_node
from agents.synthesis_agent import synthesis_node

def route_tasks(state: MarketResearchState) -> list:
    """
    Pure Python keyword routing - decides which agents actually need
    to run based on the query. Zero LLM tokens spent on this decision.
    """
    query = state["query"].lower()
    tasks = []

    financial_keywords = [
        "earnings", "revenue", "profit", "financial", "balance sheet",
        "cash flow", "ratio", "p/e", "ebt", "cogs", "sg&a", "margin"
    ]
    macro_keywords = [
        "macro", "fed", "rate", "gdp", "inflation", "interest rate",
        "sector", "economy", "economic"
    ]
    tech_keywords = [
        "technology", "tech", "cloud", "digital", "ai", "automation",
        "partnership", "it strategy", "innovation", "cybersecurity"
    ]
    news_keywords = [
        "news", "recent", "announced", "development", "latest"
    ]

    if any(k in query for k in financial_keywords):
        tasks.append("financial_agent")
    if any(k in query for k in macro_keywords):
        tasks.append("macro_agent")
    if any(k in query for k in tech_keywords):
        tasks.append("tech_agent")
    if any(k in query for k in news_keywords):
        tasks.append("news_agent")

    # news_agent and rag_agent always run — news provides the company overview
    # section which must appear in every report regardless of query topic
    if "news_agent" not in tasks:
        tasks.append("news_agent")
    tasks.append("rag_agent")

    # If the query is broad/ambiguous and matched nothing specific,
    # run everything rather than risk missing relevant context
    specific_tasks = [t for t in tasks if t not in ("rag_agent", "news_agent")]
    if len(specific_tasks) == 0:
        tasks = ["news_agent", "tech_agent", "financial_agent", "macro_agent", "rag_agent"]

    print(f"  [Router] Running agents: {tasks}")
    return tasks


def build_graph():
    g = StateGraph(MarketResearchState)

    g.add_node("decomposition", decomposition_node)
    g.add_node("router", lambda state: {})   # pass-through node, routing happens in edges
    g.add_node("news_agent", news_node)
    g.add_node("tech_agent", tech_strategy_node)
    g.add_node("financial_agent", financial_analysis_node)
    g.add_node("macro_agent", macro_node)
    g.add_node("rag_agent", rag_retrieval_node)
    g.add_node("synthesis", synthesis_node)

    g.set_entry_point("decomposition")
    g.add_edge("decomposition", "router")

    # Conditional fan-out - only selected agents run
    g.add_conditional_edges(
        "router",
        route_tasks,
        {
            "news_agent":      "news_agent",
            "tech_agent":       "tech_agent",
            "financial_agent": "financial_agent",
            "macro_agent":     "macro_agent",
            "rag_agent":       "rag_agent",
        }
    )

    # Every agent that does run converges into synthesis
    for agent in ["news_agent", "tech_agent", "financial_agent", "macro_agent", "rag_agent"]:
        g.add_edge(agent, "synthesis")

    g.add_edge("synthesis", END)

    return g.compile()
