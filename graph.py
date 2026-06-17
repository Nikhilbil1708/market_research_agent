from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from state import MarketResearchState
from agents.news_agent import news_node
from agents.financial_agent import financial_analysis_node
from agents.macro_agent import macro_node
from agents.rag_node import rag_retrieval_node
from agents.synthesis_agent import synthesis_node

def build_graph():
    g = StateGraph(MarketResearchState)

    g.add_node("news_agent", news_node)
    g.add_node("financial_agent", financial_analysis_node)
    g.add_node("macro_agent", macro_node)
    g.add_node("rag_agent", rag_retrieval_node)
    g.add_node("synthesis", synthesis_node)

    g.set_entry_point("news_agent")
    g.add_edge("news_agent", "financial_agent")
    g.add_edge("financial_agent", "macro_agent")
    g.add_edge("macro_agent", "rag_agent")
    g.add_edge("rag_agent", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile()