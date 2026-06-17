from dotenv import load_dotenv
load_dotenv()

from state import MarketResearchState
from agents.news_agent import news_node

state = MarketResearchState(
    query="JPMorgan Q1 2025 earnings",
    ticker="JPM",
    messages=[], news_analysis="", financial_data="",
    macro_context="", rag_context="", final_report="",
    confidence_score=0.0, sources=[]
)

result = news_node(state)
print(result["news_analysis"])