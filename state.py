from typing import TypedDict, List, Annotated
import operator

class MarketResearchState(TypedDict):
    query: str
    ticker: str
    sub_queries: dict
    messages: Annotated[List, operator.add]
    news_analysis: str
    company_overview_context: str
    tech_strategy_context: str
    financial_data: str
    financial_charts: List[str]
    chart_paths: Annotated[list, operator.add]
    macro_context: str
    rag_context: str
    final_report: str
    confidence_score: float
    sources: Annotated[List[str], operator.add]