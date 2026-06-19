from typing import TypedDict, List, Annotated
import operator

class MarketResearchState(TypedDict):
    query: str
    ticker: str
    messages: Annotated[List, operator.add]
    news_analysis: str
    company_overview_context: str
    tech_strategy_context: str
    financial_data: str
    financial_metrics: dict
    chart_paths: List[str]
    company_pe: float
    competitor_pe: dict
    macro_context: str
    rag_context: str
    final_report: str
    confidence_score: float
    sources: List[str]