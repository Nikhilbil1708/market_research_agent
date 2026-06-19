import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from state import MarketResearchState
import re

llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)

def calculate_confidence_score(state: dict) -> float:
    """
    Pure Python confidence calculation based on concrete signals -
    no LLM call needed, no hallucinated number.
    """
    score = 0.4   # baseline

    financial_data = state.get("financial_data", "") or ""
    if financial_data and "not available" not in financial_data.lower():
        score += 0.15

    if len(state.get("sources", [])) >= 5:
        score += 0.15
    elif len(state.get("sources", [])) >= 2:
        score += 0.08

    if state.get("rag_context"):
        score += 0.15

    if state.get("company_pe") is not None:
        score += 0.10

    metrics = state.get("financial_metrics", {})
    if metrics and any(v for v in metrics.get("revenue", []) if v is not None):
        score += 0.05

    return min(round(score, 2), 1.0)

def synthesis_node(state: MarketResearchState) -> dict:
    prompt = f"""...
    [REMOVE the confidence score instruction line from the prompt entirely -
    do not ask the LLM to produce a confidence number anymore]
    ...
    """
    result = llm.invoke(prompt)
    return {
        "final_report": result.content,
        "confidence_score": calculate_confidence_score(state)   # ← computed in Python
    }