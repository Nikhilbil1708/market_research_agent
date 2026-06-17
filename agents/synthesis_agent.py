import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_anthropic import ChatAnthropic
from state import MarketResearchState
import re

llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)

def extract_confidence(text: str) -> float:
    match = re.search(r"confidence[:\s]+([0-9.]+)", text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.75

def synthesis_node(state: MarketResearchState) -> dict:
    prompt = f"""You are a senior JPMC market analyst.
Synthesize this research into a structured report.

News & Developments: {state['news_analysis']}
Financial Analysis:  {state['financial_data']}
Macro Context:       {state['macro_context']}
Retrieved Context:   {state['rag_context']}

Output format:
1. Executive summary (2-3 sentences)
2. Key findings with evidence
3. Risk factors
4. Data confidence score (0.0 to 1.0)
5. Sources cited
"""
    result = llm.invoke(prompt)
    return {
        "final_report": result.content,
        "confidence_score": extract_confidence(result.content)
    }