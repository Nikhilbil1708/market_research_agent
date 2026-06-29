import os
from langchain_anthropic import ChatAnthropic


def get_llm(tier: str = "fast") -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0
    )
