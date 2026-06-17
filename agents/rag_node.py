import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import MarketResearchState
from rag.vector_store import load_vector_store
from rag.retriever import build_retriever

def rag_retrieval_node(state: MarketResearchState) -> dict:
    vectorstore = load_vector_store()
    retriever = build_retriever(vectorstore)

    query = f"{state['ticker']} {state['query']}"
    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )
    return {"rag_context": context}