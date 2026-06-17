from langchain_cohere import CohereRerank
from langchain_core.documents import Document

def build_retriever(vectorstore):
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=4
    )

    class SimpleRetriever:
        def invoke(self, query: str):
            docs = base_retriever.invoke(query)
            compressed = reranker.compress_documents(docs, query)
            return compressed

    return SimpleRetriever()