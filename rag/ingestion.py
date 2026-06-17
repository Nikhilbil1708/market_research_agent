from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import date

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

def ingest_documents(sources: list) -> list:
    docs = []
    for src in sources:
        loader = PyPDFLoader(src) if src.endswith(".pdf") else WebBaseLoader(src)
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        for chunk in chunks:
            chunk.metadata["source"] = src
            chunk.metadata["indexed_at"] = str(date.today())
        docs.extend(chunks)
    return docs