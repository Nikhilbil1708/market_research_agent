from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
PERSIST_DIR = "./chroma_db"

def build_vector_store(docs):
    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIR
    )

def load_vector_store():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )