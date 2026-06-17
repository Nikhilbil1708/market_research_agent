import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import glob
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOWNLOAD_DIR = "data"
PERSIST_DIR  = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_indexed_sources() -> set:
    """
    Returns the set of source file paths already indexed
    in the Chroma vector store. Prevents re-indexing.
    """
    try:
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        results = vs.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sources
    except Exception:
        return set()


def index_new_documents(company_name: str) -> int:
    """
    Finds all PDFs in data\ that belong to this company
    and indexes only the ones not already in the vector store.
    Returns the number of new chunks indexed.
    """
    from rag.ingestion import ingest_documents
    from rag.vector_store import load_vector_store, build_vector_store

    safe_company = company_name.lower().replace(" ", "_")

    # Find all PDFs for this company in data\
    all_pdfs = glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf"))
    company_pdfs = [
        p for p in all_pdfs
        if safe_company in os.path.basename(p).lower()
        or company_name.lower().split()[0] in os.path.basename(p).lower()
    ]

    if not company_pdfs:
        print(f"  [RAG] No PDFs found for {company_name} in {DOWNLOAD_DIR}\\")
        return 0

    # Filter out already-indexed files
    already_indexed = get_indexed_sources()
    new_pdfs = [p for p in company_pdfs if p not in already_indexed]

    if not new_pdfs:
        print(f"  [RAG] All {len(company_pdfs)} PDFs for {company_name} already indexed")
        return 0

    print(f"  [RAG] Indexing {len(new_pdfs)} new PDFs for {company_name}:")
    for p in new_pdfs:
        print(f"        - {os.path.basename(p)}")

    docs = ingest_documents(new_pdfs)

    # Add to existing vector store if it exists, else create new
    if os.path.exists(PERSIST_DIR):
        vs = load_vector_store()
        vs.add_documents(docs)
    else:
        build_vector_store(docs)

    print(f"  [RAG] Indexed {len(docs)} chunks successfully")
    return len(docs)


def run_scraper_for_company(company_name: str) -> bool:
    """
    Runs the scraper agent for a specific company.
    Returns True if at least one file was downloaded.
    """
    from scraper_agent import agent_executor

    instruction = (
        f"Download the latest annual report (10-K), "
        f"the two most recent quarterly reports (10-Q), "
        f"and the latest earnings presentation for {company_name}. "
        f"Skip any documents already saved."
    )

    print(f"\n  [Scraper] Starting download for {company_name}...")

    # Count files before
    before = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf")))

    try:
        result = agent_executor.invoke({"input": instruction})
        print(f"  [Scraper] {result['output']}")
    except Exception as e:
        print(f"  [Scraper] Error: {e}")

    # Count files after
    after = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf")))
    new_files = after - before

    print(f"  [Scraper] Downloaded {len(new_files)} new file(s)")
    return len(new_files) > 0


def run_research(intent: dict) -> dict:
    """
    Runs the full LangGraph research pipeline for the given intent.
    Returns the full result state.
    """
    from graph import build_graph

    print(f"\n  [Research] Starting agents for {intent['company_name']}...")
    graph = build_graph()

    result = graph.invoke({
        "query"          : intent["query"],
        "ticker"         : intent["ticker"],
        "messages"       : [],
        "news_analysis"  : "",
        "financial_data" : "",
        "macro_context"  : "",
        "rag_context"    : "",
        "final_report"   : "",
        "confidence_score": 0.0,
        "sources"        : []
    })

    return result


def run_full_pipeline(intent: dict) -> str:
    """
    Master function. Runs all four stages in sequence:
    1. Scrape documents for the company
    2. Index new documents into vector store
    3. Run research agents
    4. Generate PDF report

    Returns the path to the generated PDF.
    """
    from pdf_generator import generate_pdf

    company = intent["company_name"]
    ticker  = intent["ticker"]
    query   = intent["query"]

    print("\n" + "="*60)
    print(f"PIPELINE STARTING: {company} ({ticker})")
    print("="*60)

    # ── Stage 1: Scrape ──────────────────────────────────────────
    print("\n[Stage 1/4] Scraping financial documents...")
    run_scraper_for_company(company)

    # ── Stage 2: Index ───────────────────────────────────────────
    print("\n[Stage 2/4] Indexing new documents into RAG...")
    chunks_added = index_new_documents(company)
    if chunks_added == 0:
        print("  [RAG] Proceeding with existing vector store")

    # ── Stage 3: Research ────────────────────────────────────────
    print("\n[Stage 3/4] Running research agents...")
    result = run_research(intent)

    print("\n" + "-"*60)
    print("RESEARCH REPORT")
    print("-"*60)
    print(result["final_report"])
    print(f"\nConfidence Score : {result['confidence_score']:.2f}")
    print(f"Sources cited    : {len(result['sources'])}")

    # ── Stage 4: Generate PDF ────────────────────────────────────
    print("\n[Stage 4/4] Generating PDF report...")
    pdf_path = generate_pdf(
        final_report     = result["final_report"],
        query            = query,
        ticker           = ticker,
        confidence_score = result["confidence_score"],
        sources          = result["sources"]
    )
    print(f"PDF saved: {pdf_path}")

    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE: {company}")
    print("="*60)

    return pdf_path