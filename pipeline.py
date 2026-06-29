import sys
import os
import re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import glob
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from guardrails import run_rag_guardrail, run_output_guardrails

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


def index_new_documents(company_name: str, ticker: str = "") -> int:
    from rag.ingestion import ingest_documents
    from rag.vector_store import load_vector_store, build_vector_store

    all_pdfs = glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf"))

    # Build match terms — ticker is most reliable, name parts as fallback
    match_terms = []

    if ticker:
        match_terms.append(ticker.lower())

    name_clean = re.sub(r"[^a-z0-9\s]", "", company_name.lower())
    name_parts = [
        w for w in name_clean.split()
        if w not in ("the", "inc", "corp", "group", "co", "ltd", "llc", "plc")
        and len(w) > 3
    ]
    match_terms.extend(name_parts)

    def matches_company(filepath: str) -> bool:
        basename = os.path.basename(filepath).lower()
        return any(term in basename for term in match_terms)

    company_pdfs = [p for p in all_pdfs if matches_company(p)]

    if not company_pdfs:
        print(f"  [RAG] No PDFs found for {company_name} ({ticker})")
        print(f"  [RAG] Match terms used: {match_terms}")
        print(f"  [RAG] All files in data folder:")
        for f in all_pdfs:
            print(f"        - {os.path.basename(f)}")
        return 0

    already_indexed = get_indexed_sources()
    new_pdfs        = [p for p in company_pdfs if p not in already_indexed]

    if not new_pdfs:
        print(f"  [RAG] All {len(company_pdfs)} PDFs for {company_name} already indexed")
        return 0

    print(f"  [RAG] Indexing {len(new_pdfs)} new PDFs for {company_name}:")
    for p in new_pdfs:
        print(f"        - {os.path.basename(p)}")

    docs = ingest_documents(new_pdfs)

    if os.path.exists(PERSIST_DIR):
        vs = load_vector_store()
        vs.add_documents(docs)
    else:
        build_vector_store(docs)

    print(f"  [RAG] Indexed {len(docs)} chunks")
    return len(docs)
   


def run_scraper_for_company(company_name: str, ticker: str) -> bool:
    from scraper_agent import scrape_company

    print(f"\n  [Scraper] Starting download for {company_name} ({ticker})...")
    before = (
        set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf"))) |
        set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.html")))
    )

    try:
        response = scrape_company(ticker, company_name)
        print(f"  [Scraper] Response: {response}")
    except Exception as scraper_error:
        print(f"  [Scraper] FAILED: {scraper_error}")
        import traceback
        traceback.print_exc()
        return False

    after = (
        set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf"))) |
        set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.html")))
    )
    new_files = after - before

    if new_files:
        print(f"  [Scraper] Downloaded {len(new_files)} new file(s):")
        for f in new_files:
            print(f"            - {os.path.basename(f)}")
    else:
        print("  [Scraper] WARNING: No new files downloaded")

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
        "sub_queries"    : {},
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
    chunks_added = index_new_documents(company, ticker)

    print("\n" + "="*60)
    print(f"PIPELINE STARTING: {company} ({ticker})")
    print("="*60)

    # ── Stage 1: Scrape ──────────────────────────────────────────
    print("\n[Stage 1/4] Scraping financial documents...")
    scrape_success = False

    try:
        scrape_success = run_scraper_for_company(company, ticker)
    except Exception as scrape_error:
        print(f" [Scraper] ERROR: {scrape_error}")
        import traceback
        traceback.print_exc()
        scrape_success = False

    if not scrape_success: 
        print("\n  [WARNING] No documents were downloaded.")
        print("  [WARNING] RAG will use existing vector store only.")
        print("  [WARNING] Research quality may be limited.")
        print("  [WARNING] To manually add documents:")
        print(f"            1. Download PDFs from the company IR page")
        print(f"            2. Save them to: {os.path.abspath(DOWNLOAD_DIR)}")
        print(f"            3. Run: python main.py")
        print("\n  [WARNING] Continuing with existing vector store data.")

    # ── Stage 2: Index ───────────────────────────────────────────
    print("\n[Stage 2/4] Indexing new documents into RAG...")
    chunks_added = 0                                # ← defined before try block

    try:
        chunks_added = index_new_documents(company, ticker)
    except Exception as index_error:               # ← named index_error not e
        print(f"  [RAG] Indexing error: {index_error}")
        import traceback
        traceback.print_exc()

    if chunks_added == 0:
        print("  [RAG] Proceeding with existing vector store")

    run_rag_guardrail(PERSIST_DIR)

    # ── Stage 3: Research ────────────────────────────────────────
    print("\n[Stage 3/4] Running research agents...")
    result = {}                                     # ← defined before try block

    try:
        result = run_research(intent)
    except Exception as research_error:            # ← named research_error not e
        print(f"  [Research] FAILED: {research_error}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "-"*60)
    print("RESEARCH REPORT")
    print("-"*60)
    print(result["final_report"])
    print(f"\nConfidence Score : {result.get('confidence_score', 0.0):.2f}")
    print(f"Sources cited    : {len(result.get('sources', []))}")

    run_output_guardrails(result)

    # ── Stage 4: Generate PDF ────────────────────────────────────
    print("\n[Stage 4/4] Generating PDF report...")
    pdf_path = ""                                   # ← defined before try block

    try:
        pdf_path = generate_pdf(
            final_report     = result["final_report"],
            query            = query,
            ticker           = ticker,
            confidence_score = result.get("confidence_score", 0.0),
            sources          = result.get("sources", []),
            financial_charts = result.get("chart_paths", [])
        )
        print(f"PDF saved: {pdf_path}")
    except Exception as pdf_error:                 # ← named pdf_error not e
        print(f"  [PDF] Generation failed: {pdf_error}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE: {company}")
    print("="*60)

    return pdf_path