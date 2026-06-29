from dotenv import load_dotenv
load_dotenv()

from graph import build_graph
from rag.ingestion import ingest_documents
from rag.vector_store import build_vector_store
from pdf_generator import generate_pdf 
def setup_rag():
    sources = [
        # Core Financials
        r"C:\Users\visha\projects\market_research_agent\data\4q25earnings-report.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\corp-q1-2026.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\corp-q2-2025.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\corp-q3-2025.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\corp-10k-2025.pdf",
    

        #Earnings Calls Transcripts
        r"C:\Users\visha\projects\market_research_agent\data\jpm-2q25-earnings-call-transcript.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\jpm-3q25-earnings-call-transcript.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\jpm-4q25-earnings-call-transcript.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\1q26-earnings-transcript.pdf",

        #Strategy and Context
        r"C:\Users\visha\projects\market_research_agent\data\annualreport-2025.pdf",
        r"C:\Users\visha\projects\market_research_agent\data\proxy-statement2026.pdf",

        #Macro Context
        r"C:\Users\visha\projects\market_research_agent\data\fomcminutes20260128.pdf",
        
        #Web URLs
        "https://www.jpmorganchase.com/",
    ]
    docs = ingest_documents(sources)
    build_vector_store(docs)
    print(f"RAG pipeline ready: {len(docs)} chunks indexed.")

if __name__ == "__main__":
    # Uncomment the line below on first run only, then comment it out again
    # setup_rag()

    graph = build_graph()

    result = graph.invoke({
        "query": "What were JPMorgan's key growth drivers and risks in 2025?",
        "ticker": "JPM",
        "sub_queries": {},
        "messages": [],
        "news_analysis": "",
        "financial_data": "",
        "macro_context": "",
        "rag_context": "",
        "final_report": "",
        "confidence_score": 0.0,
        "sources": []
    })

    print("\n===== RESEARCH REPORT =====\n")
    print(result["final_report"])
    print(f"\nConfidence: {result['confidence_score']}")
    print(f"\nSources: {len(result['sources'])} cited")

    #Generate PDF
    print("\nGenerating PDF report...")
    pdf_path = generate_pdf(
        final_report    = result["final_report"],
        query           = "What are JPMorgan's key growth drivers and risks in 2025?",
        ticker          = "JPM",
        confidence_score= result["confidence_score"],
        sources         = result["sources"]
    )
    print(f"PDF saved to: {pdf_path}")