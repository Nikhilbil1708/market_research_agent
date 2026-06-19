import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import re
import json
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

BFSI_TICKERS = {
    "jpmorgan":         "JPM",  "jp morgan":      "JPM",
    "goldman sachs":    "GS",   "goldman":        "GS",
    "morgan stanley":   "MS",
    "bank of america":  "BAC",  "bofa":           "BAC",
    "wells fargo":      "WFC",
    "citigroup":        "C",    "citi":           "C",
    "blackrock":        "BLK",
    "american express": "AXP",  "amex":           "AXP",
    "charles schwab":   "SCHW",
    "us bancorp":       "USB",
    "hsbc":             "HSBC",
    "barclays":         "BCS",
    "deutsche bank":    "DB",
    "ubs":              "UBS",
    "prudential":       "PRU",
    "metlife":          "MET",
    "aig":              "AIG",
    "berkshire":        "BRK.B",
    "visa":             "V",
    "mastercard":       "MA",
    "paypal":           "PYPL",
}


def extract_research_intent(user_instruction: str) -> dict:
    prompt = f"""You are a financial research assistant.
A user has given you this research instruction:
"{user_instruction}"

Extract and return ONLY a valid JSON object, nothing else:
{{
    "company_name": "full legal company name",
    "ticker": "stock ticker symbol",
    "query": "specific well-formed research question",
    "sector": "BFSI sector (banking, insurance, payments, asset management)"
}}

Rules:
- Make the query specific and analytical
- Include time periods mentioned by the user in the query
- If no company is mentioned set company_name to "UNCLEAR"
"""
    result = llm.invoke(prompt)
    text = re.sub(r"```json|```", "", result.content).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"company_name": "UNCLEAR", "ticker": "", "query": "", "sector": ""}


def resolve_ticker(company_name: str, extracted_ticker: str) -> str:
    lookup = company_name.lower().strip()
    for key, ticker in BFSI_TICKERS.items():
        if key in lookup or lookup in key:
            return ticker
    return extracted_ticker.upper() if extracted_ticker else "UNKNOWN"


def confirm_with_user(intent: dict) -> str:
    print("\n" + "="*60)
    print("RESEARCH INTENT")
    print("="*60)
    print(f"  Company  : {intent['company_name']}")
    print(f"  Ticker   : {intent['ticker']}")
    print(f"  Sector   : {intent['sector']}")
    print(f"  Query    : {intent['query']}")
    print("="*60)
    print("\nThis will:")
    print("  1. Scrape and download financial documents")
    print("  2. Index them into the RAG vector store")
    print("  3. Run news, financial, and macro research agents")
    print("  4. Generate a PDF report")
    return input("\nProceed? (yes / no / edit): ").strip().lower()


def handle_edit(intent: dict) -> dict:
    print("\nWhich field to edit?")
    print("  1. Company name")
    print("  2. Ticker")
    print("  3. Query")
    choice = input("Enter 1, 2 or 3: ").strip()
    if choice == "1":
        intent["company_name"] = input("Company name: ").strip()
    elif choice == "2":
        intent["ticker"] = input("Ticker: ").strip().upper()
    elif choice == "3":
        intent["query"] = input("Research query: ").strip()
    return intent


def try_python_only_resolution(user_instruction: str) -> dict | None:
    """
    Attempts to resolve company name, ticker, and query using pure
    Python dictionary matching - zero LLM tokens. Returns None if
    no known company is found in the instruction, signaling the
    caller to fall back to the LLM extractor.
    """
    lookup = user_instruction.lower()

    for key, ticker in BFSI_TICKERS.items():
        if key in lookup:
            return {
                "company_name": key.title(),
                "ticker": ticker,
                "query": user_instruction.strip(),
                "sector": "Banking"
            }

    return None


def run_interface():
    ...
    while True:
        user_input = input("Research instruction: ").strip()
        ...
        # Step 1 — Try pure Python resolution first, zero tokens
        intent = try_python_only_resolution(user_input)

        if intent is None:
            # Step 1b — Fall back to LLM only if Python couldn't resolve it
            print("\nExtracting research intent...")
            intent = extract_research_intent(user_input)

            if intent["company_name"] == "UNCLEAR":
                print("Could not identify a company. Please mention the company name.\n")
                continue

            intent["ticker"] = resolve_ticker(
                intent["company_name"],
                intent.get("ticker", "")
            )
        else:
            print(f"\nResolved directly: {intent['company_name']} ({intent['ticker']}) — no LLM call needed")

        # Step 2 — Confirm
        decision = confirm_with_user(intent)

        if decision == "no":
            print("Cancelled.\n")
            continue

        if decision == "edit":
            intent = handle_edit(intent)
            if input("Proceed? (yes/no): ").strip().lower() != "yes":
                continue

        # Step 3 — Run full pipeline
        from pipeline import run_full_pipeline

        try:
            pdf_path = run_full_pipeline(intent)
            print(f"\nDone. Report saved to: {pdf_path}")
        except Exception as e:
            print(f"\nPipeline error: {e}")
            import traceback
            traceback.print_exc()

        print("\nReady for next instruction.\n")


if __name__ == "__main__":
    run_interface()