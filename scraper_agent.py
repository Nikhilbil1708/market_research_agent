import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import re
import json
import requests
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from scraper_filter import filter_documents
import scraper_config

# ── Configuration ────────────────────────────────────────────────────

DOWNLOAD_DIR = "data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

KNOWN_IR_PAGES = {
    "jpmorgan":         "https://www.jpmorganchase.com/ir",
    "jp morgan":        "https://www.jpmorganchase.com/ir",
    "goldman sachs":    "https://www.goldmansachs.com/investor-relations",
    "goldman":          "https://www.goldmansachs.com/investor-relations",
    "morgan stanley":   "https://www.morganstanley.com/about-us/investor-relations",
    "bank of america":  "https://investor.bankofamerica.com",
    "bofa":             "https://investor.bankofamerica.com",
    "wells fargo":      "https://www.wellsfargo.com/about/investor-relations",
    "citigroup":        "https://www.citigroup.com/global/en/homepage/finance.html",
    "citi":             "https://www.citigroup.com/global/en/homepage/finance.html",
    "blackrock":        "https://ir.blackrock.com",
    "american express": "https://ir.americanexpress.com",
    "amex":             "https://ir.americanexpress.com",
    "metlife":          "https://investor.metlife.com",
    "prudential":       "https://investor.prudential.com",
    "aig":              "https://www.aig.com/investor-relations",
    "hsbc":             "https://www.hsbc.com/investors",
    "barclays":         "https://home.barclays/investor-relations",
    "charles schwab":   "https://www.aboutschwab.com/investor-relations",
    "us bancorp":       "https://ir.usbank.com",
    "mastercard":       "https://investor.mastercard.com",
    "visa":             "https://investor.visa.com",
    "paypal":           "https://investor.paypal.com",
}

COMPANY_CIKS = {
    "JPM":   "0000019617",
    "GS":    "0000886982",
    "MS":    "0000895421",
    "BAC":   "0000070858",
    "WFC":   "0000072971",
    "C":     "0000831001",
    "BLK":   "0001364742",
    "AXP":   "0000004962",
    "SCHW":  "0000316709",
    "USB":   "0000036104",
    "HSBC":  "0000083246",
    "PRU":   "0001137774",
    "MET":   "0001099219",
    "AIG":   "0000005272",
    "MA":    "0001141391",
    "V":     "0001403161",
    "PYPL":  "0001633917",
}

TARGET_DOC_KEYWORDS = [
    "annual report", "10-k", "10k",
    "quarterly report", "10-q", "10q",
    "earnings", "press release",
    "investor presentation", "investor day",
    "supplement", "proxy", "def 14a", "8-k", "8k"
]


# ── Tool 1: Fetch Earnings Transcripts ──────────────────────────────

def fetch_earnings_transcripts(ticker: str, company_name: str) -> str:
    """
    Fetches earnings call transcript links from Motley Fool.
    Returns a JSON list of transcript links.
    """
    print(f"  [Transcripts] Searching for {company_name} transcripts...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        )
    }

    search_url = (
        f"https://www.fool.com/search/solr.aspx"
        f"?q={company_name.replace(' ', '+')}+earnings+call+transcript"
        f"&filter=ArticleTypes%3AEarningsCallTranscript"
    )

    links = []

    try:
        resp = requests.get(search_url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.select("a[href]"):
            href  = a["href"]
            label = a.get_text(strip=True).lower()

            if (
                "transcript" in label
                and ticker.lower() in label
                and "earnings" in label
            ):
                full_url = (
                    href if href.startswith("http")
                    else f"https://www.fool.com{href}"
                )
                links.append({
                    "label":    a.get_text(strip=True),
                    "url":      full_url,
                    "doc_type": "earnings_transcript",
                    "year":     str(scraper_config.CURRENT_YEAR)
                })

                if len(links) >= 4:
                    break

    except Exception as transcript_error:
        print(f"  [Transcripts] Error: {transcript_error}")

    print(f"  [Transcripts] Found {len(links)} transcripts")
    return json.dumps(links)


# ── Tool 2: Find IR Page ─────────────────────────────────────────────

def find_ir_page(company_name: str) -> str:
    """
    Returns the known investor relations page URL for a company,
    or a not-found message if the company is not in KNOWN_IR_PAGES.
    """
    lookup = company_name.lower().strip()

    for key, url in KNOWN_IR_PAGES.items():
        if key in lookup or lookup in key:
            print(f"  [IR Page] Found known IR page for {company_name}: {url}")
            return url

    return (
        f"No known IR page for {company_name}. "
        f"Add it to KNOWN_IR_PAGES in scraper_agent.py if needed."
    )


# ── Tool 3: Fetch SEC EDGAR Links ────────────────────────────────────

def fetch_sec_edgar_links(ticker: str) -> str:
    """
    Fetches 10-K, 10-Q, and 8-K filing links from SEC EDGAR.
    Returns a JSON list of document links.
    """
    ticker = ticker.upper().strip()
    cik    = COMPANY_CIKS.get(ticker)

    if not cik:
        print(f"  [SEC EDGAR] No CIK found for {ticker}")
        return json.dumps([])

    print(f"  [SEC EDGAR] Fetching filings for {ticker} (CIK: {cik})")

    headers = {
        "User-Agent": "market-research-agent contact@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }

    links = []

    for doc_type, max_count in [("10-K", 2), ("10-Q", 4), ("8-K", 6)]:
        url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}"
            f"&type={doc_type}&dateb=&owner=include"
            f"&count={max_count}&search_text="
        )

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")

            filing_count = 0

            for row in soup.select("tr"):
                cells = row.select("td")
                if len(cells) < 4:
                    continue

                form_type = cells[0].get_text(strip=True)
                date      = cells[3].get_text(strip=True)

                if form_type not in (doc_type, f"{doc_type}/A"):
                    continue

                try:
                    filing_year = int(date[:4])
                    if filing_year < scraper_config.OLDEST_YEAR_ALLOWED:
                        print(f"  [SEC EDGAR] Skipping {form_type} {date} — too old")
                        continue
                except ValueError:
                    continue

                link_tag = cells[1].find("a", href=True)
                if not link_tag:
                    continue

                index_url = "https://www.sec.gov" + link_tag["href"]

                try:
                    index_resp = requests.get(index_url, headers=headers, timeout=15)
                    index_soup = BeautifulSoup(index_resp.text, "html.parser")

                    doc_table = index_soup.find("table", {"class": "tableFile"})
                    if not doc_table:
                        continue

                    for doc_row in doc_table.select("tr"):
                        doc_cells = doc_row.select("td")
                        if len(doc_cells) < 4:
                            continue

                        doc_desc = doc_cells[1].get_text(strip=True).lower()
                        doc_link = doc_cells[2].find("a", href=True)

                        if not doc_link:
                            continue

                        href = doc_link["href"]

                        if any(k in doc_desc for k in [
                            "10-k", "10-q", "annual report",
                            "quarterly report", "form 10"
                        ]):
                            if href.startswith("/ix?doc="):
                                href = href.replace("/ix?doc=", "")

                            full_url = "https://www.sec.gov" + href
                            links.append({
                                "label":    f"{form_type} {date} — {ticker}",
                                "url":      full_url,
                                "doc_type": form_type,
                                "date":     date,
                                "ticker":   ticker,
                                "year":     date[:4]
                            })
                            print(
                                f"  [SEC EDGAR] Found: {form_type} "
                                f"dated {date} → {full_url[:60]}..."
                            )
                            filing_count += 1
                            break

                except Exception as index_error:
                    print(f"  [SEC EDGAR] Index error: {index_error}")
                    continue

                if filing_count >= max_count:
                    break

        except Exception as fetch_error:
            print(f"  [SEC EDGAR] Error fetching {doc_type}: {fetch_error}")

    print(f"  [SEC EDGAR] Total links found: {len(links)}")
    return json.dumps(links)


# ── Helper: Per-URL headers ──────────────────────────────────────────

def get_headers_for_url(url: str) -> dict:
    if "sec.gov" in url.lower():
        return {
            "User-Agent": "market-research-agent contact@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*"
    }


# ── Tool 4: Download Filtered PDFs ───────────────────────────────────

def download_filtered_pdfs(raw_links_json: str, company_name: str) -> str:
    """
    Applies filter rules from scraper_config.py and scraper_filter.py,
    then downloads approved documents into the data folder.
    Returns a summary of what was downloaded and what was skipped.
    """
    try:
        raw_links = json.loads(raw_links_json)
    except json.JSONDecodeError:
        return "Error: could not parse links JSON"

    if not raw_links:
        return "No links provided to download"

    approved = filter_documents(raw_links)

    if not approved:
        return (
            "No documents passed the filter criteria. "
            "Check scraper_config.py to adjust document types or age limits."
        )

    downloaded = []
    failed     = []
    skipped    = []

    for doc in approved:
        url      = doc["url"]
        doc_type = doc.get("doc_type", "unknown")
        year     = doc.get("year", scraper_config.CURRENT_YEAR)

        headers = get_headers_for_url(url)

        safe_company = re.sub(r"[^a-zA-Z0-9]", "_", company_name.lower()).strip("_")
        safe_type    = re.sub(r"[^a-zA-Z0-9]", "_", doc_type.lower()).strip("_")
        file_ext     = ".pdf" if url.lower().endswith(".pdf") else ".html"
        filename     = f"{safe_company}_{safe_type}_{year}{file_ext}"
        filepath     = os.path.join(DOWNLOAD_DIR, filename)

        if os.path.exists(filepath):
            from scraper_filter import is_already_downloaded
            if is_already_downloaded(filename):
                skipped.append(f"{filename} — already downloaded")
                print(f"  [Download] Skipping (exists): {filename}")
                continue

        try:
            print(f"  [Download] Downloading: {filename}")
            resp = requests.get(url, headers=headers, timeout=60, stream=True)
            resp.raise_for_status()

            content_length = int(resp.headers.get("Content-Length", 0))
            max_bytes      = scraper_config.MAX_FILE_SIZE_MB * 1024 * 1024
            if content_length > max_bytes:
                size_mb = content_length // 1024 // 1024
                msg = f"{filename} — too large ({size_mb} MB)"
                failed.append(msg)
                print(f"  [Download] Skipping: {msg}")
                continue

            content_type = resp.headers.get("Content-Type", "")
            is_pdf_ext   = url.lower().endswith(".pdf")
            is_htm_ext   = url.lower().endswith((".htm", ".html"))

            if "pdf" not in content_type.lower() and not (is_pdf_ext or is_htm_ext):
                failed.append(f"{filename} — not a recognized document type ({content_type})")
                continue

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            size_kb = os.path.getsize(filepath) // 1024
            downloaded.append(f"{filename} ({size_kb} KB)")
            print(f"  [Download] Saved: {filename} ({size_kb} KB)")

        except requests.exceptions.Timeout:
            failed.append(f"{filename} — request timed out")
            print(f"  [Download] Timeout: {filename}")
        except requests.exceptions.HTTPError as http_error:
            failed.append(f"{filename} — HTTP error {http_error.response.status_code}")
            print(f"  [Download] HTTP error: {filename} — {http_error}")
        except Exception as download_error:
            failed.append(f"{filename} — {str(download_error)}")
            print(f"  [Download] Failed: {filename} — {download_error}")

    lines = [f"Downloaded {len(downloaded)} file(s):"]
    lines += [f"  - {d}" for d in downloaded]

    if skipped:
        lines.append(f"\nSkipped {len(skipped)} (already exist):")
        lines += [f"  - {s}" for s in skipped]

    if failed:
        lines.append(f"\nFailed {len(failed)}:")
        lines += [f"  - {f}" for f in failed]

    lines.append(f"\nFiles saved to: {os.path.abspath(DOWNLOAD_DIR)}")
    return "\n".join(lines)


# ── Tool 5: Scrape PDF Links ─────────────────────────────────────────

def scrape_pdf_links(ir_url: str) -> str:
    """
    Scrapes an investor relations page for document links using Playwright.
    Used as fallback when SEC EDGAR returns no results.
    Returns a JSON string of link objects.
    """
    print(f"  [Scraper] Scraping: {ir_url}")

    base_domain = f"{urlparse(ir_url).scheme}://{urlparse(ir_url).netloc}"
    pdf_links   = []

    SCRAPE_KEYWORDS = [
        "annual report", "10-k", "10k",
        "quarterly report", "10-q", "10q",
        "earnings", "press release",
        "investor presentation", "investor day",
        "supplement", "proxy", "def 14a",
        "8-k", "8k", "transcript"
    ]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()

            page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                )
            })

            try:
                page.goto(ir_url, timeout=30000, wait_until="networkidle")
                time.sleep(2)
                html = page.content()

            except Exception as page_error:
                print(f"  [Scraper] Page load error: {page_error}")
                try:
                    page.goto(ir_url, timeout=30000, wait_until="domcontentloaded")
                    time.sleep(3)
                    html = page.content()
                    print(f"  [Scraper] Retried with domcontentloaded — ok")
                except Exception as retry_error:
                    print(f"  [Scraper] Retry failed: {retry_error}")
                    browser.close()
                    return json.dumps([])

            browser.close()

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href        = a["href"].strip()
            label       = a.get_text(strip=True)
            label_lower = label.lower()
            href_lower  = href.lower()

            is_pdf      = href_lower.endswith(".pdf")
            is_relevant = any(
                k in label_lower or k in href_lower
                for k in SCRAPE_KEYWORDS
            )

            if not (is_pdf or is_relevant) or not label:
                continue

            if href.startswith("/"):
                href = urljoin(base_domain, href)
            elif not href.startswith("http"):
                href = urljoin(ir_url, href)

            if any(href.startswith(p) for p in ("mailto:", "javascript:", "#")):
                continue

            pdf_links.append({
                "label":    label,
                "url":      href,
                "doc_type": "unknown",
                "year":     str(scraper_config.CURRENT_YEAR)
            })

        seen         = set()
        unique_links = []
        for link in pdf_links:
            if link["url"] not in seen:
                seen.add(link["url"])
                unique_links.append(link)

        print(f"  [Scraper] Found {len(unique_links)} raw document links")
        return json.dumps(unique_links[:50])

    except Exception as scrape_error:
        print(f"  [Scraper] Fatal error scraping {ir_url}: {scrape_error}")
        return json.dumps([])


# ── Main Scraping Pipeline ───────────────────────────────────────────

def scrape_company(ticker: str, company_name: str) -> str:
    """
    Runs the full 5-step document scraping pipeline for a company.
    Returns a summary of what was downloaded, skipped, and failed.
    """
    ticker = ticker.upper().strip()

    # Step 1: Look up known IR page (used only as fallback in Step 5)
    ir_url_result = find_ir_page(company_name)
    ir_found      = "No known IR page" not in ir_url_result
    ir_url        = ir_url_result if ir_found else None

    # Step 2: Fetch SEC EDGAR filings — primary source
    sec_links_json = fetch_sec_edgar_links(ticker)
    sec_links      = json.loads(sec_links_json)

    # Step 3: Fetch earnings transcripts — always run alongside SEC
    transcripts_json = fetch_earnings_transcripts(ticker, company_name)
    transcripts      = json.loads(transcripts_json)

    # Step 4: Download combined links from Steps 2 + 3
    all_links = sec_links + transcripts
    summary   = download_filtered_pdfs(json.dumps(all_links), company_name)

    # Step 5: Fallback — only if SEC returned nothing and IR page exists
    if not sec_links and ir_found:
        print("\n  [Pipeline] SEC returned no results — falling back to IR page scraping")
        scraped_json  = scrape_pdf_links(ir_url)
        scraped_links = json.loads(scraped_json)
        if scraped_links:
            fallback_summary = download_filtered_pdfs(scraped_json, company_name)
            summary += "\n\n[Fallback IR scrape results]\n" + fallback_summary
        else:
            summary += "\n\nFallback IR scrape also returned no documents."

    return summary


# ── Main Interface ───────────────────────────────────────────────────

def run_scraper():
    print("\n" + "="*60)
    print("BFSI DOCUMENT SCRAPER")
    print("="*60)
    print("Document controls are set in scraper_config.py")
    print(f"Allowed types  : {[k for k, v in scraper_config.DOCUMENT_TYPES.items() if v]}")
    print(f"Oldest year    : {scraper_config.OLDEST_YEAR_ALLOWED}")
    print(f"Max per company: {scraper_config.MAX_FILES_PER_COMPANY}")
    print("="*60)
    print("\nEnter a ticker and company name to scrape. Type 'quit' to exit.\n")

    while True:
        ticker = input("Ticker symbol (e.g. JPM): ").strip().upper()

        if ticker.lower() in ("quit", "exit", "q"):
            print("Exiting scraper.")
            break

        if not ticker:
            continue

        company_name = input("Company name (e.g. JPMorgan Chase): ").strip()

        if not company_name:
            print("Company name is required.")
            continue

        print(f"\nStarting scraper: {ticker} / {company_name}\n")

        try:
            response = scrape_company(ticker, company_name)
            print("\n" + "="*60)
            print("SCRAPER SUMMARY")
            print("="*60)
            print(response)
            print("\nReady for next company.\n")

        except Exception as e:
            print(f"\nScraper error: {e}")
            import traceback
            traceback.print_exc()
            print("\nTry again.\n")


if __name__ == "__main__":
    run_scraper()
