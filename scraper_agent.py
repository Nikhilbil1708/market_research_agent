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
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from scraper_filter import filter_documents     # ← filter integration
import scraper_config                           # ← config integration

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

# SEC EDGAR CIK numbers — unique identifier for each company
# Find any company's CIK at: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
COMPANY_CIKS = {
    "JPM":   "0000019617",   # JPMorgan Chase
    "GS":    "0000886982",   # Goldman Sachs
    "MS":    "0000895421",   # Morgan Stanley
    "BAC":   "0000070858",   # Bank of America
    "WFC":   "0000072971",   # Wells Fargo
    "C":     "0000831001",   # Citigroup
    "BLK":   "0001364742",   # BlackRock
    "AXP":   "0000004962",   # American Express
    "SCHW":  "0000316709",   # Charles Schwab
    "USB":   "0000036104",   # US Bancorp
    "HSBC":  "0000083246",   # HSBC
    "PRU":   "0001137774",   # Prudential
    "MET":   "0001099219",   # MetLife
    "AIG":   "0000005272",   # AIG
    "MA":    "0001141391",   # Mastercard
    "V":     "0001403161",   # Visa
    "PYPL":  "0001633917",   # PayPal
}

# Keywords used to identify relevant document links on IR pages
TARGET_DOC_KEYWORDS = [
    "annual report", "10-k", "10k",
    "quarterly report", "10-q", "10q",
    "earnings", "press release",
    "investor presentation", "investor day",
    "supplement", "proxy", "def 14a", "8-k", "8k"
]

llm = ChatAnthropic(
    model=os.getenv("RESEARCH_MODEL", "claude-sonnet-4-6"),
    temperature=0
)


# ── Tool 1: Fetch Earnings Transcripts ─────────────────────────────────────────────

@tool
def fetch_earnings_transcripts(ticker: str, company_name: str) -> str:
    """
    Fetches earnings call transcript links from Motley Fool.
    These are free and publicly accessible.
    Returns a JSON list of transcript links.
    """
    import requests
    from bs4 import BeautifulSoup

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

                if len(links) >= 4:   # last 4 transcripts
                    break

    except Exception as transcript_error:
        print(f"  [Transcripts] Error: {transcript_error}")

    print(f"  [Transcripts] Found {len(links)} transcripts")
    return json.dumps(links)


# ── Tool 2: Find IR Page ─────────────────────────────────────────────

@tool
def find_ir_page(company_name: str) -> str:
    """
    Finds the investor relations page URL for a given BFSI company.
    Only checks the known IR pages dictionary — used as a fallback
    when SEC EDGAR has no CIK entry for this company.
    Returns the IR page URL as a string, or a not-found message.
    """
    lookup = company_name.lower().strip()

    # Check known pages first
    for key, url in KNOWN_IR_PAGES.items():
        if key in lookup or lookup in key:
            print(f"  [IR Page] Found known IR page for {company_name}: {url}")
            return url


    return (
        f"No known IR page for {company_name}. "
        f"Add it to KNOWN_IR_PAGES in scraper_agent.py if needed."
    )


# ── Tool 3: Fetch SEC EDGAR Links ─────────────────────────────────────────

@tool
def fetch_sec_edgar_links(ticker: str) -> str:
    """
    Fetches 10-K and 10-Q filing links directly from SEC EDGAR
    using the company's unique CIK number for precise identification.
    Returns a JSON list of document links with labels and URLs.
    """
    import requests
    from bs4 import BeautifulSoup

    ticker = ticker.upper().strip()
    cik    = COMPANY_CIKS.get(ticker)

    if not cik:
        print(f"  [SEC EDGAR] No CIK found for {ticker} — trying name search")
        return json.dumps([])

    print(f"  [SEC EDGAR] Fetching filings for {ticker} (CIK: {cik})")

    headers = {
        "User-Agent": "market-research-agent contact@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }

    links = []

    for doc_type, max_count in [("10-K", 2), ("10-Q", 4), ("8-K",  6)]:
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

                form_type  = cells[0].get_text(strip=True)
                date       = cells[3].get_text(strip=True)

                # Only accept exact form types
                if form_type not in (doc_type, f"{doc_type}/A"):
                    continue

                # Enforce year filter from scraper_config
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

                # Visit filing index page to find the actual document
                try:
                    index_resp = requests.get(
                        index_url, headers=headers, timeout=15
                    )
                    index_soup = BeautifulSoup(index_resp.text, "html.parser")

                    # Look for the primary document — prefer HTM over PDF
                    # as SEC filings are usually HTM format
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

                        # Only take the primary document
                        if any(k in doc_desc for k in [
                            "10-k", "10-q", "annual report",
                            "quarterly report", "form 10"
                        ]):
                            # Strip the inline XBRL viewer wrapper if present
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
                            break  # one document per filing

                except Exception as index_error:
                    print(f"  [SEC EDGAR] Index error: {index_error}")
                    continue

                if filing_count >= max_count:
                    break

        except Exception as fetch_error:
            print(f"  [SEC EDGAR] Error fetching {doc_type}: {fetch_error}")

    print(f"  [SEC EDGAR] Total links found: {len(links)}")
    return json.dumps(links)

@tool
def download_filtered_pdfs(raw_links_json: str, company_name: str) -> str:
    """
    Takes raw scraped links as a JSON string, applies all rules from
    scraper_config.py and scraper_filter.py, then downloads only the
    approved documents into the data folder.
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

        # Safety net — strip inline viewer wrapper if it ever appears
        if "/ix?doc=" in url:
            url = url.replace("https://www.sec.gov/ix?doc=", "https://www.sec.gov")

        doc_type = doc.get("doc_type", "unknown")
        year     = doc.get("year", scraper_config.CURRENT_YEAR)

        headers = get_headers_for_url(url)   # ← per-URL headers now

        safe_company = re.sub(r"[^a-zA-Z0-9]", "_", company_name.lower()).strip("_")
        safe_type    = re.sub(r"[^a-zA-Z0-9]", "_", doc_type.lower()).strip("_")
        filename     = f"{safe_company}_{safe_type}_{year}.pdf"
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
            max_bytes       = scraper_config.MAX_FILE_SIZE_MB * 1024 * 1024
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

            # SEC filings are often .htm — save with .pdf extension is misleading
            # but content is text-based and will still ingest fine downstream
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            size_kb = os.path.getsize(filepath) // 1024
            downloaded.append(f"{filename} ({size_kb} KB)")
            print(f"  [Download] Saved: {filename} ({size_kb} KB)")

        except requests.exceptions.Timeout:
            failed.append(f"{filename} — request timed out")
        except requests.exceptions.HTTPError as http_error:
            failed.append(f"{filename} — HTTP error {http_error.response.status_code}")
            print(f"  [Download] HTTP error: {filename} — {http_error}")
        except Exception as download_error:
            failed.append(f"{filename} — {str(download_error)}")
            print(f"  [Download] Failed: {filename} — {download_error}")

    lines = [f"Downloaded {len(downloaded)} file(s):"]
    lines += [f"  ✓ {d}" for d in downloaded]

    if skipped:
        lines.append(f"\nSkipped {len(skipped)} (already exist):")
        lines += [f"  – {s}" for s in skipped]

    if failed:
        lines.append(f"\nFailed {len(failed)}:")
        lines += [f"  ✗ {f}" for f in failed]

    lines.append(f"\nFiles saved to: {os.path.abspath(DOWNLOAD_DIR)}")
    return "\n".join(lines)


# ── Tool 4: Download Filtered PDFs ───────────────────────────────────

def get_headers_for_url(url: str) -> dict:
    """Returns the correct headers depending on which site is being accessed."""
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


@tool
def download_filtered_pdfs(raw_links_json: str, company_name: str) -> str:
    """
    Takes raw scraped links as a JSON string, applies all rules from
    scraper_config.py and scraper_filter.py, then downloads only the
    approved documents into the data folder.
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

        headers = get_headers_for_url(url)   # ← now actually called, per URL

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
            max_bytes       = scraper_config.MAX_FILE_SIZE_MB * 1024 * 1024
            if content_length > max_bytes:
                size_mb = content_length // 1024 // 1024
                msg = f"{filename} — too large ({size_mb} MB)"
                failed.append(msg)
                print(f"  [Download] Skipping: {msg}")
                continue

            content_type = resp.headers.get("Content-Type", "")
            is_pdf_ext    = url.lower().endswith(".pdf")
            is_htm_ext    = url.lower().endswith((".htm", ".html"))

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


# ── Tool 5: Scrape PDF Links ───────────────────────────────────

@tool
def scrape_pdf_links(ir_url: str) -> str:
    """
    Scrapes an investor relations page and returns a JSON list of
    document links found on the page with their labels and URLs.
    Used as a fallback when SEC EDGAR returns no results.
    Uses Playwright to handle JavaScript-rendered pages.
    Returns a JSON string of link objects.
    """
    import time
    from urllib.parse import urljoin, urlparse
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup

    print(f"  [Scraper] Scraping: {ir_url}")

    base_domain = (
        f"{urlparse(ir_url).scheme}://{urlparse(ir_url).netloc}"
    )
    pdf_links = []

    # Keywords that identify relevant financial documents
    TARGET_DOC_KEYWORDS = [
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

            # Set headers to look like a real browser
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
                page.goto(
                    ir_url,
                    timeout=30000,
                    wait_until="networkidle"
                )
                time.sleep(2)   # wait for JS to finish rendering
                html = page.content()

            except Exception as page_error:
                print(f"  [Scraper] Page load error: {page_error}")
                # Try with domcontentloaded instead of networkidle
                # which is less strict and works on more sites
                try:
                    page.goto(
                        ir_url,
                        timeout=30000,
                        wait_until="domcontentloaded"
                    )
                    time.sleep(3)
                    html = page.content()
                    print(f"  [Scraper] Retried with domcontentloaded — ok")
                except Exception as retry_error:
                    print(f"  [Scraper] Retry failed: {retry_error}")
                    browser.close()
                    return json.dumps([])

            browser.close()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href        = a["href"].strip()
            label       = a.get_text(strip=True)
            label_lower = label.lower()
            href_lower  = href.lower()

            # Check if link is a PDF or a relevant document
            is_pdf      = href_lower.endswith(".pdf")
            is_relevant = any(
                k in label_lower or k in href_lower
                for k in TARGET_DOC_KEYWORDS
            )

            if not (is_pdf or is_relevant):
                continue

            if not label:
                continue

            # Resolve relative URLs to absolute
            if href.startswith("/"):
                href = urljoin(base_domain, href)
            elif not href.startswith("http"):
                href = urljoin(ir_url, href)

            # Skip mailto, javascript, and anchor links
            if any(
                href.startswith(p)
                for p in ("mailto:", "javascript:", "#")
            ):
                continue

            pdf_links.append({
                "label":    label,
                "url":      href,
                "doc_type": "unknown",   # scraper_filter will classify
                "year":     str(scraper_config.CURRENT_YEAR)
            })

        # Deduplicate by URL
        seen         = set()
        unique_links = []
        for link in pdf_links:
            if link["url"] not in seen:
                seen.add(link["url"])
                unique_links.append(link)

        print(f"  [Scraper] Found {len(unique_links)} raw document links")

        # Cap at 50 before passing to filter
        return json.dumps(unique_links[:50])

    except Exception as scrape_error:
        print(f"  [Scraper] Fatal error scraping {ir_url}: {scrape_error}")
        return json.dumps([])
    

# ── Tools list — GOES HERE, after all tools, before agent setup ──────
tools = [
    find_ir_page,
    fetch_sec_edgar_links,
    fetch_earnings_transcripts,
    scrape_pdf_links,
    download_filtered_pdfs
]

# ── Agent Setup ──────────────────────────────────────────────────────
from langchain.agents import create_agent

SYSTEM_PROMPT = """You are a financial document scraper agent for BFSI research.

Follow these steps in order:

Step 1 - Call find_ir_page with the company name to check if a known
         investor relations page exists for this company. This is used
         only as a fallback later in Step 5 — note the result but do
         not download anything yet.

Step 2 - Call fetch_sec_edgar_links with the TICKER SYMBOL (not the
         company name, e.g. "MS" not "Morgan Stanley") to fetch 10-K,
         10-Q, and 8-K filings directly from SEC EDGAR using the
         company's CIK number. This is your primary and most reliable
         source — always try this first.

Step 3 - Call fetch_earnings_transcripts with the TICKER SYMBOL and
         COMPANY NAME to fetch earnings call transcripts from Motley Fool.
         Run this regardless of whether Step 2 succeeded, since
         transcripts are a separate document type.

Step 4 - Combine the JSON links returned from Step 2 and Step 3 into
         a single JSON list. Call download_filtered_pdfs with this
         combined list and the company name. The filter rules in
         scraper_config.py are applied automatically — do not filter
         or select documents yourself.

Step 5 - ONLY if Step 2 (fetch_sec_edgar_links) returned an empty list
         AND the IR page from Step 1 was found, call scrape_pdf_links
         with that IR page URL to find documents directly on the
         company website. Then call download_filtered_pdfs again with
         whatever links this returns.

Important rules:
- Always use the ticker symbol, not the company name, when calling
  fetch_sec_edgar_links
- SEC EDGAR (Step 2) is always more reliable than scraping a company
  website — never skip it or call Step 5 before Step 2
- Do not attempt to filter, classify, or select documents yourself —
  download_filtered_pdfs handles all of that automatically
- If both Step 2 and Step 5 return no documents, report this clearly
  to the user instead of pretending something was downloaded
- After all steps, summarize exactly what was downloaded, what was
  skipped, and what failed
"""

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# ── Helper to invoke agent ───────────────────────────────────────────

def invoke_scraper(instruction: str) -> str:
    """
    Invokes the scraper agent with a plain English instruction.
    Returns the final response as a string.
    """
    result = agent_executor.invoke({
        "messages": [("user", instruction)]
    })
    # create_react_agent returns messages list — get last message
    return result["messages"][-1].content


# ── Main Interface ───────────────────────────────────────────────────

def run_scraper():
    print("\n" + "="*60)
    print("BFSI DOCUMENT SCRAPER AGENT")
    print("="*60)
    print("Document controls are set in scraper_config.py")
    print(f"Allowed types  : {[k for k,v in scraper_config.DOCUMENT_TYPES.items() if v]}")
    print(f"Oldest year    : {scraper_config.OLDEST_YEAR_ALLOWED}")
    print(f"Max per company: {scraper_config.MAX_FILES_PER_COMPANY}")
    print("="*60)
    print("\nType a company name or instruction. Type 'quit' to exit.\n")

    while True:
        user_input = input("Scraper instruction: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting scraper.")
            break

        if not user_input:
            continue

        print(f"\nStarting scraper: {user_input}\n")

        try:
            response = invoke_scraper(user_input)
            print("\n" + "="*60)
            print("SCRAPER SUMMARY")
            print("="*60)
            print(response)
            print("\nReady for next instruction.\n")

        except Exception as e:
            print(f"\nScraper error: {e}")
            import traceback
            traceback.print_exc()
            print("\nTry again.\n")


if __name__ == "__main__":
    run_scraper()