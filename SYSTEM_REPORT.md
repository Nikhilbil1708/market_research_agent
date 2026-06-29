# Market Research Agent — System Report

---

## 1. Introduction

The Market Research Agent is an automated, multi-agent research system designed for the Banking, Financial Services, and Insurance (BFSI) sector. Given a company name and a research question, it autonomously gathers data from multiple sources, analyzes it across specialized domains, and produces a structured PDF report — with no manual intervention between query and output.

### What it does

- Accepts a natural language research instruction (e.g. "What are JPMorgan's key growth drivers and risks in 2025?")
- Resolves the company name and ticker symbol automatically
- Downloads and indexes the latest financial filings (10-K, 10-Q, 8-K, earnings transcripts)
- Runs parallel research agents across news, financial data, technology strategy, and macroeconomic context
- Synthesizes all findings into a structured, sourced report
- Generates financial charts and exports a PDF

### Technology stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM | Anthropic Claude Haiku 4.5 |
| Web search | Tavily Search API |
| Financial data | yfinance |
| Document storage | ChromaDB (vector store) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Reranking | Cohere rerank-english-v3.0 |
| PDF scraping | SEC EDGAR, BeautifulSoup |
| Charts | matplotlib |
| Output | PDF (pdf_generator.py) |

---

## 2. Architecture

### 2.1 Agent overview

The system is composed of seven nodes wired together in a LangGraph StateGraph:

```
decomposition_agent
       |
     router
    /  |  |  \
news  fin  tech  macro
agent agent agent agent
    \  |  |  /
      rag_agent
         |
    synthesis_agent
         |
       [END]
```

Each agent reads from and writes to a shared state object (`MarketResearchState`). No agent calls another directly — all communication happens through state.

### 2.2 Shared state (MarketResearchState)

```
query                   Original research question
ticker                  Stock ticker symbol (e.g. JPM)
sub_queries             Focused per-agent search queries (from decomposition)
messages                Conversation history
news_analysis           Output of news_agent
tech_strategy_context   Output of tech_agent
financial_data          Output of financial_agent (prose)
financial_metrics       Structured financial data (revenue, EBITDA, COGS, SG&A)
financial_charts        Chart file paths
chart_paths             Accumulated chart paths
macro_context           Output of macro_agent
rag_context             Retrieved document chunks from ChromaDB
final_report            Synthesized report text
confidence_score        Data quality score (0.0 - 1.0)
sources                 Accumulated list of all cited URLs and data sources
company_pe              P/E ratio of the researched company
competitor_pe           P/E ratios of sector competitors
```

### 2.3 Agent responsibilities

#### Decomposition Agent (`agents/decomposition_agent.py`)
- Entry point of the graph
- Makes one Haiku LLM call to break the user's query into four focused sub-queries — one per search-based agent
- If LLM returns invalid JSON, falls back to using the original query for all agents
- Writes `sub_queries` dict to state

#### Router (`graph.py → route_tasks()`)
- Pure Python — zero LLM calls
- Keyword-matches the original query to decide which agents to run
- `news_agent` and `rag_agent` always run regardless of query content
- If no specific keywords match, runs all agents

#### News Agent (`agents/news_agent.py`)
- Always runs — provides the company overview section required in every report
- Calls Tavily Search API (max 8 results) using `sub_queries["news"]`
- One Haiku LLM call extracts: Company Overview, Recent News, Important Events
- Writes `news_analysis` to state; appends URLs to `sources`

#### Financial Agent (`agents/financial_agent.py`)
- Runs when query contains financial keywords (earnings, revenue, margin, etc.)
- Three-tier data pipeline, no LLM unless last resort:
  1. **yfinance** — fetches real Revenue, EBITDA (derived), COGS (Interest Expense proxy), SG&A
  2. **HTML table parsing** — pandas-based fallback from downloaded SEC filings
  3. **LLM extraction** — last resort only, extracts figures from RAG context text
- Fetches P/E ratio for the company and sector competitors
- Generates 4 matplotlib charts (Revenue, EBITDA, COGS, SG&A) — annual and quarterly
- Writes `financial_metrics`, `chart_paths`, `company_pe`, `competitor_pe` to state

#### Tech Agent (`agents/tech_agent.py`)
- Runs when query contains technology keywords (AI, cloud, digital, cybersecurity, etc.)
- Makes **two** Tavily searches: one for AI/ML initiatives, one for cloud/partnerships
- One Haiku LLM call extracts named initiatives and vendor partnerships
- Writes `tech_strategy_context` to state

#### Macro Agent (`agents/macro_agent.py`)
- Runs when query contains macro keywords (rate, inflation, GDP, sector, etc.)
- One Tavily search (max 6 results) using `sub_queries["macro"]`
- One Haiku LLM call extracts interest rate impact, sector conditions, economic data
- Writes `macro_context` to state

#### RAG Agent (`agents/rag_node.py`)
- Always runs — provides grounding from indexed company documents
- No LLM call; uses two external components:
  - **ChromaDB** MMR search: fetches 20 candidates, returns 8 diverse relevant chunks
  - **Cohere Rerank**: scores all 8 chunks against the query, returns top 4
- Writes `rag_context` (raw PDF text chunks) to state

#### Synthesis Agent (`agents/synthesis_agent.py`)
- Runs after all other agents converge
- Receives all agent outputs from state and combines them into one prompt
- One Haiku LLM call produces a structured 7-section report:
  1. Company Overview
  2. Technology and IT Strategy
  3. Financial Summary
  4. Macro and Sector Context
  5. Key Findings
  6. Risk Factors
  7. Sources
- Runs `calculate_confidence_score()` — pure Python, no LLM
- Writes `final_report` and `confidence_score` to state

### 2.4 RAG pipeline architecture

Documents go through a two-phase pipeline:

**Indexing phase (runs once per company):**
```
PDF / HTML / URL
      |
  PyPDFLoader / WebBaseLoader
      |
  RecursiveCharacterTextSplitter
  (chunk_size=1000, overlap=200)
      |
  all-MiniLM-L6-v2 (local, 384-dim vectors)
      |
  ChromaDB (persisted to ./chroma_db)
```

**Retrieval phase (runs every query):**
```
Query string
      |
  all-MiniLM-L6-v2 embedding
      |
  ChromaDB MMR search (fetch_k=20 -> k=8)
      |
  Cohere rerank-english-v3.0 (8 -> top 4)
      |
  rag_context (4 most relevant chunks)
```

MMR (Maximal Marginal Relevance) is used instead of plain similarity search to ensure the 8 retrieved chunks are both relevant and diverse — preventing duplicate content from appearing multiple times.

### 2.5 Document scraping pipeline

Triggered once per company before the research agents run:

```
Step 1 — Look up known IR page URL (fallback only)
Step 2 — SEC EDGAR: fetch 10-K (x2), 10-Q (x4), 8-K (x6) filing links
Step 3 — Earnings transcripts scraper
Step 4 — Download all PDFs from Steps 2 + 3
Step 5 — Fallback: scrape IR page directly if EDGAR returned nothing
```

Downloaded files are saved to `./data/` and filtered by `OLDEST_YEAR_ALLOWED` from `scraper_config.py`.

### 2.6 LLM call summary

| Agent | LLM calls | Always? | Approx. input tokens |
|---|---|---|---|
| decomposition_agent | 1 | Yes | ~200 |
| interface_agent (intent extraction) | 1 | Only for unknown companies | ~200 |
| news_agent | 1 | Yes | ~3,000 |
| macro_agent | 1 | If macro keywords match | ~2,300 |
| tech_agent | 1 | If tech keywords match | ~6,000 |
| financial_agent | 1 | Last resort only | ~2,000 |
| synthesis_agent | 1 | Yes | ~3,000–5,000 |
| **Total per run** | **3–7** | | **~14,000–18,000** |

Estimated cost per full pipeline run: **~$0.014–$0.018** (Haiku pricing).

---

## 3. Process Flow

### Step 1 — User input
The user types a research instruction into the terminal via `interface_agent.py → run_interface()`.

```
Research instruction: What are JPMorgan's key growth drivers in 2025?
```

### Step 2 — Intent resolution
The system first attempts pure Python resolution using a hardcoded `BFSI_TICKERS` dictionary. If the company is found (e.g. "jpmorgan" → `JPM`), no LLM is used. If the company is unknown, a Haiku LLM call extracts the company name, ticker, query, and sector from the raw instruction.

### Step 3 — Display resolved intent
The resolved intent is printed to the terminal:
```
Company  : Jpmorgan
Ticker   : JPM
Sector   : Banking
Query    : What are JPMorgan's key growth drivers in 2025?
```

### Step 4 — Input guardrails
Two blocking checks run before the pipeline starts (see Section 4).

### Step 5 — Stage 1: Document scraping
`scraper_agent.scrape_company()` downloads the latest 10-K, 10-Q, 8-K filings and earnings transcripts from SEC EDGAR into `./data/`. If scraping fails, the pipeline continues using whatever is already in the vector store.

### Step 6 — Stage 2: Document indexing
New PDFs are chunked, embedded, and added to ChromaDB. Already-indexed documents are skipped. The RAG staleness guardrail runs here (see Section 4).

### Step 7 — Query decomposition
`decomposition_node` makes one Haiku call to generate four focused sub-queries tailored to each agent's domain. These are stored in `state["sub_queries"]`.

### Step 8 — Routing
`route_tasks()` keyword-matches the original query to determine which agents run. `news_agent` and `rag_agent` always run. The decomposition does not affect routing — only the search queries agents use.

### Step 9 — Parallel agent execution
LangGraph fires all selected agents concurrently. Each agent reads from state, performs its research, and writes its output back to state. Agents do not communicate with each other directly.

### Step 10 — Synthesis
Once all agents complete, `synthesis_node` collects every agent's output from state, builds a single combined prompt (~3,000–5,000 tokens), and makes one Haiku call to produce the structured 7-section final report.

### Step 11 — Output guardrails
Four non-blocking checks run on the research output (see Section 4). Warnings are printed but PDF generation always proceeds.

### Step 12 — PDF generation
`generate_pdf()` renders the final report, confidence score, source list, and the four financial charts into a PDF file saved to disk.

---

## 4. Guardrails

All guardrails are implemented in `guardrails.py` and called from `interface_agent.py` and `pipeline.py`.

### 4.1 Input guardrails (blocking — pipeline aborts if these fail)

#### Guardrail 1 — Ticker validation
**Where:** `interface_agent.py`, before pipeline starts
**Method:** Calls `yf.Ticker(ticker).fast_info.last_price` via yfinance. No LLM.
**Behaviour:** If the ticker has no market data, the pipeline aborts immediately with the message:
> "Irrelevant query - Can't be done"

**Purpose:** Catches typos and unknown tickers before spending any API credits.

#### Guardrail 2 — Query relevance
**Where:** `interface_agent.py`, before pipeline starts
**Method:** Pure Python blocklist check against clearly off-topic topics (recipes, poems, weather, sports scores, etc.) and a minimum word count check (≥ 3 words). No LLM.
**Behaviour:** Any query not matching the blocklist passes — the check is intentionally permissive so that any legitimate question about the company (culture, leadership, ESG, legal, products, etc.) is allowed through. Only blatantly irrelevant queries are blocked.
**Behaviour on fail:** Pipeline aborts with the message:
> "Irrelevant query - Can't be done"

### 4.2 Pipeline guardrail (warning only)

#### Guardrail 6 — RAG staleness
**Where:** `pipeline.py`, after Stage 2 (indexing)
**Method:** Checks `os.path.getmtime("chroma_db")` against a 7-day threshold. No LLM.
**Behaviour on fail:** Prints a warning. Pipeline continues.
> "[WARN] RAG staleness: Vector store last updated N day(s) ago. Consider re-indexing."

**Purpose:** Alerts the user when the indexed documents may be too old to reflect recent earnings or events.

### 4.3 Output guardrails (warning only — PDF is always generated)

All four output guardrails run after Stage 3 (research agents complete) and before Stage 4 (PDF generation).

#### Guardrail 3 — Confidence score threshold
**Method:** Checks `confidence_score >= 0.65`. The confidence score is computed deterministically from data signals (no LLM):

| Signal | Points |
|---|---|
| Baseline | +0.40 |
| news_analysis populated | +0.10 |
| Sources ≥ 5 | +0.15 |
| Sources ≥ 2 (but < 5) | +0.08 |
| RAG context populated | +0.15 |
| P/E ratio found | +0.10 |
| Real revenue data found | +0.10 |
| Charts generated | +0.05 |
| **Maximum** | **1.00** |

**Behaviour on fail:** Prints a warning identifying the score and threshold.

#### Guardrail 4 — Empty section detection
**Method:** Counts occurrences of "Not researched for this query." in the final report. Triggers if count > 2.
**Behaviour on fail:** Warns that multiple sections are empty, suggesting the query may need broadening or the routing may have missed relevant agents.

#### Guardrail 5 — Source minimum
**Method:** Checks `len(sources) >= 2`.
**Behaviour on fail:** Warns that the report has fewer than 2 cited sources, indicating limited research coverage.

#### Guardrail 7 — LLM output length
**Method:** Splits the final report on numbered section headings using regex and checks that no section exceeds 600 characters.
**Behaviour on fail:** Lists which sections exceeded the limit, indicating the synthesis model ignored the output constraints in its prompt.

### 4.4 Guardrail summary

| # | Guardrail | Stage | Blocking | Method |
|---|---|---|---|---|
| 1 | Ticker validation | Input | Yes | yfinance |
| 2 | Query relevance | Input | Yes | Python blocklist |
| 6 | RAG staleness | Post-indexing | No | File mtime |
| 3 | Confidence threshold | Post-research | No | Python arithmetic |
| 4 | Empty section detection | Post-research | No | String count |
| 5 | Source minimum | Post-research | No | List length |
| 7 | LLM output length | Post-research | No | Regex |
