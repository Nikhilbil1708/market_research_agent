"""
Tests for the query decomposition agent.
No real LLM calls — decomposition LLM is mocked with a fixed JSON response.
All other agent LLMs and search tools are also mocked.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from unittest.mock import MagicMock, patch, call

TICKER = "JPM"
QUERY  = "What are JPM's revenue growth, AI strategy, and interest rate exposure?"

FAKE_SUB_QUERIES = {
    "news":      "JPM latest earnings leadership news 2025",
    "financial": "JPM revenue EBITDA earnings growth 2024 2025",
    "tech":      "JPM AI machine learning cloud digital strategy 2025",
    "macro":     "JPM interest rate federal reserve inflation exposure 2025",
}

# Decomposition LLM returns this JSON
fake_decomp = MagicMock()
fake_decomp.content = """{
    "news":      "JPM latest earnings leadership news 2025",
    "financial": "JPM revenue EBITDA earnings growth 2024 2025",
    "tech":      "JPM AI machine learning cloud digital strategy 2025",
    "macro":     "JPM interest rate federal reserve inflation exposure 2025"
}"""

# Generic LLM stub for all other agents
fake_llm = MagicMock()
fake_llm.content = "Mocked agent output."

fake_agent_response = MagicMock()
fake_agent_response.content = "Mocked agent output."

fake_search_result = [{"url": "https://example.com", "content": "Some article text about JPM."}]


def sep(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ── Test 1: decomposition_node parses JSON correctly ──────────────────
sep("Test 1: decomposition_node — valid JSON response")

with patch("agents.decomposition_agent.llm") as mock_llm:
    mock_llm.invoke.return_value = fake_decomp
    from agents.decomposition_agent import decomposition_node

    state  = {"query": QUERY, "ticker": TICKER, "sub_queries": {}}
    result = decomposition_node(state)

    assert "sub_queries" in result, "sub_queries missing from result"
    for key in ("news", "financial", "tech", "macro"):
        assert key in result["sub_queries"], f"Missing key: {key}"
        assert TICKER in result["sub_queries"][key], f"Ticker not in sub_query for {key}"
        print(f"  [OK] {key:10}: {result['sub_queries'][key]}")

print("  PASSED")


# ── Test 2: decomposition_node fallback on bad JSON ───────────────────
sep("Test 2: decomposition_node — invalid JSON fallback")

bad_response = MagicMock()
bad_response.content = "Sorry I cannot do that right now."

with patch("agents.decomposition_agent.llm") as mock_llm:
    mock_llm.invoke.return_value = bad_response
    result = decomposition_node({"query": QUERY, "ticker": TICKER, "sub_queries": {}})

    assert "sub_queries" in result
    for key in ("news", "financial", "tech", "macro"):
        assert key in result["sub_queries"]
        assert TICKER in result["sub_queries"][key], f"Ticker missing in fallback for {key}"
        print(f"  [OK] fallback {key}: {result['sub_queries'][key]}")

print("  PASSED")


# ── Test 3: news_agent uses sub_queries["news"] ───────────────────────
sep("Test 3: news_agent picks up sub_queries['news']")

with patch("agents.news_agent.llm") as mock_llm, \
     patch("agents.news_agent.search_tool") as mock_search:

    mock_llm.invoke.return_value  = fake_agent_response
    mock_search.invoke.return_value = fake_search_result

    from agents.news_agent import news_node
    news_node({
        "ticker": TICKER,
        "query": QUERY,
        "sub_queries": FAKE_SUB_QUERIES,
        "sources": []
    })

    called_with = mock_search.invoke.call_args[0][0]
    assert FAKE_SUB_QUERIES["news"] in called_with, \
        f"Expected sub_query in search call, got: {called_with}"
    print(f"  [OK] search called with: {called_with}")

print("  PASSED")


# ── Test 4: macro_agent uses sub_queries["macro"] ────────────────────
sep("Test 4: macro_agent picks up sub_queries['macro']")

with patch("agents.macro_agent.llm") as mock_llm, \
     patch("agents.macro_agent.search_tool") as mock_search:

    mock_llm.invoke.return_value    = fake_agent_response
    mock_search.invoke.return_value = fake_search_result

    from agents.macro_agent import macro_node
    macro_node({
        "ticker": TICKER,
        "query": QUERY,
        "sub_queries": FAKE_SUB_QUERIES,
        "sources": []
    })

    called_with = mock_search.invoke.call_args[0][0]
    assert FAKE_SUB_QUERIES["macro"] in called_with, \
        f"Expected sub_query in search call, got: {called_with}"
    print(f"  [OK] search called with: {called_with}")

print("  PASSED")


# ── Test 5: tech_agent uses sub_queries["tech"] ──────────────────────
sep("Test 5: tech_agent picks up sub_queries['tech']")

with patch("agents.tech_agent.llm") as mock_llm, \
     patch("agents.tech_agent.search_tool") as mock_search:

    mock_llm.invoke.return_value    = fake_agent_response
    mock_search.invoke.return_value = fake_search_result

    from agents.tech_agent import tech_strategy_node
    tech_strategy_node({
        "ticker": TICKER,
        "query": QUERY,
        "sub_queries": FAKE_SUB_QUERIES,
        "sources": []
    })

    all_calls = [c[0][0] for c in mock_search.invoke.call_args_list]
    assert any(FAKE_SUB_QUERIES["tech"] in c for c in all_calls), \
        f"Expected tech sub_query in one of search calls: {all_calls}"
    print(f"  [OK] search calls: {all_calls}")

print("  PASSED")


# ── Test 6: agents fall back to original query when sub_queries empty ─
sep("Test 6: agents fall back to original query when sub_queries is empty")

with patch("agents.news_agent.llm") as mock_llm, \
     patch("agents.news_agent.search_tool") as mock_search:

    mock_llm.invoke.return_value    = fake_agent_response
    mock_search.invoke.return_value = fake_search_result

    news_node({
        "ticker": TICKER,
        "query": QUERY,
        "sub_queries": {},      # empty — no decomposition ran
        "sources": []
    })

    called_with = mock_search.invoke.call_args[0][0]
    assert TICKER in called_with
    assert QUERY in called_with or TICKER in called_with
    print(f"  [OK] fallback search called with: {called_with}")

print("  PASSED")


# ── Test 7: full graph flow ───────────────────────────────────────────
sep("Test 7: full graph — decomposition > router > agents > synthesis")

patches = [
    patch("agents.decomposition_agent.llm"),
    patch("agents.news_agent.llm"),
    patch("agents.news_agent.search_tool"),
    patch("agents.macro_agent.llm"),
    patch("agents.macro_agent.search_tool"),
    patch("agents.tech_agent.llm"),
    patch("agents.tech_agent.search_tool"),
    patch("agents.synthesis_agent.llm"),
    patch("agents.rag_node.load_vector_store"),
    patch("agents.rag_node.build_retriever"),
]

mocks = [p.start() for p in patches]

decomp_mock, news_llm, news_search, macro_llm, macro_search, \
tech_llm, tech_search, synth_llm, mock_vs, mock_retriever = mocks

decomp_mock.invoke.return_value = fake_decomp

for m in (news_llm, macro_llm, tech_llm, synth_llm):
    m.invoke.return_value = fake_agent_response

for m in (news_search, macro_search, tech_search):
    m.invoke.return_value = fake_search_result

mock_retriever.return_value.invoke.return_value = []

from graph import build_graph

graph  = build_graph()
result = graph.invoke({
    "query"          : QUERY,
    "ticker"         : TICKER,
    "sub_queries"    : {},
    "messages"       : [],
    "news_analysis"  : "",
    "financial_data" : "",
    "macro_context"  : "",
    "rag_context"    : "",
    "final_report"   : "",
    "confidence_score": 0.0,
    "sources"        : [],
    "chart_paths"    : [],
    "financial_charts": [],
    "company_overview_context": "",
    "tech_strategy_context"   : "",
})

for p in patches:
    p.stop()

assert result["sub_queries"] == FAKE_SUB_QUERIES, \
    f"sub_queries not propagated through graph: {result['sub_queries']}"
assert result["final_report"] == "Mocked agent output."
assert decomp_mock.invoke.call_count == 1, "Decomposition LLM should be called exactly once"

print(f"  [OK] sub_queries propagated: {list(result['sub_queries'].keys())}")
print(f"  [OK] final_report present")
print(f"  [OK] decomposition LLM called exactly once")
print("  PASSED")


sep("ALL TESTS PASSED — no LLM calls were made")
