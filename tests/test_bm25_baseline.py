"""
tests/test_bm25_baseline.py

Unit tests for BM25Retriever (src/retrieval/bm25_baseline.py).

Run with:
    pytest tests/test_bm25_baseline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the project root is on sys.path so imports work without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.bm25_baseline import BM25Retriever  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"text": "Revenue increased by 12% in Q3 driven by strong cloud demand.", "chunk_index": 0},
    {"text": "Operating expenses rose due to increased headcount in R&D.", "chunk_index": 1},
    {"text": "Net income declined as a result of one-time restructuring charges.", "chunk_index": 2},
    {"text": "The company repurchased 5 million shares in the open market.", "chunk_index": 3},
    {"text": "Gross margin improved by 2 percentage points year-over-year.", "chunk_index": 4},
]


@pytest.fixture
def built_retriever():
    retriever = BM25Retriever()
    retriever.build(SAMPLE_CHUNKS)
    return retriever


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBM25Search:
    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_search_returns_exactly_top_k(self, built_retriever, top_k):
        results = built_retriever.search("revenue increased", top_k=top_k)
        expected = min(top_k, len(SAMPLE_CHUNKS))
        assert len(results) == expected

        # Check required keys
        for r in results:
            assert "text" in r
            assert "chunk_index" in r
            assert "score" in r

    def test_search_returns_all_if_top_k_exceeds_corpus(self, built_retriever):
        results = built_retriever.search("revenue", top_k=10)
        assert len(results) == len(SAMPLE_CHUNKS)

    def test_exact_match_ranks_first(self, built_retriever):
        # Query matching the exact words of chunk 1
        query = "Operating expenses rose due to increased headcount"
        results = built_retriever.search(query, top_k=1)
        assert results, "Expected at least one result"
        top_result = results[0]
        assert top_result["chunk_index"] == 1
        assert "Operating expenses rose" in top_result["text"]


class TestBM25Build:
    def test_empty_corpus_raises_error(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="empty corpus"):
            retriever.build([])

    def test_search_before_build_raises_error(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="not built"):
            retriever.search("query")
