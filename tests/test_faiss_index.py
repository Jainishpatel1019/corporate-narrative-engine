"""
tests/test_faiss_index.py

Unit tests for FAISSIndex (src/retrieval/faiss_index.py).

Run with:
    pytest tests/test_faiss_index.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make sure the project root is on sys.path so imports work without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.faiss_index import FAISSIndex  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"text": "Revenue increased by 12% in Q3 driven by strong cloud demand.", "chunk_index": 0},
    {"text": "Operating expenses rose due to increased headcount in R&D.", "chunk_index": 1},
    {"text": "Net income declined as a result of one-time restructuring charges.", "chunk_index": 2},
    {"text": "The company repurchased 5 million shares in the open market.", "chunk_index": 3},
    {"text": "Gross margin improved by 2 percentage points year-over-year.", "chunk_index": 4},
    {"text": "Cash and cash equivalents stood at $4.2 billion at quarter end.", "chunk_index": 5},
    {"text": "Forward guidance was raised for the full fiscal year.", "chunk_index": 6},
]


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    """Build an index once for all tests in this module."""
    tmp = tmp_path_factory.mktemp("faiss_data")
    index_path = str(tmp / "test.faiss")

    idx = FAISSIndex()
    idx.build(SAMPLE_CHUNKS, index_path=index_path)
    return idx, index_path


# ---------------------------------------------------------------------------
# Test 1 — build() creates both files on disk
# ---------------------------------------------------------------------------

class TestBuildCreatesFiles:
    def test_faiss_index_file_exists(self, built_index):
        _, index_path = built_index
        assert os.path.isfile(index_path), "FAISS index file should be created by build()"

    def test_metadata_json_file_exists(self, built_index):
        _, index_path = built_index
        meta_path = str(Path(index_path).with_suffix(".json"))
        assert os.path.isfile(meta_path), "Metadata JSON file should be created by build()"

    def test_metadata_json_is_valid(self, built_index):
        _, index_path = built_index
        meta_path = str(Path(index_path).with_suffix(".json"))
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == len(SAMPLE_CHUNKS)
        for item in data:
            assert "text" in item
            assert "chunk_index" in item


# ---------------------------------------------------------------------------
# Test 2 — search() returns exactly top_k results
# ---------------------------------------------------------------------------

class TestSearchReturnsTopK:
    @pytest.mark.parametrize("top_k", [1, 3, 5, 7])
    def test_exact_top_k_results(self, built_index, top_k):
        idx, _ = built_index
        results = idx.search("financial performance", top_k=top_k)
        expected = min(top_k, len(SAMPLE_CHUNKS))
        assert len(results) == expected, (
            f"Expected {expected} results for top_k={top_k}, got {len(results)}"
        )

    def test_results_have_required_keys(self, built_index):
        idx, _ = built_index
        results = idx.search("earnings", top_k=3)
        for result in results:
            assert "text" in result
            assert "chunk_index" in result
            assert "score" in result


# ---------------------------------------------------------------------------
# Test 3 — scores are within the cosine similarity range [0, 1]
# ---------------------------------------------------------------------------

class TestScoresInValidRange:
    def test_all_scores_between_0_and_1(self, built_index):
        idx, _ = built_index
        results = idx.search("quarterly revenue", top_k=len(SAMPLE_CHUNKS))
        assert results, "Expected at least one result"
        for result in results:
            score = result["score"]
            assert 0.0 <= score <= 1.0, (
                f"Score {score} is outside the valid cosine similarity range [0, 1]"
            )

    def test_scores_are_sorted_descending(self, built_index):
        idx, _ = built_index
        results = idx.search("operating expenses", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results should be sorted by descending score"
        )


# ---------------------------------------------------------------------------
# Test 4 — exact text of a chunk returns that chunk as top result
# ---------------------------------------------------------------------------

class TestExactTextMatchIsTopResult:
    @pytest.mark.parametrize("chunk", SAMPLE_CHUNKS)
    def test_exact_chunk_text_is_top_result(self, built_index, chunk):
        idx, _ = built_index
        results = idx.search(chunk["text"], top_k=1)
        assert results, "Expected at least one result"
        top = results[0]
        assert top["text"] == chunk["text"], (
            f"Expected top result to be the exact queried chunk.\n"
            f"  Query:    {chunk['text']!r}\n"
            f"  Got:      {top['text']!r}"
        )

    def test_exact_chunk_score_is_near_one(self, built_index):
        """Cosine similarity of a vector with itself should be ≈ 1.0."""
        idx, _ = built_index
        chunk = SAMPLE_CHUNKS[0]
        results = idx.search(chunk["text"], top_k=1)
        assert results[0]["score"] >= 0.99, (
            f"Self-similarity score should be close to 1.0, got {results[0]['score']}"
        )


# ---------------------------------------------------------------------------
# Test 5 — load() round-trip preserves search behaviour
# ---------------------------------------------------------------------------

class TestLoadRoundTrip:
    def test_load_and_search(self, built_index, tmp_path):
        _, index_path = built_index

        # Load into a fresh instance
        fresh_idx = FAISSIndex()
        fresh_idx.load(index_path)

        results = fresh_idx.search("cash equivalents", top_k=3)
        assert len(results) == 3

    def test_load_missing_index_raises(self, tmp_path):
        idx = FAISSIndex()
        with pytest.raises(FileNotFoundError):
            idx.load(str(tmp_path / "nonexistent.faiss"))

    def test_search_before_load_raises(self):
        idx = FAISSIndex()
        with pytest.raises(RuntimeError):
            idx.search("anything")
