"""
tests/test_reranker.py

Unit tests for Reranker (src/retrieval/reranker.py).
NLIModel is mocked so no model weights are downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chunk(text: str, chunk_index: int, semantic_score: float) -> dict:
    return {"text": text, "chunk_index": chunk_index, "semantic_score": semantic_score}


SAMPLE_CHUNKS = [
    make_chunk("Revenue grew by 15% year over year.", 0, 0.85),
    make_chunk("Operating costs increased due to headcount.", 1, 0.70),
    make_chunk("Net income fell due to restructuring.", 2, 0.60),
    make_chunk("Cash reserves are at an all-time high.", 3, 0.50),
]


@pytest.fixture
def mock_nli_constant(monkeypatch):
    """Mock NLIModel.predict to always return a constant contradiction score of 0.5."""
    mock = MagicMock()
    mock.predict.return_value = {"label": "neutral", "score": 0.5}
    monkeypatch.setattr("retrieval.reranker.NLIModel", lambda: mock)
    return mock


@pytest.fixture
def mock_nli_controlled(monkeypatch):
    """
    Mock NLIModel.predict to return different scores based on chunk index stored
    in a closure so tests can control per-chunk NLI scores precisely.

    Scores are provided as a list keyed by the order chunks are passed in.
    """
    scores = []
    call_count = [0]

    mock = MagicMock()

    def _predict(query, text):
        idx = call_count[0] % len(scores) if scores else 0
        result = scores[idx] if scores else {"label": "neutral", "score": 0.5}
        call_count[0] += 1
        return result

    mock.predict.side_effect = _predict
    monkeypatch.setattr("retrieval.reranker.NLIModel", lambda: mock)
    return mock, scores, call_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRerankerOutputStructure:
    def test_all_five_keys_present(self, mock_nli_constant):
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.score("revenue performance", SAMPLE_CHUNKS)

        required_keys = {"text", "chunk_index", "semantic_score", "contradiction_score", "combined_score"}
        for result in results:
            assert required_keys == set(result.keys()), (
                f"Expected keys {required_keys}, got {set(result.keys())}"
            )

    def test_output_length_matches_input(self, mock_nli_constant):
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.score("query", SAMPLE_CHUNKS)
        assert len(results) == len(SAMPLE_CHUNKS)

    def test_empty_input_returns_empty_list(self, mock_nli_constant):
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.score("query", [])
        assert results == []


class TestRerankerSorting:
    def test_results_sorted_by_combined_score_descending(self, mock_nli_constant):
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.score("revenue performance", SAMPLE_CHUNKS)

        scores = [r["combined_score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )

    def test_combined_score_formula_is_correct(self, mock_nli_constant):
        """Verify the 0.6/0.4 weighting is applied correctly."""
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.score("query", [make_chunk("text", 0, 0.8)])

        r = results[0]
        expected = 0.6 * r["semantic_score"] + 0.4 * r["contradiction_score"]
        assert abs(r["combined_score"] - expected) < 1e-9, (
            f"combined_score {r['combined_score']} != expected {expected}"
        )


class TestRerankerScoreLogic:
    def test_high_semantic_and_high_nli_beats_high_semantic_only(self, monkeypatch):
        """
        Chunk A: semantic=0.7, nli=0.9 → combined = 0.42 + 0.36 = 0.78
        Chunk B: semantic=0.9, nli=0.1 → combined = 0.54 + 0.04 = 0.58
        Chunk A should rank first despite lower semantic score.
        """
        chunk_a = make_chunk("Earnings improved significantly.", 0, 0.7)
        chunk_b = make_chunk("Revenue at record levels.", 1, 0.9)

        nli_scores = [
            {"label": "entailment", "score": 0.9},   # chunk_a
            {"label": "neutral", "score": 0.1},      # chunk_b
        ]
        call_count = [0]

        mock = MagicMock()
        def _predict(query, text):
            result = nli_scores[call_count[0] % len(nli_scores)]
            call_count[0] += 1
            return result
        mock.predict.side_effect = _predict
        monkeypatch.setattr("retrieval.reranker.NLIModel", lambda: mock)

        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.score("earnings improvement", [chunk_a, chunk_b])

        assert results[0]["chunk_index"] == 0, (
            f"Expected chunk_a (index 0) first, got chunk_index={results[0]['chunk_index']}"
        )
        assert results[0]["combined_score"] > results[1]["combined_score"]

    def test_contradiction_score_is_nli_probability(self, monkeypatch):
        """contradiction_score in output should be the raw NLI probability."""
        mock = MagicMock()
        mock.predict.return_value = {"label": "contradiction", "score": 0.88}
        monkeypatch.setattr("retrieval.reranker.NLIModel", lambda: mock)

        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.score("query", [make_chunk("some text", 0, 0.5)])
        assert results[0]["contradiction_score"] == pytest.approx(0.88)
