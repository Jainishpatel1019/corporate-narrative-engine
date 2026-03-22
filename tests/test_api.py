"""
tests/test_api.py

Unit tests for the FastAPI service (src/api.py).
We mock the internal methods of NLIModel and FAISSIndex to avoid model loading.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import api to access globals and app
import api  # noqa: E402

client = TestClient(api.app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_backends(monkeypatch):
    """
    Patch the NLI predict and FAISS search methods directly on the global
    instances in api.py, avoiding any real model loading.
    """
    # Mock NLI
    mock_predict = MagicMock(return_value={"label": "entailment", "score": 0.95})
    monkeypatch.setattr(api.nli_model, "predict", mock_predict)

    # Mock FAISS (defaults to unloaded state like the real class)
    api.faiss_index._index = None
    mock_search = MagicMock(return_value=[{"text": "Sample", "chunk_index": 0, "score": 0.99}])
    monkeypatch.setattr(api.faiss_index, "search", mock_search)

    yield mock_predict, mock_search

    # Reset state
    api.faiss_index._index = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200_and_keys(self):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False  # default state
        assert isinstance(data["latency_ms"], float)


class TestDetectEndpoint:
    def test_detect_returns_correct_response_schema(self, mock_backends):
        mock_pred, _ = mock_backends

        response = client.post("/detect", json={"text_a": "A", "text_b": "B"})
        assert response.status_code == 200

        data = response.json()
        assert data["label"] == "entailment"
        assert data["score"] == 0.95
        assert isinstance(data["latency_ms"], float)

        mock_pred.assert_called_once_with("A", "B")


class TestSearchEndpoint:
    def test_search_returns_503_when_index_not_loaded(self):
        # By default in the fixture, api.faiss_index._index is None
        response = client.post("/search", json={"query": "test query"})
        assert response.status_code == 503
        
        data = response.json()
        assert "Index not loaded yet" in data["detail"]

    def test_search_returns_top_k_results(self, mock_backends):
        _, mock_search = mock_backends

        # Simulate index being loaded
        api.faiss_index._index = MagicMock()

        # Customise the mock return for this test
        expected_results = [
            {"text": "A", "chunk_index": 0, "score": 0.9},
            {"text": "B", "chunk_index": 1, "score": 0.8},
            {"text": "C", "chunk_index": 2, "score": 0.7},
        ]
        mock_search.return_value = expected_results

        response = client.post("/search", json={"query": "growth", "top_k": 3})
        assert response.status_code == 200

        data = response.json()
        assert len(data["results"]) == 3
        assert data["results"] == expected_results
        assert isinstance(data["latency_ms"], float)

        mock_search.assert_called_once_with("growth", top_k=3)
