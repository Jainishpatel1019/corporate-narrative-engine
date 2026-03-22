"""
tests/test_nli.py

Unit tests for NLIModel (src/models/nli.py).

The HuggingFace model and tokenizer are mocked so these tests run instantly
in CI without downloading any weights.

Run with:
    pytest tests/test_nli.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Helpers to build a realistic mock backend
# ---------------------------------------------------------------------------

VALID_LABELS = {"contradiction", "entailment", "neutral"}
_LABEL_LIST = ["contradiction", "entailment", "neutral"]


def _make_mock_logits(label: str, high_prob: float = 0.95) -> torch.Tensor:
    """Return a (1, 3) logit tensor biased toward *label*."""
    low = (1.0 - high_prob) / (len(_LABEL_LIST) - 1)
    probs = [low] * len(_LABEL_LIST)
    probs[_LABEL_LIST.index(label)] = high_prob
    # Convert probabilities → logits via log (softmax inverse approximation)
    logits = [p for p in probs]
    return torch.tensor([logits])


def _make_mock_model(label: str = "contradiction", high_prob: float = 0.95):
    """Return a mock AutoModelForSequenceClassification-like object."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    # id2label mirrors what DeBERTa NLI config returns
    mock_model.config.id2label = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}

    # Make model(**encoding) return an object with .logits
    mock_output = MagicMock()
    mock_output.logits = _make_mock_logits(label, high_prob)

    # Support batches: repeat logits along batch dim dynamically
    def _call(**kwargs):
        input_ids = kwargs.get("input_ids", torch.zeros(1, 1))
        batch_size = input_ids.shape[0]
        mock_output.logits = _make_mock_logits(label, high_prob).expand(batch_size, -1)
        return mock_output

    mock_model.side_effect = _call
    mock_model.__call__ = _call
    return mock_model


def _make_mock_tokenizer():
    """Return a mock tokenizer."""
    tok = MagicMock()

    def _encode(texts_a, texts_b, **kwargs):
        batch_size = len(texts_a) if isinstance(texts_a, list) else 1
        # Return a dict-like object with .items() so **encoding works
        encoding = {
            "input_ids": torch.zeros(batch_size, 8, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
        }
        return encoding

    tok.side_effect = _encode
    tok.__call__ = _encode
    return tok


# ---------------------------------------------------------------------------
# Fixture: patch out _NLIBackend so no weights are downloaded
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_nli_backend(monkeypatch):
    """
    Replace _NLIBackend._load with a lightweight mock so that:
    - No HuggingFace downloads happen.
    - The tokenizer and model behave realistically for our tests.
    """
    # Clear any cached singleton before each test
    import models.nli as nli_module
    nli_module._NLIBackend._instance = None

    def _mock_load(self):
        self.tokenizer = _make_mock_tokenizer()
        self.model = _make_mock_model("contradiction")
        self.model.eval()
        self.labels = _LABEL_LIST

    monkeypatch.setattr("models.nli._NLIBackend._load", _mock_load)

    yield

    # Reset singleton after test
    nli_module._NLIBackend._instance = None


# ---------------------------------------------------------------------------
# Tests: predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_dict_with_required_keys(self):
        from models.nli import NLIModel
        nli = NLIModel()
        result = nli.predict("Revenue grew 10%.", "The company lost money.")
        assert isinstance(result, dict)
        assert "label" in result, "predict() result must contain 'label'"
        assert "score" in result, "predict() result must contain 'score'"

    def test_score_is_float_between_0_and_1(self):
        from models.nli import NLIModel
        nli = NLIModel()
        result = nli.predict("Revenue grew 10%.", "The company lost money.")
        score = result["score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score {score} is outside [0, 1]"

    def test_label_is_valid(self):
        from models.nli import NLIModel
        nli = NLIModel()
        result = nli.predict("Revenue grew 10%.", "The company lost money.")
        assert result["label"] in VALID_LABELS, (
            f"Label '{result['label']}' is not one of {VALID_LABELS}"
        )


# ---------------------------------------------------------------------------
# Tests: predict_batch()
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_list_of_dicts(self):
        from models.nli import NLIModel
        nli = NLIModel()
        pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        results = nli.predict_batch(pairs)
        assert isinstance(results, list)
        assert len(results) == len(pairs)

    def test_each_result_has_required_keys(self):
        from models.nli import NLIModel
        nli = NLIModel()
        pairs = [("premise one", "hypothesis one"), ("premise two", "hypothesis two")]
        for result in nli.predict_batch(pairs):
            assert "label" in result
            assert "score" in result

    def test_each_score_in_valid_range(self):
        from models.nli import NLIModel
        nli = NLIModel()
        pairs = [("A", "B"), ("C", "D")]
        for result in nli.predict_batch(pairs):
            assert 0.0 <= result["score"] <= 1.0

    def test_each_label_is_valid(self):
        from models.nli import NLIModel
        nli = NLIModel()
        pairs = [("X", "Y"), ("P", "Q")]
        for result in nli.predict_batch(pairs):
            assert result["label"] in VALID_LABELS

    def test_empty_pairs_returns_empty_list(self):
        from models.nli import NLIModel
        nli = NLIModel()
        assert nli.predict_batch([]) == []

    def test_batch_length_matches_input(self):
        from models.nli import NLIModel
        nli = NLIModel()
        pairs = [(f"premise {i}", f"hyp {i}") for i in range(10)]
        results = nli.predict_batch(pairs)
        assert len(results) == 10


# ---------------------------------------------------------------------------
# Tests: singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_model_loaded_once(self):
        """Two NLIModel instances must share the same backend object."""
        from models.nli import NLIModel, _get_backend
        nli1 = NLIModel()
        nli2 = NLIModel()
        # Trigger backend creation via predict
        nli1.predict("A", "B")
        nli2.predict("C", "D")
        assert _get_backend() is _get_backend(), "Backend singleton should be the same object"
