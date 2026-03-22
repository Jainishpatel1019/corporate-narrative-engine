"""
tests/test_finbert.py

Unit tests for FinBERTSentiment (src/models/finbert.py).
The HuggingFace model and tokenizer are mocked to avoid network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Fixture: patch out _FinBERTBackend
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_finbert_backend(monkeypatch):
    """
    Mock the backend to prevent weights downloading.

    Mock logic:
    - tokenizer reads the text:
        - if "good" in text -> token 0 (POSITIVE)
        - if "bad" in text -> token 1 (NEGATIVE)
        - otherwise -> token 2 (NEUTRAL)
    - model reads this token and returns logits accordingly.
    """
    import models.finbert as finbert_module
    finbert_module._FinBERTBackend._instance = None

    def _mock_load(self):
        self.tokenizer = MagicMock()

        def _encode(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            seq_len = 8
            input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

            for i, text in enumerate(texts if isinstance(texts, list) else [texts]):
                t = text.lower()
                if "good" in t:
                    input_ids[i, 0] = 0
                elif "bad" in t:
                    input_ids[i, 0] = 1
                else:
                    input_ids[i, 0] = 2

            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            }

        self.tokenizer.side_effect = _encode
        self.tokenizer.__call__ = _encode

        self.model = MagicMock()
        self.model.eval.return_value = self.model
        self.model.config.id2label = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}

        def _model_call(**kwargs):
            input_ids = kwargs["input_ids"]
            batch_size = input_ids.shape[0]
            logits = torch.zeros(batch_size, 3)

            for i in range(batch_size):
                label_idx = input_ids[i, 0].item()
                logits[i, label_idx] = 10.0  # Force softmax ~1.0 for this class

            mock_output = MagicMock()
            mock_output.logits = logits
            return mock_output

        self.model.side_effect = _model_call
        self.model.__call__ = _model_call

        self.labels = ["positive", "negative", "neutral"]

    monkeypatch.setattr("models.finbert._FinBERTBackend._load", _mock_load)

    yield

    finbert_module._FinBERTBackend._instance = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_dict_with_required_keys(self):
        from models.finbert import FinBERTSentiment
        finbert = FinBERTSentiment()
        result = finbert.predict("This is a random text.")
        assert isinstance(result, dict)
        assert "label" in result
        assert "score" in result

    def test_predict_returns_valid_label(self):
        from models.finbert import FinBERTSentiment
        finbert = FinBERTSentiment()
        result = finbert.predict("Earnings look very good this quarter.")
        assert result["label"] in FinBERTSentiment.VALID_LABELS

    def test_predict_score_is_between_0_and_1(self):
        from models.finbert import FinBERTSentiment
        finbert = FinBERTSentiment()
        result = finbert.predict("Things are super bad.")
        assert 0.0 <= result["score"] <= 1.0


class TestSentimentShift:
    def test_sentiment_shift_positive_when_after_is_more_positive(self):
        from models.finbert import FinBERTSentiment
        finbert = FinBERTSentiment()

        before = ["The market is bad.", "Profits are bad."]
        after = ["Things are good now.", "Revenue is good."]

        shift = finbert.sentiment_shift(before, after)
        assert shift > 0.0, f"Expected positive shift, got {shift}"

    def test_sentiment_shift_zero_for_identical_inputs(self):
        from models.finbert import FinBERTSentiment
        finbert = FinBERTSentiment()

        texts = ["Revenue is stable.", "Costs are managed well.", "Not sure what to expect."]

        shift = finbert.sentiment_shift(texts, texts)
        assert shift == 0.0, f"Expected zero shift for identical inputs, got {shift}"

    def test_sentiment_shift_negative_when_after_is_more_negative(self):
        from models.finbert import FinBERTSentiment
        finbert = FinBERTSentiment()

        before = ["Q1 was good."]
        after = ["Q2 is very bad."]

        shift = finbert.sentiment_shift(before, after)
        assert shift < 0.0, f"Expected negative shift, got {shift}"

    def test_sentiment_shift_handles_empty_lists(self):
        from models.finbert import FinBERTSentiment
        finbert = FinBERTSentiment()

        shift = finbert.sentiment_shift([], ["good"])
        assert shift > 0.0

        shift2 = finbert.sentiment_shift(["bad"], [])
        assert shift2 > 0.0
