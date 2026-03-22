"""
src/models/finbert.py

Provides FinBERTSentiment — a financial sentiment classifier backed by
ProsusAI/finbert from HuggingFace.

The pipeline is loaded once at module level (singleton) and reused across
all FinBERTSentiment instances.

Label mapping:
  positive | negative | neutral
"""

from __future__ import annotations

from typing import ClassVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "ProsusAI/finbert"
_BATCH_SIZE = 32
_FALLBACK_LABELS = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class _FinBERTBackend:
    """Holds the tokenizer and model; loaded exactly once."""

    _instance: ClassVar[_FinBERTBackend | None] = None

    def __new__(cls) -> _FinBERTBackend:
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._load()
            cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self.model.eval()

        # Build ordered label list from model config
        id2label: dict[int, str] = self.model.config.id2label  # type: ignore[assignment]
        self.labels: list[str] = [id2label[i].lower() for i in sorted(id2label)]

    @torch.no_grad()
    def infer(self, texts: list[str]) -> list[dict]:
        """
        Run inference on *texts* (batched) and return a list of dicts.

        Returns
        -------
        list[dict]
            Each dict has keys ``"label"`` and ``"score"``.
        """
        results: list[dict] = []

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start : batch_start + _BATCH_SIZE]

            encoding = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            logits = self.model(**encoding).logits  # (B, num_labels)
            probs = torch.softmax(logits, dim=-1)   # (B, num_labels)

            for prob_row in probs:
                best_idx = int(prob_row.argmax().item())
                label = self.labels[best_idx] if best_idx < len(self.labels) else _FALLBACK_LABELS[best_idx]
                score = float(prob_row[best_idx].item())
                # Clamp to [0, 1] to defend against floating-point drift
                score = max(0.0, min(1.0, score))
                results.append({"label": label, "score": score})

        return results


def _get_backend() -> _FinBERTBackend:
    """Return the cached FinBERT backend, initialising it on the first call."""
    return _FinBERTBackend()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FinBERTSentiment:
    """
    Stateless wrapper around FinBERT for sequence classification and
    sentiment shift calculation.
    """

    VALID_LABELS: ClassVar[frozenset[str]] = frozenset({"positive", "negative", "neutral"})

    def predict(self, text: str) -> dict:
        """
        Predict the financial sentiment of *text*.

        Returns
        -------
        dict
            ``{"label": "positive"|"negative"|"neutral", "score": float}``
        """
        return _get_backend().infer([text])[0]

    def _compute_numeric_score(self, results: list[dict]) -> float:
        """
        Map categorical labels to a numeric scale and return the mean.
        +1.0 for positive, -1.0 for negative, 0.0 for neutral.
        """
        if not results:
            return 0.0

        total = 0.0
        for r in results:
            label = r["label"]
            if label == "positive":
                total += 1.0
            elif label == "negative":
                total -= 1.0

        return total / len(results)

    def sentiment_shift(self, texts_before: list[str], texts_after: list[str]) -> float:
        """
        Compute the mean sentiment shift from *texts_before* to *texts_after*.

        Each text is scored: positive=+1, negative=-1, neutral=0.
        The returned shift is `mean(scores_after) - mean(scores_before)`.

        Returns
        -------
        float
            Positive value indicates sentiment became more positive.
            Negative value indicates sentiment became more negative.
        """
        backend = _get_backend()

        res_before = backend.infer(texts_before) if texts_before else []
        res_after = backend.infer(texts_after) if texts_after else []

        score_before = self._compute_numeric_score(res_before)
        score_after = self._compute_numeric_score(res_after)

        return score_after - score_before
