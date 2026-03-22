"""
src/models/finbert.py — FinBERT sentiment analysis for financial text.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_NAME = "ProsusAI/finbert"
_BATCH_SIZE = 32


class _FinBERTBackend:
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
        id2label = self.model.config.id2label
        self.labels = [id2label[i].lower() for i in sorted(id2label)]

    @torch.no_grad()
    def infer(self, texts: list[str]) -> list[dict]:
        results: list[dict] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            for row in probs:
                idx = int(row.argmax().item())
                results.append({"label": self.labels[idx], "score": max(0.0, min(1.0, float(row[idx].item())))})
        return results


def _get_backend() -> _FinBERTBackend:
    return _FinBERTBackend()


class FinBERTSentiment:
    VALID_LABELS = frozenset({"positive", "negative", "neutral"})

    def predict(self, text: str) -> dict:
        return _get_backend().infer([text])[0]

    def sentiment_shift(self, texts_before: list[str], texts_after: list[str]) -> float:
        def score(results):
            if not results:
                return 0.0
            total = sum(1.0 if r["label"] == "positive" else (-1.0 if r["label"] == "negative" else 0.0) for r in results)
            return total / len(results)

        before = _get_backend().infer(texts_before) if texts_before else []
        after = _get_backend().infer(texts_after) if texts_after else []
        return score(after) - score(before)
