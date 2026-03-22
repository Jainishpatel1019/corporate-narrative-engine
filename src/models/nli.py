"""
src/models/nli.py — DeBERTa NLI model for contradiction detection.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_BATCH_SIZE = 32


class _NLIBackend:
    _instance: ClassVar[_NLIBackend | None] = None

    def __new__(cls) -> _NLIBackend:
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
    def infer(self, pairs: list[tuple[str, str]]) -> list[dict]:
        results: list[dict] = []
        for start in range(0, len(pairs), _BATCH_SIZE):
            batch = pairs[start : start + _BATCH_SIZE]
            enc = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            for row in probs:
                idx = int(row.argmax().item())
                results.append({"label": self.labels[idx], "score": max(0.0, min(1.0, float(row[idx].item())))})
        return results


def _get_backend() -> _NLIBackend:
    return _NLIBackend()


class NLIModel:
    VALID_LABELS = frozenset({"contradiction", "entailment", "neutral"})

    def predict(self, text_a: str, text_b: str) -> dict:
        return _get_backend().infer([(text_a, text_b)])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        if not pairs:
            return []
        return _get_backend().infer(pairs)
