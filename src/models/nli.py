"""
src/models/nli.py

Provides NLIModel — a Natural Language Inference classifier backed by
cross-encoder/nli-deberta-v3-base from HuggingFace.

The pipeline is loaded once at module level (singleton) and reused across all
NLIModel instances, mirroring the @st.cache_resource pattern used in the
Streamlit layer.

Label mapping follows the standard three-class NLI convention:
  contradiction | entailment | neutral
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_BATCH_SIZE = 32

# The DeBERTa NLI fine-tune uses this label order (id2label).
# We resolve dynamically from model config, but keep a fallback.
_FALLBACK_LABELS = ["contradiction", "entailment", "neutral"]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class _NLIBackend:
    """Holds the tokenizer and model; loaded exactly once."""

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

        # Build ordered label list from model config (id → label, sorted by id)
        id2label: dict[int, str] = self.model.config.id2label  # type: ignore[assignment]
        self.labels: list[str] = [id2label[i].lower() for i in sorted(id2label)]

    @torch.no_grad()
    def infer(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Run inference on *pairs* (batched) and return per-pair result dicts.

        Returns
        -------
        list[dict]
            Each dict has keys ``"label"`` and ``"score"``.
        """
        results: list[dict] = []

        for batch_start in range(0, len(pairs), _BATCH_SIZE):
            batch = pairs[batch_start : batch_start + _BATCH_SIZE]
            texts_a = [p[0] for p in batch]
            texts_b = [p[1] for p in batch]

            encoding = self.tokenizer(
                texts_a,
                texts_b,
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
                # Clamp to [0, 1] to defend against floating-point edge cases
                score = max(0.0, min(1.0, score))
                results.append({"label": label, "score": score})

        return results


def _get_backend() -> _NLIBackend:
    """Return the cached NLI backend, initialising it on the first call."""
    return _NLIBackend()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class NLIModel:
    """
    Thin, stateless wrapper around the NLI backend.

    Examples
    --------
    >>> nli = NLIModel()
    >>> nli.predict("Revenue grew 10%.", "The company lost money.")
    {'label': 'contradiction', 'score': 0.97}

    >>> nli.predict_batch([("A", "B"), ("C", "D")])
    [{'label': ..., 'score': ...}, {'label': ..., 'score': ...}]
    """

    # Valid label set exposed for downstream assertions / tests
    VALID_LABELS: ClassVar[frozenset[str]] = frozenset(
        {"contradiction", "entailment", "neutral"}
    )

    def predict(self, text_a: str, text_b: str) -> dict:
        """
        Predict the NLI relationship between *text_a* (premise) and
        *text_b* (hypothesis).

        Parameters
        ----------
        text_a:
            The premise text (e.g. a sentence from an SEC filing).
        text_b:
            The hypothesis text (e.g. a claim to verify).

        Returns
        -------
        dict
            ``{"label": "contradiction"|"entailment"|"neutral", "score": float}``
            where ``score`` is the softmax probability of the predicted label.
        """
        return _get_backend().infer([(text_a, text_b)])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Predict NLI labels for multiple *pairs* in batches of 32.

        Parameters
        ----------
        pairs:
            A list of (premise, hypothesis) string tuples.

        Returns
        -------
        list[dict]
            One result dict per pair, same format as :meth:`predict`.
        """
        if not pairs:
            return []
        return _get_backend().infer(pairs)
