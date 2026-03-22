"""
src/retrieval/reranker.py

Provides Reranker — combines semantic similarity scores from FAISS with
NLI-based contradiction scores to produce a final combined ranking.

combined_score = 0.6 * semantic_score + 0.4 * contradiction_score

The contradiction_score here is used as an *entailment/relevance* signal:
it is the softmax probability of the predicted NLI label, regardless of
whether the label is entailment, neutral, or contradiction.  This keeps
the signal numerically meaningful — a high-confidence NLI prediction
(in any direction) indicates the chunk is topically related to the query.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.nli import NLIModel  # noqa: E402

# Weighting constants
_SEMANTIC_WEIGHT = 0.6
_CONTRADICTION_WEIGHT = 0.4


class Reranker:
    """
    Reranks retrieved chunks using a linear combination of FAISS semantic
    scores and NLI confidence scores.

    Usage
    -----
    >>> reranker = Reranker()
    >>> ranked = reranker.score(query, chunks)
    """

    def __init__(self) -> None:
        self._nli = NLIModel()

    def score(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Rerank *chunks* by combining semantic and NLI scores.

        Parameters
        ----------
        query:
            The search query used to retrieve the chunks.
        chunks:
            List of dicts, each containing at minimum:
              - ``"text"``          : chunk text
              - ``"chunk_index"``   : original position in the corpus
              - ``"semantic_score"``: cosine similarity score from FAISS (0–1)

        Returns
        -------
        list[dict]
            Same chunks augmented with ``"contradiction_score"`` and
            ``"combined_score"``, sorted by ``combined_score`` descending.
            Each result has exactly the keys:
              text, chunk_index, semantic_score, contradiction_score, combined_score
        """
        if not chunks:
            return []

        results: list[dict] = []
        for chunk in chunks:
            nli_result = self._nli.predict(query, chunk["text"])
            contradiction_score = nli_result["score"]  # probability of predicted label

            semantic_score = float(chunk["semantic_score"])
            combined_score = (
                _SEMANTIC_WEIGHT * semantic_score
                + _CONTRADICTION_WEIGHT * contradiction_score
            )

            results.append(
                {
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "semantic_score": semantic_score,
                    "contradiction_score": float(contradiction_score),
                    "combined_score": combined_score,
                }
            )

        results.sort(key=lambda r: r["combined_score"], reverse=True)
        return results
