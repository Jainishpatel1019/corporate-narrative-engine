"""
src/models/reranker.py — Combined semantic + NLI reranking.
"""

from __future__ import annotations

from src.models.nli import NLIModel

_SEMANTIC_W = 0.6
_NLI_W = 0.4


class Reranker:
    def __init__(self) -> None:
        self._nli = NLIModel()

    def score(self, query: str, chunks: list[dict]) -> list[dict]:
        """Rerank *chunks* by combined semantic + NLI score."""
        if not chunks:
            return []

        results = []
        for c in chunks:
            nli = self._nli.predict(query, c["text"])
            sem = float(c.get("semantic_score", c.get("score", 0.0)))
            nli_score = nli["score"]
            combined = _SEMANTIC_W * sem + _NLI_W * nli_score
            results.append({
                "text": c["text"],
                "chunk_index": c.get("chunk_index", 0),
                "semantic_score": sem,
                "contradiction_score": nli_score,
                "combined_score": combined,
            })
        results.sort(key=lambda r: r["combined_score"], reverse=True)
        return results
