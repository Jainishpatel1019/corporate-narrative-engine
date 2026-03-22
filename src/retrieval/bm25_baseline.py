"""
src/retrieval/bm25_baseline.py

Provides BM25Retriever — a baseline retrieval class using the rank_bm25 library.
Tokenizes text using simple whitespace split and lowercasing.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Build and query a BM25 index over a set of text chunks.

    Typical usage
    -------------
    >>> retriever = BM25Retriever()
    >>> retriever.build(chunks)
    >>> results = retriever.search("revenue growth", top_k=3)
    """

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase and split on whitespace."""
        return text.lower().split()

    def build(self, chunks: list[dict]) -> None:
        """
        Index *chunks* using BM25Okapi.

        Parameters
        ----------
        chunks:
            List of dicts. Each dict must contain at least a "text" key.
            A "chunk_index" key is used if present; otherwise the list
            position is used.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index with an empty corpus.")

        self.chunks = chunks
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the BM25 index for the *top_k* chunks most relevant to *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Each element contains keys "text", "chunk_index", and "score"
            (BM25 Okapi score).
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not built. Call build() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[dict] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "chunk_index": chunk.get("chunk_index", idx),
                    "score": float(scores[idx]),
                }
            )

        return results
