"""
src/models/embedder.py — Sentence-BERT embeddings + FAISS index for SignalEdge.
"""

from __future__ import annotations

import json
import os
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed *texts* → (N, D) float32 array."""
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype(np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vectors)
    return vectors


class Embedder:
    """Build and query a FAISS inner-product index over text chunks."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, Any]] = []

    def build(self, chunks: list[dict], index_path: str | None = None) -> np.ndarray:
        """Embed chunks, build FAISS index. Optionally save to disk. Returns embeddings."""
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        normalize(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._metadata = [
            {"text": c["text"], "chunk_index": c.get("chunk_index", i)}
            for i, c in enumerate(chunks)
        ]

        if index_path:
            os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
            faiss.write_index(self._index, index_path)
            meta_path = index_path.rsplit(".", 1)[0] + ".json"
            with open(meta_path, "w") as f:
                json.dump(self._metadata, f)

        return embeddings

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")

        q = embed_texts([query])
        normalize(q)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            m = self._metadata[idx]
            results.append({"text": m["text"], "chunk_index": m["chunk_index"], "score": float(score)})
        return results
