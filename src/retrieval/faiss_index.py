"""
src/retrieval/faiss_index.py

Provides FAISSIndex — a thin wrapper around faiss.IndexFlatIP that embeds
text chunks with sentence-transformers, normalises vectors for cosine
similarity, and persists both the index and chunk metadata to disk.

The SentenceTransformer model is loaded once at module import time (singleton)
so the expensive model-load cost is paid only once, mirroring the behaviour
of @st.cache_resource used in the Streamlit layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level singleton — load once, reuse across all FAISSIndex instances.
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer model, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> np.ndarray:
    """Embed *texts* and return an (N, D) float32 array."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise *vectors* in-place and return them (for cosine similarity)."""
    faiss.normalize_L2(vectors)
    return vectors


def _metadata_path(index_path: str) -> str:
    """Derive the companion JSON path from *index_path*."""
    return str(Path(index_path).with_suffix(".json"))


# ---------------------------------------------------------------------------
# FAISSIndex
# ---------------------------------------------------------------------------

class FAISSIndex:
    """
    Build, persist, load, and query a FAISS inner-product index that stores
    L2-normalised sentence embeddings (equivalent to cosine similarity).

    Typical usage
    -------------
    >>> idx = FAISSIndex()
    >>> idx.build(chunks, index_path="data/processed/sec_index.faiss")
    >>> idx.load("data/processed/sec_index.faiss")
    >>> results = idx.search("revenue decline", top_k=3)
    """

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict], index_path: str) -> None:
        """
        Embed and index *chunks*, then write the FAISS index and metadata.

        Parameters
        ----------
        chunks:
            List of dicts produced by SlidingWindowChunker.  Each dict must
            contain at least a ``"text"`` key.  A ``"chunk_index"`` key is
            used if present; otherwise the list position is used.
        index_path:
            File path for the serialised FAISS index (e.g. ``"index.faiss"``).
            The companion metadata file is written to the same directory with
            the same stem and a ``.json`` extension.
        """
        if not chunks:
            raise ValueError("chunks must be a non-empty list")

        texts = [c["text"] for c in chunks]
        embeddings = _embed(texts)
        _normalize(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)  # type: ignore[arg-type]

        # Persist index
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        faiss.write_index(index, index_path)

        # Persist metadata
        metadata = [
            {
                "text": c["text"],
                "chunk_index": c.get("chunk_index", i),
            }
            for i, c in enumerate(chunks)
        ]
        with open(_metadata_path(index_path), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

        # Keep in-memory state consistent so the caller can search immediately
        self._index = index
        self._metadata = metadata

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, index_path: str) -> None:
        """
        Load a previously built index and its metadata from *index_path*.

        Parameters
        ----------
        index_path:
            Path to the ``.faiss`` file created by :meth:`build`.
        """
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        meta_path = _metadata_path(index_path)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the index for the *top_k* chunks most similar to *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Each element contains keys ``"text"``, ``"chunk_index"``, and
            ``"score"`` (cosine similarity in [0, 1]).
        """
        if self._index is None:
            raise RuntimeError("Index is not loaded. Call build() or load() first.")

        query_vec = _embed([query])  # shape (1, D)
        _normalize(query_vec)

        effective_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, effective_k)  # type: ignore[arg-type]

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "text": meta["text"],
                    "chunk_index": meta["chunk_index"],
                    "score": float(score),
                }
            )
        return results
