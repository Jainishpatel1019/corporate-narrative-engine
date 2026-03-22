"""
src/ingestion/chunker.py — Sliding window text chunker using tiktoken.
"""

from __future__ import annotations

import tiktoken

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


class SlidingWindowChunker:
    """Split text into overlapping token-based chunks."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[dict]:
        """Return a list of chunk dicts with 'text' and 'chunk_index' keys."""
        if not text or not text.strip():
            return []

        tokens = self.enc.encode(text)
        chunks: list[dict] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0

        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            chunk_text = self.enc.decode(window)
            chunks.append({"text": chunk_text, "chunk_index": idx})
            idx += 1
            if start + self.chunk_size >= len(tokens):
                break

        return chunks
