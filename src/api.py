"""
src/api.py

FastAPI service exposing corporate narrative engine capabilities.
Provides /detect for NLI pairs, /search for semantic retrieval, and /health.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure src/ is on the path so we can import internal modules standalone
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.nli import NLIModel  # noqa: E402
from retrieval.faiss_index import FAISSIndex  # noqa: E402

app = FastAPI(title="Corporate Narrative Engine API")

# Global instances (lazy-loading backends underneath)
nli_model = NLIModel()
faiss_index = FAISSIndex()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    text_a: str
    text_b: str


class DetectResponse(BaseModel):
    label: str
    score: float
    latency_ms: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    results: list[dict]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe. Reports whether the FAISS index is loaded."""
    t0 = time.perf_counter()
    is_loaded = faiss_index._index is not None
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "ok",
        "model_loaded": is_loaded,
        "latency_ms": latency_ms,
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    """Run cross-encoder NLI on a premise and hypothesis."""
    t0 = time.perf_counter()
    result = nli_model.predict(req.text_a, req.text_b)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "label": result["label"],
        "score": result["score"],
        "latency_ms": latency_ms,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Retrieve top-k semantic matches from the FAISS index."""
    t0 = time.perf_counter()

    if faiss_index._index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded yet. Call load() via admin interface or script.",
        )

    try:
        results = faiss_index.search(req.query, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "results": results,
        "latency_ms": latency_ms,
    }
