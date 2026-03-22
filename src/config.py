"""
src/config.py — Central configuration for SignalEdge.
"""

import os

# ── API Keys ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

# ── LLM ───────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Tickers ───────────────────────────────────────────────────────────────
PINNED_TICKERS = ["AAPL", "TSLA", "GOOGL", "NVDA", "AMZN"]

# ── Database ──────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("SIGNALEDGE_DB", "signaledge.db")

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 150
CHUNK_OVERLAP = 25

# ── NLI / Contradiction Thresholds ────────────────────────────────────────
NLI_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.7

# ── SEC EDGAR ─────────────────────────────────────────────────────────────
SEC_USER_AGENT = "Jainish Patel jainishpatel153@gmail.com"
SEC_RATE_LIMIT = 10  # requests per second
EDGAR_BASE_URL = "https://data.sec.gov"

# ── FMP ───────────────────────────────────────────────────────────────────
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
