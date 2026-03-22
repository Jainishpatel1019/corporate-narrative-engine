"""
src/utils/helpers.py — Shared utility functions for SignalEdge.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime


def make_id(*parts: str) -> str:
    """Deterministic UUID-style ID from input parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def new_uuid() -> str:
    return uuid.uuid4().hex[:24]


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def extract_topic(text: str) -> str:
    """Simple keyword-based topic classifier."""
    text_lower = text.lower()
    topics = {
        "Revenue": ["revenue", "sales", "top line", "top-line"],
        "Guidance": ["guidance", "outlook", "forecast", "expect"],
        "Legal": ["legal", "litigation", "lawsuit", "regulatory", "compliance"],
        "Operations": ["operations", "operating", "costs", "expenses", "margin"],
        "Projects": ["project", "initiative", "expansion", "investment", "capex"],
    }
    for topic, keywords in topics.items():
        if any(k in text_lower for k in keywords):
            return topic
    return "General"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truncate_text(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
