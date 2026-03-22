"""
src/ingestion/fmp_client.py — Financial Modeling Prep API client.
"""

from __future__ import annotations

import requests

from src.config import FMP_API_KEY, FMP_BASE_URL


class FMPClient:
    """Fetch earnings transcripts and company profiles from FMP."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._call_count = 0

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        self._call_count += 1
        if params is None:
            params = {}
        params["apikey"] = FMP_API_KEY
        url = f"{FMP_BASE_URL}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            return None

    def get_transcripts(self, ticker: str, limit: int = 4) -> list[dict]:
        """Return the last *limit* earnings call transcripts."""
        data = self._get(f"earning_call_transcript/{ticker.upper()}")
        if not data or not isinstance(data, list):
            return []
        results = []
        for item in data[:limit]:
            results.append(
                {
                    "ticker": ticker.upper(),
                    "quarter": item.get("quarter"),
                    "year": item.get("year"),
                    "date": item.get("date"),
                    "content": item.get("content", ""),
                }
            )
        return results

    def get_company_profile(self, ticker: str) -> dict | None:
        """Return basic company info: {name, sector, description}."""
        data = self._get(f"profile/{ticker.upper()}")
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        p = data[0]
        return {
            "name": p.get("companyName", ticker.upper()),
            "sector": p.get("sector", ""),
            "description": p.get("description", ""),
        }
