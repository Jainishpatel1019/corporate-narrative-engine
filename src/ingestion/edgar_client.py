"""
src/ingestion/edgar_client.py — SEC EDGAR API client for SignalEdge.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from src.config import EDGAR_BASE_URL, SEC_RATE_LIMIT, SEC_USER_AGENT


class EDGARClient:
    """Fetch SEC filings metadata and documents from EDGAR."""

    HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
        self._last_request_time: float = 0.0

    # ── Rate limiting ─────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / SEC_RATE_LIMIT
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, retries: int = 3) -> requests.Response:
        for attempt in range(retries):
            self._rate_limit()
            try:
                resp = self._session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (429, 500, 503):
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except requests.exceptions.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return resp  # type: ignore[possibly-undefined]

    # ── CIK lookup ────────────────────────────────────────────────────────

    def get_cik(self, ticker: str) -> str | None:
        """Resolve *ticker* to a zero-padded 10-digit CIK string."""
        try:
            resp = self._get(self.COMPANY_TICKERS_URL)
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        except Exception:
            pass
        return None

    # ── Filing list ───────────────────────────────────────────────────────

    def get_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict]:
        """Return recent filings for *ticker*, filtered to *form_types*."""
        if form_types is None:
            form_types = ["10-K", "10-Q", "8-K"]

        cik = self.get_cik(ticker)
        if not cik:
            return []

        url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
        try:
            resp = self._get(url)
            data = resp.json()
        except Exception:
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        doc_urls = recent.get("primaryDocument", [])

        results: list[dict] = []
        for i, form in enumerate(forms):
            if form in form_types:
                acc = accessions[i].replace("-", "")
                results.append(
                    {
                        "accession_number": accessions[i],
                        "form_type": form,
                        "filed_date": dates[i] if i < len(dates) else None,
                        "url": f"{EDGAR_BASE_URL}/Archives/edgar/data/{cik}/{acc}/{doc_urls[i]}"
                        if i < len(doc_urls)
                        else None,
                        "cik": cik,
                    }
                )
                if len(results) >= limit:
                    break
        return results

    # ── Download filing text ──────────────────────────────────────────────

    def download_filing_text(self, url: str) -> str:
        """Download the raw HTML/text of a filing given its URL."""
        try:
            resp = self._get(url)
            return resp.text
        except Exception:
            return ""
