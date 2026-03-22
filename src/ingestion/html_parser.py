"""
src/ingestion/html_parser.py — Extract relevant sections from SEC filing HTML.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


class SECHTMLParser:
    """Extract MD&A, Risk Factors, and other key sections from SEC HTML."""

    SECTION_PATTERNS = {
        "mda": [
            r"management.s discussion and analysis",
            r"item\s*7[\.\s]",
        ],
        "risk_factors": [
            r"risk factors",
            r"item\s*1a[\.\s]",
        ],
        "financial_statements": [
            r"financial statements",
            r"item\s*8[\.\s]",
        ],
    }

    def parse(self, html: str) -> dict[str, str]:
        """
        Return a dict of {section_name: plain_text} extracted from *html*.
        Falls back to the full stripped text if no sections are found.
        """
        if not html or not html.strip():
            return {}

        soup = BeautifulSoup(html, "html.parser")
        full_text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        full_text = re.sub(r"\s+", " ", full_text).strip()

        if len(full_text) < 100:
            return {"full_text": full_text} if full_text else {}

        sections: dict[str, str] = {}
        text_lower = full_text.lower()

        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    start = match.start()
                    # Grab up to 10,000 chars from section start
                    snippet = full_text[start : start + 10000]
                    sections[section_name] = snippet
                    break

        if not sections:
            # No recognisable sections — return truncated full text
            sections["full_text"] = full_text[:15000]

        return sections
