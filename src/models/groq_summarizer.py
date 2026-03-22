"""
src/models/groq_summarizer.py — LLM-based contradiction summarization via Groq.
"""

from __future__ import annotations

from src.config import GROQ_API_KEY, GROQ_MODEL

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment, misc]


class GroqSummarizer:
    """Generate 2-sentence analyst summaries for contradiction pairs."""

    SYSTEM_PROMPT = (
        "You are a financial analyst specializing in corporate narrative analysis. "
        "Be concise and specific."
    )

    def __init__(self) -> None:
        if Groq is not None:
            self._client = Groq(api_key=GROQ_API_KEY)
        else:
            self._client = None

    def summarize(
        self,
        claim_a: str,
        claim_b: str,
        ticker: str,
        doc_a_type: str = "filing",
        doc_b_type: str = "transcript",
    ) -> str:
        """Return a 2-sentence summary of the contradiction."""
        if self._client is None:
            return self._fallback(ticker, doc_a_type, doc_b_type)

        user_prompt = (
            f"Company: {ticker}. Earlier statement ({doc_a_type}): '{claim_a}'. "
            f"Later statement ({doc_b_type}): '{claim_b}'. In exactly 2 sentences: "
            f"(1) explain the contradiction and (2) state the likely market implication."
        )

        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self._fallback(ticker, doc_a_type, doc_b_type)

    @staticmethod
    def _fallback(ticker: str, doc_a_type: str, doc_b_type: str) -> str:
        return (
            f"A narrative inconsistency was detected for {ticker} between "
            f"the {doc_a_type} and {doc_b_type}. Further analysis is recommended "
            f"to assess potential market impact."
        )
