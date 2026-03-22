"""
src/pipeline/processor.py — End-to-end company analysis pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from src.config import CONTRADICTION_THRESHOLD, NLI_THRESHOLD
from src.database import (
    get_db,
    init_db,
    insert_claim,
    insert_contradiction,
    insert_document,
    insert_signal,
    upsert_company,
)
from src.ingestion.chunker import SlidingWindowChunker
from src.ingestion.edgar_client import EDGARClient
from src.ingestion.fmp_client import FMPClient
from src.ingestion.html_parser import SECHTMLParser
from src.utils.helpers import extract_topic, make_id, new_uuid, now_iso


class CompanyProcessor:
    """Run the full SignalEdge pipeline for a single company."""

    def __init__(self) -> None:
        self.edgar = EDGARClient()
        self.fmp = FMPClient()
        self.parser = SECHTMLParser()
        self.chunker = SlidingWindowChunker()

    def _cb(self, callback: Callable | None, step: int, msg: str) -> None:
        if callback:
            callback(step, msg)

    def process(self, ticker: str, progress_callback: Callable | None = None) -> dict:
        """
        Run the 10-step pipeline for *ticker*.
        Returns a summary dict with counts.
        """
        init_db()
        ticker = ticker.upper()
        summary = {"ticker": ticker, "documents": 0, "claims": 0, "contradictions": 0, "signals": 0}

        # ── STEP 1: Company profile ───────────────────────────────────
        self._cb(progress_callback, 1, f"Fetching company profile for {ticker}...")
        profile = self.fmp.get_company_profile(ticker)
        name = profile["name"] if profile else ticker
        sector = profile.get("sector", "") if profile else ""

        with get_db() as conn:
            upsert_company(conn, ticker=ticker, name=name, cik="",
                           sector=sector, status="processing",
                           last_processed=None, is_pinned=ticker in __import__("src.config", fromlist=["PINNED_TICKERS"]).PINNED_TICKERS)

        # ── STEP 2: SEC filings ───────────────────────────────────────
        self._cb(progress_callback, 2, f"Fetching SEC filings for {ticker}...")
        filings = self.edgar.get_filings(ticker, limit=8)

        filing_docs = []
        for f in filings:
            doc_id = make_id(ticker, f["form_type"], f.get("filed_date", ""))
            raw = ""
            if f.get("url"):
                raw = self.edgar.download_filing_text(f["url"])
            with get_db() as conn:
                insert_document(conn, id=doc_id, ticker=ticker,
                                doc_type=f["form_type"],
                                period=f.get("filed_date", "")[:7],
                                filed_date=f.get("filed_date"),
                                source_url=f.get("url", ""),
                                raw_text=raw[:50000], processed=False)
            filing_docs.append({"id": doc_id, "type": f["form_type"],
                                "date": f.get("filed_date", ""), "text": raw})
            summary["documents"] += 1

        # ── STEP 3: Earnings transcripts ──────────────────────────────
        self._cb(progress_callback, 3, f"Fetching earnings transcripts for {ticker}...")
        transcripts = self.fmp.get_transcripts(ticker, limit=4)

        transcript_docs = []
        for t in transcripts:
            doc_id = make_id(ticker, "transcript", str(t.get("date", "")))
            with get_db() as conn:
                insert_document(conn, id=doc_id, ticker=ticker,
                                doc_type="transcript",
                                period=f"Q{t.get('quarter', '?')} {t.get('year', '?')}",
                                filed_date=t.get("date"),
                                source_url="", raw_text=t.get("content", "")[:50000],
                                processed=False)
            transcript_docs.append({"id": doc_id, "type": "transcript",
                                    "date": t.get("date", ""), "text": t.get("content", "")})
            summary["documents"] += 1

        # ── STEP 4: Parse HTML sections ───────────────────────────────
        self._cb(progress_callback, 4, "Parsing SEC filing sections...")
        for doc in filing_docs:
            sections = self.parser.parse(doc["text"])
            doc["sections"] = sections

        # ── STEP 5: Chunk all texts ───────────────────────────────────
        self._cb(progress_callback, 5, "Chunking documents...")
        filing_chunks = []
        for doc in filing_docs:
            sections = doc.get("sections", {"full_text": doc["text"]})
            for section_name, text in sections.items():
                chunks = self.chunker.chunk(text)
                for c in chunks:
                    c["doc_id"] = doc["id"]
                    c["doc_type"] = doc["type"]
                    c["doc_date"] = doc["date"]
                    c["section"] = section_name
                filing_chunks.append(c)

        transcript_chunks = []
        for doc in transcript_docs:
            chunks = self.chunker.chunk(doc["text"])
            for c in chunks:
                c["doc_id"] = doc["id"]
                c["doc_type"] = "transcript"
                c["doc_date"] = doc["date"]
                c["section"] = "transcript"
            transcript_chunks.append(c)

        # ── STEP 6: Embed chunks and store claims ─────────────────────
        self._cb(progress_callback, 6, "Embedding and storing claims...")
        all_chunks = filing_chunks + transcript_chunks
        for c in all_chunks:
            claim_id = make_id(c["doc_id"], str(c["chunk_index"]))
            c["claim_id"] = claim_id
            with get_db() as conn:
                insert_claim(conn, id=claim_id, doc_id=c["doc_id"],
                             ticker=ticker, chunk_text=c["text"],
                             chunk_index=c["chunk_index"],
                             section=c.get("section", ""),
                             embedding_json="",
                             sentiment_label="", sentiment_score=0.0)
            summary["claims"] += 1

        # ── STEP 7: NLI contradiction detection ──────────────────────
        self._cb(progress_callback, 7, "Detecting contradictions (NLI)...")
        try:
            from src.models.nli import NLIModel
            from src.models.finbert import FinBERTSentiment
            from src.models.groq_summarizer import GroqSummarizer

            nli = NLIModel()
            finbert = FinBERTSentiment()
            summarizer = GroqSummarizer()
        except Exception:
            nli = finbert = summarizer = None

        if nli and filing_chunks and transcript_chunks:
            # Compare top filing chunks against transcript chunks
            pairs_to_check = []
            for fc in filing_chunks[:50]:  # limit to avoid explosion
                for tc in transcript_chunks[:50]:
                    pairs_to_check.append((fc, tc))

            for fc, tc in pairs_to_check[:200]:  # cap at 200 pairs
                try:
                    result = nli.predict(fc["text"], tc["text"])
                    if result["label"] == "contradiction" and result["score"] > NLI_THRESHOLD:
                        # Compute sentiment shift
                        sent_shift = 0.0
                        if finbert:
                            sent_shift = finbert.sentiment_shift([fc["text"]], [tc["text"]])

                        combined = 0.6 * result["score"] + 0.4 * abs(sent_shift)
                        topic = extract_topic(fc["text"] + " " + tc["text"])

                        # Generate LLM summary
                        llm_summary = ""
                        if summarizer:
                            llm_summary = summarizer.summarize(
                                fc["text"], tc["text"], ticker,
                                fc.get("doc_type", "filing"), tc.get("doc_type", "transcript"),
                            )

                        contra_id = new_uuid()
                        with get_db() as conn:
                            insert_contradiction(
                                conn, id=contra_id, ticker=ticker,
                                claim_a_id=fc.get("claim_id", ""),
                                claim_b_id=tc.get("claim_id", ""),
                                claim_a_text=fc["text"][:2000],
                                claim_b_text=tc["text"][:2000],
                                doc_a_type=fc.get("doc_type", "filing"),
                                doc_b_type=tc.get("doc_type", "transcript"),
                                doc_a_date=fc.get("doc_date", ""),
                                doc_b_date=tc.get("doc_date", ""),
                                topic=topic,
                                nli_label=result["label"],
                                nli_score=result["score"],
                                semantic_score=0.0,
                                combined_score=combined,
                                llm_summary=llm_summary,
                                created_at=now_iso(),
                            )
                        summary["contradictions"] += 1
                except Exception:
                    continue

        # ── STEP 8: Compute CAR for contradictions ────────────────────
        self._cb(progress_callback, 8, "Computing market signals (CAR)...")
        from src.backtest.signals import SignalGenerator
        sig_gen = SignalGenerator()

        # ── STEP 9: Generate signals ──────────────────────────────────
        self._cb(progress_callback, 9, "Generating trading signals...")
        with get_db() as conn:
            contras = conn.execute(
                "SELECT * FROM contradictions WHERE ticker = ?", (ticker,)
            ).fetchall()

            for c in contras:
                c = dict(c)
                event_date = c.get("doc_b_date") or c.get("doc_a_date") or datetime.now().strftime("%Y-%m-%d")
                sig = sig_gen.generate_signal(
                    ticker=ticker,
                    contradiction_score=c.get("combined_score", 0.0),
                    sentiment_shift=0.0,
                    event_date=event_date,
                    contradiction_id=c["id"],
                    description=c.get("llm_summary", "")[:500],
                )
                insert_signal(conn, **sig)
                summary["signals"] += 1

        # ── STEP 10: Mark complete ────────────────────────────────────
        self._cb(progress_callback, 10, f"Processing complete for {ticker}!")
        with get_db() as conn:
            conn.execute(
                "UPDATE companies SET status = 'processed', last_processed = ? WHERE ticker = ?",
                (datetime.now().strftime("%Y-%m-%d"), ticker),
            )

        return summary
