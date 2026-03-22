"""
src/database.py — SQLite schema and helper functions for SignalEdge.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

from src.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    cik TEXT,
    sector TEXT,
    status TEXT DEFAULT 'pending',
    last_processed DATE,
    is_pinned BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    ticker TEXT,
    doc_type TEXT,
    period TEXT,
    filed_date DATE,
    source_url TEXT,
    raw_text TEXT,
    processed BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    doc_id TEXT,
    ticker TEXT,
    chunk_text TEXT,
    chunk_index INT,
    section TEXT,
    embedding_json TEXT,
    sentiment_label TEXT,
    sentiment_score FLOAT
);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    ticker TEXT,
    claim_a_id TEXT,
    claim_b_id TEXT,
    claim_a_text TEXT,
    claim_b_text TEXT,
    doc_a_type TEXT,
    doc_b_type TEXT,
    doc_a_date DATE,
    doc_b_date DATE,
    topic TEXT,
    nli_label TEXT,
    nli_score FLOAT,
    semantic_score FLOAT,
    combined_score FLOAT,
    llm_summary TEXT,
    created_at DATETIME
);

CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    ticker TEXT,
    signal_type TEXT,
    confidence FLOAT,
    price FLOAT,
    price_change_pct FLOAT,
    description TEXT,
    event_date DATE,
    car_1d FLOAT,
    car_3d FLOAT,
    car_5d FLOAT,
    contradiction_id TEXT
);

CREATE TABLE IF NOT EXISTS market_data (
    ticker TEXT,
    date DATE,
    close FLOAT,
    volume INT,
    abnormal_return FLOAT,
    PRIMARY KEY (ticker, date)
);
"""


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Return a new SQLite connection with row factory enabled."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db(db_path: str | None = None):
    """Context manager that yields a connection and commits on success."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create all tables if they do not exist."""
    with get_db(db_path) as conn:
        conn.executescript(_SCHEMA)


# ── Generic helpers ───────────────────────────────────────────────────────

def upsert_company(conn: sqlite3.Connection, **kw) -> None:
    conn.execute(
        """INSERT INTO companies (ticker, name, cik, sector, status, last_processed, is_pinned)
           VALUES (:ticker, :name, :cik, :sector, :status, :last_processed, :is_pinned)
           ON CONFLICT(ticker) DO UPDATE SET
             name=excluded.name, cik=excluded.cik, sector=excluded.sector,
             status=excluded.status, last_processed=excluded.last_processed,
             is_pinned=excluded.is_pinned""",
        kw,
    )


def insert_document(conn: sqlite3.Connection, **kw) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO documents
           (id, ticker, doc_type, period, filed_date, source_url, raw_text, processed)
           VALUES (:id, :ticker, :doc_type, :period, :filed_date, :source_url, :raw_text, :processed)""",
        kw,
    )


def insert_claim(conn: sqlite3.Connection, **kw) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO claims
           (id, doc_id, ticker, chunk_text, chunk_index, section, embedding_json,
            sentiment_label, sentiment_score)
           VALUES (:id, :doc_id, :ticker, :chunk_text, :chunk_index, :section,
                   :embedding_json, :sentiment_label, :sentiment_score)""",
        kw,
    )


def insert_contradiction(conn: sqlite3.Connection, **kw) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO contradictions
           (id, ticker, claim_a_id, claim_b_id, claim_a_text, claim_b_text,
            doc_a_type, doc_b_type, doc_a_date, doc_b_date, topic,
            nli_label, nli_score, semantic_score, combined_score,
            llm_summary, created_at)
           VALUES (:id, :ticker, :claim_a_id, :claim_b_id, :claim_a_text, :claim_b_text,
                   :doc_a_type, :doc_b_type, :doc_a_date, :doc_b_date, :topic,
                   :nli_label, :nli_score, :semantic_score, :combined_score,
                   :llm_summary, :created_at)""",
        kw,
    )


def insert_signal(conn: sqlite3.Connection, **kw) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO signals
           (id, ticker, signal_type, confidence, price, price_change_pct,
            description, event_date, car_1d, car_3d, car_5d, contradiction_id)
           VALUES (:id, :ticker, :signal_type, :confidence, :price, :price_change_pct,
                   :description, :event_date, :car_1d, :car_3d, :car_5d, :contradiction_id)""",
        kw,
    )


def insert_market_data(conn: sqlite3.Connection, **kw) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO market_data
           (ticker, date, close, volume, abnormal_return)
           VALUES (:ticker, :date, :close, :volume, :abnormal_return)""",
        kw,
    )


def count_table(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def fetch_all(conn: sqlite3.Connection, table: str, where: str = "", params: tuple = ()) -> list[dict]:
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]
