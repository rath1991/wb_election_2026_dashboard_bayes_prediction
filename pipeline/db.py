"""
Database layer using DuckDB.
Local: wb_elections.duckdb (file in project root)
Render: set DB_PATH env var to a persistent disk path, e.g. /data/wb_elections.duckdb
"""
import os
import json
import duckdb
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent.parent / "wb_elections.duckdb"


def get_db_path() -> str:
    return os.environ.get("DB_PATH", str(_DEFAULT_DB))


def get_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(get_db_path())
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            date DATE PRIMARY KEY,
            tmc_p5 INTEGER, tmc_p25 INTEGER, tmc_p50 INTEGER, tmc_p75 INTEGER, tmc_p95 INTEGER,
            bjp_p5 INTEGER, bjp_p25 INTEGER, bjp_p50 INTEGER, bjp_p75 INTEGER, bjp_p95 INTEGER,
            p_tmc_win DOUBLE, p_hung DOUBLE, p_bjp_win DOUBLE,
            sir_uncertainty_band INTEGER,
            prev_tmc_p50 INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regional_signals (
            id INTEGER PRIMARY KEY,
            date DATE, region VARCHAR,
            signal_strength DOUBLE,
            article_count INTEGER,
            sir_adjusted_mu DOUBLE,
            sir_sigma DOUBLE,
            top_articles_json VARCHAR,
            UNIQUE (date, region)
        )
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS regional_signals_seq START 1
    """)
    # Migration: add top_articles_json column if it doesn't exist yet
    try:
        cols_df = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='regional_signals'").fetchdf()
        if "top_articles_json" not in cols_df["column_name"].tolist():
            conn.execute("ALTER TABLE regional_signals ADD COLUMN top_articles_json VARCHAR")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bjp_conditions (
            id INTEGER PRIMARY KEY,
            date DATE, condition_key VARCHAR,
            status VARCHAR,
            confidence DOUBLE,
            evidence_json VARCHAR,
            UNIQUE (date, condition_key)
        )
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS bjp_conditions_seq START 1
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            date DATE, source VARCHAR, url VARCHAR,
            headline VARCHAR, body_snippet VARCHAR,
            credibility_score DOUBLE,
            is_noise BOOLEAN DEFAULT FALSE,
            region_tags VARCHAR,
            signal_tags VARCHAR,
            bjp_conditions_hit VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS articles_seq START 1
    """)


# ── forecasts ──────────────────────────────────────────────────────────────

def upsert_forecast(conn: duckdb.DuckDBPyConnection, forecast_date: date, data: dict):
    row = {"date": str(forecast_date)}
    for k, v in data.items():
        if k == "tmc_distribution":
            continue
        row[k] = v

    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?" for _ in row])
    updates = ", ".join(f"{k} = excluded.{k}" for k in row if k != "date")
    conn.execute(
        f"INSERT INTO forecasts ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT (date) DO UPDATE SET {updates}",
        list(row.values())
    )


def get_latest_forecast(conn: duckdb.DuckDBPyConnection) -> dict | None:
    result = conn.execute(
        "SELECT * FROM forecasts ORDER BY date DESC LIMIT 1"
    ).fetchdf()
    if result.empty:
        return None
    row = result.iloc[0].to_dict()
    # Convert date to string for JSON safety
    row["date"] = str(row["date"])
    return row


def get_forecast_history(conn: duckdb.DuckDBPyConnection, days: int = 30) -> pd.DataFrame:
    since = date.today() - timedelta(days=days)
    df = conn.execute(
        "SELECT * FROM forecasts WHERE date >= ? ORDER BY date",
        [str(since)]
    ).fetchdf()
    return df


# ── regional_signals ────────────────────────────────────────────────────────

def upsert_regional_signals(conn: duckdb.DuckDBPyConnection, signal_date: date, signals: dict):
    for region, sig in signals.items():
        conn.execute("""
            INSERT INTO regional_signals (id, date, region, signal_strength, article_count, sir_adjusted_mu, sir_sigma, top_articles_json)
            VALUES (nextval('regional_signals_seq'), ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (date, region) DO UPDATE SET
                signal_strength = excluded.signal_strength,
                article_count = excluded.article_count,
                top_articles_json = excluded.top_articles_json
        """, [
            str(signal_date), region,
            round(sig.get("signal_strength", 0.0), 4),
            sig.get("article_count", 0),
            sig.get("sir_adjusted_mu"),
            sig.get("sir_sigma"),
            json.dumps(sig.get("top_articles", [])),
        ])


def get_regional_signals(conn: duckdb.DuckDBPyConnection, days: int = 7) -> pd.DataFrame:
    since = date.today() - timedelta(days=days)
    return conn.execute(
        "SELECT * FROM regional_signals WHERE date >= ? ORDER BY date",
        [str(since)]
    ).fetchdf()


def get_latest_regional_signals_with_articles(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Return today's (or most recent) regional signals with top_articles for citations."""
    result = conn.execute("""
        SELECT region, signal_strength, article_count, top_articles_json
        FROM regional_signals
        WHERE date = (SELECT MAX(date) FROM regional_signals)
        ORDER BY region
    """).fetchdf()
    if result.empty:
        return []
    rows = result.to_dict("records")
    for row in rows:
        raw = row.get("top_articles_json") or "[]"
        row["top_articles"] = json.loads(raw) if isinstance(raw, str) else []
    return rows


# ── bjp_conditions ──────────────────────────────────────────────────────────

def upsert_bjp_conditions(conn: duckdb.DuckDBPyConnection, cond_date: date, conditions: dict):
    for cond_key, data in conditions.items():
        conn.execute("""
            INSERT INTO bjp_conditions (id, date, condition_key, status, confidence, evidence_json)
            VALUES (nextval('bjp_conditions_seq'), ?, ?, ?, ?, ?)
            ON CONFLICT (date, condition_key) DO UPDATE SET
                status = excluded.status,
                confidence = excluded.confidence,
                evidence_json = excluded.evidence_json
        """, [
            str(cond_date), cond_key,
            data.get("status", "red"),
            round(data.get("confidence", 0.0), 4),
            data.get("evidence_json", "[]"),
        ])


def get_bjp_conditions(conn: duckdb.DuckDBPyConnection, cond_date: date | None = None) -> list[dict]:
    if cond_date is None:
        latest = get_latest_forecast(conn)
        if not latest:
            return []
        cond_date = latest["date"]
    result = conn.execute(
        "SELECT * FROM bjp_conditions WHERE date = ?", [str(cond_date)]
    ).fetchdf()
    return result.to_dict("records")


# ── articles ────────────────────────────────────────────────────────────────

def insert_articles(conn: duckdb.DuckDBPyConnection, articles: list[dict]):
    for a in articles:
        conn.execute("""
            INSERT INTO articles
                (id, date, source, url, headline, body_snippet, credibility_score,
                 is_noise, region_tags, signal_tags, bjp_conditions_hit)
            VALUES (nextval('articles_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            a.get("date", str(date.today())),
            a.get("source", "")[:200],
            a.get("url", "")[:500],
            a.get("headline", "")[:500],
            a.get("body_snippet", "")[:800],
            float(a.get("credibility_score", 0.5)),
            bool(a.get("is_noise", False)),
            json.dumps(a.get("region_tags", [])),
            json.dumps(a.get("signal_tags", {})),
            json.dumps(a.get("bjp_conditions_hit", [])),
        ])


def get_recent_articles(
    conn: duckdb.DuckDBPyConnection, days: int = 2, min_credibility: float = 0.5
) -> pd.DataFrame:
    since = date.today() - timedelta(days=days)
    df = conn.execute("""
        SELECT * FROM articles
        WHERE date >= ?
          AND is_noise = FALSE
          AND credibility_score >= ?
        ORDER BY credibility_score DESC
        LIMIT 100
    """, [str(since), min_credibility]).fetchdf()

    if df.empty:
        return df

    for col in ["region_tags", "signal_tags", "bjp_conditions_hit"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
    return df
