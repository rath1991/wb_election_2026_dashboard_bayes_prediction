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


def _ensure_schema(conn: duckdb.DuckDBPyConnection):  # noqa: C901
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
            date DATE, source VARCHAR, source_tier INTEGER DEFAULT 4,
            url VARCHAR, headline VARCHAR, body_snippet VARCHAR,
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            date DATE PRIMARY KEY,
            tmc_probability DOUBLE,
            bjp_probability DOUBLE,
            market_shift_24h DOUBLE,
            volume_usd DOUBLE,
            source VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turnout_live (
            id INTEGER PRIMARY KEY,
            date DATE, phase INTEGER, district VARCHAR,
            ac_name VARCHAR, turnout_percent DOUBLE,
            absolute_votes BIGINT,
            turnout_vs_2021 DOUBLE,
            turnout_vs_2024 DOUBLE,
            source VARCHAR,
            UNIQUE (date, ac_name)
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS turnout_live_seq START 1")

    # ── Phase turnout (official VTR + post-scrutiny) ─────────────────────────
    # Stores both initial VTR-app estimates and post-scrutiny revised figures.
    # NEVER truncated — new rows upserted by (phase, ac_name).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS phase_turnout (
            id INTEGER PRIMARY KEY,
            phase INTEGER,
            district VARCHAR,
            ac_no INTEGER,
            ac_name VARCHAR,
            region VARCHAR,
            total_electors BIGINT,
            initial_turnout_percent DOUBLE,
            post_scrutiny_turnout_percent DOUBLE,
            votes_polled BIGINT,
            male_turnout_percent DOUBLE,
            female_turnout_percent DOUBLE,
            turnout_2021 DOUBLE,
            turnout_2024ls DOUBLE,
            turnout_swing_vs_2021 DOUBLE,
            source VARCHAR,
            source_type VARCHAR,
            confidence_score DOUBLE DEFAULT 0.9,
            notes VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (phase, ac_name)
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS phase_turnout_seq START 1")

    # ── Booth-level turnout (Form 17C from agents / VTR booth data) ──────────
    # source_type: 'official_vtr' | 'form17c' | 'party_agent' | 'media'
    # source_party: 'TMC' | 'BJP' | 'INC' | 'CPM' | 'official' | 'media'
    # Cross-check: single-party Form 17C treated as confidence_score 0.5 until verified.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS booth_turnout (
            id INTEGER PRIMARY KEY,
            phase INTEGER,
            ac_no INTEGER,
            ac_name VARCHAR,
            booth_no INTEGER,
            polling_station_name VARCHAR,
            total_electors INTEGER,
            votes_polled INTEGER,
            male_votes INTEGER,
            female_votes INTEGER,
            other_votes INTEGER,
            turnout_percent DOUBLE,
            turnout_vs_2021 DOUBLE,
            source VARCHAR,
            source_type VARCHAR,
            source_party VARCHAR,
            confidence_score DOUBLE DEFAULT 0.5,
            notes VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (phase, ac_name, booth_no)
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS booth_turnout_seq START 1")

    # ── Booth master (static enriched reference table) ───────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS booth_master (
            id INTEGER PRIMARY KEY,
            district VARCHAR,
            ac_no INTEGER,
            ac_name VARCHAR,
            region VARCHAR,
            booth_no INTEGER,
            polling_station_name VARCHAR,
            polling_station_location VARCHAR,
            urban_rural VARCHAR,
            total_electors INTEGER,
            male_electors INTEGER,
            female_electors INTEGER,
            third_gender_electors INTEGER,
            sir_deleted_count INTEGER,
            minority_flag BOOLEAN DEFAULT FALSE,
            sc_st_flag BOOLEAN DEFAULT FALSE,
            matua_flag BOOLEAN DEFAULT FALSE,
            border_flag BOOLEAN DEFAULT FALSE,
            sensitive_flag BOOLEAN DEFAULT FALSE,
            violence_history_flag BOOLEAN DEFAULT FALSE,
            UNIQUE (ac_no, booth_no)
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS booth_master_seq START 1")

    # Migration: add source_tier to articles if missing
    try:
        cols_df = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='articles'").fetchdf()
        if "source_tier" not in cols_df["column_name"].tolist():
            conn.execute("ALTER TABLE articles ADD COLUMN source_tier INTEGER DEFAULT 4")
    except Exception:
        pass


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

def upsert_market_data(conn: duckdb.DuckDBPyConnection, data: dict):
    if not data:
        return
    prev = conn.execute(
        "SELECT tmc_probability FROM market_data ORDER BY date DESC LIMIT 1"
    ).fetchone()
    shift = None
    if prev and data.get("tmc_probability") is not None:
        shift = round(data["tmc_probability"] - prev[0], 4)
    conn.execute("""
        INSERT INTO market_data (date, tmc_probability, bjp_probability, market_shift_24h, volume_usd, source)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (date) DO UPDATE SET
            tmc_probability = excluded.tmc_probability,
            bjp_probability = excluded.bjp_probability,
            market_shift_24h = excluded.market_shift_24h,
            volume_usd = excluded.volume_usd
    """, [
        str(date.today()),
        data.get("tmc_probability"),
        data.get("bjp_probability"),
        shift,
        data.get("volume_usd"),
        data.get("source", "Polymarket"),
    ])


def get_latest_market_data(conn: duckdb.DuckDBPyConnection) -> dict | None:
    result = conn.execute(
        "SELECT * FROM market_data ORDER BY date DESC LIMIT 1"
    ).fetchdf()
    if result.empty:
        return None
    row = result.iloc[0].to_dict()
    row["date"] = str(row["date"])
    return row


# ── phase_turnout ────────────────────────────────────────────────────────────

def upsert_phase_turnout(conn: duckdb.DuckDBPyConnection, rows: list[dict]):
    """Upsert AC/district-level turnout. Safe to call repeatedly — no deletes."""
    for r in rows:
        swing = None
        if r.get("post_scrutiny_turnout_percent") and r.get("turnout_2021"):
            swing = round(r["post_scrutiny_turnout_percent"] - r["turnout_2021"], 2)
        elif r.get("initial_turnout_percent") and r.get("turnout_2021"):
            swing = round(r["initial_turnout_percent"] - r["turnout_2021"], 2)
        conn.execute("""
            INSERT INTO phase_turnout
                (id, phase, district, ac_no, ac_name, region, total_electors,
                 initial_turnout_percent, post_scrutiny_turnout_percent, votes_polled,
                 male_turnout_percent, female_turnout_percent,
                 turnout_2021, turnout_2024ls, turnout_swing_vs_2021,
                 source, source_type, confidence_score, notes, updated_at)
            VALUES (nextval('phase_turnout_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (phase, ac_name) DO UPDATE SET
                post_scrutiny_turnout_percent = COALESCE(excluded.post_scrutiny_turnout_percent, post_scrutiny_turnout_percent),
                initial_turnout_percent       = COALESCE(excluded.initial_turnout_percent, initial_turnout_percent),
                turnout_swing_vs_2021         = COALESCE(excluded.turnout_swing_vs_2021, turnout_swing_vs_2021),
                votes_polled                  = COALESCE(excluded.votes_polled, votes_polled),
                male_turnout_percent          = COALESCE(excluded.male_turnout_percent, male_turnout_percent),
                female_turnout_percent        = COALESCE(excluded.female_turnout_percent, female_turnout_percent),
                source                        = excluded.source,
                confidence_score              = excluded.confidence_score,
                notes                         = COALESCE(excluded.notes, notes),
                updated_at                    = CURRENT_TIMESTAMP
        """, [
            r.get("phase"), r.get("district"), r.get("ac_no"), r.get("ac_name"),
            r.get("region"), r.get("total_electors"),
            r.get("initial_turnout_percent"), r.get("post_scrutiny_turnout_percent"),
            r.get("votes_polled"),
            r.get("male_turnout_percent"), r.get("female_turnout_percent"),
            r.get("turnout_2021"), r.get("turnout_2024ls"), swing,
            r.get("source", "manual"), r.get("source_type", "manual"),
            r.get("confidence_score", 0.9), r.get("notes"),
        ])


def get_phase_turnout(conn: duckdb.DuckDBPyConnection, phase: int | None = None) -> pd.DataFrame:
    if phase is not None:
        return conn.execute(
            "SELECT * FROM phase_turnout WHERE phase = ? ORDER BY district, ac_name", [phase]
        ).fetchdf()
    return conn.execute("SELECT * FROM phase_turnout ORDER BY phase, district, ac_name").fetchdf()


def get_phase_turnout_summary(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """District-level summary grouped for dashboard display."""
    return conn.execute("""
        SELECT phase, district, region,
               AVG(post_scrutiny_turnout_percent) AS avg_post_scrutiny,
               AVG(initial_turnout_percent)        AS avg_initial,
               AVG(turnout_2021)                   AS avg_2021,
               AVG(turnout_swing_vs_2021)           AS avg_swing,
               COUNT(*) AS ac_count
        FROM phase_turnout
        GROUP BY phase, district, region
        ORDER BY phase, avg_swing DESC NULLS LAST
    """).fetchdf()


# ── booth_turnout ─────────────────────────────────────────────────────────────

def upsert_booth_turnout(conn: duckdb.DuckDBPyConnection, rows: list[dict]):
    """Upsert Form 17C or VTR booth-level data. Never deletes existing rows."""
    for r in rows:
        conn.execute("""
            INSERT INTO booth_turnout
                (id, phase, ac_no, ac_name, booth_no, polling_station_name,
                 total_electors, votes_polled, male_votes, female_votes, other_votes,
                 turnout_percent, turnout_vs_2021,
                 source, source_type, source_party, confidence_score, notes)
            VALUES (nextval('booth_turnout_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (phase, ac_name, booth_no) DO UPDATE SET
                votes_polled     = COALESCE(excluded.votes_polled, votes_polled),
                turnout_percent  = COALESCE(excluded.turnout_percent, turnout_percent),
                source_type      = excluded.source_type,
                source_party     = excluded.source_party,
                confidence_score = GREATEST(excluded.confidence_score, confidence_score),
                notes            = COALESCE(excluded.notes, notes)
        """, [
            r.get("phase"), r.get("ac_no"), r.get("ac_name"), r.get("booth_no"),
            r.get("polling_station_name"),
            r.get("total_electors"), r.get("votes_polled"),
            r.get("male_votes"), r.get("female_votes"), r.get("other_votes"),
            r.get("turnout_percent"), r.get("turnout_vs_2021"),
            r.get("source"), r.get("source_type", "manual"),
            r.get("source_party", "unknown"),
            r.get("confidence_score", 0.5), r.get("notes"),
        ])


def get_booth_turnout(conn: duckdb.DuckDBPyConnection, ac_name: str | None = None) -> pd.DataFrame:
    if ac_name:
        return conn.execute(
            "SELECT * FROM booth_turnout WHERE ac_name = ? ORDER BY booth_no", [ac_name]
        ).fetchdf()
    return conn.execute("SELECT * FROM booth_turnout ORDER BY phase, ac_name, booth_no").fetchdf()


# ── booth_master ──────────────────────────────────────────────────────────────

def upsert_booth_master(conn: duckdb.DuckDBPyConnection, rows: list[dict]):
    for r in rows:
        conn.execute("""
            INSERT INTO booth_master
                (id, district, ac_no, ac_name, region, booth_no,
                 polling_station_name, polling_station_location, urban_rural,
                 total_electors, male_electors, female_electors, third_gender_electors,
                 sir_deleted_count, minority_flag, sc_st_flag, matua_flag,
                 border_flag, sensitive_flag, violence_history_flag)
            VALUES (nextval('booth_master_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ac_no, booth_no) DO UPDATE SET
                sir_deleted_count    = COALESCE(excluded.sir_deleted_count, sir_deleted_count),
                sensitive_flag       = excluded.sensitive_flag,
                violence_history_flag = excluded.violence_history_flag
        """, [
            r.get("district"), r.get("ac_no"), r.get("ac_name"), r.get("region"),
            r.get("booth_no"), r.get("polling_station_name"), r.get("polling_station_location"),
            r.get("urban_rural"), r.get("total_electors"),
            r.get("male_electors"), r.get("female_electors"), r.get("third_gender_electors"),
            r.get("sir_deleted_count"),
            bool(r.get("minority_flag")), bool(r.get("sc_st_flag")),
            bool(r.get("matua_flag")), bool(r.get("border_flag")),
            bool(r.get("sensitive_flag")), bool(r.get("violence_history_flag")),
        ])


def insert_articles(conn: duckdb.DuckDBPyConnection, articles: list[dict]):
    for a in articles:
        conn.execute("""
            INSERT INTO articles
                (id, date, source, source_tier, url, headline, body_snippet,
                 credibility_score, is_noise, region_tags, signal_tags, bjp_conditions_hit)
            VALUES (nextval('articles_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            a.get("date", str(date.today())),
            a.get("source", "")[:200],
            int(a.get("source_tier", 4)),
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
