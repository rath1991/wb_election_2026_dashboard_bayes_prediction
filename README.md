# WB 2026 Elections Prediction Dashboard

Bayesian day-over-day seat forecast for West Bengal 2026 Assembly Elections.

## Setup

### 1. Install dependencies
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env with your keys
```

Required:
- `ANTHROPIC_KEY` — Claude API (claude-sonnet-4-6 for noise filtering)
- `SUPABASE_URL` + `SUPABASE_KEY` — free at supabase.com

Optional:
- `NEWS_API_KEY` — newsapi.org (free tier, 100 req/day)
- `YOUTUBE_API_KEY` — Google Cloud Console

### 3. Create Supabase tables
In your Supabase project → SQL Editor, run:

```sql
CREATE TABLE IF NOT EXISTS forecasts (
    date DATE PRIMARY KEY,
    tmc_p5 INT, tmc_p25 INT, tmc_p50 INT, tmc_p75 INT, tmc_p95 INT,
    bjp_p5 INT, bjp_p25 INT, bjp_p50 INT, bjp_p75 INT, bjp_p95 INT,
    p_tmc_win REAL, p_hung REAL, p_bjp_win REAL,
    sir_uncertainty_band INT,
    prev_tmc_p50 INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS regional_signals (
    id BIGSERIAL PRIMARY KEY,
    date DATE, region TEXT,
    signal_strength REAL,
    article_count INT,
    sir_adjusted_mu REAL,
    sir_sigma REAL,
    UNIQUE(date, region)
);

CREATE TABLE IF NOT EXISTS bjp_conditions (
    id BIGSERIAL PRIMARY KEY,
    date DATE, condition_key TEXT,
    status TEXT,
    confidence REAL,
    evidence_json JSONB,
    UNIQUE(date, condition_key)
);

CREATE TABLE IF NOT EXISTS articles (
    id BIGSERIAL PRIMARY KEY,
    date DATE, source TEXT, url TEXT,
    headline TEXT, body_snippet TEXT,
    credibility_score REAL,
    is_noise BOOLEAN DEFAULT FALSE,
    region_tags JSONB,
    signal_tags JSONB,
    bjp_conditions_hit JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 4. Generate first forecast
```bash
python scheduler.py --once
```

### 5. Run dashboard
```bash
streamlit run app.py
```

---

## Bayesian Model

**Prior**: `μ = 0.60 × hist_avg + 0.40 × LS2024_region`  
**SIR adjustment**: `δ = -deletion_rate × minority_share × 0.80`  
**Uncertainty**: `σ = sqrt(σ_prior² + σ_SIR² + 0.04²)`  
**Daily update**: Normal-Normal conjugate (news weight ~31%, prior ~69%)  
**Seat range**: Monte Carlo 10,000 simulations → p5/p50/p95

---

## Deploy to Render

1. Push to GitHub
2. Create new Render project → "New Blueprint" → point to repo
3. Render reads `render.yaml` and creates:
   - Web service (Streamlit dashboard)
   - Cron job (daily pipeline at 07:00 IST)
4. Add env vars in Render dashboard

---

## Data Sources

- `data/ecidata_2021/2016/2011.csv` — ECI historical results (synthetic approximation; replace with actual ECI data)
- `data/ls2024_regional.csv` — 2024 Lok Sabha results by region
- `data/sir_deletions.csv` — SIR voter deletion data by constituency (from ECI/The Wire analysis)
- `data/wb_constituencies.csv` — 294 WB assembly constituencies with region mapping

> **Note**: ECI historical CSV files are approximations generated from regional distributions. For production accuracy, replace with actual ECI booth-level data from [results.eci.gov.in](https://results.eci.gov.in).
