"""
Bayesian seat range model for WB 2026 elections.

KEY DESIGN: Indian elections use FPTP with 3+ parties, so TMC can win with ~40-45%
vote share. We model P(TMC wins constituency) directly in logit space rather than
comparing vote share against 0.5.

Pipeline:
  build_priors()           → per-constituency win probability from historical win rates + 2024 LS
  apply_sir_adjustment()   → systematic downward shift in logit space
  bayesian_update()        → Normal-Normal conjugate update from daily news signals (in logit space)
  monte_carlo()            → 10,000 simulations → seat distribution + win probs
  run_full_pipeline()      → orchestrates all steps
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logit, expit  # logit = log(p/(1-p)), expit = sigmoid

DATA_DIR = Path(__file__).parent.parent / "data"

# Signal noise in logit space (news is a very noisy proxy for true probability)
TAU_LOGIT = 1.20
# Max logit shift per day from a strong signal (+/-1.0)
SIGNAL_SCALE_LOGIT = 0.30
# Prior σ used for Bayesian update (per-constituency uncertainty)
SIGMA_LOGIT_PRIOR = 1.20

# Monte Carlo uncertainty decomposed into two components:
# 1. GLOBAL: election-wave factor correlated across ALL constituencies (BJP/TMC wave)
#    This is the dominant driver of wide confidence bands (±30-45 seat CI).
# 2. CONSTITUENCY: seat-specific noise (local factors, candidate quality, etc.)
SIGMA_LOGIT_GLOBAL = 0.65
SIGMA_LOGIT_CONSTITUENCY = 0.80
# SIR global model uncertainty (logit space)
GLOBAL_SIR_SIGMA_LOGIT = 0.16

# SIR lean factors — grounded in actual WB voter deletion composition:
# 91 lakh total deleted: ~63 lakh Hindu (69%), ~28 lakh Muslim/minority (31%)
# Muslim deleted voters: ~85% TMC lean (historical Muslim-TMC alignment)
# Hindu deleted voters: ~45% TMC lean (split electorate; BJP has Hindu vote share)
# Baseline assumption: if voters weren't deleted, 50% would have voted TMC on average.
# Only the EXCESS over 50% creates an asymmetric impact on TMC.
TMC_LEAN_MINORITY = 0.85   # Muslim/minority deleted voters → TMC lean
TMC_LEAN_MAJORITY = 0.45   # Hindu deleted voters → slight BJP lean

# SIR uncertainty: we don't know exact lean factors; model as ±15% of deletion_rate
SIR_LEAN_SIGMA_FACTOR = 0.15

# Fixed Left/Others seats
LEFT_OTHERS_FIXED = 15
MAJORITY = 148


def _safe_logit(p: pd.Series, eps: float = 0.01) -> pd.Series:
    return logit(p.clip(eps, 1 - eps))



# Calibrated regional TMC win probabilities as of 2026 baseline.
# Derived from: 2021 assembly (0.50 wt) + 2016 (0.30 wt) + 2011 (0.20 wt) + 2024 LS recency.
# Actual 2021: north=46%, jangal=36%, med=63%, urban=91%, rural=85% (TMC 213/294 total).
# 2024 LS: TMC 29/42 seats; north gained, jangal weakened, urban/medinipur strong.

# Calibrated so that Monte Carlo with σ_logit=1.50 produces expected TMC baseline ~185 seats.
# With 120 rural + 68 urban + 54 NB + 27 med + 25 jangal = 294 total (rural is inflated vs reality).
# Values reflect: 2021 assembly + 2024 LS recency + current 2026 structural baseline.
# Pre-SIR baseline. SIR adjustment then reduces TMC further.
REGIONAL_WIN_PRIOR: dict[str, float] = {
    "north_bengal":       0.47,  # BJP competitive; TMC improved in 2024 LS → ~47%
    "jangalmahal":        0.40,  # BJP stronghold; BJP held 2/4 LS → ~40%
    "medinipur":          0.66,  # TMC advantage; won Medinipur LS 2024 → ~66%
    "urban_kolkata":      0.88,  # TMC stronghold; 100% LS 2024 → ~88%
    "south_bengal_rural": 0.58,  # Calibrated down for inflated seat count (120 vs true ~105)
}
# Within-region variation: seats differ from regional mean (some are safe, some swing)
# σ in probability space before logit transform
WITHIN_REGION_SIGMA = 0.10


def build_priors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Compute prior win probability P(TMC wins constituency).

    Uses calibrated regional baselines (REGIONAL_WIN_PRIOR) plus constituency-level
    variation seeded from historical data to differentiate seats within each region.

    μ_logit = logit(win_prior_i)
    σ_logit = SIGMA_LOGIT_PRIOR (constant)
    """
    constituencies = pd.read_csv(data_dir / "wb_constituencies.csv")

    # Read historical data to add constituency-level deviation from regional mean
    e2021 = pd.read_csv(data_dir / "ecidata_2021.csv")[["constituency_id", "winner"]]
    e2016 = pd.read_csv(data_dir / "ecidata_2016.csv")[["constituency_id", "winner"]]
    e2011 = pd.read_csv(data_dir / "ecidata_2011.csv")[["constituency_id", "winner"]]

    e2021["w21"] = (e2021["winner"] == "tmc").astype(float)
    e2016["w16"] = (e2016["winner"] == "tmc").astype(float)
    e2011["w11"] = (e2011["winner"] == "tmc").astype(float)

    df = (constituencies
          .merge(e2021[["constituency_id", "w21"]], on="constituency_id")
          .merge(e2016[["constituency_id", "w16"]], on="constituency_id")
          .merge(e2011[["constituency_id", "w11"]], on="constituency_id"))

    # Constituency-level win signal (raw, before calibration)
    df["win_raw"] = (0.50 * df["w21"] + 0.30 * df["w16"] + 0.20 * df["w11"]).clip(0.02, 0.98)

    # Regional mean from raw data (will be used to compute deviation)
    region_means = df.groupby("region")["win_raw"].mean().rename("region_mean_raw")
    df = df.merge(region_means, on="region")

    # Calibrated: regional baseline + constituency deviation from raw mean
    df["regional_baseline"] = df["region"].map(REGIONAL_WIN_PRIOR)
    df["constituency_deviation"] = (df["win_raw"] - df["region_mean_raw"]).clip(-0.30, 0.30)
    df["win_prior"] = (df["regional_baseline"] + df["constituency_deviation"] * 0.5).clip(0.05, 0.95)

    df["mu_logit"] = _safe_logit(df["win_prior"])
    df["sigma_logit"] = SIGMA_LOGIT_PRIOR

    return df[["constituency_id", "name", "region", "win_prior", "mu_logit", "sigma_logit"]]


def apply_sir_adjustment(priors: pd.DataFrame, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    SIR creates a systematic reduction in TMC's win probability.

    Factual basis: 91 lakh voters deleted statewide — 63 lakh Hindu (~69%), 28 lakh Muslim (~31%).
    Both communities are affected, but with different TMC lean:
      - Muslim deleted voters: ~85% TMC lean  (TMC_LEAN_MINORITY)
      - Hindu deleted voters:  ~45% TMC lean  (TMC_LEAN_MAJORITY, slight BJP lean)

    Per constituency, minority_share = fraction of deleted voters who are Muslim/minority.
    (Proxied by the local Muslim population share — higher in Murshidabad/Malda, lower elsewhere.)

    Blended TMC lean among all deleted voters:
      tmc_lean = minority_share × 0.85 + (1 - minority_share) × 0.45
               = 0.45 + minority_share × 0.40

    Excess lean over the 50% neutral baseline:
      excess_lean = tmc_lean - 0.50
                  = minority_share × 0.40 - 0.05

    TMC win probability reduction (in probability space):
      delta_p = -deletion_rate × excess_lean

    We then convert EXACTLY to logit space (no linear approximation):
      win_adj = win_prior - delta_p
      mu_adj  = logit(win_adj)

    This avoids the LOGIT_SCALE≈4 linearization which only holds near p=0.5.

    Uncertainty (in logit space via delta method):
      sigma_sir ≈ deletion_rate × SIR_LEAN_SIGMA_FACTOR / (win_adj × (1 - win_adj))
    """
    sir = pd.read_csv(data_dir / "sir_deletions.csv")[["constituency_id", "deletion_rate", "minority_share"]]
    df = priors.merge(sir, on="constituency_id", how="left")
    df["deletion_rate"] = df["deletion_rate"].fillna(0.04)
    df["minority_share"] = df["minority_share"].fillna(0.12)

    # Blended TMC lean across Hindu + Muslim deleted voters
    tmc_lean = TMC_LEAN_MAJORITY + df["minority_share"] * (TMC_LEAN_MINORITY - TMC_LEAN_MAJORITY)
    excess_lean = tmc_lean - 0.50  # impact relative to neutral 50% baseline

    # Adjust in probability space (exact), then convert to logit
    win_p = expit(df["mu_logit"])
    delta_p = df["deletion_rate"] * excess_lean
    win_adj_p = (win_p - delta_p).clip(0.02, 0.98)
    df["delta_p"] = delta_p
    df["mu_adj"] = _safe_logit(win_adj_p)
    df["delta_logit"] = df["mu_adj"] - df["mu_logit"]

    # SIR uncertainty in logit space via delta method: σ_logit ≈ σ_p / (p(1-p))
    sigma_sir_p = df["deletion_rate"] * SIR_LEAN_SIGMA_FACTOR
    sigma_sir_logit = sigma_sir_p / (win_adj_p * (1 - win_adj_p))
    df["sigma_sir_logit"] = sigma_sir_logit

    df["sigma_adj"] = np.sqrt(
        df["sigma_logit"] ** 2 +
        df["sigma_sir_logit"] ** 2 +
        GLOBAL_SIR_SIGMA_LOGIT ** 2
    )
    df["win_adj"] = expit(df["mu_adj"])

    return df


def bayesian_update(adj_priors: pd.DataFrame, regional_signals: dict) -> pd.DataFrame:
    """
    Normal-Normal conjugate update in logit space.

    regional_signals: {region: {"signal_strength": float ∈ [-1,+1], "article_count": int}}

    Observation in logit space: obs = mu_adj + signal_strength × SIGNAL_SCALE_LOGIT
    Since TAU_LOGIT(0.5) >> sigma_adj(~0.45): news gets ~45% weight, prior ~55%.
    """
    updated = adj_priors.copy()
    updated["mu_post"] = updated["mu_adj"]
    updated["sigma_post"] = updated["sigma_adj"]

    precision_obs = 1.0 / (TAU_LOGIT ** 2)

    for region, sig in regional_signals.items():
        mask = updated["region"] == region
        if not mask.any() or sig.get("article_count", 0) == 0:
            continue

        s = float(sig.get("signal_strength", 0.0))
        obs_logit = updated.loc[mask, "mu_adj"] + s * SIGNAL_SCALE_LOGIT

        precision_prior = 1.0 / updated.loc[mask, "sigma_adj"] ** 2
        denom = precision_prior + precision_obs

        updated.loc[mask, "mu_post"] = (
            precision_prior * updated.loc[mask, "mu_adj"] +
            precision_obs * obs_logit
        ) / denom
        updated.loc[mask, "sigma_post"] = np.sqrt(1.0 / denom)

    updated["win_post"] = expit(updated["mu_post"])
    return updated


def monte_carlo(posteriors: pd.DataFrame, n_sim: int = 10_000) -> dict:
    """
    Simulate n_sim elections using two-level uncertainty:

    1. Global swing ε_global ~ N(0, σ_global²): correlated wave affecting ALL constituencies.
       Models BJP/TMC election-wide wave (2019 BJP wave, 2021 TMC wave).
       This dominates the width of the confidence interval.

    2. Constituency noise ε_i ~ N(0, σ_constituency²): seat-specific factors.

    z_eff_ik = mu_post_i + ε_global_k + ε_constituency_ik + ε_SIR_ik

    TMC wins constituency if z_eff > 0.
    """
    mu = posteriors["mu_post"].values
    delta_sir = posteriors.get("delta_logit", pd.Series(0.0, index=posteriors.index)).values
    sigma_sir = posteriors.get("sigma_sir_logit", pd.Series(0.0, index=posteriors.index)).values
    n_seats = len(mu)

    rng = np.random.default_rng(seed=2026)

    # Global swing: (n_sim, 1) — same for all constituencies in each simulation
    z_global = rng.normal(0, SIGMA_LOGIT_GLOBAL, size=(n_sim, 1))

    # Constituency-specific noise: (n_sim, n_seats)
    z_local = rng.normal(0, SIGMA_LOGIT_CONSTITUENCY, size=(n_sim, n_seats))

    # SIR uncertainty: (n_sim, n_seats)
    z_sir = rng.normal(0, sigma_sir, size=(n_sim, n_seats))

    z_eff = mu + z_global + z_local + z_sir
    tmc_wins = (z_eff > 0).sum(axis=1)
    bjp_wins = np.clip(294 - tmc_wins - LEFT_OTHERS_FIXED, 0, 294 - LEFT_OTHERS_FIXED)

    # SIR-only band: compare with/without SIR delta applied
    z_no_sir = mu - delta_sir + z_global + z_local
    tmc_no_sir = (z_no_sir > 0).sum(axis=1)
    sir_band = int(np.percentile(tmc_no_sir - tmc_wins, 75)) + int(np.percentile(tmc_wins, 95) - np.percentile(tmc_wins, 5)) // 4

    return {
        # Primary display: p25-p75 (central 50% range — "likely" seats)
        "tmc_p25": int(np.percentile(tmc_wins, 25)),
        "tmc_p50": int(np.percentile(tmc_wins, 50)),
        "tmc_p75": int(np.percentile(tmc_wins, 75)),
        # Full 90% CI (p5-p95)
        "tmc_p5": int(np.percentile(tmc_wins, 5)),
        "tmc_p95": int(np.percentile(tmc_wins, 95)),
        "bjp_p25": int(np.percentile(bjp_wins, 25)),
        "bjp_p50": int(np.percentile(bjp_wins, 50)),
        "bjp_p75": int(np.percentile(bjp_wins, 75)),
        "bjp_p5": int(np.percentile(bjp_wins, 5)),
        "bjp_p95": int(np.percentile(bjp_wins, 95)),
        "p_tmc_win": float(np.mean(tmc_wins >= MAJORITY)),
        "p_hung": float(np.mean((tmc_wins < MAJORITY) & (tmc_wins >= 120))),
        "p_bjp_win": float(np.mean(bjp_wins >= MAJORITY)),
        "sir_uncertainty_band": sir_band,
    }


def run_full_pipeline(regional_signals: dict, data_dir: Path = DATA_DIR) -> tuple[dict, pd.DataFrame]:
    """Entry point: returns (forecast_dict, constituency_posteriors)."""
    priors = build_priors(data_dir)
    adj = apply_sir_adjustment(priors, data_dir)
    post = bayesian_update(adj, regional_signals)
    results = monte_carlo(post)
    return results, post
