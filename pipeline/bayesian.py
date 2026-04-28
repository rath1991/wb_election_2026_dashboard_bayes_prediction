"""
Bayesian seat range model for WB 2026 elections.

KEY DESIGN: Indian elections use FPTP with 3+ parties, so TMC can win with ~40-45%
vote share. We model P(TMC wins constituency) directly in logit space rather than
comparing vote share against 0.5.

Pipeline:
  build_priors()                  → per-constituency TMC win probability from REAL 2021+2024 margins
  apply_sir_adjustment()          → systematic downward shift from voter deletion (SIR)
  apply_organizational_factors()  → BJP RSS mobilization + CM-face vacuum + IPAC shutdown
  bayesian_update()               → Normal-Normal conjugate update from daily news signals (in logit space)
  monte_carlo()                   → 10,000 simulations → seat distribution + win probs
  run_full_pipeline()             → orchestrates all steps

DATA SOURCES (real, not synthetic):
  2021 assembly margins: tecoholic/Election2021 GitHub (ECI candidate-level data)
  2024 LS margins:       ECI results portal / Wikipedia (all 42 WB seats hardcoded)
  SIR deletions:         TOI/Indian Express district+AC level reporting
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logit, expit  # logit = log(p/(1-p)), expit = sigmoid

DATA_DIR = Path(__file__).parent.parent / "data"

# Signal noise per source tier (logit space) — lower τ = signal is trusted more.
# Framework weight: news/social = 3%, polls/markets = 5% of total model.
# Tier 1 (ECI official) is near-truth; Tier 6 (social) is very noisy.
TAU_LOGIT_BY_TIER = {
    1: 0.40,   # ECI official — near ground truth
    2: 0.80,   # Tier-1 news (Indian Express, The Hindu, NDTV) — reliable reporting
    3: 1.10,   # Regional Bengali news — good but some bias
    4: 1.50,   # Aggregated (Google News, GDELT) — noisy aggregation
    5: 2.00,   # Prediction markets — sentiment only, not ground truth
    6: 2.50,   # Social / YouTube — very high noise
}
TAU_LOGIT = 1.20                # legacy default (used if tier not specified)
# Max logit shift per day from a strong signal (+/-1.0)
SIGNAL_SCALE_LOGIT = 0.30
# Prior σ used for Bayesian update (per-constituency uncertainty)
SIGMA_LOGIT_PRIOR = 1.20
# Polymarket: market odds enter as very weak observation
TAU_LOGIT_MARKET = 2.00        # high noise — sentiment, not booth data

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

# ── Organizational factor constants ──────────────────────────────────────────
#
# SOURCE: BJP deployed Sunil Bansal (national gen. sec.) + Bhupendra Yadav from 2022
#   onwards for WB 2026 groundwork. RSS expanded from 1,320 → 1,823 shakhas (+38%)
#   in Madhya Banga Prant alone; 1.75 lakh voter-awareness meetings across ~250/294
#   constituencies (BJP internal estimate, reported in India Today / The Print 2024).
#
# HISTORICAL CAP (why effect is bounded):
#   WB 2021: BJP had full RSS mobilization + Sunil Bansal yet won only 77/294 seats
#            (target was 130+). WB politics resists the standard RSS mobilization playbook.
#   2024 LS: BJP deployed Bansal; TMC still won 29/42 seats. RSS advantage did not
#            translate proportionally. Estimate: WB conversion rate ~25% of UP equivalent.
#
# RSS effect on TMC win probability per region (negative = hurts TMC):
RSS_MOBILIZATION_DELTA_P: dict[str, float] = {
    "north_bengal":        -0.030,  # BJP shakha network deepest; Matua belt + hill seat work
    "jangalmahal":         -0.020,  # BJP tribal stronghold; booth-level management targeted
    "medinipur":           -0.015,  # Suvendu home turf; RSS + personal network combined
    "urban_kolkata":       -0.012,  # RSS drives Hindu consolidation (bhadralok + migrant workers).
                                    # PARTIALLY OFFSET by Muslim counter-mobilization: RSS visibility
                                    # in 35%+ Muslim wards hardens TMC consolidation there.
                                    # Net: small non-zero BJP gain in Hindu-majority urban wards.
    "south_bengal_rural":  -0.018,  # RSS worked 250/294 constituencies — rural penetration real.
                                    # Higher Hindu population share than urban Kolkata reduces
                                    # counter-mobilization intensity. Larger net BJP gain vs urban.
}
# ±50% uncertainty on the RSS delta (WB conversion rate from organization → votes is empirically noisy)
RSS_SIGMA_FACTOR = 0.50

# BJP CM face vacuum effect on TMC win probability (positive = helps TMC):
#
# SOURCE: As of 2026, BJP has NOT declared a CM candidate for WB.
#   Suvendu Adhikari (Leader of Opposition) and Sukanta Majumdar (state president)
#   both positioning themselves. Dilip Ghosh (former state chief, mass connect) was
#   sidelined after 2021 loss — creating demoralization in old guard cadre.
#   (Ref: Indian Express 2024 "BJP Bengal CM face dilemma", Anandabazar Patrika 2025)
#
# Historical precedent:
#   2021: BJP had no state CM face ("Modi-for-PM" campaign) → 77 seats.
#   2016: No CM face → 10 seats.
#   Mamata's personal vote (CM face brand) is worth ~5-8% in urban swing seats.
#
CM_FACE_VACUUM_DELTA_P: dict[str, float] = {
    "north_bengal":        0.010,   # Some Modi-wave residual; CM face less decisive here
    "jangalmahal":         0.010,   # Rural/tribal vote — CM face less sensitive
    "medinipur":           0.015,   # Suvendu's local pull partially offsets vacuum
    "urban_kolkata":       0.030,   # Highest — Mamata's personal vote vs. BJP anonymity
    "south_bengal_rural":  0.020,   # Minority + TMC loyalty; no BJP face compounds BJP weakness
}
# ±40% uncertainty (Suvendu could be declared as CM face before election, partially closing gap)
CM_FACE_SIGMA_FACTOR = 0.40

# IPAC shutdown effect on TMC win probability (negative = hurts TMC):
#
# SOURCE: IPAC (I-PAC, Indian Political Action Committee) shut down operations mid-election
#   April 2026. Founder Vinesh Chandel arrested; Rishiraj Singh summoned. Mamata personally
#   visited Pratik Jain's home after January 2026 ED raid — shows IPAC was irreplaceable.
#   IPAC responsibility: booth-level committees, social media strategy, rally planning,
#   Didi Ke Bolo feedback loop, Duare Sarkar coordination. (Ref: Red Mic video April 2026,
#   India Today / Wire reporting on IPAC shutdown).
#
# IPAC was the reason TMC recovered from 2019 disaster to win 213/294 in 2021.
# Its mid-election shutdown removes TMC's primary campaign infrastructure.
#
# Effect magnitude: comparable to RSS mobilization (both are ground-level org factors).
# Largest in urban_kolkata/south_bengal_rural where IPAC ran tightest operations.
# Smaller in north_bengal/jangalmahal where BJP was already strong regardless.
IPAC_SHUTDOWN_DELTA_P: dict[str, float] = {
    "north_bengal":        -0.010,  # BJP-competitive; IPAC loss less decisive vs BJP structural advantage
    "jangalmahal":         -0.012,  # Rural tribal belt; IPAC handled booth targeting here
    "medinipur":           -0.018,  # Key swing seats; IPAC's micro-management was critical
    "urban_kolkata":       -0.025,  # IPAC most critical: urban coordination, social media, feedback loops
    "south_bengal_rural":  -0.022,  # IPAC's rural booth network was TMC's backbone in minority belt
}
# ±40% uncertainty: some IPAC workers may continue informally; TMC state apparatus partly compensates
IPAC_SIGMA_FACTOR = 0.40

# Seat-type SIR conditioning: SIR benefits BJP ONLY in BJP-competitive seats.
# SOURCE: In Malda-Murshidabad (Muslim-majority), competition is Congress vs TMC — not BJP.
#   If Muslim votes are deleted there, Congress gains, not BJP. Nandagram (95% Muslim
#   deletions) and Nadia are mixed seats where tactical Muslim vote matters for BJP competition.
#   (Ref: Dr. Kartikeya Batra analysis, The Red Mic, April 2026)
#
# At minority_share=0.20: BJP fully competitive → full SIR delta applies (factor=1.0)
# At minority_share=0.50: BJP partially competitive → factor ~0.63
# At minority_share=0.80: BJP non-competitive (Congress fight) → factor=0.25 (floor)
SIR_BJP_COMPETITIVE_THRESHOLD = 0.20  # above this, start scaling down
SIR_BJP_COMPETITIVE_RATE = 1.25       # rate of scale-down per unit of minority_share
SIR_BJP_COMPETITIVE_FLOOR = 0.25      # minimum factor even in fully Muslim-majority seats

# Fixed Left/Others/Congress seats — raised from 15 to 20.
# Congress wins several Malda/Murshidabad seats (Muslim-majority) where competition
# is Congress vs TMC, not BJP. SIR-conditioning reduces BJP attribution there, but
# the floor is captured here. (2021: Congress+Left won ~12 seats; 2026 baseline ~20
# with Congress recovering slightly in their belt.)
LEFT_OTHERS_FIXED = 20
MAJORITY = 148


def _safe_logit(p: pd.Series, eps: float = 0.01) -> pd.Series:
    return logit(p.clip(eps, 1 - eps))



# Sigma for converting margin (percentage points) to logit probability.
# At sigma=7pp: a TMC lead of 7pp → logistic(1.0) = 73% TMC win probability.
# Reflects empirical uncertainty in seat-level vote share conversion.
SIGMA_MARGIN_PP = 0.07

# Margin formula weights (from user's reproducible model spec):
#   M_ac = 0.45 * ls_2024_margin + 0.25 * assembly_2021_margin + [adjustments applied later]
# Weights sum to 0.70 for the base data component; remaining 0.30 is applied via
# SIR, org, turnout adjustments in subsequent pipeline steps.
WEIGHT_LS_2024 = 0.45
WEIGHT_ASSEMBLY_2021 = 0.25


def build_priors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Compute prior TMC win probability using REAL 2021 + 2024 data.

    Formula (user-specified reproducible model):
      base_margin = 0.45 * ls_2024_tmc_minus_bjp + 0.25 * assembly_2021_tmc_minus_bjp
      p_tmc_raw   = logistic(base_margin / σ)      σ = 7 percentage points

    Others adjustment (Congress/Left strongholds):
      In seats where Congress+Left got >20% in 2021 (Murshidabad/Malda), reduce
      p_tmc by scaling factor to account for three-way competition. This prevents
      the model from assigning TMC wins in seats Congress will actually win.

    Data sources (real, not synthetic):
      ecidata_2021.csv:  real 2021 WB assembly results from tecoholic/Election2021 GitHub
      ls2024_real.csv:   real 2024 LS seat results (all 42 WB seats) mapped to each AC
    """
    constituencies = pd.read_csv(data_dir / "wb_constituencies.csv")

    # Real 2021 assembly data
    e2021 = pd.read_csv(data_dir / "ecidata_2021.csv")[[
        "constituency_id", "tmc_voteshare", "bjp_voteshare",
        "left_voteshare", "cong_voteshare", "tmc_minus_bjp_margin",
    ]]

    # Real 2024 LS data (LS seat level, mapped to each AC)
    ls2024 = pd.read_csv(data_dir / "ls2024_real.csv")[[
        "constituency_id", "ls24_tmc_minus_bjp", "ls24_winner",
    ]]

    df = constituencies.merge(e2021, on="constituency_id", how="left")
    df = df.merge(ls2024, on="constituency_id", how="left")

    # Fill missing values with regional fallbacks
    regional_fallback_margin = {
        "north_bengal":       -0.04,  # BJP slightly ahead
        "jangalmahal":        -0.10,  # BJP stronghold
        "medinipur":          +0.08,  # TMC advantage
        "urban_kolkata":      +0.30,  # TMC stronghold
        "south_bengal_rural": +0.18,  # TMC dominant
    }
    for col, fallback_map in [
        ("tmc_minus_bjp_margin", regional_fallback_margin),
        ("ls24_tmc_minus_bjp",   regional_fallback_margin),
    ]:
        df[col] = df[col].fillna(df["region"].map(fallback_map))

    df["left_voteshare"] = df["left_voteshare"].fillna(0.06)
    df["cong_voteshare"] = df["cong_voteshare"].fillna(0.04)

    # Load minority_share from SIR data — proxy for Congress recovery potential
    sir_min = pd.read_csv(data_dir / "sir_deletions.csv")[["constituency_id", "minority_share"]]
    df = df.merge(sir_min, on="constituency_id", how="left")
    df["minority_share"] = df["minority_share"].fillna(0.12)

    # ── Base margin: weighted combination of 2024 LS and 2021 assembly ──────────
    df["base_margin"] = (
        WEIGHT_LS_2024      * df["ls24_tmc_minus_bjp"] +
        WEIGHT_ASSEMBLY_2021 * df["tmc_minus_bjp_margin"]
    )

    # ── Others adjustment: Congress/Left strongholds ─────────────────────────────
    # Two sources of Congress/Left competition:
    # (a) 2021 vote share — but Congress was wiped out in 2021 (won 0 seats),
    #     so this understates their 2026 recovery in Muslim-majority seats.
    # (b) Minority share proxy: in Muslim-majority areas (Murshidabad, Malda),
    #     Congress is recovering in 2026 as they were suppressed in 2021.
    #     High minority_share → higher Congress probability of winning the seat.
    df["others_2021"] = df["left_voteshare"] + df["cong_voteshare"]
    # Direct signal from 2021 data
    others_2021_signal = ((df["others_2021"] - 0.10).clip(lower=0.0) * 1.2).clip(upper=0.30)
    # Congress recovery signal from minority_share (Murshidabad/Malda effect)
    # minority_share > 0.30 → significant Congress competition for the seat
    congress_recovery = ((df["minority_share"] - 0.30).clip(lower=0.0) * 1.5).clip(upper=0.40)
    df["p_others_adj"] = (others_2021_signal + congress_recovery).clip(upper=0.45)

    # ── Convert margin to TMC win probability ───────────────────────────────────
    # logistic(margin / σ): margin=0 → 50%, margin=+7pp → 73%, margin=-7pp → 27%
    df["p_tmc_raw"] = expit(df["base_margin"] / SIGMA_MARGIN_PP)

    # Apply others scaling: in three-way seats, both TMC and BJP probabilities shrink
    df["win_prior"] = (df["p_tmc_raw"] * (1 - df["p_others_adj"])).clip(0.05, 0.95)

    df["mu_logit"] = _safe_logit(df["win_prior"])
    df["sigma_logit"] = SIGMA_LOGIT_PRIOR

    return df[[
        "constituency_id", "name", "region",
        "base_margin", "p_tmc_raw", "p_others_adj",
        "win_prior", "mu_logit", "sigma_logit",
    ]]


def apply_sir_adjustment(priors: pd.DataFrame, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    SIR creates a systematic reduction in TMC's win probability.

    Factual basis: ~89–91 lakh voters removed statewide (EC data: ~89L total roll fall at 11.62%;
    27.16L deleted after adjudication from 60.06L under adjudication pool — Indian Express Apr 2026).
    Composition: ~63 lakh Hindu (~69%), ~28 lakh Muslim (~31%).
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

    BJP competitiveness conditioning (KEY: SIR only helps BJP in BJP-competitive seats):
      In Muslim-majority seats (Malda/Murshidabad), competition is Congress vs TMC — not BJP.
      If Muslim votes are deleted there, Congress gains, not BJP. Seat-type factor:
        bjp_competitive_factor = (1 - (minority_share - 0.20).clip(0) × 1.25).clip(0.25, 1.0)
      This scales the delta down in Muslim-majority seats, preventing over-attribution to BJP.
      (Source: Dr. Kartikeya Batra, The Red Mic, April 2026)

    TMC win probability reduction (in probability space):
      delta_p = deletion_rate × excess_lean × bjp_competitive_factor

    We then convert EXACTLY to logit space (no linear approximation):
      win_adj = win_prior - delta_p
      mu_adj  = logit(win_adj)

    Uncertainty (in logit space via delta method):
      sigma_sir ≈ deletion_rate × SIR_LEAN_SIGMA_FACTOR / (win_adj × (1 - win_adj))
    """
    sir = pd.read_csv(data_dir / "sir_deletions.csv")[["constituency_id", "deletion_count", "deletion_rate", "minority_share", "sir_severity"]]
    df = priors.merge(sir, on="constituency_id", how="left")
    df["deletion_rate"] = df["deletion_rate"].fillna(0.04)
    df["minority_share"] = df["minority_share"].fillna(0.12)

    # Blended TMC lean across Hindu + Muslim deleted voters
    tmc_lean = TMC_LEAN_MAJORITY + df["minority_share"] * (TMC_LEAN_MINORITY - TMC_LEAN_MAJORITY)
    excess_lean = tmc_lean - 0.50  # impact relative to neutral 50% baseline

    # BJP competitiveness factor: reduces SIR delta in Muslim-majority/non-BJP-competitive seats
    bjp_competitive_factor = (
        1.0 - (df["minority_share"] - SIR_BJP_COMPETITIVE_THRESHOLD).clip(lower=0.0) * SIR_BJP_COMPETITIVE_RATE
    ).clip(lower=SIR_BJP_COMPETITIVE_FLOOR, upper=1.0)

    # Adjust in probability space (exact), then convert to logit
    win_p = expit(df["mu_logit"])
    delta_p = df["deletion_rate"] * excess_lean * bjp_competitive_factor
    win_adj_p = (win_p - delta_p).clip(0.02, 0.98)
    df["delta_p"] = delta_p
    df["mu_adj"] = _safe_logit(win_adj_p)
    df["delta_logit"] = df["mu_adj"] - df["mu_logit"]

    # SIR uncertainty in logit space via delta method: σ_logit ≈ σ_p / (p(1-p))
    sigma_sir_p = df["deletion_rate"] * SIR_LEAN_SIGMA_FACTOR
    sigma_sir_logit = sigma_sir_p / (win_adj_p * (1 - win_adj_p))

    # SIR pressure scaling: high deletion_rate → additional uncertainty beyond lean uncertainty.
    # In seats where deletions >> competitive margin (Samserganj 29.6%, Lalgola 22%, Bhabanipur 25%),
    # small errors in lean assumptions compound. Scale sigma up proportionally.
    # Source: real AC-level data (Indian Express Apr 2026) confirms deletion/margin ratios of 5–47x.
    # At deletion_rate=0.10: +25% sigma. At deletion_rate≥0.20: +50% sigma (capped).
    pressure_scale = 1.0 + (df["deletion_rate"] / 0.10).clip(upper=2.0) * 0.25
    sigma_sir_logit = sigma_sir_logit * pressure_scale
    df["sigma_sir_logit"] = sigma_sir_logit

    df["sigma_adj"] = np.sqrt(
        df["sigma_logit"] ** 2 +
        df["sigma_sir_logit"] ** 2 +
        GLOBAL_SIR_SIGMA_LOGIT ** 2
    )
    df["win_adj"] = expit(df["mu_adj"])

    return df


def apply_organizational_factors(adj_priors: pd.DataFrame) -> pd.DataFrame:
    """
    Apply structural BJP organizational factors as a permanent prior adjustment.

    Three effects modeled:

    1. RSS mobilization (BJP advantage, capped by WB historical conversion rate):
       BJP expanded from 1,320 → 1,823 shakhas (+38%, Madhya Banga Prant), conducted
       1.75 lakh voter-awareness meetings across ~250/294 constituencies. Sunil Bansal
       + Bhupendra Yadav redeployed from 2022 for WB 2026 groundwork.
       Cap: WB 2021 BJP won 77 seats (target 130+) despite full RSS mobilization;
       2024 LS TMC won 29/42 with Bansal deployed → conversion rate ~25% of UP equivalent.
       Effect: negative delta on TMC win probability in BJP-competitive regions.

    2. BJP CM face vacuum (TMC advantage):
       BJP has not declared a CM candidate. Suvendu Adhikari vs. Sukanta Majumdar rivalry.
       Dilip Ghosh sidelined post-2021 → cadre demoralization. Mamata Banerjee's personal
       vote (strong incumbent CM face) historically worth 5-8% in urban swing seats.
       Effect: positive delta on TMC win probability, strongest in urban_kolkata.

    3. IPAC shutdown (TMC disadvantage):
       IPAC (Vinesh Chandel arrested; Rishiraj Singh summoned) shut down mid-election April 2026.
       IPAC ran booth-level committees, social media strategy, rally planning, and Duare Sarkar
       coordination — the very infrastructure that reversed 2019 damage in 2021.
       Effect: negative delta on TMC win probability, largest in urban_kolkata and south_bengal_rural
       where IPAC's operations were most critical.

    All effects applied in probability space (exact), then converted to logit.
    Uncertainty propagated in quadrature and folded into sigma_adj.
    """
    df = adj_priors.copy()

    rss_delta = df["region"].map(RSS_MOBILIZATION_DELTA_P).fillna(0.0)
    cm_delta = df["region"].map(CM_FACE_VACUUM_DELTA_P).fillna(0.0)
    ipac_delta = df["region"].map(IPAC_SHUTDOWN_DELTA_P).fillna(0.0)
    net_delta_p = rss_delta + cm_delta + ipac_delta

    win_p = expit(df["mu_adj"])
    win_org_p = (win_p + net_delta_p).clip(0.02, 0.98)

    df["org_rss_delta_p"] = rss_delta
    df["org_cm_delta_p"] = cm_delta
    df["org_ipac_delta_p"] = ipac_delta
    df["org_net_delta_p"] = net_delta_p
    df["mu_org"] = _safe_logit(win_org_p)

    # Uncertainty: sigma in probability space → delta-method to logit
    sigma_rss_p = df["region"].map(
        {r: abs(v) * RSS_SIGMA_FACTOR for r, v in RSS_MOBILIZATION_DELTA_P.items()}
    ).fillna(0.0)
    sigma_cm_p = df["region"].map(
        {r: abs(v) * CM_FACE_SIGMA_FACTOR for r, v in CM_FACE_VACUUM_DELTA_P.items()}
    ).fillna(0.0)
    sigma_ipac_p = df["region"].map(
        {r: abs(v) * IPAC_SIGMA_FACTOR for r, v in IPAC_SHUTDOWN_DELTA_P.items()}
    ).fillna(0.0)
    sigma_org_p = np.sqrt(sigma_rss_p ** 2 + sigma_cm_p ** 2 + sigma_ipac_p ** 2)
    sigma_org_logit = sigma_org_p / (win_org_p * (1 - win_org_p))

    df["sigma_adj"] = np.sqrt(df["sigma_adj"] ** 2 + sigma_org_logit ** 2)
    df["mu_adj"] = df["mu_org"]
    df["win_adj"] = expit(df["mu_adj"])

    return df


def bayesian_update(
    adj_priors: pd.DataFrame,
    regional_signals: dict,
    market_data: dict | None = None,
) -> pd.DataFrame:
    """
    Normal-Normal conjugate update in logit space.

    regional_signals: {region: {
        "signal_strength": float ∈ [-1,+1],
        "article_count": int,
        "avg_source_tier": float   ← NEW: mean tier of contributing articles
    }}

    market_data: optional Polymarket odds {
        "tmc_probability": float,   ← converted to logit → statewide obs
        "bjp_probability": float,
    }

    TAU varies by source tier:
      Tier 1 (official) τ=0.40 → very strong signal
      Tier 2 (quality news) τ=0.80 → reliable
      Tier 6 (social) τ=2.50 → nearly ignored

    Weights per framework: news/social 3%, markets 5% of total model.
    The structural prior (history + SIR + org factors) dominates at ~92%.
    """
    updated = adj_priors.copy()
    updated["mu_post"] = updated["mu_adj"]
    updated["sigma_post"] = updated["sigma_adj"]

    for region, sig in regional_signals.items():
        mask = updated["region"] == region
        if not mask.any() or sig.get("article_count", 0) == 0:
            continue

        s = float(sig.get("signal_strength", 0.0))
        # Use tier-specific TAU — higher tier sources are trusted more
        avg_tier = float(sig.get("avg_source_tier", 4.0))
        # Interpolate: pick the TAU for the nearest integer tier
        tier_int = max(1, min(6, round(avg_tier)))
        tau = TAU_LOGIT_BY_TIER.get(tier_int, TAU_LOGIT)

        obs_logit = updated.loc[mask, "mu_adj"] + s * SIGNAL_SCALE_LOGIT
        precision_prior = 1.0 / updated.loc[mask, "sigma_adj"] ** 2
        precision_obs   = 1.0 / (tau ** 2)
        denom = precision_prior + precision_obs

        updated.loc[mask, "mu_post"] = (
            precision_prior * updated.loc[mask, "mu_adj"] +
            precision_obs   * obs_logit
        ) / denom
        updated.loc[mask, "sigma_post"] = np.sqrt(1.0 / denom)

    # ── Polymarket signal (statewide, very weak) ──────────────────────────────
    if market_data and market_data.get("tmc_probability"):
        tmc_mkt = float(market_data["tmc_probability"])
        tmc_mkt = np.clip(tmc_mkt, 0.05, 0.95)
        obs_market_logit = float(logit(tmc_mkt))
        precision_market = 1.0 / (TAU_LOGIT_MARKET ** 2)

        precision_prior = 1.0 / updated["sigma_post"] ** 2
        denom = precision_prior + precision_market
        updated["mu_post"] = (
            precision_prior * updated["mu_post"] +
            precision_market * obs_market_logit
        ) / denom
        updated["sigma_post"] = np.sqrt(1.0 / denom)

    updated["win_post"] = expit(updated["mu_post"])
    return updated


def monte_carlo(posteriors: pd.DataFrame, n_sim: int = 10_000) -> dict:
    """
    Simulate n_sim elections using three-way seat competition (TMC / BJP / Others).

    For each constituency in each simulation:
    1. Draw if seat goes to Others (Congress/Left) with fixed probability p_others_adj
    2. If not Others: TMC wins if z_eff > 0 (two-party logit contest)

    Two-level uncertainty:
    1. Global swing ε_global ~ N(0, σ_global²): election-wide wave correlated across all seats.
    2. Constituency noise ε_i ~ N(0, σ_constituency²): seat-specific factors.

    z_eff_ik = mu_post_i + ε_global_k + ε_constituency_ik + ε_SIR_ik
    """
    mu = posteriors["mu_post"].values
    delta_sir = posteriors.get("delta_logit", pd.Series(0.0, index=posteriors.index)).values
    sigma_sir = posteriors.get("sigma_sir_logit", pd.Series(0.0, index=posteriors.index)).values
    # Per-AC probability that seat goes to Congress/Left/Others
    p_others = posteriors.get("p_others_adj", pd.Series(0.0, index=posteriors.index)).values
    n_seats = len(mu)

    rng = np.random.default_rng(seed=2026)

    # Global swing: (n_sim, 1) — same for all constituencies in each simulation
    z_global = rng.normal(0, SIGMA_LOGIT_GLOBAL, size=(n_sim, 1))

    # Constituency-specific noise: (n_sim, n_seats)
    z_local = rng.normal(0, SIGMA_LOGIT_CONSTITUENCY, size=(n_sim, n_seats))

    # SIR uncertainty: (n_sim, n_seats)
    z_sir = rng.normal(0, sigma_sir, size=(n_sim, n_seats))

    # Three-way sampling: first draw which seats go to Others
    r_others = rng.uniform(0, 1, size=(n_sim, n_seats))
    seat_to_others = (r_others < p_others)  # (n_sim, n_seats) bool

    # TMC wins among non-Others seats if z_eff > 0
    z_eff = mu + z_global + z_local + z_sir
    tmc_wins_all = (z_eff > 0)
    tmc_wins = (tmc_wins_all & ~seat_to_others).sum(axis=1)
    others_wins = seat_to_others.sum(axis=1)
    bjp_wins = 294 - tmc_wins - others_wins

    # SIR-only band: compare with/without SIR delta applied
    z_no_sir = mu - delta_sir + z_global + z_local
    tmc_no_sir = ((z_no_sir > 0) & ~seat_to_others).sum(axis=1)
    sir_band = int(np.percentile(tmc_no_sir - tmc_wins, 75)) + \
               int(np.percentile(tmc_wins, 95) - np.percentile(tmc_wins, 5)) // 4

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
        "others_p50": int(np.percentile(others_wins, 50)),
        "p_tmc_win": float(np.mean(tmc_wins >= MAJORITY)),
        "p_hung": float(np.mean((tmc_wins < MAJORITY) & (tmc_wins >= 120))),
        "p_bjp_win": float(np.mean(bjp_wins >= MAJORITY)),
        "sir_uncertainty_band": sir_band,
    }


def run_full_pipeline(
    regional_signals: dict,
    data_dir: Path = DATA_DIR,
    market_data: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Entry point: returns (forecast_dict, constituency_posteriors)."""
    priors = build_priors(data_dir)
    adj = apply_sir_adjustment(priors, data_dir)
    adj = apply_organizational_factors(adj)
    post = bayesian_update(adj, regional_signals, market_data=market_data)
    results = monte_carlo(post)
    return results, post
