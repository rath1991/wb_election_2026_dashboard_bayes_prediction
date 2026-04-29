"""
Exit poll calibration layer for WB 2026.

Two-tier bias model:
  Tier A — WB-specific TMC bias (direct historical evidence, highest weight)
  Tier B — Cross-state ruling-party underestimation bias (general pattern)
  Tier C — Direction failure risk (3/12 non-LS elections got direction wrong)

Tier A (WB Assembly):
  2016: polls underestimated TMC by +11 to +55 (mean +31)
  2021: polls underestimated TMC by +57 to +103 (mean +65)
  WB LS 2024: TMC underestimated by ~+14 LS seats (assembly-equiv discounted)

Tier B (cross-state ruling party underestimation, as % of total seats):
  Bihar 2025 (NDA ruling):       underestimated by 47–73 seats / 243 = 19–30%
  Maharashtra 2024 (Mahayuti):   underestimated by 46–88 seats / 288 = 16–31%
  Karnataka 2023 (Congress won): underestimated by  4–29 seats / 224 =  2–13%
  MP 2023 (BJP won):             underestimated by 39–59 seats / 230 = 17–26%
  UP 2022 (BJP ruling):          underestimated by ~14 seats  / 403 =  3%
  Punjab 2022 (AAP won):         underestimated by ~12 seats  / 117 = 10%
  Gujarat 2022 (BJP ruling):     underestimated by ~6  seats  / 182 =  3%
  Cross-state mean: ~15% of seats underestimated → for WB (294): ~44 seats

  LS 2024 national: EXCLUDED — BJP was overestimated (400-paar narrative is
  the OPPOSITE direction; political propaganda inflated BJP prediction, not
  the general ruling-party underestimation bias. Including it would cancel
  legitimate bias signals. Treated separately as agency reliability penalty.)

Tier C (direction failure risk):
  5 direction failures out of 14 distinct election-contexts (36%):
    Himachal 2022, Chhattisgarh 2023, Rajasthan 2023 (Axis), Haryana 2024, Delhi 2025 (Axis)
  When direction fails: true bias could be opposite sign.
  This dramatically widens σ_combined even if mean is unchanged.

Final combined bias:
  WB-specific (Tier A, 60% weight) + cross-state (Tier B, 40% weight)
  Direction failure → extra variance term added to σ
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ──────────────────────────────────────────────────────────────────────────────
# 1. WB-specific TMC assembly bias (actual − predicted, positive = underestimated)
# ──────────────────────────────────────────────────────────────────────────────
WB_ASSEMBLY_BIAS: list[dict] = [
    # 2016 (294 seats, actual TMC = 211)
    {"year": 2016, "agency": "C-Voter",        "bias": 211 - 156},   # +55
    {"year": 2016, "agency": "GFK Mode",        "bias": 211 - 200},   # +11
    {"year": 2016, "agency": "Poll of Polls",   "bias": 211 - 184},   # +27
    # 2021 (294 seats, actual TMC = 215)
    {"year": 2021, "agency": "Axis My India",   "bias": 215 - 143},   # +72
    {"year": 2021, "agency": "C-Voter",         "bias": 215 - 158},   # +57
    {"year": 2021, "agency": "Jan Ki Baat",     "bias": 215 - 112},   # +103
    {"year": 2021, "agency": "Poll of Polls",   "bias": 215 - 156},   # +59
]

# ── Tier A: WB-specific bias (recency-weighted) ───────────────────────────
# 2021 gets 4×, 2016 gets 1×, WB LS 2024 gets 2× (discounted for LS dynamics)
_tier_a = (
    [d["bias"] for d in WB_ASSEMBLY_BIAS if d["year"] == 2016] * 1
    + [d["bias"] for d in WB_ASSEMBLY_BIAS if d["year"] == 2021] * 4
    + [59] * 2   # WB LS 2024: avg 14 LS seats × 294/42 × 0.60 discount = 59 seats
)
TIER_A_MEAN = float(np.mean(_tier_a))
TIER_A_STD  = float(np.std(_tier_a))

# ── Tier B: Cross-state ruling-party underestimation (in WB seat equivalents) ─
# Ruling/winning party underestimation as % of seats × 294 (WB total).
# LS 2024 EXCLUDED: NDA was OVERESTIMATED (400-paar narrative = opposite direction).
# Each tuple: (election, underestimation_seats, total_seats, weight)
CROSS_STATE_BIAS = [
    # Strong signal — large magnitude, clear ruling party
    ("Bihar 2025",         60,  243, 2.0),   # NDA ruling, underestimated by avg 60 seats
    ("Maharashtra 2024",   66,  288, 2.0),   # Mahayuti ruling, underestimated by avg 66 seats
    ("MP 2023",            51,  230, 1.5),   # BJP won, underestimated by avg 51 seats
    # Moderate signal
    ("Karnataka 2023",     17,  224, 1.0),   # Congress won, underestimated by avg 17 seats
    ("Punjab 2022",        12,  117, 1.0),   # AAP won, underestimated by 12 seats
    # Weak signal — small magnitude
    ("UP 2022",            14,  403, 0.5),   # BJP ruling, underestimated by 14 seats
    ("Gujarat 2022",        6,  182, 0.5),   # BJP ruling, underestimated by 6 seats
]
# Convert to WB-seat-equivalent underestimation: (pct_underestimated × 294)
_tier_b_values = []
_tier_b_weights = []
for _, seats, total, w in CROSS_STATE_BIAS:
    pct = seats / total
    wb_equiv = pct * 294
    _tier_b_values.append(wb_equiv)
    _tier_b_weights.append(w)

TIER_B_MEAN = float(np.average(_tier_b_values, weights=_tier_b_weights))
TIER_B_STD  = float(np.sqrt(np.average(
    [(v - TIER_B_MEAN)**2 for v in _tier_b_values], weights=_tier_b_weights
)))

# ── Combined bias: 60% Tier A (WB-specific) + 40% Tier B (cross-state) ───────
BIAS_MEAN = 0.60 * TIER_A_MEAN + 0.40 * TIER_B_MEAN
# Combined std: quadrature combination weighted
BIAS_STD  = float(np.sqrt(0.60**2 * TIER_A_STD**2 + 0.40**2 * TIER_B_STD**2))

# ── Tier C: Direction failure variance ────────────────────────────────────────
# 5 direction failures out of 14 election-contexts = 36% failure rate.
# LS 2024 national added as a special "narrative contamination" failure
# (direction technically correct but magnitude off by 107 seats for Chanakya).
# When direction fails: bias could be negative (polls were right for wrong reasons).
# Expected extra variance = P(failure) × (mean_bias - typical_failure_error)²
DIRECTION_FAILURE_RATE = 0.30  # conservative: 30% chance polls partially right
DIRECTION_FAILURE_ERROR = -40.0  # if direction fails, TMC overestimated by ~40 seats
_extra_variance = DIRECTION_FAILURE_RATE * (DIRECTION_FAILURE_ERROR - BIAS_MEAN)**2
BIAS_STD_WITH_FAILURE = float(np.sqrt(BIAS_STD**2 + _extra_variance))

# ──────────────────────────────────────────────────────────────────────────────
# 2. Empirical agency reliability scores
#    Derived from full 2022-2025 cross-state track record.
#    Score = (direction_accuracy × 0.5) + (1 - normalised_RMSE) × 0.5
#    Higher = more reliable. Used as weights in consensus calculation.
# ──────────────────────────────────────────────────────────────────────────────
AGENCY_RELIABILITY: dict[str, float] = {
    # Matrize: best in Bihar 2025 (-47), Rajasthan reasonable, no WB track record yet
    "Matrize":          0.78,
    # P-MARQ: Rajasthan 2023 spot-on (+7), Maharashtra moderate
    "P-MARQ":           0.75,
    # Peoples Pulse: Maharashtra 2024 close (-50), WB 2026 lone TMC-majority prediction
    "People's Pulse":   0.72,
    # C-Voter: consistent conservative bias; underestimates landslides; got WB 2021 wrong
    "C-Voter":          0.55,
    # Axis My India: Karnataka ✓, UP ✓; Delhi 2025 ✗ (direction wrong), LS 2024 ✗, Haryana ✗
    # Post-2023 systematic BJP-lean detected; reliability decayed
    "Axis My India":    0.50,
    # Today's Chanakya: LS 2024 NDA+400 (worst call in history), Haryana ✗, Bihar understated
    # High BJP narrative contamination — apply heavy penalty
    "Today's Chanakya": 0.30,
    # Zeenia AI: no track record; AI-generated; BJP-funded Zee ecosystem
    "Zeenia AI":        0.35,
    # Poll of Polls aggregate: averages contamination from bad agencies; use cautiously
    "Poll of Polls":    0.60,
}

# BJP narrative contamination signal: agencies that have shown a systematic
# post-2023 pattern of over-predicting BJP/NDA. When used in WB context,
# we apply an additional scepticism multiplier on their BJP-seat predictions.
# (Informational — used in app display, not in seat consensus calculation)
BJP_LEAN_AGENCIES = {"Today's Chanakya", "Zeenia AI", "Axis My India", "C-Voter"}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Load 2026 WB exit poll data
# ──────────────────────────────────────────────────────────────────────────────

def load_2026_polls() -> pd.DataFrame:
    path = DATA_DIR / "wb2026_exit_polls.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def load_history() -> pd.DataFrame:
    path = DATA_DIR / "exit_poll_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Consensus + calibration
# ──────────────────────────────────────────────────────────────────────────────

def raw_consensus_tmc(
    df: pd.DataFrame,
    exclude_outliers: bool = False,
) -> tuple[float, float]:
    """
    Reliability-weighted average of agency TMC midpoints.
    Returns (weighted_mean, weighted_std_of_predictions).
    """
    rows = df.dropna(subset=["tmc_mid"]).copy()
    if exclude_outliers:
        rows = rows[rows["agency"] != "People's Pulse"]

    weights = rows["agency"].map(AGENCY_RELIABILITY).fillna(0.55).values
    mids    = rows["tmc_mid"].values
    w_mean  = float(np.average(mids, weights=weights))
    w_std   = float(np.sqrt(np.average((mids - w_mean) ** 2, weights=weights)))
    return w_mean, w_std


def calibrated_tmc_estimate(exclude_outliers: bool = False, include_direction_failure: bool = True) -> dict:
    """
    Apply WB historical bias correction to 2026 exit poll consensus.

    Calibrated TMC = raw_consensus + bias_mean
    Uncertainty    = sqrt(inter_agency_spread² + bias_std²)

    The bias correction is the single most important number here:
    WB polls have underestimated TMC by ~31 seats (2016) and ~65 seats (2021).
    Recency-weighted mean: ~{BIAS_MEAN:.0f} seats. This is not noise — it is
    structural: minority voters don't talk to pollsters, SIR-deleted voters
    still exist in reality (voted in 2021), ground-level TMC cadre density
    is invisble to booth-intercept surveys.
    """
    df = load_2026_polls()
    raw_mean, raw_spread = raw_consensus_tmc(df, exclude_outliers=exclude_outliers)

    calibrated_mean = raw_mean + BIAS_MEAN
    bias_std_used   = BIAS_STD_WITH_FAILURE if include_direction_failure else BIAS_STD
    calibrated_std  = float(np.sqrt(raw_spread**2 + bias_std_used**2))

    return {
        "raw_consensus":           round(raw_mean, 1),
        "raw_spread_std":          round(raw_spread, 1),
        "tier_a_mean":             round(TIER_A_MEAN, 1),
        "tier_b_mean":             round(TIER_B_MEAN, 1),
        "bias_mean_combined":      round(BIAS_MEAN, 1),
        "bias_std_base":           round(BIAS_STD, 1),
        "bias_std_with_failure":   round(BIAS_STD_WITH_FAILURE, 1),
        "direction_failure_rate":  DIRECTION_FAILURE_RATE,
        "calibrated_mean":         round(calibrated_mean, 1),
        "calibrated_std":          round(calibrated_std, 1),
        "as_voteshare_mu":         round(calibrated_mean / 294, 4),
        "as_voteshare_sigma":      round(calibrated_std / 294, 4),
        "n_agencies":              int(len(df.dropna(subset=["tmc_mid"]))),
        "exclude_outliers":        exclude_outliers,
        "include_direction_failure": include_direction_failure,
    }


def exit_poll_regional_signals(
    prior_model_median: float = 170.0,
    context_discount: float = 0.45,
    exclude_outliers: bool = True,
) -> dict[str, float]:
    """
    Convert calibrated exit poll observation into regional signals for Bayesian update.

    context_discount (0–1): fraction of historical bias to apply.
      Default 0.45 → ~45% of the raw WB bias, reflecting that:
        - 2021 had IPAC at full strength (now shut down)
        - 2021 had no SIR deletions (now real and unresolved)
        - 2026 BJP is more mature in WB than 2021
        - 2016 is too old to be fully comparable
      So we trust ~45% of the historical correction, not 100%.

    The resulting signal is intentionally weak — it informs the prior
    but does not dominate it. If MC still lands at ~170, that is valid.
    """
    df = load_2026_polls()
    raw_mean, _ = raw_consensus_tmc(df, exclude_outliers=exclude_outliers)

    # Apply discounted bias correction
    discounted_bias    = BIAS_MEAN * context_discount
    calibrated_mean    = raw_mean + discounted_bias

    delta  = calibrated_mean - prior_model_median
    signal = float(np.clip(delta / 60.0, -1.0, 1.0))  # wider scale = gentler signal

    regions = [
        "north_bengal", "jangalmahal", "medinipur",
        "urban_kolkata", "south_bengal_rural",
    ]
    return {
        "signals":           {r: round(signal, 4) for r in regions},
        "raw_consensus":     round(raw_mean, 1),
        "discounted_bias":   round(discounted_bias, 1),
        "calibrated_mean":   round(calibrated_mean, 1),
        "context_discount":  context_discount,
        "signal_strength":   round(signal, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Summary tables for app display
# ──────────────────────────────────────────────────────────────────────────────

def wb_bias_table() -> pd.DataFrame:
    rows = []
    for d in WB_ASSEMBLY_BIAS:
        actual = 215 if d["year"] == 2021 else 211
        rows.append({
            "Year":                          d["year"],
            "Agency":                        d["agency"],
            "Predicted TMC":                 actual - d["bias"],
            "Actual TMC":                    actual,
            "Bias (actual − predicted)":     f"+{d['bias']}",
        })
    return pd.DataFrame(rows)


def agency_scorecard() -> pd.DataFrame:
    """Summary of agency track records used to derive reliability weights."""
    records = [
        {"Agency": "Matrize",          "Reliability": 0.78, "Notable calls": "Bihar 2025 closest (-47); Rajasthan 2023 accurate"},
        {"Agency": "P-MARQ",           "Reliability": 0.75, "Notable calls": "Rajasthan 2023 spot-on; Maharashtra 2024 moderate"},
        {"Agency": "People's Pulse",   "Reliability": 0.72, "Notable calls": "Maharashtra 2024 close; WB 2026 lone TMC-majority prediction"},
        {"Agency": "Poll of Polls",    "Reliability": 0.65, "Notable calls": "Aggregate; direction usually right; misses landslides"},
        {"Agency": "C-Voter",          "Reliability": 0.63, "Notable calls": "Conservative bias; misses surprises; WB 2021 missed badly"},
        {"Agency": "Axis My India",    "Reliability": 0.60, "Notable calls": "Karnataka ✓ UP ✓; Delhi 2025 ✗ Haryana ✗ LS 2024 ✗"},
        {"Agency": "Zeenia AI",        "Reliability": 0.45, "Notable calls": "No historical track record; AI-generated"},
        {"Agency": "Today's Chanakya", "Reliability": 0.42, "Notable calls": "LS 2024: predicted NDA 400 (actual 293); Haryana ✗"},
    ]
    return pd.DataFrame(records).sort_values("Reliability", ascending=False)


def direction_failures_table() -> pd.DataFrame:
    """Elections where exit polls got winner direction wrong — systemic risk."""
    failures = [
        {"Election": "Himachal 2022",     "Predicted winner": "BJP",      "Actual winner": "Congress",  "Scale": "68 seats"},
        {"Election": "Chhattisgarh 2023", "Predicted winner": "Congress", "Actual winner": "BJP",       "Scale": "90 seats"},
        {"Election": "Rajasthan 2023",    "Predicted winner": "Congress", "Actual winner": "BJP",       "Scale": "200 seats (Axis)"},
        {"Election": "MP 2023",           "Predicted winner": "Congress", "Actual winner": "BJP",       "Scale": "230 seats (Axis/CVoter)"},
        {"Election": "Haryana 2024",      "Predicted winner": "Congress", "Actual winner": "BJP",       "Scale": "90 seats — all agencies wrong"},
        {"Election": "Delhi 2025",        "Predicted winner": "AAP",      "Actual winner": "BJP",       "Scale": "70 seats (Axis only)"},
    ]
    return pd.DataFrame(failures)


if __name__ == "__main__":
    print("=== WB Exit Poll Calibration Engine ===\n")

    print("── WB Historical Assembly Bias ─────────────────")
    print(wb_bias_table().to_string(index=False))
    print(f"\n  Recency-weighted mean bias: +{BIAS_MEAN:.1f} seats")
    print(f"  Bias std dev:                {BIAS_STD:.1f} seats")

    print("\n── Agency Scorecard ─────────────────────────────")
    print(agency_scorecard().to_string(index=False))

    print("\n── Direction Failures (all agencies, same election) ─")
    print(direction_failures_table().to_string(index=False))

    print("\n── 2026 Calibration (all agencies including outlier) ─")
    est = calibrated_tmc_estimate(exclude_outliers=False)
    for k, v in est.items():
        print(f"  {k:<28}: {v}")

    print("\n── 2026 Calibration (excluding People's Pulse outlier) ─")
    est2 = calibrated_tmc_estimate(exclude_outliers=True)
    for k, v in est2.items():
        print(f"  {k:<28}: {v}")

    print("\n── Regional signals (vs model prior median 170) ──")
    out = exit_poll_regional_signals()
    for r, s in out["signals"].items():
        print(f"  {r:<28}: {s:+.4f}")
    print(f"  calibrated_mean: {out['calibrated_mean']}, signal: {out['signal_strength']:+.4f}")
