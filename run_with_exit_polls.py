"""
Exit poll calibration encoded as a distributional observation at the
SCENARIO level — not per-constituency.

The calibrated EP distribution N(μ_cal, σ_cal) gives its own scenario
probabilities. These are blended with the structural model's MC output
using a credibility-weighted mixture. The MC seat median is unchanged;
only the scenario probability mass shifts.

python3 run_with_exit_polls.py
"""
import sys, numpy as np
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.bayesian import (
    build_priors, apply_sir_adjustment, apply_organizational_factors,
    bayesian_update, monte_carlo, DATA_DIR,
)
from pipeline.exit_poll_calibration import (
    load_2026_polls, raw_consensus_tmc, BIAS_MEAN, BIAS_STD,
)

TOTAL_SEATS      = 294
CONTEXT_DISCOUNT = 0.45
CONTEXT_SIGMA    = 20.0        # ±20 seat epistemic uncertainty on 2026 applicability
EP_BLEND_WEIGHT  = 0.15        # exit poll gets 15% credibility; structural gets 85%

# ── Structural baseline ───────────────────────────────────────────────────────
priors = build_priors(DATA_DIR)
adj    = apply_sir_adjustment(priors, DATA_DIR)
adj    = apply_organizational_factors(adj)
post   = bayesian_update(adj, {})
mc     = monte_carlo(post)

# ── Exit poll calibrated distribution ────────────────────────────────────────
df_polls               = load_2026_polls()
raw_mean, raw_spread   = raw_consensus_tmc(df_polls, exclude_outliers=True)
discounted_bias_mean   = BIAS_MEAN * CONTEXT_DISCOUNT          # +29.5
discounted_bias_std    = BIAS_STD  * CONTEXT_DISCOUNT          # +10.3
calibrated_mean        = raw_mean + discounted_bias_mean        # ~161
sigma_ep               = float(np.sqrt(
    raw_spread**2 + discounted_bias_std**2 + CONTEXT_SIGMA**2
))                                                              # ~22.7 seats

ep_dist = norm(loc=calibrated_mean, scale=sigma_ep)

# ── Scenario definitions (seat thresholds) ────────────────────────────────────
# BJP Surge:  TMC < 148 (BJP majority territory)
# Status Quo: 148 ≤ TMC ≤ 195
# TMC Wave:   TMC > 195
SURGE_CEIL   = 148
WAVE_FLOOR   = 195

# Structural model scenario probabilities (from MC distribution)
mc_tmc_samples = None  # We use percentile proxies below
p_surge_mc   = mc["p_bjp_win"] + mc["p_hung"]  # proxy: P(TMC < 148)
p_wave_mc    = float(np.clip(1 - mc["p_tmc_win"] - mc["p_hung"] - mc["p_bjp_win"] + mc["p_tmc_win"], 0, 1))
# Better: compute directly from percentile distribution
# Use normal approximation to MC distribution
mc_mu    = mc["tmc_p50"]
mc_sigma = (mc["tmc_p95"] - mc["tmc_p5"]) / (2 * 1.645)  # 90% CI → σ
mc_dist  = norm(loc=mc_mu, scale=mc_sigma)

p_surge_mc   = float(mc_dist.cdf(SURGE_CEIL))
p_statusq_mc = float(mc_dist.cdf(WAVE_FLOOR) - mc_dist.cdf(SURGE_CEIL))
p_wave_mc    = float(1 - mc_dist.cdf(WAVE_FLOOR))

# Exit poll scenario probabilities from calibrated distribution
p_surge_ep   = float(ep_dist.cdf(SURGE_CEIL))
p_statusq_ep = float(ep_dist.cdf(WAVE_FLOOR) - ep_dist.cdf(SURGE_CEIL))
p_wave_ep    = float(1 - ep_dist.cdf(WAVE_FLOOR))

# Blended scenario probabilities (credibility-weighted mixture)
W_S = 1 - EP_BLEND_WEIGHT
W_E = EP_BLEND_WEIGHT

p_surge_blend   = W_S * p_surge_mc   + W_E * p_surge_ep
p_statusq_blend = W_S * p_statusq_mc + W_E * p_statusq_ep
p_wave_blend    = W_S * p_wave_mc    + W_E * p_wave_ep

# CI: convolve structural MC distribution with EP uncertainty (adds noise to tails)
# Combined σ = sqrt(σ_mc² + W_E² × σ_ep²) — EP blend adds noise to the extremes
sigma_combined = float(np.sqrt(mc_sigma**2 + (W_E * sigma_ep)**2))
combined_dist  = norm(loc=mc_mu, scale=sigma_combined)

p5_comb  = int(combined_dist.ppf(0.05))
p25_comb = int(combined_dist.ppf(0.25))
p50_comb = int(combined_dist.ppf(0.50))
p75_comb = int(combined_dist.ppf(0.75))
p95_comb = int(combined_dist.ppf(0.95))

# ── Print ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("WB 2026 — EXIT POLL: DISTRIBUTIONAL SCENARIO UPDATE")
print("="*65)

print(f"""
CALIBRATED EXIT POLL DISTRIBUTION
  Raw consensus (down-weighted BJP-lean agencies):   {raw_mean:.1f} seats
  Discounted bias correction (+{discounted_bias_mean:.1f}):            {calibrated_mean:.1f} seats
  Combined uncertainty (σ_ep):                      ±{sigma_ep:.1f} seats
  → EP distribution: N({calibrated_mean:.0f}, {sigma_ep:.0f}) in seat space
  EP credibility weight in blend:                   {EP_BLEND_WEIGHT:.0%}
  Structural model weight:                          {W_S:.0%}
""")

print(f"SCENARIO PROBABILITIES")
print(f"{'─'*65}")
print(f"{'Scenario':<28} {'Structural':>12} {'EP only':>10} {'Blended':>10} {'Δ':>6}")
print(f"{'─'*65}")
for name, p_mc, p_ep, p_bl in [
    ("🔴 BJP Surge  (<148 TMC)",   p_surge_mc,   p_surge_ep,   p_surge_blend),
    ("🟡 Status Quo (148–195)",    p_statusq_mc, p_statusq_ep, p_statusq_blend),
    ("🟢 TMC Wave   (>195)",       p_wave_mc,    p_wave_ep,    p_wave_blend),
]:
    delta = p_bl - p_mc
    print(f"  {name:<26} {p_mc:>11.1%} {p_ep:>9.1%} {p_bl:>9.1%} {delta:>+5.1%}")
print(f"{'─'*65}")

print(f"""
CI COMPARISON (structural MC vs blended)
  {'Percentile':<20} {'Structural':>12} {'Blended':>12} {'Δ':>6}""")
for label, bv, ev in [
    ("p5  (floor)",   mc["tmc_p5"],  p5_comb),
    ("p25 (likely)",  mc["tmc_p25"], p25_comb),
    ("p50 (median)",  mc["tmc_p50"], p50_comb),
    ("p75 (likely)",  mc["tmc_p75"], p75_comb),
    ("p95 (ceiling)", mc["tmc_p95"], p95_comb),
]:
    print(f"  {label:<20} {bv:>12} {ev:>12} {ev-bv:>+6}")
print(f"  {'─'*52}")
print(f"  {'90% CI width':<20} {mc['tmc_p95']-mc['tmc_p5']:>12} {p95_comb-p5_comb:>12} {(p95_comb-p5_comb)-(mc['tmc_p95']-mc['tmc_p5']):>+6}")

print(f"""
INTERPRETATION
  Calibrated EP estimate ({calibrated_mean:.0f} seats) is {mc_mu - calibrated_mean:.0f} below structural median ({mc_mu})
  At {EP_BLEND_WEIGHT:.0%} EP credibility: blended median stays at {p50_comb} seats.
  EP's wider uncertainty (σ={sigma_ep:.0f}) adds noise to tails: CI widens by {(p95_comb-p5_comb)-(mc['tmc_p95']-mc['tmc_p5'])} seats.
  BJP Surge probability: {p_surge_mc:.1%} → {p_surge_blend:.1%} ({p_surge_blend-p_surge_mc:+.1%})
  TMC Wave probability:  {p_wave_mc:.1%} → {p_wave_blend:.1%} ({p_wave_blend-p_wave_mc:+.1%})
  Blended scenarios sum: {p_surge_blend+p_statusq_blend+p_wave_blend:.1%}
""")
