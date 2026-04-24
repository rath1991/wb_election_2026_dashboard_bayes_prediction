"""
WB 2026 Elections Prediction Dashboard
Streamlit app — 6 pages: Headline Forecast, Regional Tracker, BJP Pathway, Scenario Analysis, News Feed, Methodology
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="WB 2026 Elections Dashboard",
    layout="wide",
    page_icon="🗳️",
    initial_sidebar_state="expanded",
)

# ─── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main .block-container { padding-top: 1.5rem; }
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .sir-banner {
    background: linear-gradient(90deg, #7c3aed22, #1e1e2e);
    border-left: 3px solid #7c3aed;
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
  }
  .condition-green  { color: #22c55e; font-weight: 600; }
  .condition-yellow { color: #f59e0b; font-weight: 600; }
  .condition-red    { color: #ef4444; font-weight: 600; }
  .forecast-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 1.2rem;
    border: 1px solid #2d2d4e;
  }
</style>
""", unsafe_allow_html=True)

SIR_BANNER = (
    "⚠️  <b>SIR Impact:</b> 91 lakh voters deleted (12% of electorate), "
    "concentrated in Muslim-majority / TMC stronghold districts "
    "(Samserganj 74k, Murshidabad 460k, Kolkata 700k, N24P 330k, Malda 240k). "
    "This creates <b>systematic downward bias</b> in TMC estimates — confidence bands are intentionally wider."
)

# ─── DB helpers ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _db():
    from pipeline.db import get_conn
    return get_conn()


def _seed_baseline_if_empty():
    """Seed a neutral baseline forecast on first load so the dashboard always shows numbers."""
    from pipeline.db import get_latest_forecast, upsert_forecast, upsert_regional_signals
    from pipeline.bayesian import run_full_pipeline
    conn = _db()
    if get_latest_forecast(conn) is not None:
        return
    signals = {}  # neutral — no news signals
    forecast, _ = run_full_pipeline(signals)
    forecast["prev_tmc_p50"] = None
    upsert_forecast(conn, date.today(), forecast)
    upsert_regional_signals(conn, date.today(), {})


_seed_baseline_if_empty()


@st.cache_data(ttl=1800, show_spinner=False)
def _forecast_latest():
    from pipeline.db import get_latest_forecast
    return get_latest_forecast(_db())


@st.cache_data(ttl=1800, show_spinner=False)
def _forecast_history():
    from pipeline.db import get_forecast_history
    return get_forecast_history(_db(), days=30)


@st.cache_data(ttl=1800, show_spinner=False)
def _regional_signals():
    from pipeline.db import get_regional_signals
    return get_regional_signals(_db(), days=7)


@st.cache_data(ttl=1800, show_spinner=False)
def _bjp_conditions():
    from pipeline.db import get_bjp_conditions
    return get_bjp_conditions(_db())


@st.cache_data(ttl=1800, show_spinner=False)
def _articles(min_cred):
    from pipeline.db import get_recent_articles
    return get_recent_articles(_db(), days=2, min_credibility=min_cred)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗳️ WB 2026")
    st.caption(f"Model updated: {date.today()}")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Headline Forecast", "Regional Tracker", "BJP Pathway", "Scenario Analysis", "News Feed", "Methodology"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Predictions are probabilistic estimates based on historical data, "
        "2024 LS trends, SIR voter deletion data, and daily news signals. "
        "High uncertainty — do not treat as ground truth."
    )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: HEADLINE FORECAST
# ─────────────────────────────────────────────────────────────────────────────
if page == "Headline Forecast":
    st.title("West Bengal 2026 — Seat Forecast")
    st.markdown(f'<div class="sir-banner">{SIR_BANNER}</div>', unsafe_allow_html=True)

    forecast = _forecast_latest()

    if not forecast:
        st.stop()

    # Delta calculation
    prev = forecast.get("prev_tmc_p50")
    delta_str, delta_val = None, 0
    if prev:
        delta_val = forecast["tmc_p50"] - int(prev)
        delta_str = f"{'▲ +' if delta_val >= 0 else '▼ '}{abs(delta_val)} vs yesterday"

    # ── Top metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p25 = forecast.get("tmc_p25", forecast["tmc_p5"])
        p75 = forecast.get("tmc_p75", forecast["tmc_p95"])
        st.metric(
            "TMC Seats (median)",
            forecast["tmc_p50"],
            delta_str,
            delta_color="normal" if delta_val >= 0 else "inverse",
        )
        st.caption(f"Likely: **{p25}–{p75}** · 90% CI: {forecast['tmc_p5']}–{forecast['tmc_p95']}")
    with c2:
        bjp_p25 = forecast.get("bjp_p25", forecast["bjp_p5"])
        bjp_p75 = forecast.get("bjp_p75", forecast["bjp_p95"])
        st.metric("BJP Seats (median)", forecast["bjp_p50"])
        st.caption(f"Likely: **{bjp_p25}–{bjp_p75}** · 90% CI: {forecast['bjp_p5']}–{forecast['bjp_p95']}")
    with c3:
        st.metric("P(TMC Majority)", f"{forecast['p_tmc_win']:.0%}")
        st.caption(f"P(Hung): {forecast['p_hung']:.0%} | P(BJP): {forecast['p_bjp_win']:.0%}")
    with c4:
        band = forecast.get("sir_uncertainty_band", 0)
        st.metric("SIR Uncertainty", f"{band} seats wide")
        st.caption("CI width attributable to voter roll deletions")

    st.markdown("")

    # ── Gauge + Probability donut ────────────────────────────────────────────
    col_gauge, col_donut = st.columns([3, 2])

    with col_gauge:
        tmc_mid = forecast["tmc_p50"]
        ref = int(prev) if prev else tmc_mid
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=tmc_mid,
            delta={"reference": ref, "increasing": {"color": "#22c55e"}, "decreasing": {"color": "#ef4444"}},
            title={"text": "TMC Seats (median)", "font": {"size": 16}},
            number={"font": {"size": 48}},
            gauge={
                "axis": {"range": [50, 250], "tickwidth": 1},
                "bar": {"color": "#22c55e", "thickness": 0.25},
                "bgcolor": "#1e1e2e",
                "borderwidth": 0,
                "steps": [
                    {"range": [50, 120],  "color": "rgba(239,68,68,0.08)"},
                    {"range": [120, 148], "color": "rgba(245,158,11,0.08)"},
                    {"range": [148, 250], "color": "rgba(34,197,94,0.08)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 2},
                    "thickness": 0.8,
                    "value": 148,
                },
            },
        ))
        p25 = forecast.get("tmc_p25", forecast["tmc_p5"])
        p75 = forecast.get("tmc_p75", forecast["tmc_p95"])
        fig.add_annotation(
            text=f"Likely range: {p25}–{p75}  ·  90% CI: {forecast['tmc_p5']}–{forecast['tmc_p95']}",
            xref="paper", yref="paper", x=0.5, y=-0.05,
            showarrow=False, font={"size": 12, "color": "#94a3b8"},
        )
        fig.update_layout(height=320, margin=dict(t=30, b=40, l=20, r=20),
                          paper_bgcolor="#0e1117", font={"color": "#f8fafc"})
        st.plotly_chart(fig, use_container_width=True, key="gauge_tmc")

    with col_donut:
        probs = [
            forecast["p_tmc_win"] * 100,
            forecast["p_hung"] * 100,
            forecast["p_bjp_win"] * 100,
        ]
        labels = ["TMC Majority", "Hung Assembly", "BJP Majority"]
        colors = ["#22c55e", "#f59e0b", "#ef4444"]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=probs, hole=0.60,
            marker={"colors": colors, "line": {"color": "#0e1117", "width": 2}},
            textinfo="label+percent",
            textfont={"size": 12},
            rotation=90,
        ))
        fig2.update_layout(
            title={"text": "Win Probability", "font": {"size": 14}},
            height=320, showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="#0e1117", font={"color": "#f8fafc"},
        )
        st.plotly_chart(fig2, use_container_width=True, key="donut_probs")

    # ── 30-day trend ─────────────────────────────────────────────────────────
    history = _forecast_history()
    if not history.empty and len(history) > 1:
        st.subheader("30-Day Forecast Trend")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p95"],
            name="TMC p95", line=dict(color="rgba(34,197,94,0.3)", width=1),
            fill=None,
        ))
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p5"],
            name="TMC p5", line=dict(color="rgba(34,197,94,0.3)", width=1),
            fill="tonexty", fillcolor="rgba(34,197,94,0.08)",
        ))
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p50"],
            name="TMC median",
            line=dict(color="#22c55e", width=2.5),
        ))
        if "bjp_p50" in history.columns:
            fig3.add_trace(go.Scatter(
                x=history["date"], y=history["bjp_p50"],
                name="BJP median",
                line=dict(color="#ef4444", width=2, dash="dot"),
            ))
        fig3.add_hline(y=148, line_dash="dash", line_color="#94a3b8",
                       annotation_text="Majority (148)", annotation_position="right")
        fig3.update_layout(
            height=300, yaxis_range=[80, 240],
            xaxis_title="Date", yaxis_title="Seats",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font={"color": "#f8fafc"}, legend={"bgcolor": "#1e1e2e"},
            margin=dict(t=10, b=40, l=40, r=80),
        )
        st.plotly_chart(fig3, use_container_width=True, key="history_line")

        # P(TMC win) trend
        if "p_tmc_win" in history.columns:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=history["date"],
                y=(history["p_tmc_win"] * 100).round(1),
                name="P(TMC majority)",
                line=dict(color="#22c55e", width=2),
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.08)",
            ))
            fig4.add_trace(go.Scatter(
                x=history["date"],
                y=(history["p_bjp_win"] * 100).round(1),
                name="P(BJP majority)",
                line=dict(color="#ef4444", width=2),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.08)",
            ))
            fig4.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
            fig4.update_layout(
                height=200, yaxis_range=[0, 100],
                xaxis_title="Date", yaxis_title="Probability %",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font={"color": "#f8fafc"}, legend={"bgcolor": "#1e1e2e"},
                margin=dict(t=10, b=40, l=40, r=10),
            )
            st.plotly_chart(fig4, use_container_width=True, key="regional_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: REGIONAL TRACKER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Regional Tracker":
    st.title("Regional Signal Tracker")
    st.caption("Green = news signals favor TMC, Red = signals favor BJP. Based on Claude-filtered articles.")

    signals_df = _regional_signals()

    REGIONS = ["north_bengal", "jangalmahal", "medinipur", "urban_kolkata", "south_bengal_rural"]
    LABELS = {
        "north_bengal": "North Bengal",
        "jangalmahal": "Jangalmahal",
        "medinipur": "Medinipur",
        "urban_kolkata": "Urban Kolkata",
        "south_bengal_rural": "South Bengal Rural",
    }

    # 2024 LS baseline by region (averaged from ls2024_regional.csv)
    LS2024 = {
        "north_bengal":       {"tmc": 0.44, "bjp": 0.43, "note": "BJP lost Cooch Behar in 2024; TMC recovered"},
        "jangalmahal":        {"tmc": 0.44, "bjp": 0.47, "note": "BJP retained Purulia/Bishnupur only"},
        "medinipur":          {"tmc": 0.49, "bjp": 0.40, "note": "TMC won Medinipur LS 2024"},
        "urban_kolkata":      {"tmc": 0.51, "bjp": 0.37, "note": "TMC stronghold, BJP weakened"},
        "south_bengal_rural": {"tmc": 0.52, "bjp": 0.31, "note": "TMC dominant; SIR hits hardest here"},
    }
    SIR_IMPACT = {
        "north_bengal": "medium",
        "jangalmahal": "low",
        "medinipur": "low-medium",
        "urban_kolkata": "high",
        "south_bengal_rural": "critical",
    }

    if not signals_df.empty:
        # Heatmap
        dates = sorted(signals_df["date"].unique())[-7:]
        z, text = [], []
        for region in REGIONS:
            row, row_text = [], []
            for d in dates:
                vals = signals_df[(signals_df["region"] == region) & (signals_df["date"] == d)]["signal_strength"].values
                v = float(vals[0]) if len(vals) > 0 else 0.0
                row.append(v)
                row_text.append(f"{v:+.2f}")
            z.append(row)
            text.append(row_text)

        fig = go.Figure(go.Heatmap(
            z=z, x=dates, y=[LABELS[r] for r in REGIONS],
            colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#22c55e"]],
            zmid=0, zmin=-1, zmax=1,
            text=text, texttemplate="%{text}",
            colorbar={"title": "Signal", "tickvals": [-1, 0, 1], "ticktext": ["BJP↑", "Neutral", "TMC↑"]},
        ))
        fig.update_layout(
            title="7-Day Signal Heatmap", height=300,
            margin=dict(t=40, b=10),
            paper_bgcolor="#0e1117", font={"color": "#f8fafc"},
        )
        st.plotly_chart(fig, use_container_width=True, key="bjp_pathway_bar")
    else:
        st.info("No regional signal data yet. Run pipeline first.")

    # Regional cards
    st.subheader("Region Detail — vs 2024 LS Baseline")
    cols = st.columns(len(REGIONS))
    for i, region in enumerate(REGIONS):
        ls = LS2024[region]
        sir = SIR_IMPACT[region]
        sir_colors = {"low": "#22c55e", "low-medium": "#84cc16", "medium": "#f59e0b", "high": "#f97316", "critical": "#ef4444"}

        latest_signal = 0.0
        if not signals_df.empty:
            sub = signals_df[signals_df["region"] == region].sort_values("date")
            if not sub.empty:
                latest_signal = float(sub.iloc[-1]["signal_strength"])

        with cols[i]:
            st.markdown(f"**{LABELS[region]}**")
            st.markdown(
                f'<span style="color:{sir_colors.get(sir,"#94a3b8")}">SIR: {sir.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"2024 LS: TMC {ls['tmc']:.0%} / BJP {ls['bjp']:.0%}")
            color = "normal" if latest_signal >= 0 else "inverse"
            st.metric("Today signal", f"{latest_signal:+.2f}", delta_color=color)
            st.caption(ls["note"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: BJP PATHWAY SCORECARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "BJP Pathway":
    st.title("BJP Win Pathway Scorecard")
    st.markdown(
        "All **7 conditions** must align simultaneously for BJP to reach majority. "
        "Each missing condition geometrically reduces the probability."
    )

    CONDITIONS_META = {
        "north_bengal_sweep_35plus": {
            "label": "North Bengal sweep (35+ of ~54 seats)",
            "desc": "BJP's 2019 base. Must dominate border districts, Matua belt, hill seats. "
                    "2024 LS: TMC recovered Cooch Behar — trend has reversed.",
            "ls2024": "BJP lost ground. TMC won Cooch Behar (defeated Nisith Pramanik).",
        },
        "jangalmahal_hold_18plus": {
            "label": "Jangalmahal hold (18+ of ~25 seats)",
            "desc": "Tribal + anti-incumbency zone. BJP won majority here in 2021. "
                    "2024 LS: BJP retained only Purulia and Bishnupur.",
            "ls2024": "BJP retained 2/4 Jangalmahal LS seats. Weakened from 2019.",
        },
        "medinipur_majority": {
            "label": "Medinipur belt majority",
            "desc": "Symbolic and organizational battleground. TMC won Medinipur LS seat in 2024.",
            "ls2024": "TMC won Medinipur constituency in 2024 LS — BJP reversal needed.",
        },
        "urban_kolkata_gain_10plus": {
            "label": "Urban Kolkata net +10 gain",
            "desc": "Governance/corruption narrative must convert Kolkata middle class. "
                    "TMC historically dominant here.",
            "ls2024": "TMC won all 4 Kolkata-area LS seats with increased margin.",
        },
        "minority_fragmentation": {
            "label": "Minority vote fragmentation",
            "desc": "Left/ISF/Congress draws 15%+ in key Muslim-majority seats, "
                    "splitting the anti-BJP vote. Without SIR, this is the only realistic path.",
            "ls2024": "Minority consolidation held for TMC in 2024 LS — fragmentation did NOT occur.",
        },
        "sir_voter_suppression_effective": {
            "label": "SIR voter suppression effective",
            "desc": "91 lakh deleted voters (heavily Muslim/minority) cannot vote. "
                    "If courts intervene or voters find workarounds, this condition fails.",
            "ls2024": "SIR implemented post-2024 LS. No 2024 baseline — pure 2026 structural factor.",
        },
        "anti_incumbency_national": {
            "label": "National anti-incumbency wave",
            "desc": "Central BJP govt narrative overrides state TMC incumbency advantage. "
                    "2024 LS showed TMC resurgence at national level.",
            "ls2024": "BJP national narrative failed in 2024 WB — TMC won 29/42 seats.",
        },
    }

    conditions = _bjp_conditions()
    conditions_by_key = {c["condition_key"]: c for c in conditions} if conditions else {}
    green = [k for k, v in conditions_by_key.items() if v.get("status") == "green"]
    yellow = [k for k, v in conditions_by_key.items() if v.get("status") == "yellow"]

    st.progress(len(green) / len(CONDITIONS_META), text=f"{len(green)}/7 conditions met (🟢 {len(green)} green, 🟡 {len(yellow)} yellow)")
    st.markdown("---")

    for key, meta in CONDITIONS_META.items():
        data = conditions_by_key.get(key, {"status": "red", "confidence": 0.05, "evidence_json": "[]"})
        status = data.get("status", "red")
        confidence = float(data.get("confidence", 0.05))
        icons = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
        icon = icons.get(status, "⚫")

        with st.expander(f"{icon} {meta['label']}   —   confidence: **{confidence:.0%}**"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(meta["desc"])
                st.caption(f"📊 2024 LS context: {meta['ls2024']}")
            with col_b:
                cmap = {"green": "#22c55e", "yellow": "#f59e0b", "red": "#ef4444"}
                fig_conf = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={"text": "Confidence %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": cmap.get(status, "#6b7280")},
                        "bgcolor": "#1e1e2e",
                    },
                    number={"suffix": "%"},
                ))
                fig_conf.update_layout(height=150, margin=dict(t=30, b=10, l=10, r=10),
                                       paper_bgcolor="#0e1117", font={"color": "#f8fafc"})
                st.plotly_chart(fig_conf, use_container_width=True, key=f"bjp_cond_conf_{key}")

            try:
                snippets = json.loads(data.get("evidence_json", "[]"))
                if snippets:
                    st.markdown("**Evidence from today's news:**")
                    for s in snippets[:3]:
                        st.markdown(f"› {s.get('headline', '')} *(cred: {s.get('credibility', 0):.1f})*")
                else:
                    st.caption("No direct evidence in recent news.")
            except Exception:
                pass

    st.markdown("---")
    n = len(green)
    verdict = {
        0: "🔴 BJP pathway essentially closed.",
        1: "🔴 BJP has no realistic path.",
        2: "🔴 BJP at 2/7 — very unlikely.",
        3: "🟡 BJP at 3/7 — hung possible, majority very unlikely.",
        4: "🟡 BJP at 4/7 — competitive race, hung likely.",
        5: "🟠 BJP at 5/7 — close race, majority possible.",
        6: "🟠 BJP at 6/7 — BJP favored if last condition breaks their way.",
        7: "🟢 BJP at 7/7 — BJP majority likely.",
    }
    st.info(verdict.get(n, ""))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Scenario Analysis":
    st.title("Scenario Analysis")
    st.caption("Three structural scenarios based on how key variables resolve.")

    forecast = _forecast_latest()

    SCENARIOS = {
        "TMC Wave": {
            "color": "#22c55e",
            "tmc_low": 195, "tmc_high": 220, "tmc_mid": 208,
            "bjp_low": 58, "bjp_high": 80, "bjp_mid": 70,
            "prob": "20–30%",
            "sir": "SIR rolls partially/fully restored by court order",
            "conditions": [
                "Court orders restore significant voter names",
                "Minority consolidation >85% behind TMC",
                "North Bengal partial recovery by TMC",
                "BJP organizational failure on booth management",
            ],
        },
        "Status Quo": {
            "color": "#f59e0b",
            "tmc_low": 165, "tmc_high": 195, "tmc_mid": 180,
            "bjp_low": 82, "bjp_high": 110, "bjp_mid": 96,
            "prob": "45–55%",
            "sir": "SIR as-is (12% deletion, concentrated in minority areas)",
            "conditions": [
                "SIR impact as-is — minority voters largely absent",
                "Minority vote broadly consolidates for TMC",
                "BJP holds North Bengal + Jangalmahal base",
                "No major new swing factor emerges",
            ],
        },
        "BJP Surge": {
            "color": "#ef4444",
            "tmc_low": 128, "tmc_high": 155, "tmc_mid": 142,
            "bjp_low": 118, "bjp_high": 148, "bjp_mid": 133,
            "prob": "15–25%",
            "sir": "Full SIR suppression + minority fragmentation",
            "conditions": [
                "SIR fully effective + Left/ISF draws 15%+ in Muslim seats",
                "North Bengal sweep: BJP wins 33-35+ seats",
                "Jangalmahal hold: 18+ of 25",
                "Urban anti-incumbency converts 10+ seats",
                "Broad anti-TMC wave driven by governance issues",
            ],
        },
    }

    cols = st.columns(3)
    for i, (name, s) in enumerate(SCENARIOS.items()):
        with cols[i]:
            st.markdown(f"### {name}")
            st.markdown(f"**Probability: {s['prob']}**")
            st.markdown(f"*{s['sir']}*")

            fig = go.Figure()
            parties = ["TMC", "BJP", "Left/Others"]
            mids = [s["tmc_mid"], s["bjp_mid"], 16]
            errs = [
                (s["tmc_high"] - s["tmc_low"]) // 2,
                (s["bjp_high"] - s["bjp_low"]) // 2,
                3,
            ]
            fig.add_trace(go.Bar(
                x=parties, y=mids,
                error_y=dict(type="data", array=errs, visible=True, color="#94a3b8"),
                marker_color=[s["color"], "#6b7280", "#3b82f6"],
                text=[f"{m}±{e}" for m, e in zip(mids, errs)],
                textposition="outside",
            ))
            fig.add_hline(y=148, line_dash="dash", line_color="#94a3b8",
                          annotation_text="Majority", annotation_position="right")
            fig.update_layout(
                height=280, yaxis_range=[0, 240],
                showlegend=False,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font={"color": "#f8fafc"}, margin=dict(t=10, b=10, l=10, r=60),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"scenario_bar_{i}")

            st.markdown(f"TMC: **{s['tmc_low']}–{s['tmc_high']}**  \nBJP: **{s['bjp_low']}–{s['bjp_high']}**")
            st.markdown("**Key requirements:**")
            for cond in s["conditions"]:
                st.markdown(f"• {cond}")

    st.markdown("---")
    st.subheader("SIR Swing Analysis")
    st.markdown("""
| SIR Scenario | TMC Seat Impact | Mechanism |
|---|---|---|
| **Full roll restoration** (courts intervene) | **+15 to +25** seats | Deleted minority voters re-enfranchised in Murshidabad, Malda, N24P, Kolkata |
| **Partial restoration** (50% restored) | **+8 to +12** seats | Partial court relief, some constituencies restored |
| **As-is** (baseline) | **0** (reference) | 91L deleted voters absent; our baseline forecast assumes this |
| **As-is + fragmentation** | **-10 to -18** seats | Left/ISF splits vote in Muslim-majority seats on top of deletions |

> Samserganj alone: 74,000 deletions in a 95%-Muslim constituency (~25% of total electorate). If deletions hold, this seat almost certainly flips.
""")

    if forecast:
        col_x, col_y = st.columns(2)
        with col_x:
            st.metric("Current model TMC (p50)", forecast["tmc_p50"])
            st.caption("Assumes SIR as-is")
        with col_y:
            restored_est = forecast["tmc_p50"] + 18
            st.metric("Estimated TMC if rolls restored", f"~{restored_est}")
            st.caption("Rule-of-thumb: +15–25 if courts act")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: NEWS FEED
# ─────────────────────────────────────────────────────────────────────────────
elif page == "News Feed":
    st.title("Filtered News Feed")
    st.caption("Noise-filtered by Claude. Only non-noise articles above credibility threshold shown.")

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        region_filter = st.multiselect(
            "Filter by region",
            ["north_bengal", "jangalmahal", "medinipur", "urban_kolkata", "south_bengal_rural", "statewide"],
            default=[],
            placeholder="All regions",
        )
    with col_f2:
        signal_filter = st.multiselect(
            "Filter by signal",
            ["tmc_momentum", "bjp_momentum", "sir_impact", "minority_consolidation", "turnout_signal"],
            default=[],
            placeholder="All signals",
        )
    with col_f3:
        min_cred = st.slider("Min credibility", 0.0, 1.0, 0.5, 0.05)

    articles_df = _articles(min_cred)

    if articles_df.empty:
        st.info("No articles in the last 2 days. Check back after the next pipeline run (07:00 IST daily).")
        st.stop()

    # Apply region filter
    if region_filter:
        def has_region(tags):
            if not tags:
                return False
            return any(r in tags for r in region_filter)
        articles_df = articles_df[articles_df["region_tags"].apply(has_region)]

    # Apply signal filter
    if signal_filter:
        def has_signal(sig_tags):
            if not sig_tags:
                return False
            return any(abs(float(sig_tags.get(s, 0))) > 0.15 for s in signal_filter)
        articles_df = articles_df[articles_df["signal_tags"].apply(has_signal)]

    st.caption(f"Showing {len(articles_df)} articles")

    for _, row in articles_df.head(50).iterrows():
        cred = float(row.get("credibility_score", 0))
        cred_icon = "🟢" if cred >= 0.7 else "🟡" if cred >= 0.5 else "🔴"
        headline = row.get("headline", "No title")[:100]
        source = row.get("source", "")
        tags = row.get("region_tags") or []
        tags_str = " · ".join(tags) if tags else "—"

        with st.expander(f"{cred_icon} {headline}  |  *{source}*"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(row.get("body_snippet", ""))
                url = row.get("url", "")
                if url and url.startswith("http"):
                    st.markdown(f"[Read full article ↗]({url})")
            with col_b:
                st.metric("Credibility", f"{cred:.2f}")
                st.caption(f"Regions: {tags_str}")

                sigs = row.get("signal_tags") or {}
                relevant = {k: float(v) for k, v in sigs.items() if abs(float(v)) > 0.15}
                if relevant:
                    st.markdown("**Signals:**")
                    for k, v in sorted(relevant.items(), key=lambda x: -abs(x[1])):
                        direction = "▲" if v > 0 else "▼"
                        st.markdown(f"`{direction} {k}: {v:+.2f}`")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Methodology":
    st.title("How This Model Works")
    st.caption("A plain-English walkthrough of the Bayesian formulation, assumptions, and limitations.")

    st.markdown(f'<div class="sir-banner">{SIR_BANNER}</div>', unsafe_allow_html=True)

    # ── Why is this hard ──
    st.header("Why predicting WB 2026 is unusually difficult")
    st.markdown("""
Three compounding sources of uncertainty make this harder than a typical election forecast:

| Challenge | What it means |
|-----------|--------------|
| **No reliable polls** | No credible, regularly updated WB-specific polling exists. We're navigating without a GPS. |
| **SIR voter deletions** | ~91 lakh (9.1M) voters deleted from rolls during the Summary Revision. Deletions concentrate in Muslim-majority TMC strongholds — a systematic bias, not random noise. |
| **Media spin** | BJP's media ecosystem systematically overstates momentum. Raw headlines, taken at face value, would make BJP look far stronger than structural data supports. |
""")

    st.divider()

    # ── Step 1: Prior ──
    st.header("Step 1 — The Starting Point (Prior)")
    st.markdown("""
**What do we believe before reading today's news?**

We anchor on **historical win rates by region** — not vote shares. This matters because West Bengal uses
first-past-the-post (FPTP): a party can win a seat with 40% of votes if opposition is split.
Modelling "TMC gets >50% of votes" would be wrong. We model "TMC wins the seat."

We work in **logit space** — a transformation that lets probabilities live on an infinite number
line so Gaussian math applies naturally:

```
logit(p) = log(p / (1 − p))

logit(0.50) =  0.00  →  50% win (coin flip)
logit(0.73) = +1.00  →  73% win (TMC favored)
logit(0.27) = −1.00  →  27% win (TMC struggling)
logit(0.88) = +2.00  →  88% win (near-certain)
```

Our calibrated regional baselines:
""")

    prior_df = pd.DataFrame([
        {"Region": "North Bengal", "Seats": 54, "TMC Win Prior": "47%", "Logit": "+0.12", "Basis": "BJP inroads 2019–21; partial TMC recovery 2024 LS"},
        {"Region": "Jangalmahal", "Seats": 20, "TMC Win Prior": "40%", "Logit": "−0.41", "Basis": "BJP stronghold 2019–21; TMC partial comeback"},
        {"Region": "Medinipur", "Seats": 42, "TMC Win Prior": "66%", "Logit": "+0.66", "Basis": "Swings with state trend; TMC holds edge"},
        {"Region": "Urban Kolkata", "Seats": 51, "TMC Win Prior": "88%", "Logit": "+2.00", "Basis": "TMC near-monopoly; BJP urban gains reversed"},
        {"Region": "South Bengal Rural", "Seats": "~127", "TMC Win Prior": "58%", "Logit": "+0.32", "Basis": "Large, diverse; includes Matua belt + minority areas"},
    ])
    st.dataframe(prior_df, use_container_width=True, hide_index=True)

    st.markdown("""
The 2024 Lok Sabha result (TMC 45.8%, BJP 38.7%, TMC won 29/42 seats) is applied as a
**40% weight recency signal** on top of older assembly history (2021 → 50%, 2016 → 30%, 2011 → 20%).

Prior uncertainty: **σ = 1.20 logit units** — reflecting genuine election-to-election swing variance.
In probability terms, this says "TMC probably wins Urban Kolkata ~88% of the time, but the range is wide."
""")

    st.divider()

    # ── Step 2: SIR ──
    st.header("Step 2 — SIR Adjustment")
    st.markdown("""
**Accounting for the systematic effect of voter deletions.**

For each constituency we apply a deterministic downward logit shift:

```
logit_shift = −deletion_rate × minority_share × 0.80 × 4
```

Where:
- `deletion_rate` = fraction of voters deleted in that constituency
- `minority_share` = fraction of electorate that is Muslim/minority (proxies who was deleted)
- `0.80` = TMC lean factor — our estimate that 80% of deleted minority voters would have voted TMC
- `4` = logit scaling constant (converts vote-share shift to the right magnitude in log-odds)

**Effect:** Shifts the TMC point estimate *down* in minority-heavy seats AND widens the
uncertainty band (since we don't know exactly how deleted voters would have voted).
The "SIR band" shown on the dashboard (±27 seats) is this uncertainty made visible.

Extreme example — **Samserganj**: 74,000 deletions in a constituency with 95% Muslim electorate,
~25% of total voters gone. With this adjustment, TMC's win probability in Samserganj falls sharply.

**Key assumption:** 80% TMC lean among deleted voters. If the true lean is 60%, TMC's median seat
count shifts up by roughly 10–15 seats from the current baseline.
""")

    st.divider()

    # ── Step 3: Bayesian Update ──
    st.header("Step 3 — Daily Bayesian Update from News")
    st.markdown("""
**How today's headlines move the forecast — modestly.**

Each day the pipeline: fetches articles → Claude filters noise → Claude scores regional signals.
Each region gets a signal score in [−1, +1], where +1 = strong TMC momentum, −1 = strong BJP momentum.

We then do a **Normal-Normal conjugate Bayesian update**. Here's the full worked example for
Urban Kolkata with a mild positive signal of +0.35:
""")

    with st.expander("Worked example — Urban Kolkata, signal = +0.35", expanded=True):
        st.markdown("""
**Prior** (from Steps 1 & 2):
```
μ_prior = +1.90   (logit of ~87% TMC win rate after SIR adjustment)
σ_prior =  1.20   (our uncertainty — same as observation noise by design)
```

**Convert news signal to observation:**
```
obs = μ_prior + signal × 0.30   (0.30 = signal scaling constant)
obs = 1.90 + 0.35 × 0.30 = +1.995
τ   = 1.20                       (observation noise — news is genuinely noisy)
```

**Precision-weighted update:**
```
precision_prior = 1 / σ²  = 1 / 1.44 = 0.694
precision_news  = 1 / τ²  = 1 / 1.44 = 0.694

posterior_μ = (0.694 × 1.90 + 0.694 × 1.995) ÷ (0.694 + 0.694)
            = +1.947   ← barely moved from 1.90

posterior_σ = sqrt(1 / (0.694 + 0.694)) = 0.849
```

**Result:** Urban Kolkata logit moves from 1.90 → 1.947, i.e. 87.1% → 87.5% win probability.
A small nudge — exactly the right behaviour for one day of news.
""")

    st.markdown("""
**Why doesn't news move things more?**

Because `τ = σ`, both prior and news have equal precision, so each gets **50% weight**.
This is intentional — a single day of BJP-favoring headlines should not collapse the forecast.

| Observation noise τ | News weight | Logit shift from signal=+0.35 |
|---------------------|------------|-------------------------------|
| 0.60 (very trusted) | 80% | +0.28 |
| **1.20 (current)**  | **50%** | **+0.05** |
| 2.40 (very noisy)   | 20% | +0.02 |

We chose τ = 1.20 deliberately: nudge, don't swing.
""")

    st.divider()

    # ── Step 4: Monte Carlo ──
    st.header("Step 4 — Monte Carlo Simulation (10,000 runs)")
    st.markdown("""
**Translating per-constituency probabilities into a seat distribution.**

Elections have *correlated* uncertainty — if there's a surprise anti-incumbency wave, it hits
all TMC seats simultaneously, not independently. We model this with two layers of randomness:

```python
for each simulation k in 10,000:

    # One global shock — same for all 294 seats (the "wave")
    global_shock = Normal(0, σ_global=0.65)

    for each constituency i:
        # Local noise — candidate quality, local issues, caste arithmetic
        logit_i = Normal(μ_post_i, σ_constituency=0.80) + global_shock
        tmc_wins_i = (logit_i > 0)   # wins if logit > 0 ↔ probability > 50%

    tmc_seats[k] = sum(tmc_wins_i)
```

The **global shock is the key driver of wide confidence intervals.** A −1.5 global shock
shifts all constituencies simultaneously, collapsing TMC seats. A +1.5 shock does the opposite.
This is why the 90% CI spans 90–270 seats even though the median is stable at ~192.

**Output metrics:**

| Metric | Meaning |
|--------|---------|
| p50 (median) | TMC wins this many seats in the middle scenario |
| p25–p75 (likely range) | Middle 50% of simulated outcomes |
| p5–p95 (90% CI) | Full plausible range, excluding extreme 5% tails each side |
| P(TMC majority) | Fraction of simulations where TMC ≥ 148 seats |
| P(Hung assembly) | TMC < 148 but ≥ 100; no party majority |
| P(BJP majority) | BJP ≥ 148 seats |
| SIR uncertainty band | Extra seat range attributable specifically to SIR model uncertainty |
""")

    st.divider()

    # ── Assumptions & Limitations ──
    st.header("Assumptions & Limitations")

    st.markdown("""
#### Key assumptions
""")
    assumptions_df = pd.DataFrame([
        {"Assumption", "Value", "If wrong…"},
    ])
    st.markdown("""
| Assumption | Value | If wrong… |
|-----------|-------|-----------|
| SIR TMC lean factor | 80% | If 60% lean → TMC median ~+10–15 seats |
| Global swing σ | 0.65 logit | If lower → CI narrows; if higher → CI widens |
| News observation noise τ | 1.20 | If lower → news gets more daily weight |
| Left + Others fixed seats | ~15 | If Left collapses → BJP gains; if Left surges → TMC loses |
| 2024 LS weight on prior | 40% | Assembly elections often diverge from LS trends |
| No reliable polls available | — | If credible polls emerge, they should replace the prior |
""")

    st.markdown("""
#### What this model cannot do
- Predict black-swan events (EC postponement, major scandal 2 weeks before voting)
- Capture hyper-local caste dynamics within individual constituencies
- Account for TMC's ground operation quality (historically superior booth management)
- Replace actual polling data if and when it becomes available
- Model tactical voting between Left, ISF, and Congress in minority-heavy seats

#### Disclaimer
This is a probabilistic forecast tool for analytical purposes only. Election outcomes are
inherently uncertain. A **73.7% TMC win probability means BJP wins in roughly 1 in 4 simulated
elections** — not that BJP winning is impossible. Treat ranges as plausible scenarios, not predictions.

*Data sources: ECI historical results, 2024 Lok Sabha regional data, SIR deletion estimates
from ECI/The Wire reporting, daily news via GDELT + NewsAPI + YouTube.*
""")
