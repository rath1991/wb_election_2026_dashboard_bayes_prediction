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
import threading
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─── Background scheduler (runs pipeline daily at 07:00 IST inside web service) ─
def _start_scheduler():
    import time
    import pytz
    from datetime import datetime

    def _run():
        from scheduler import run_pipeline
        tz = pytz.timezone("Asia/Kolkata")
        last_run_date = None
        while True:
            now = datetime.now(tz)
            today = now.date()
            if last_run_date != today and now.hour >= 7:
                try:
                    run_pipeline()
                except Exception as e:
                    print(f"[scheduler] pipeline error: {e}")
                last_run_date = today
            time.sleep(300)  # check every 5 min

    t = threading.Thread(target=_run, daemon=True)
    t.start()

# Only start once per process (not on every Streamlit rerun)
if not st.session_state.get("_scheduler_started"):
    st.session_state["_scheduler_started"] = True
    _start_scheduler()

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


@st.cache_data(ttl=1800, show_spinner=False)
def _signals_with_articles():
    from pipeline.db import get_latest_regional_signals_with_articles
    return get_latest_regional_signals_with_articles(_db())


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗳️ WB 2026")
    st.caption(f"Model updated: {date.today()}")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Headline Forecast", "Regional Tracker", "BJP Pathway", "Scenario Analysis", "News Feed", "Manual Input", "Methodology"],
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
    st.markdown("---")
    if st.session_state.get("admin_authenticated"):
        st.caption("Admin access active")
        if st.button("📰 Run News Pipeline", use_container_width=True, type="primary"):
            st.session_state["run_pipeline_triggered"] = True
            st.rerun()
        if st.button("🔓 Sign out", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.rerun()
    else:
        if st.button("🔐 Admin Login", use_container_width=True):
            st.session_state["show_admin_login"] = True


# ─── Admin login dialog ──────────────────────────────────────────────────────
if st.session_state.get("show_admin_login") and not st.session_state.get("admin_authenticated"):
    with st.container(border=True):
        st.subheader("Admin Login")
        username = st.text_input("Username", key="admin_user_input")
        password = st.text_input("Password", type="password", key="admin_pass_input")
        col_login, col_cancel = st.columns(2)
        with col_login:
            if st.button("Login", type="primary", use_container_width=True):
                valid_user = os.environ.get("ADMIN_USERNAME", "admin")
                valid_pass = os.environ.get("ADMIN_PASSWORD", "wb2026")
                if username == valid_user and password == valid_pass:
                    st.session_state["admin_authenticated"] = True
                    st.session_state["show_admin_login"] = False
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state["show_admin_login"] = False
                st.rerun()

# ─── Pipeline trigger ────────────────────────────────────────────────────────
if st.session_state.get("run_pipeline_triggered") and st.session_state.get("admin_authenticated"):
    st.session_state["run_pipeline_triggered"] = False
    with st.status("Running pipeline — this takes 1–2 minutes...", expanded=True) as status:
        try:
            st.write("Fetching news from GDELT, NewsAPI.ai, YouTube...")
            from pipeline.fetchers import fetch_all
            articles = fetch_all(days_back=1)
            st.write(f"Fetched {len(articles)} articles.")

            st.write("Running Claude noise filter + signal extraction...")
            from pipeline.filter_extract import filter_and_extract, aggregate_regional_signals, extract_bjp_conditions
            enriched = filter_and_extract(articles)
            signal_count = sum(1 for a in enriched if not a.get("is_noise"))
            st.write(f"{signal_count} signal articles, {len(enriched)-signal_count} filtered as noise.")

            st.write("Storing articles...")
            from pipeline.db import (get_conn, insert_articles, upsert_forecast,
                                      upsert_regional_signals, upsert_bjp_conditions, get_latest_forecast)
            conn = get_conn()
            insert_articles(conn, enriched)

            st.write("Aggregating regional signals...")
            signals = aggregate_regional_signals(enriched)
            upsert_regional_signals(conn, date.today(), signals)

            conditions = extract_bjp_conditions(enriched)
            upsert_bjp_conditions(conn, date.today(), conditions)

            st.write("Running Bayesian model...")
            from pipeline.bayesian import run_full_pipeline
            prev = get_latest_forecast(conn)
            forecast_new, _ = run_full_pipeline(signals)
            forecast_new["prev_tmc_p50"] = prev["tmc_p50"] if prev else None
            upsert_forecast(conn, date.today(), forecast_new)

            status.update(label=f"Pipeline complete — TMC median: {forecast_new['tmc_p50']} seats", state="complete")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            status.update(label=f"Pipeline failed: {e}", state="error")
            st.exception(e)

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

    # ── TMC vs BJP trend ──────────────────────────────────────────────────────
    history = _forecast_history()
    st.subheader("TMC vs BJP — Seat Forecast Over Time")
    if history.empty or len(history) < 1:
        st.caption("Run the pipeline at least once to start building the trend chart.")
    else:
        fig3 = go.Figure()

        # TMC 90% CI band
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p95"],
            name="TMC 90% CI", line=dict(width=0), showlegend=False,
        ))
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p5"],
            name="TMC 90% CI", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(34,197,94,0.12)",
        ))
        # TMC likely range (p25-p75)
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p75"],
            name="TMC likely", line=dict(width=0), showlegend=False,
        ))
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p25"],
            name="TMC likely range", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(34,197,94,0.22)",
        ))
        # TMC median
        fig3.add_trace(go.Scatter(
            x=history["date"], y=history["tmc_p50"],
            name="TMC median",
            line=dict(color="#22c55e", width=3),
            mode="lines+markers", marker=dict(size=6),
        ))

        if "bjp_p50" in history.columns:
            # BJP 90% CI band
            fig3.add_trace(go.Scatter(
                x=history["date"], y=history["bjp_p95"],
                name="BJP 90% CI", line=dict(width=0), showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=history["date"], y=history["bjp_p5"],
                name="BJP 90% CI", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(239,68,68,0.10)",
            ))
            # BJP likely range
            fig3.add_trace(go.Scatter(
                x=history["date"], y=history["bjp_p75"],
                name="BJP likely", line=dict(width=0), showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=history["date"], y=history["bjp_p25"],
                name="BJP likely range", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(239,68,68,0.20)",
            ))
            # BJP median
            fig3.add_trace(go.Scatter(
                x=history["date"], y=history["bjp_p50"],
                name="BJP median",
                line=dict(color="#ef4444", width=3),
                mode="lines+markers", marker=dict(size=6),
            ))

        fig3.add_hline(y=148, line_dash="dash", line_color="#94a3b8",
                       annotation_text="Majority threshold (148)",
                       annotation_position="top right",
                       annotation_font_color="#94a3b8")
        fig3.update_layout(
            height=380,
            yaxis=dict(title="Seats", range=[40, 260], gridcolor="#2d2d4e"),
            xaxis=dict(title="Date", gridcolor="#2d2d4e"),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font={"color": "#f8fafc"},
            legend=dict(bgcolor="#1e1e2e", bordercolor="#2d2d4e", borderwidth=1,
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=40, b=40, l=40, r=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig3, use_container_width=True, key="history_line")

        # Summary delta table
        if len(history) >= 2:
            latest = history.iloc[-1]
            prev_row = history.iloc[-2]
            d1, d2 = st.columns(4)
            d1.metric("TMC median", int(latest["tmc_p50"]),
                      delta=int(latest["tmc_p50"] - prev_row["tmc_p50"]))
            d2.metric("BJP median", int(latest["bjp_p50"]),
                      delta=int(latest["bjp_p50"] - prev_row["bjp_p50"]),
                      delta_color="inverse")
            d3, d4 = st.columns(4), st.columns(4)
            st.caption(f"vs {prev_row['date'].strftime('%d %b') if hasattr(prev_row['date'], 'strftime') else prev_row['date']}")

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


    # ── Citation expander ──────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📰 What drove today's forecast update?", expanded=False):
        st.caption(
            "Articles fetched from NewsAPI.ai and YouTube, filtered by Claude (claude-sonnet-4-6). "
            "Credibility scores: 0.8+ = major outlets (NDTV, Telegraph India, The Hindu, ABP Ananda), "
            "0.5–0.8 = regional/verified sources, <0.5 = blogs/unverified."
        )
        region_labels = {
            "north_bengal": "North Bengal",
            "jangalmahal": "Jangalmahal",
            "medinipur": "Medinipur",
            "urban_kolkata": "Urban Kolkata",
            "south_bengal_rural": "South Bengal Rural",
        }
        signals_data = _signals_with_articles()
        if not signals_data:
            st.info("No article data available yet. Run the news pipeline to populate citations.")
        else:
            for row in signals_data:
                region = row.get("region", "")
                strength = float(row.get("signal_strength", 0.0))
                article_count = int(row.get("article_count", 0))
                top = row.get("top_articles", [])

                direction = "→ TMC favorable" if strength > 0.05 else ("→ BJP favorable" if strength < -0.05 else "→ neutral")
                color = "#22c55e" if strength > 0.05 else ("#ef4444" if strength < -0.05 else "#94a3b8")
                label = region_labels.get(region, region)

                st.markdown(
                    f"**{label}** &nbsp; "
                    f'<span style="color:{color}">signal: {strength:+.3f} ({direction})</span>'
                    f" &nbsp; _{article_count} articles_",
                    unsafe_allow_html=True,
                )
                if top:
                    for art in top[:3]:
                        headline = art.get("headline", "")
                        source = art.get("source", "")
                        url = art.get("url", "")
                        cred = float(art.get("credibility", 0.0))
                        net_sig = float(art.get("net_signal", 0.0))
                        sig_dir = "↑ TMC" if net_sig > 0.02 else ("↓ BJP" if net_sig < -0.02 else "≈ neutral")
                        link = f"[{headline}]({url})" if url else headline
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;• {link} &nbsp; "
                            f"_({source}, cred {cred:.2f}, signal {net_sig:+.3f} {sig_dir})_",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("&nbsp;&nbsp;&nbsp;_No articles with signal data yet._")
                st.markdown("")


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
        "rss_organizational_mobilization": {
            "label": "RSS/organizational mobilization effective",
            "desc": "BJP has deployed 1,823 shakhas (up from 1,320, +38% in Madhya Banga Prant alone) "
                    "and conducted 1.75 lakh voter-awareness meetings across ~250 of 294 constituencies. "
                    "Sunil Bansal's panna pramukh system targets 80,000 booths. "
                    "GREEN if RSS shakha attendance is high and booth-agent coverage exceeds 70% of targets. "
                    "Estimated seat impact if green: +8–15 seats (concentrated in North Bengal + Jangalmahal). "
                    "Ceiling: RSS mobilization consolidates existing BJP voters; limited ability to convert genuine TMC voters.",
            "ls2024": "RSS was active in 2024 LS but TMC still won 29/42 — mobilization alone is insufficient without other conditions.",
        },
        "bjp_clear_cm_face": {
            "label": "BJP declares credible CM face",
            "desc": "BJP has NOT declared a CM candidate. Suvendu Adhikari (LoP) and Sukanta Majumdar "
                    "(state president) both vie for the role. Dilip Ghosh sidelining has demoralized "
                    "South Bengal booth workers. Without a popular Bengali CM face, BJP votes convert "
                    "poorly to seats — voters are voting AGAINST TMC, not FOR a BJP alternative. "
                    "Historical impact: parties without CM face in Bengal underperform vote-to-seat "
                    "conversion by ~15–20%. Estimated seat impact if resolved: +10–18 seats statewide. "
                    "This is BJP's single largest unresolved structural drag.",
            "ls2024": "BJP had no CM face in 2021 either — won 77 seats despite 38% vote share (poor conversion). Pattern repeating.",
        },
    }

    conditions = _bjp_conditions()
    conditions_by_key = {c["condition_key"]: c for c in conditions} if conditions else {}
    green = [k for k, v in conditions_by_key.items() if v.get("status") == "green"]
    yellow = [k for k, v in conditions_by_key.items() if v.get("status") == "yellow"]
    total_conditions = len(CONDITIONS_META)

    st.progress(len(green) / total_conditions, text=f"{len(green)}/{total_conditions} conditions met (🟢 {len(green)} green, 🟡 {len(yellow)} yellow)")

    # Structural analysis box
    with st.expander("📋 Structural Factor Analysis — BJP's Organizational vs Leadership Trade-off", expanded=False):
        st.markdown("""
**BJP has deployed an unprecedented organizational machine but faces an unresolved leadership vacuum.**

| Factor | Direction | Seat Impact | Status |
|--------|-----------|-------------|--------|
| Bansal/Yadav panna pramukh system | BJP ▲ | +5–10 seats | Active (80,000 booths targeted) |
| RSS shakha expansion (+500 shakhas, 1.75L meetings) | BJP ▲ | +8–15 seats | Active (concentrated NB + Jangalmahal) |
| No CM face declared (Suvendu vs Sukanta unresolved) | BJP ▼ | −10–18 seats | 🔴 Unresolved |
| TMC booth management still superior | BJP ▼ | −5–8 seats | Structural |
| **Net effect** | **Slight BJP −** | **−2 to −6 seats** | Organizational gains offset by leadership gap |

**Why the CM face matters more than it seems:**
In FPTP elections, marginal seats are decided by the 3–5% of voters who are persuadable.
These voters need a reason to vote *for* someone, not just *against* the incumbent.
BJP's organizational machine can bring their base to the booth (turnout efficiency),
but cannot manufacture the "positive vote" that undecided voters need to cross over.
Mamata Banerjee remains the only statewide figure with genuine cross-community appeal.
Until BJP presents a credible Bengali alternative, their ceiling remains structurally capped at ~130–150 seats
even in a favorable scenario.
""")

    st.markdown("---")
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
                        hl = s.get("headline", "")
                        src = s.get("source", "")
                        url = s.get("url", "")
                        cred = s.get("credibility", 0)
                        link = f"[{hl}]({url})" if url else hl
                        src_str = f" — *{src}*" if src else ""
                        st.markdown(f"› {link}{src_str} *(cred: {cred:.1f})*")
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

    # All scenarios enforce TMC + BJP + Others(15) = 294
    # bjp_low  = 294 - tmc_high - 15
    # bjp_high = 294 - tmc_low  - 15
    # bjp_mid  = 294 - tmc_mid  - 15
    SCENARIOS = {
        "TMC Wave": {
            "color": "#22c55e",
            "tmc_low": 200, "tmc_high": 230, "tmc_mid": 215,
            "bjp_low":  49, "bjp_high":  79, "bjp_mid":  64,
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
            "tmc_low": 170, "tmc_high": 205, "tmc_mid": 187,
            "bjp_low":  74, "bjp_high": 109, "bjp_mid": 92,
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
            "tmc_low": 130, "tmc_high": 158, "tmc_mid": 144,
            "bjp_low": 121, "bjp_high": 149, "bjp_mid": 135,
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
            mids = [s["tmc_mid"], s["bjp_mid"], 15]
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

    # ── Pipeline Flowchart ──
    st.header("Pipeline Flowchart")
    st.caption("How a single day's run flows from raw data to the seat forecast you see on the dashboard.")
    st.graphviz_chart("""
digraph pipeline {
    rankdir=TB
    node [fontname="Arial" fontsize=13 margin="0.3,0.2" style=filled]
    edge [fontname="Arial" fontsize=11 color="#94a3b8"]

    subgraph cluster_data {
        label="INPUTS (fixed, historical)"
        style=filled color="#1e1e2e" fontcolor="#94a3b8" fontsize=12
        ECI  [label="ECI Results\\n2011 / 2016 / 2021" shape=cylinder fillcolor="#2d2d4e" fontcolor="#f8fafc"]
        LS24 [label="2024 Lok Sabha\\nRegional Results"   shape=cylinder fillcolor="#2d2d4e" fontcolor="#f8fafc"]
        SIR  [label="SIR Deletion Data\\n(91L deleted voters)"  shape=cylinder fillcolor="#2d2d4e" fontcolor="#f8fafc"]
    }

    subgraph cluster_prior {
        label="STEP 1 — BUILD PRIOR (once per model version)"
        style=filled color="#1a1a3e" fontcolor="#94a3b8" fontsize=12
        P1 [label="Constituency win rate\\nfrom ECI (2021×0.5 + 2016×0.3 + 2011×0.2)\\nWeights sum to 1.0" shape=box fillcolor="#3b3b6e" fontcolor="#f8fafc"]
        P2 [label="Regional baseline calibration\\n(REGIONAL_WIN_PRIOR)\\nIncorporates 2024 LS trends" shape=box fillcolor="#3b3b6e" fontcolor="#f8fafc"]
        P3 [label="Convert to LOGIT space\\nlogit(p) = log(p / 1−p)\\nUnbounded: 0.5→0, 0.88→+2.0, 0.27→−1.0" shape=box fillcolor="#4a4a7e" fontcolor="#f8fafc"]
    }

    subgraph cluster_sir {
        label="STEP 2 — SIR ADJUSTMENT (in PROBABILITY space)"
        style=filled color="#2a1a1a" fontcolor="#94a3b8" fontsize=12
        S1 [label="Blended TMC lean of deleted voters\\nMuslim share × 0.85 + Hindu share × 0.45\\n(63L Hindu + 28L Muslim deleted statewide)" shape=box fillcolor="#5e2d2d" fontcolor="#f8fafc"]
        S2 [label="excess_lean = tmc_lean − 0.50\\ndelta_p = deletion_rate × excess_lean\\n(Only excess over neutral baseline matters)" shape=box fillcolor="#5e2d2d" fontcolor="#f8fafc"]
        S3 [label="win_adj = win_prior − delta_p\\nThen convert EXACTLY to logit\\n(NOT ×4 approximation)" shape=box fillcolor="#6e3d3d" fontcolor="#f8fafc"]
    }

    subgraph cluster_org {
        label="STEP 2b — ORGANIZATIONAL FACTORS (structural, baked-in)"
        style=filled color="#1a1a2a" fontcolor="#94a3b8" fontsize=12
        G1 [label="RSS Mobilization (BJP advantage)\\n1,823 shakhas (+38%), 1.75L meetings\\nSunil Bansal + Bhupendra Yadav deployed\\nCap: 2021 BJP won 77 despite full RSS; 2024 LS TMC won 29/42" shape=box fillcolor="#2d2d5e" fontcolor="#f8fafc"]
        G2 [label="BJP CM Face Vacuum (TMC advantage)\\nNo declared CM candidate\\nSuvendu vs Sukanta rivalry; Dilip Ghosh sidelined\\nMamata personal vote vs BJP anonymity" shape=box fillcolor="#2d2d5e" fontcolor="#f8fafc"]
        G3 [label="Net delta applied in probability space\\nRSS effect (negative for TMC) vs CM vacuum (positive for TMC)\\nThen convert exactly to logit + propagate uncertainty" shape=box fillcolor="#3d3d6e" fontcolor="#f8fafc"]
    }

    subgraph cluster_news {
        label="STEP 3 — DAILY NEWS UPDATE"
        style=filled color="#1a2a1a" fontcolor="#94a3b8" fontsize=12
        N1 [label="Fetch: GDELT + NewsAPI.ai + YouTube\\n~300 articles/day" shape=box fillcolor="#2d5e2d" fontcolor="#f8fafc"]
        N2 [label="Claude AI filter\\nNoise removed (BJP PR, duplicates)\\nSignal scored −1 to +1 per region" shape=box fillcolor="#2d5e2d" fontcolor="#f8fafc"]
        N3 [label="Bayesian update (logit space)\\nobs = mu_adj + signal × 0.30\\nPrecision-weighted: news gets ~50% weight\\nPrior gets ~50% weight  (τ = σ = 1.20)" shape=box fillcolor="#3d6e3d" fontcolor="#f8fafc"]
    }

    subgraph cluster_mc {
        label="STEP 4 — MONTE CARLO (10,000 simulations)"
        style=filled color="#2a1a2a" fontcolor="#94a3b8" fontsize=12
        M1 [label="Draw GLOBAL SHOCK once per simulation\\nε_global ~ Normal(0, σ=0.65) ← in LOGIT space\\nNOT a probability — logit shift applied to all 294 seats\\n(models correlated election wave: 2019 BJP wave, 2021 TMC wave)" shape=box fillcolor="#5e2d5e" fontcolor="#f8fafc"]
        M2 [label="For each of 294 constituencies:\\nlogit_i = mu_post_i + ε_global + Normal(0, σ=0.80)\\n                                         ↑ constituency noise\\nTMC wins seat if logit_i > 0  (i.e. prob > 50%)" shape=box fillcolor="#5e2d5e" fontcolor="#f8fafc"]
        M3 [label="Count TMC seats won in this simulation\\nBJP = 294 − TMC − 15 (Left/Others fixed)\\nRepeat 10,000 times → distribution" shape=box fillcolor="#6e3d6e" fontcolor="#f8fafc"]
    }

    subgraph cluster_out {
        label="OUTPUT"
        style=filled color="#1a2a2a" fontcolor="#94a3b8" fontsize=12
        O1 [label="p50 = median seats (most likely outcome)\\np25−p75 = likely range (middle 50% of sims)\\np5−p95 = 90% CI (full uncertainty range)\\nP(TMC majority), P(Hung), P(BJP majority)" shape=box fillcolor="#2d5e5e" fontcolor="#f8fafc"]
    }

    ECI  -> P1
    LS24 -> P2
    P1   -> P2
    P2   -> P3
    SIR  -> S1
    P3   -> S1 [label="win_prior (probability)"]
    S1   -> S2
    S2   -> S3
    S3   -> G1
    G1   -> G2
    G2   -> G3
    G3   -> N3 [label="mu_adj, sigma_adj\\n(LOGIT space)"]
    N1   -> N2
    N2   -> N3 [label="signal ∈ [−1, +1]"]
    N3   -> M1 [label="mu_post per constituency\\n(still in LOGIT space)"]
    M1   -> M2
    M2   -> M3
    M3   -> M1 [label="next simulation" style=dashed]
    M3   -> O1 [label="after 10,000 sims"]
}
""", use_container_width=True)

    st.info(
        "**Key clarification on logit space:** Every number the model works with internally "
        "(mu_prior, mu_adj, mu_post, global shock, constituency noise) is in **logit space** — "
        "an unbounded scale where 0 = 50% win chance, +2 ≈ 88%, −2 ≈ 12%. "
        "The global shock of σ=0.65 is NOT a 65% probability — it is a logit-space standard deviation. "
        "Only the final step converts logit back to probability via sigmoid(x) = 1/(1+e^−x)."
    )

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
**How the weights work (they do sum to 1):**

The historical win rate is a weighted blend of past elections:
```
win_hist = 0.50 × (won in 2021) + 0.30 × (won in 2016) + 0.20 × (won in 2011)
```
These three weights sum to 1.0. `win_hist` is a number between 0 and 1 per constituency.

The 2024 Lok Sabha result then adjusts the *regional baseline* — not added on top as a separate weight.
The REGIONAL_WIN_PRIOR values shown above already incorporate both historical assembly trends and
2024 LS outcomes as a calibration exercise, not a mechanical formula. This avoids the confusion of
having weights from two separate steps that appear to sum to more than 1.

Prior uncertainty: **σ = 1.20 logit units** — reflecting genuine election-to-election swing variance.
""")

    st.divider()

    # ── Step 2: SIR ──
    st.header("Step 2 — SIR Adjustment")
    st.markdown("""
**Accounting for the systematic effect of voter deletions — both Hindu and Muslim.**

The previous version of this model incorrectly assumed only Muslim voters were deleted.
The actual composition of 91 lakh deletions statewide: **63 lakh Hindu (~69%), 28 lakh Muslim (~31%)**.
Both matter for the forecast, with different TMC lean:

| Deleted voter group | Count | TMC lean | Reasoning |
|---------------------|-------|----------|-----------|
| Muslim / minority | ~28L (31%) | **85%** | Historical Muslim-TMC alignment in WB |
| Hindu | ~63L (69%) | **45%** | Split electorate; BJP has significant Hindu vote share |

**Blended TMC lean** per constituency (using local minority share among deleted voters):
```
tmc_lean = minority_share × 0.85 + (1 − minority_share) × 0.45
         = 0.45 + minority_share × 0.40
```

At WB average (30% minority among deleted):
```
tmc_lean = 0.45 + 0.30 × 0.40 = 0.57
```

**Excess lean** over the neutral 50% baseline (the part that actually harms TMC):
```
excess_lean = tmc_lean − 0.50 = minority_share × 0.40 − 0.05
```

**TMC win probability reduction** (computed in probability space, then converted exactly to logit):
```
win_adj = win_prior − deletion_rate × excess_lean
mu_adj  = logit(win_adj)           ← exact conversion, no approximation
```

**Why exact conversion instead of "× 4"?**
The × 4 shortcut is the derivative of logit at p = 0.5: `d/dp[log(p/(1−p))] = 1/(p(1−p)) = 4 at p=0.5`.
It's a linear approximation that breaks down away from 0.5 (e.g. at p=0.88 for Urban Kolkata,
the true scaling factor is 1/(0.88×0.12) ≈ 9.5, not 4). We use the exact logit conversion instead.

**Effect:** SIR now impacts ALL constituencies proportionally to their deletion rate,
not just Muslim-heavy ones. The overall TMC impact is smaller than the minority-only model
(since Hindus at 45% lean barely exceed the 50% neutral baseline), but more accurate.

Extreme example — **Samserganj**: 95% Muslim constituency, 25% deletion rate.
```
tmc_lean    = 0.45 + 0.95 × 0.40 = 0.83
excess_lean = 0.83 − 0.50 = 0.33
delta_p     = 0.25 × 0.33 = −0.083   → TMC win prob drops ~8pp in this seat
```
""")

    st.divider()

    # ── Step 2b: Organizational Factors ──
    st.header("Step 2b — Organizational Factor Adjustment")
    st.markdown("""
**Baking in structural BJP organizational facts that don't change day-to-day.**

These are historical facts known before any news pipeline runs. They are modeled as
permanent prior adjustments — applied after SIR, before the daily Bayesian update.
""")

    col_rss, col_cm = st.columns(2)
    with col_rss:
        st.markdown("#### RSS Mobilization (BJP advantage)")
        st.markdown("""
**Source:** BJP deployed Sunil Bansal (national general secretary, architect of UP 2017/2022 victories)
and Bhupendra Yadav from 2022 onwards for WB 2026 groundwork.
RSS expanded from **1,320 → 1,823 shakhas (+38%)** in Madhya Banga Prant alone.
Conducted **1.75 lakh voter-awareness meetings** across ~250 of 294 constituencies.
*(India Today, The Print, 2024)*

**Historical cap applied:** This is not UP. In WB 2021, BJP had full RSS mobilization
+ Sunil Bansal yet won only **77/294** seats (target: 130+). In 2024 LS with Bansal
deployed, TMC still won **29/42** seats. WB's political culture — strong Mamata
personal brand, minority bloc, booth-level TMC machine — resists the standard RSS
playbook. We apply ~25% of the equivalent UP conversion rate.

| Region | RSS effect on TMC win prob | Reasoning |
|--------|---------------------------|-----------|
| North Bengal | −3.0pp | Shakha network deepest; Matua belt + hill seat targeting |
| Jangalmahal | −2.0pp | BJP tribal stronghold; booth management most effective |
| Medinipur | −1.5pp | Suvendu home turf; RSS + personal network combined |
| **Urban Kolkata** | **−1.2pp** | Hindu consolidation (bhadralok + migrants) real, but **partially offset** by Muslim counter-mobilization — RSS visibility in 35%+ Muslim wards hardens TMC alignment |
| **South Bengal Rural** | **−1.8pp** | 1.75L meetings covered rural constituencies; higher Hindu share than urban Kolkata means less counter-mobilization intensity, larger net BJP gain |

Uncertainty: **±50%** on these estimates.
""")

    with col_cm:
        st.markdown("#### BJP CM Face Vacuum (TMC advantage)")
        st.markdown("""
**Source:** As of 2026, BJP has **not declared a CM candidate** for WB.
- **Suvendu Adhikari** (Leader of Opposition) and **Sukanta Majumdar** (state president)
  are both positioning themselves for the role — creating visible rivalry.
- **Dilip Ghosh** (former state chief with mass connect) was sidelined after the 2021 loss,
  demoralizing the old-guard cadre who built BJP's Bengal network.
*(Indian Express 2024 "BJP Bengal CM face dilemma", Anandabazar Patrika 2025)*

**Historical precedent:**
- 2021: BJP ran a "Modi-for-PM, no CM face" campaign → **77 seats**.
- 2016: No state CM face → **10 seats**.
- Mamata Banerjee's personal CM-face vote is estimated at **5–8pp** in urban swing seats
  vs. a party with no recognizable state leader.

| Region | CM vacuum effect on TMC win prob |
|--------|----------------------------------|
| North Bengal | +1.0pp (Modi-wave partially offsets) |
| Jangalmahal | +1.0pp (rural/tribal, less sensitive) |
| Medinipur | +1.5pp (Suvendu local pull partially offsets) |
| Urban Kolkata | +3.0pp (Mamata personal vote strongest here) |
| South Bengal Rural | +2.0pp (minority loyalty + no BJP face) |

Uncertainty: **±40%** (Suvendu could be declared CM face before election).
""")

    st.markdown("#### Net Effect on Forecast")
    st.markdown("""
The two effects partially offset. RSS mobilization hurts TMC in BJP-competitive regions;
CM face vacuum helps TMC across the board (especially urban Kolkata).

| Region | RSS Δp | CM face Δp | **Net Δp** | Seats |
|--------|---------|------------|-----------|-------|
| North Bengal | −0.030 | +0.010 | **−0.020** | 54 |
| Jangalmahal | −0.020 | +0.010 | **−0.010** | 25 |
| Medinipur | −0.015 | +0.015 | **0.000** | 27 |
| Urban Kolkata | −0.012 | +0.030 | **+0.018** | 68 |
| South Bengal Rural | −0.018 | +0.020 | **+0.002** | 120 |

**Net seat impact vs. SIR-only baseline:** TMC median **−2 seats** (197→195).
RSS mobilization now applies to all regions including strongholds. South Bengal Rural (120 seats)
has the largest seat count and a near-zero net (RSS −1.8pp vs CM vacuum +2.0pp), making it
the swing region. Urban Kolkata remains TMC-positive (+1.8pp net) because Mamata's personal
vote effect is strongest there. Overall, RSS mobilization across all regions slightly outweighs
the CM face vacuum advantage.
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
μ_prior = +1.97   (logit of ~88% TMC win rate after SIR adjustment)
σ_prior =  1.20   (our uncertainty — same as observation noise by design)
```

**Convert news signal to observation:**
```
obs = μ_prior + signal × 0.30   (0.30 = signal scaling constant)
obs = 1.97 + 0.35 × 0.30 = +2.061
τ   = 1.20                       (observation noise — news is genuinely noisy)
```

**Precision-weighted update:**
```
precision_prior = 1 / σ²  = 1 / 1.44 = 0.694
precision_news  = 1 / τ²  = 1 / 1.44 = 0.694

posterior_μ = (0.694 × 1.97 + 0.694 × 2.061) ÷ (0.694 + 0.694)
            = +2.016   ← barely moved from 1.97

posterior_σ = sqrt(1 / (0.694 + 0.694)) = 0.849
```

**Result:** Urban Kolkata logit moves from 1.97 → 2.016, i.e. 87.8% → 88.2% win probability.
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
This is why the 90% CI spans 93–272 seats even though the median is stable at ~197.

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
    st.markdown("""
| Assumption | Value | If wrong… |
|-----------|-------|-----------|
| SIR Muslim lean | 85% | If 75% lean → TMC median ~+5–8 seats |
| SIR Hindu lean | 45% | If 50% lean → TMC median ~+8–12 seats (Hindus become neutral) |
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
inherently uncertain. A **76% TMC win probability means BJP wins in roughly 1 in 4 simulated
elections** — not that BJP winning is impossible. Treat ranges as plausible scenarios, not predictions.

*Data sources: ECI historical results, 2024 Lok Sabha regional data, SIR deletion estimates
from ECI/The Wire reporting, daily news via GDELT + NewsAPI + YouTube.*
""")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: MANUAL INPUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Manual Input":
    st.title("Manual Signal Input")
    st.caption(
        "Override or supplement automated news signals with your own ground-level assessment. "
        "Adjust regional signals and BJP conditions, then run the model to see the updated forecast."
    )
    st.markdown(f'<div class="sir-banner">{SIR_BANNER}</div>', unsafe_allow_html=True)

    REGIONS = ["north_bengal", "jangalmahal", "medinipur", "urban_kolkata", "south_bengal_rural"]
    REGION_LABELS = {
        "north_bengal":       "North Bengal (54 seats)",
        "jangalmahal":        "Jangalmahal (25 seats)",
        "medinipur":          "Medinipur (27 seats)",
        "urban_kolkata":      "Urban Kolkata (68 seats)",
        "south_bengal_rural": "South Bengal Rural (120 seats)",
    }
    BJP_CONDITIONS = {
        "north_bengal_sweep_35plus":    "North Bengal sweep (35+ seats)",
        "jangalmahal_hold_18plus":      "Jangalmahal hold (18+ of 25)",
        "medinipur_majority":           "Medinipur majority",
        "urban_kolkata_gain_10plus":    "Urban Kolkata gain (10+ seats)",
        "minority_fragmentation":       "Minority vote fragmentation",
        "sir_voter_suppression":            "SIR suppression effective",
        "anti_incumbency_national":         "National anti-incumbency wave",
        "rss_organizational_mobilization":  "RSS/organizational mobilization effective",
        "bjp_clear_cm_face":                "BJP declares credible CM face",
    }

    st.header("Regional Signal Strengths")
    st.markdown(
        "Set each region's signal: **+1.0** = strong TMC momentum, **−1.0** = strong BJP momentum, "
        "**0.0** = neutral (no news edge). These feed directly into the Bayesian update."
    )

    # Load current automated signals as defaults
    auto_signals_df = _regional_signals()
    def _get_auto_signal(region):
        if auto_signals_df.empty:
            return 0.0
        row = auto_signals_df[auto_signals_df["region"] == region]
        if row.empty:
            return 0.0
        return float(row.sort_values("date").iloc[-1]["signal_strength"])

    manual_signals = {}
    cols_r = st.columns(2)
    for i, region in enumerate(REGIONS):
        auto_val = _get_auto_signal(region)
        with cols_r[i % 2]:
            val = st.slider(
                REGION_LABELS[region],
                min_value=-1.0, max_value=1.0,
                value=float(round(auto_val, 2)),
                step=0.05,
                key=f"sig_{region}",
                help=f"Automated signal today: {auto_val:+.2f}",
            )
            manual_signals[region] = {"signal_strength": val, "article_count": 1}
            color = "#22c55e" if val > 0.1 else "#ef4444" if val < -0.1 else "#94a3b8"
            st.markdown(
                f'<div style="font-size:0.8rem;color:{color};margin-top:-12px;margin-bottom:8px">'
                f'{"TMC ▲" if val > 0.1 else "BJP ▲" if val < -0.1 else "Neutral"} {val:+.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.header("BJP Pathway Conditions")
    st.markdown("Set your assessment of each BJP condition. This updates the BJP Pathway scorecard.")

    bjp_manual = {}
    cols_b = st.columns(3)
    auto_conditions = _bjp_conditions()
    auto_cond_map = {c["condition_key"]: c for c in auto_conditions} if auto_conditions else {}

    for i, (ckey, clabel) in enumerate(BJP_CONDITIONS.items()):
        auto = auto_cond_map.get(ckey, {})
        auto_status = auto.get("status", "red")
        auto_conf   = float(auto.get("confidence", 0.1))
        with cols_b[i % 3]:
            st.markdown(f"**{clabel}**")
            status = st.selectbox(
                "Status", ["green", "yellow", "red"],
                index=["green", "yellow", "red"].index(auto_status),
                key=f"bjp_status_{ckey}",
                label_visibility="collapsed",
            )
            conf = st.slider(
                "Confidence", 0.0, 1.0, auto_conf, 0.05,
                key=f"bjp_conf_{ckey}",
                label_visibility="collapsed",
            )
            bjp_manual[ckey] = {"status": status, "confidence": conf, "evidence_json": "[]"}
            icon = "🟢" if status == "green" else "🟡" if status == "yellow" else "🔴"
            st.caption(f"{icon} {int(conf*100)}% confidence")

    st.divider()

    col_run1, col_run2, _ = st.columns([1, 1, 2])
    with col_run1:
        run_btn = st.button("▶ Run Forecast with Manual Signals", type="primary", use_container_width=True)
    with col_run2:
        save_btn = st.button("💾 Save & Update Dashboard", use_container_width=True,
                             help="Saves manual signals to DB and updates all pages")

    if run_btn or save_btn:
        from pipeline.bayesian import run_full_pipeline
        from pipeline.db import get_latest_forecast, upsert_forecast, upsert_regional_signals, upsert_bjp_conditions

        with st.spinner("Running Bayesian model with manual signals..."):
            prev = _forecast_latest()
            prev_tmc_p50 = prev["tmc_p50"] if prev else None

            forecast_m, _ = run_full_pipeline(manual_signals)
            forecast_m["prev_tmc_p50"] = prev_tmc_p50

        st.success("Forecast computed.")

        # Show results
        st.subheader("Forecast with Manual Signals")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TMC median", forecast_m["tmc_p50"],
                  delta=int(forecast_m["tmc_p50"] - prev_tmc_p50) if prev_tmc_p50 else None)
        m2.metric("TMC likely range", f"{forecast_m['tmc_p25']}–{forecast_m['tmc_p75']}")
        m3.metric("BJP median", forecast_m["bjp_p50"])
        m4.metric("P(TMC majority)", f"{forecast_m['p_tmc_win']:.1%}")

        c1, c2, c3 = st.columns(3)
        c1.metric("P(TMC majority)", f"{forecast_m['p_tmc_win']:.1%}")
        c2.metric("P(Hung)",         f"{forecast_m['p_hung']:.1%}")
        c3.metric("P(BJP majority)", f"{forecast_m['p_bjp_win']:.1%}")

        # Compare auto vs manual
        if prev:
            st.markdown("**Auto forecast vs manual override:**")
            comp_df = pd.DataFrame([
                {"Source": "Automated (last pipeline run)", "TMC p50": prev["tmc_p50"],
                 "TMC range": f"{prev['tmc_p25']}–{prev['tmc_p75']}", "P(TMC win)": f"{prev['p_tmc_win']:.1%}"},
                {"Source": "Manual override (just computed)", "TMC p50": forecast_m["tmc_p50"],
                 "TMC range": f"{forecast_m['tmc_p25']}–{forecast_m['tmc_p75']}", "P(TMC win)": f"{forecast_m['p_tmc_win']:.1%}"},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if save_btn:
            with st.spinner("Saving to database..."):
                conn = _db()
                upsert_forecast(conn, date.today(), forecast_m)
                upsert_regional_signals(conn, date.today(), manual_signals)
                upsert_bjp_conditions(conn, date.today(), bjp_manual)
                st.cache_data.clear()
            st.success("Saved. All dashboard pages now reflect your manual input.")
            st.rerun()
