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
def _market_data():
    from pipeline.db import get_latest_market_data
    return get_latest_market_data(_db())


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
    _pipeline_ok = False
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
            noise_count = len(enriched) - signal_count
            st.write(f"{signal_count} signal articles, {noise_count} filtered as noise.")
            if signal_count == 0 and len(enriched) > 0:
                st.warning(
                    "⚠️ All articles marked as noise — likely cause: **Anthropic API credit balance is zero**. "
                    "Top up at console.anthropic.com → Plans & Billing. "
                    "Pipeline will continue with neutral signals (forecast unchanged)."
                )

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

            st.write("Fetching Polymarket odds...")
            from pipeline.fetchers import fetch_polymarket
            from pipeline.db import upsert_market_data
            market = fetch_polymarket()
            if market:
                upsert_market_data(conn, market)
                st.write(f"Polymarket: TMC {market.get('tmc_probability','—')} · BJP {market.get('bjp_probability','—')}")
            else:
                st.write("Polymarket: no data available.")

            st.write("Running Bayesian model...")
            from pipeline.bayesian import run_full_pipeline
            from pipeline.db import get_latest_market_data
            prev = get_latest_forecast(conn)
            mkt = get_latest_market_data(conn)
            forecast_new, _ = run_full_pipeline(signals, market_data=mkt)
            forecast_new["prev_tmc_p50"] = prev["tmc_p50"] if prev else None
            upsert_forecast(conn, date.today(), forecast_new)
            conn.close()

            status.update(label=f"Pipeline complete — TMC median: {forecast_new['tmc_p50']} seats", state="complete")
            _pipeline_ok = True
        except Exception as e:
            status.update(label=f"Pipeline failed: {e}", state="error")
            st.exception(e)

    # st.rerun() MUST be outside try/except — it raises RerunException which
    # would otherwise be caught as a generic Exception, aborting the rerun.
    if _pipeline_ok:
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

    # ── Polymarket odds (Tier 5 — sentiment only) ──────────────────────────
    market = _market_data()
    if market and (market.get("tmc_probability") or market.get("bjp_probability")):
        st.markdown("---")
        st.caption("📊 **Prediction Market** (Polymarket) — sentiment signal only, not ground truth")
        pm1, pm2, pm3, pm4 = st.columns(4)
        tmc_mkt = market.get("tmc_probability")
        bjp_mkt = market.get("bjp_probability")
        shift   = market.get("market_shift_24h")
        vol     = market.get("volume_usd")
        with pm1:
            st.metric("Market P(TMC)", f"{tmc_mkt:.0%}" if tmc_mkt else "—",
                      delta=f"{shift:+.1%}" if shift else None)
        with pm2:
            st.metric("Market P(BJP)", f"{bjp_mkt:.0%}" if bjp_mkt else "—")
        with pm3:
            st.metric("Volume traded", f"${vol:,.0f}" if vol else "—")
        with pm4:
            st.caption(f"Source: {market.get('source','Polymarket')} · {market.get('date','')}")
        st.caption(
            "⚠️ Market odds reflect trader sentiment, not booth-level evidence. "
            "Model weight: 5% (per source hierarchy). Do not use as primary forecast."
        )

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

    st.divider()

    # ── Phase Intelligence ────────────────────────────────────────────────────
    st.subheader("Phase Turnout Intelligence")
    st.caption(
        "Official ECI VTR + post-scrutiny turnout. "
        "Phase 1 (April 23) covers North Bengal, Malda-Murshidabad belt, Jangalmahal. "
        "Phase 2 (April 29) covers South Bengal, Urban Kolkata, Medinipur."
    )

    @st.cache_data(ttl=300)
    def _phase_turnout_data():
        conn = _db()
        from pipeline.db import get_phase_turnout, get_booth_turnout
        pt = get_phase_turnout(conn)
        bt = get_booth_turnout(conn)
        return pt, bt

    pt_df, bt_df = _phase_turnout_data()

    if pt_df.empty:
        st.info("No phase turnout data yet. Run the pipeline (seeds Phase 1 district data automatically) or enter manually via Manual Input → Phase Turnout Data Entry.")
    else:
        tab_ph1, tab_ph2, tab_booth = st.tabs(["Phase 1 Turnout", "Phase 2 Turnout", "Booth-level (Form 17C)"])

        with tab_ph1:
            ph1 = pt_df[pt_df["phase"] == 1].copy() if "phase" in pt_df.columns else pd.DataFrame()
            if ph1.empty:
                st.info("No Phase 1 data yet.")
            else:
                # Turnout swing chart
                ph1_sorted = ph1.dropna(subset=["turnout_swing_vs_2021"]).sort_values("turnout_swing_vs_2021", ascending=True)
                if not ph1_sorted.empty:
                    colors = ["#22c55e" if v > 0 else "#ef4444" for v in ph1_sorted["turnout_swing_vs_2021"]]
                    fig_t = go.Figure(go.Bar(
                        x=ph1_sorted["turnout_swing_vs_2021"],
                        y=ph1_sorted["ac_name"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.1f}pp" for v in ph1_sorted["turnout_swing_vs_2021"]],
                        textposition="outside",
                        hovertemplate="%{y}: %{x:+.1f}pp vs 2021<extra></extra>",
                    ))
                    fig_t.update_layout(
                        title="Turnout swing vs 2021 (Phase 1)",
                        height=max(300, 35 * len(ph1_sorted)),
                        xaxis_title="Swing (pp)", yaxis_title="",
                        paper_bgcolor="#0e1117", font={"color": "#f8fafc"},
                        margin=dict(t=40, b=20, l=160, r=60),
                    )
                    st.plotly_chart(fig_t, use_container_width=True, key="ph1_swing_chart")

                # Summary metrics
                avg_post = ph1["post_scrutiny_turnout_percent"].dropna().mean()
                avg_swing = ph1["turnout_swing_vs_2021"].dropna().mean()
                highest = ph1.loc[ph1["post_scrutiny_turnout_percent"].idxmax()] if not ph1["post_scrutiny_turnout_percent"].dropna().empty else None
                lowest = ph1.loc[ph1["post_scrutiny_turnout_percent"].idxmin()] if not ph1["post_scrutiny_turnout_percent"].dropna().empty else None

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Phase 1 avg turnout", f"{avg_post:.1f}%" if avg_post else "—")
                m2.metric("Avg swing vs 2021", f"{avg_swing:+.1f}pp" if avg_swing else "—")
                if highest is not None:
                    m3.metric("Highest", f"{highest['ac_name']} {highest['post_scrutiny_turnout_percent']:.1f}%")
                if lowest is not None:
                    m4.metric("Lowest", f"{lowest['ac_name']} {lowest['post_scrutiny_turnout_percent']:.1f}%")

                st.info(
                    "📌 **Interpretation:** High turnout in BJP-competitive areas (Cooch Behar, Alipurduar) "
                    "may signal BJP enthusiasm. High turnout in TMC/Muslim-majority areas (Murshidabad) "
                    "may signal counter-mobilisation from SIR anger. Treat swing >+10pp in BJP belts as a "
                    "BJP signal; swing >+10pp in minority belts as a TMC counter-mobilisation signal."
                )

                st.dataframe(
                    ph1[["district","ac_name","region","post_scrutiny_turnout_percent","initial_turnout_percent",
                          "turnout_2021","turnout_swing_vs_2021","source_type","confidence_score","notes"]].rename(columns={
                        "post_scrutiny_turnout_percent": "Post-scrutiny %",
                        "initial_turnout_percent": "Initial VTR %",
                        "turnout_2021": "2021 %",
                        "turnout_swing_vs_2021": "Swing vs 2021",
                        "source_type": "Source",
                        "confidence_score": "Confidence",
                    }),
                    use_container_width=True, hide_index=True
                )

        with tab_ph2:
            ph2 = pt_df[pt_df["phase"] == 2] if "phase" in pt_df.columns else pd.DataFrame()
            if ph2.empty:
                st.info("Phase 2 data will appear here after April 29 polling. Enter via Manual Input → Phase Turnout Data Entry.")
            else:
                st.dataframe(ph2, use_container_width=True, hide_index=True)

        with tab_booth:
            if bt_df.empty:
                st.info(
                    "No booth-level data yet. Sources:\n"
                    "- **Form 17C** from party booth agents (handed at poll close)\n"
                    "- ECI VTR app booth-level update (post-scrutiny)\n\n"
                    "Enter via **Manual Input → Form 17C Booth Entry**."
                )
            else:
                st.caption(f"{len(bt_df)} booth entries | "
                           f"Sources: {', '.join(bt_df['source_type'].unique())}")
                conf_filter = st.slider("Min confidence score", 0.0, 1.0, 0.5, 0.05, key="booth_conf_filter")
                st.dataframe(
                    bt_df[bt_df["confidence_score"] >= conf_filter][[
                        "ac_name","booth_no","polling_station_name","total_electors",
                        "votes_polled","turnout_percent","turnout_vs_2021",
                        "source_type","source_party","confidence_score","notes"
                    ]].sort_values(["ac_name","booth_no"]),
                    use_container_width=True, hide_index=True,
                )


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

    # Scenarios use three-party accounting: TMC + BJP + Others = 294
    # Others = Congress (Malda/Murshidabad) + Left + ISF + GJM/Hill parties
    # Model baseline (real 2021+2024 data): TMC 168, BJP 94, Others ~32
    # Each scenario shifts Others too: SIR restoration → Congress seats shrink;
    # SIR full suppression → Congress/ISF gain more Muslim-majority seats.
    SCENARIOS = {
        "TMC Wave": {
            "color": "#22c55e",
            "tmc_low": 185, "tmc_high": 220, "tmc_mid": 200,
            "bjp_low":  46, "bjp_high":  81, "bjp_mid":  68,
            "others_mid": 26,
            "prob": "20–30%",
            "sir": "SIR rolls partially/fully restored by court order",
            "conditions": [
                "Court orders restore significant voter names before Phase 2",
                "Minority consolidation >85% behind TMC (counter-mobilisation from SIR anger)",
                "IPAC shutdown absorbed by TMC state machinery",
                "North Bengal partial recovery by TMC",
                "BJP booth management fails despite RSS groundwork",
            ],
        },
        "Status Quo": {
            "color": "#f59e0b",
            "tmc_low": 148, "tmc_high": 195, "tmc_mid": 168,
            "bjp_low":  67, "bjp_high": 114, "bjp_mid":  94,
            "others_mid": 32,
            "prob": "45–55%",
            "sir": "SIR as-is — 12.9L deletions in 142 seats, contested in court",
            "conditions": [
                "SIR impact as-is — deletions in Muslim-majority seats flow to Congress/ISF, not BJP",
                "IPAC shutdown hurts TMC operations but Mamata brand holds",
                "Minority vote broadly consolidates for TMC despite SIR anger",
                "BJP holds North Bengal + Jangalmahal base (RSS mobilisation effective)",
                "No major new swing factor emerges before counting",
            ],
        },
        "BJP Surge": {
            "color": "#ef4444",
            "tmc_low": 108, "tmc_high": 145, "tmc_mid": 126,
            "bjp_low": 118, "bjp_high": 155, "bjp_mid": 139,
            "others_mid": 29,
            "prob": "15–25%",
            "sir": "Full SIR suppression + minority fragmentation",
            "conditions": [
                "SIR fully effective in Nadia/mixed seats; tactical Muslim vote absent",
                "IPAC shutdown compounds TMC booth collapse in swing seats",
                "North Bengal sweep: BJP wins 33–35+ seats",
                "Jangalmahal hold: 18+ of 25 seats",
                "Left/ISF draws 15%+ in key Muslim-plurality seats, splitting anti-BJP vote",
                "Broad anti-incumbency wave driven by economic stagnation + youth unemployment",
            ],
        },
    }

    # ── Summary comparison chart ───────────────────────────────────────────────
    # ── Individual scenario cards with pie charts ─────────────────────────────
    cols = st.columns(3)
    for i, (name, s) in enumerate(SCENARIOS.items()):
        with cols[i]:
            st.markdown(f"### {name}")
            st.markdown(f"**Probability: {s['prob']}**")
            st.markdown(f"*{s['sir']}*")

            fig = go.Figure(go.Pie(
                labels=["TMC", "BJP", "Congress / Left / Others"],
                values=[s["tmc_mid"], s["bjp_mid"], s["others_mid"]],
                hole=0.45,
                marker_colors=["#22c55e", "#ef4444", "#3b82f6"],
                textinfo="label+value",
                textfont=dict(size=15),
                hovertemplate="%{label}: %{value} seats (%{percent})<extra></extra>",
            ))
            fig.add_annotation(
                text=f"<b>294</b><br>seats",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#f8fafc"),
            )
            fig.update_layout(
                height=300,
                paper_bgcolor="#0e1117",
                font=dict(color="#f8fafc"),
                legend=dict(bgcolor="#1e1e2e", font=dict(size=11)),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"scenario_bar_{i}")

            st.markdown(
                f"🟢 TMC: **{s['tmc_low']}–{s['tmc_high']}** (mid {s['tmc_mid']})  \n"
                f"🔴 BJP: **{s['bjp_low']}–{s['bjp_high']}** (mid {s['bjp_mid']})  \n"
                f"🔵 Others: **~{s['others_mid']}** (Congress/Left/ISF)"
            )
            st.markdown("**Key requirements:**")
            for cond in s["conditions"]:
                st.markdown(f"• {cond}")

    st.markdown("---")
    st.subheader("SIR Swing Analysis")
    st.markdown("> Samserganj alone: 74,000 deletions in a 95%-Muslim constituency (~25% of total electorate). If deletions hold, this seat almost certainly flips.")

    if forecast:
        base = forecast["tmc_p50"]
        sir_labels = [
            "Full restoration\n(courts intervene)",
            "Partial restoration\n(50% restored)",
            "As-is\n(baseline)",
            "As-is +\nminority fragmentation",
        ]
        sir_mids  = [base + 20, base + 10, base, base - 14]
        sir_lows  = [base + 15, base + 8,  base, base - 18]
        sir_highs = [base + 25, base + 12, base, base - 10]
        sir_colors = ["#22c55e", "#86efac", "#f59e0b", "#ef4444"]

        fig_sir = go.Figure()
        # Range
        fig_sir.add_trace(go.Bar(
            x=sir_labels,
            y=[h - l for h, l in zip(sir_highs, sir_lows)],
            base=sir_lows,
            marker_color=["rgba(34,197,94,0.2)", "rgba(134,239,172,0.2)",
                          "rgba(245,158,11,0.2)", "rgba(239,68,68,0.2)"],
            marker_line_color=sir_colors,
            marker_line_width=2,
            showlegend=False,
            hovertemplate="%{x}<br>TMC range: %{base}–%{y} seats<extra></extra>",
        ))
        # Midpoints
        fig_sir.add_trace(go.Scatter(
            x=sir_labels, y=sir_mids,
            mode="markers+text",
            marker=dict(color=sir_colors, size=16, symbol="diamond",
                        line=dict(color="#f8fafc", width=1)),
            text=[f"<b>{v}</b>" for v in sir_mids],
            textposition="top center",
            textfont=dict(size=15),
            showlegend=False,
        ))
        fig_sir.add_hline(y=148, line_dash="dash", line_color="#f59e0b", line_width=2,
                          annotation_text="<b>Majority — 148</b>",
                          annotation_position="top right",
                          annotation_font_color="#f59e0b")
        fig_sir.update_layout(
            height=380, barmode="overlay",
            yaxis=dict(title="TMC Seats", gridcolor="#2d2d4e",
                       range=[min(sir_lows) - 15, max(sir_highs) + 20]),
            xaxis=dict(tickfont=dict(size=12)),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#f8fafc"),
            margin=dict(t=20, b=20, l=50, r=80),
        )
        st.plotly_chart(fig_sir, use_container_width=True, key="sir_swing_chart")

        st.markdown("""
| SIR Scenario | TMC Seat Impact | Mechanism |
|---|---|---|
| **Full roll restoration** | **+15 to +25** | Deleted minority voters re-enfranchised in Murshidabad, Malda, N24P, Kolkata |
| **Partial restoration (50%)** | **+8 to +12** | Partial court relief, some constituencies restored |
| **As-is** *(baseline)* | **0** (reference) | 91L deleted voters absent; our baseline assumes this |
| **As-is + fragmentation** | **−10 to −18** | Left/ISF splits vote in Muslim-majority seats on top of deletions |
""")
        col_x, col_y = st.columns(2)
        with col_x:
            st.metric("Current model TMC (p50)", base, help="Assumes SIR as-is")
        with col_y:
            st.metric("If rolls fully restored", f"~{base + 20}", delta="+20", help="+15–25 seats if courts act")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: NEWS FEED
# ─────────────────────────────────────────────────────────────────────────────
elif page == "News Feed":
    st.title("Filtered News Feed")
    st.caption("Noise-filtered by Claude. Sources ranked by tier — Official > Tier-1 News > Regional > Aggregated > Social.")

    TIER_LABELS = {
        1: ("🏛️ Official",    "#22c55e"),
        2: ("📰 Tier-1 News", "#3b82f6"),
        3: ("📡 Regional",    "#f59e0b"),
        4: ("🌐 Aggregated",  "#94a3b8"),
        5: ("📊 Market",      "#a78bfa"),
        6: ("📱 Social",      "#6b7280"),
    }
    with st.expander("Source tier legend", expanded=False):
        for tier, (label, color) in TIER_LABELS.items():
            st.markdown(
                f'<span style="color:{color}">**{label}**</span> — '
                + {1: "ECI/CEO WB official data — ground truth",
                   2: "Indian Express, The Hindu, NDTV, Reuters, TOI, Telegraph India — high credibility",
                   3: "ABP Ananda, Zee 24 Ghanta, regional Bengali outlets — medium-high",
                   4: "Google News RSS, GDELT — aggregated, verify before trusting",
                   5: "Polymarket, prediction markets — sentiment only",
                   6: "YouTube, social media — weak signal, high noise"}[tier],
                unsafe_allow_html=True,
            )

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
        tier = int(row.get("source_tier", 4)) if "source_tier" in row.index else 4
        tier_label, tier_color = TIER_LABELS.get(tier, ("🌐 Aggregated", "#94a3b8"))
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
                st.markdown(
                    f'<span style="color:{tier_color};font-size:0.85rem">**{tier_label}**</span>',
                    unsafe_allow_html=True,
                )
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
        label="INPUTS — REAL DATA (not synthetic)"
        style=filled color="#1e1e2e" fontcolor="#94a3b8" fontsize=12
        ECI  [label="2021 Assembly Results\\nReal ECI data — 292/294 ACs\\ntecoholic/Election2021 GitHub\\nTMC%, BJP%, Left%, Cong% per AC" shape=cylinder fillcolor="#2d2d4e" fontcolor="#f8fafc"]
        LS24 [label="2024 Lok Sabha — all 42 WB seats\\nReal ECI results hardcoded\\nTMC vs BJP margin per LS seat\\nMapped to each of 294 ACs" shape=cylinder fillcolor="#2d2d4e" fontcolor="#f8fafc"]
        SIR  [label="SIR Deletion Data\\nReal district totals (TOI/IE)\\nN24P 3.25L · S24P 2.23L\\nNadia 2.09L · Purba Bdm 2.09L\\nSpecific ACs: Rajarhat-NT 65k" shape=cylinder fillcolor="#2d2d4e" fontcolor="#f8fafc"]
    }

    subgraph cluster_prior {
        label="STEP 1 — BUILD PRIOR (reproducible margin formula)"
        style=filled color="#1a1a3e" fontcolor="#94a3b8" fontsize=12
        P1 [label="AC-level base margin (percentage points):\\nM = 0.45 × ls24_tmc_minus_bjp\\n    + 0.25 × assembly21_tmc_minus_bjp\\nWeights reflect recency (2024 > 2021)" shape=box fillcolor="#3b3b6e" fontcolor="#f8fafc"]
        P2 [label="Others adjustment (Congress/ISF recovery):\\np_others = f(2021 cong+left share, minority_share)\\nHigh minority_share → Congress wins seat, not BJP\\nScales down both TMC and BJP probabilities" shape=box fillcolor="#3b3b6e" fontcolor="#f8fafc"]
        P3 [label="Convert to probability + LOGIT space:\\np_tmc_raw = logistic(M / 7pp)  ← σ=7 percentage points\\nwin_prior = p_tmc_raw × (1 − p_others)\\nmu_logit  = logit(win_prior)" shape=box fillcolor="#4a4a7e" fontcolor="#f8fafc"]
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
        G4 [label="IPAC Shutdown (TMC disadvantage)\\nIPAC shut down mid-election April 2026\\nVinesh Chandel arrested; Rishiraj Singh summoned\\nBooth committees + social media + rally plans lost\\nIPAC was what reversed 2019 damage in 2021" shape=box fillcolor="#5e2d2d" fontcolor="#f8fafc"]
        G3 [label="Net delta applied in probability space\\nRSS (BJP ▲) + CM vacuum (TMC ▲) + IPAC shutdown (BJP ▲)\\nThen convert exactly to logit + propagate uncertainty" shape=box fillcolor="#3d3d6e" fontcolor="#f8fafc"]
    }

    subgraph cluster_turnout {
        label="STEP 3a — PHASE TURNOUT (Tier 1 official data)"
        style=filled color="#1a2a3a" fontcolor="#94a3b8" fontsize=12
        T1 [label="ECI VTR App + CEO WB + post-scrutiny press notes\\nAC-wise + district turnout % (initial & post-scrutiny)\\nPhase 1: 93.19% overall; Cooch Behar 96.2%; Kalimpong 83.04%\\nForm 17C from booth agents (manual entry)" shape=box fillcolor="#2d4e5e" fontcolor="#f8fafc"]
        T2 [label="Turnout swing signal = (current − 2021) / 15pp\\nBJP-competitive belt (NB/Jangal): swing → BJP signal × 0.6\\nTMC stronghold (Kolkata/South Rural): swing → TMC signal × 0.4\\nBlended 30% into news signal before Bayesian update" shape=box fillcolor="#2d4e5e" fontcolor="#f8fafc"]
    }

    subgraph cluster_news {
        label="STEP 3b — DAILY NEWS UPDATE (source-tiered)"
        style=filled color="#1a2a1a" fontcolor="#94a3b8" fontsize=12
        N1 [label="Tier 1–3 only → Claude (~22 articles, 2 batches, ~$0.08/run)\\nTier 2: RSS Indian Express, Hindu, NDTV, TOI, Telegraph, Wire\\nTier 3: ABP Ananda, Zee 24 Ghanta (Bengali)\\nTier 4+: auto-scored by keyword (no Claude call)" shape=box fillcolor="#2d5e2d" fontcolor="#f8fafc"]
        N2 [label="Claude AI filter (tier-aware)\\nNoise removed · Credibility anchored to source tier\\nTier 1 ECI = 0.95-1.0  Tier 2 news = 0.75-0.90\\nTier 6 social = 0.20-0.45\\nSignal scored −1 to +1 per region + avg_source_tier" shape=box fillcolor="#2d5e2d" fontcolor="#f8fafc"]
        N3 [label="Bayesian update — tier-specific TAU\\nTier 1 τ=0.40 (ECI near ground truth)\\nTier 2 τ=0.80 (quality news)  Tier 6 τ=2.50 (social)\\nobs = mu_adj + (0.7×news + 0.3×turnout signal) × 0.30" shape=box fillcolor="#2d5e2d" fontcolor="#f8fafc"]
        N4 [label="Polymarket odds (Tier 5)\\nτ=2.00 statewide weak signal\\n~5% effective weight per source hierarchy" shape=box fillcolor="#1a3a1a" fontcolor="#f8fafc"]
    }

    subgraph cluster_mc {
        label="STEP 4 — MONTE CARLO (10,000 simulations)"
        style=filled color="#2a1a2a" fontcolor="#94a3b8" fontsize=12
        M1 [label="Draw GLOBAL SHOCK once per simulation\\nε_global ~ Normal(0, σ=0.65) ← in LOGIT space\\nNOT a probability — logit shift applied to all 294 seats\\n(models correlated election wave: 2019 BJP wave, 2021 TMC wave)" shape=box fillcolor="#5e2d5e" fontcolor="#f8fafc"]
        M2 [label="THREE-WAY sampling per constituency:\\n1. if rand() < p_others_adj → Others wins (Congress/ISF/Left)\\n2. else: logit_i = mu_post_i + ε_global + Normal(0, σ=0.80)\\n         TMC wins if logit_i > 0 · BJP wins if logit_i ≤ 0\\np_others varies by AC — high in Murshidabad/Malda" shape=box fillcolor="#5e2d5e" fontcolor="#f8fafc"]
        M3 [label="Count per simulation:\\nTMC seats + BJP seats + Others seats = 294\\nOthers: ~32 median (Congress Malda/Murshidabad + Left + ISF)\\nRepeat 10,000 times → full distribution" shape=box fillcolor="#6e3d6e" fontcolor="#f8fafc"]
    }

    subgraph cluster_out {
        label="OUTPUT"
        style=filled color="#1a2a2a" fontcolor="#94a3b8" fontsize=12
        O1 [label="p50 = median seats (most likely outcome)\\np25−p75 = likely range (middle 50% of sims)\\np5−p95 = 90% CI (full uncertainty range)\\nP(TMC majority), P(Hung), P(BJP majority)" shape=box fillcolor="#2d5e5e" fontcolor="#f8fafc"]
    }

    ECI  -> P1
    LS24 -> P1
    P1   -> P2
    P2   -> P3
    SIR  -> S1
    P3   -> S1 [label="win_prior (probability)"]
    S1   -> S2
    S2   -> S3
    S3   -> G1
    G1   -> G2
    G2   -> G4
    G4   -> G3
    G3   -> T1 [label="mu_adj, sigma_adj\\n(LOGIT space)"]
    T1   -> T2
    T2   -> N3 [label="turnout signal (30% blend)"]
    N1   -> N2
    N2   -> N3 [label="news signal (70% blend) + avg_tier"]
    N4   -> N3 [label="market logit obs"]
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

    # ── Source hierarchy ──
    st.header("Source Hierarchy — How We Layer Data")
    st.markdown("""
The model treats different data sources with explicitly different levels of trust.
**Fact**, **reported claim**, and **model inference** are kept separate throughout.
""")
    hierarchy_data = {
        "Priority": ["1 — Official live", "2 — Phase turnout", "3 — Historical baseline",
                     "4 — BJP/RSS/IPAC org (structural)", "5 — Quality news (Tier 2)",
                     "6 — Regional Bengali news (Tier 3)", "7 — Aggregated news (Tier 4)",
                     "8 — Opinion polls", "9 — Prediction markets", "10 — Social/YouTube"],
        "Source": ["ECI / CEO WB / Form 20 / Affidavits",
                   "ECI VTR App, CEO WB, post-scrutiny RO notes, Form 17C (booth agents)",
                   "2021 Assembly + 2024 LS assembly-segment data",
                   "RSS shakha counts, IPAC shutdown, CM-face vacuum (structural priors)",
                   "Indian Express, The Hindu, NDTV, TOI, Reuters, Telegraph India, Wire",
                   "ABP Ananda, Zee 24 Ghanta, Bartaman",
                   "Google News RSS (288 articles/run), GDELT — auto-scored, no Claude",
                   "Pre-election surveys (limited availability in WB)",
                   "Polymarket WB election market",
                   "YouTube comments, X/Twitter, WhatsApp"],
        "Claude call?": ["No", "No", "No", "No", "Yes (Tier 1–3 only, ~$0.08/run)",
                         "Yes", "No (keyword auto-score)", "No", "No", "Yes (very weak)"],
        "Model weight": ["Ground truth — overrides model",
                         "~15% (turnout swing blended 30% into Step 3 signal)",
                         "35% (historical prior, baked in)",
                         "10% (permanent prior adjustment)",
                         "~3% via τ=0.80 Bayesian update",
                         "~2% via τ=1.10 Bayesian update",
                         "~1% (credibility=0.50, TIER_WEIGHT=0.6)",
                         "Not wired (polling vacuum in WB)",
                         "~5% via τ=2.00 statewide signal",
                         "<1% via τ=2.50, near-ignored"],
        "Label": ["FACT", "FACT", "FACT", "STRUCTURAL PRIOR",
                  "REPORTED CLAIM", "REPORTED CLAIM", "WEAK SIGNAL",
                  "SENTIMENT", "SENTIMENT", "WEAK SIGNAL"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(hierarchy_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Data sources breakdown ──
    st.header("What We Fetch Each Run")
    st.markdown("""
| Source | Tier | Volume | Claude? | Cost | Notes |
|--------|------|--------|---------|------|-------|
| ECI VTR App | 1 | AC-wise turnout | No | Free | Falls back to seeded Phase 1 district data if API down |
| Form 17C (manual) | 1 | Booth-level votes | No | Free | Enter via Manual Input page; single-party entries flagged |
| Indian Express, The Hindu, NDTV, HT, TOI, Telegraph, Wire, Scroll | 2 | ~15–25 articles | **Yes** | ~$0.08/run | Full body via trafilatura; only WB-relevant sent to Claude |
| ABP Ananda, Zee 24 Ghanta | 3 | ~5–10 Bengali articles | **Yes** | included above | Bengali RSS feeds |
| Google News RSS | 4 | ~288 articles | No | Free | 7 English + 3 Bengali queries; auto-scored by keyword |
| GDELT | 4 | ~10–30 articles | No | Free | Fallback for WB-specific queries; often times out |
| Polymarket gamma API | 5 | 1 odds record | No | Free | TMC/BJP win probabilities; τ=2.00 weak statewide signal |
| YouTube | 6 | Comments | **Yes** | minimal | WB political channels; near-ignored weight τ=2.50 |

**Cost control:** Tier 4+ articles (Google News, GDELT) are **never sent to Claude**. They are auto-scored
with `credibility=0.50` and keyword-based region tags. Only Tier 1–3 (~22 articles typical) reach Claude,
costing **~$0.08 per full pipeline run** (down from ~$1.75 before this optimization).

**Three data availability tiers (from ECI framework):**
- **Available now (public):** ECI VTR app, CEO WB elector data, post-scrutiny press notes
- **Available only to candidates/agents:** Form 17C Part I (booth-level votes — enter via Manual Input)
- **Available only after counting:** Form 20 (booth-wise result — use to backtest model post-May 4)
""")

    st.divider()

    # ── Step 1: Prior ──
    st.header("Step 1 — The Starting Point (Prior)")
    st.markdown("""
**What do we believe before reading today's news?**

We use a **reproducible margin formula** grounded in real data from two elections.
Every number is traceable to a specific ECI result row — nothing is synthetic or guessed.

```
base_margin  =  0.45 × ls_2024_tmc_minus_bjp
             +  0.25 × assembly_2021_tmc_minus_bjp

p_tmc_raw    =  logistic(base_margin / 7pp)   ← σ = 7 percentage points

p_others     =  f(2021 congress+left share, minority_share)
                (Congress/ISF recovery in Murshidabad/Malda)

win_prior    =  p_tmc_raw × (1 − p_others)
```

**Data sources (real, not synthetic):**
- **2021 assembly**: 292/294 ACs from tecoholic/Election2021 (real ECI candidate-level data)
- **2024 LS**: All 42 WB Lok Sabha seat results — hardcoded from ECI/Wikipedia
- **Others proxy**: 2021 left+congress voteshare + minority_share from SIR data

**Why this formula?**

2024 LS (0.45 weight) is the most recent and direct signal of current political wind.
2021 assembly (0.25 weight) captures constituency-level structure that LS aggregates miss.
σ = 7pp: at a 7pp TMC lead, win probability = 73%. At 14pp lead = 88%. At 0pp = 50%.

**Three-way competition (Congress/Left/ISF):**

In Muslim-majority seats (Murshidabad, Malda), the contest is TMC vs Congress — not BJP.
p_others rises with minority_share: high-minority seats frequently go to Congress/ISF,
removing them from the TMC vs BJP pool before either can claim them.

**Regional baselines emerging from real data:**
""")

    prior_df = pd.DataFrame([
        {"Region": "North Bengal", "Seats": 54, "TMC Win Prior": "~44%", "p_others": "~13%", "Basis": "BJP competitive; Malda Congress seats pull Others up"},
        {"Region": "Jangalmahal", "Seats": "~28", "TMC Win Prior": "~65%", "p_others": "~9%", "Basis": "TMC won back many seats in 2021; 2024 LS mixed"},
        {"Region": "Medinipur", "Seats": "~50", "TMC Win Prior": "~47%", "p_others": "~1%", "Basis": "Close — Tamluk/Contai BJP-leaning; Ghatal TMC"},
        {"Region": "Urban Kolkata", "Seats": "~65", "TMC Win Prior": "~74%", "p_others": "~4%", "Basis": "TMC stronghold; all LS seats won with big margins"},
        {"Region": "South Bengal Rural", "Seats": "~97", "TMC Win Prior": "~62%", "p_others": "~16%", "Basis": "Murshidabad Congress seats + ISF factor"},
    ])
    st.dataframe(prior_df, use_container_width=True, hide_index=True)

    st.markdown("""
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

**BJP competitiveness conditioning — new in this version:**

SIR only helps BJP if BJP is actually competitive in that seat. In Muslim-majority seats
(Malda, Murshidabad, Uttar Dinajpur), deleted Muslim votes shift to **Congress or ISF — not BJP**.
BJP was never going to win those seats regardless. We scale down the SIR delta accordingly:

```
bjp_competitive_factor = (1 − (minority_share − 0.20).clip(0) × 1.25).clip(0.25, 1.0)
```

| Minority share | BJP factor | Interpretation |
|---------------|-----------|----------------|
| 0.10 (Hindu-majority) | 1.00 | Full SIR benefit — BJP competitive |
| 0.27 (WB average) | 0.91 | Slight reduction |
| 0.50 (Nandagram-type, tactical voting) | 0.63 | Significant reduction |
| 0.80 (Murshidabad) | 0.25 | Minimal BJP benefit — Congress/ISF fight |

*(Source: Dr. Kartikeya Batra analysis, The Red Mic, April 2026)*

**TMC win probability reduction** (computed in probability space, then converted exactly to logit):
```
delta_p = deletion_rate × excess_lean × bjp_competitive_factor
win_adj = win_prior − delta_p
mu_adj  = logit(win_adj)           ← exact conversion, no approximation
```

**Why exact conversion instead of "× 4"?**
The × 4 shortcut is the derivative of logit at p = 0.5: `d/dp[log(p/(1−p))] = 1/(p(1−p)) = 4 at p=0.5`.
It's a linear approximation that breaks down away from 0.5 (e.g. at p=0.88 for Urban Kolkata,
the true scaling factor is 1/(0.88×0.12) ≈ 9.5, not 4). We use the exact logit conversion instead.

Extreme example — **Samserganj** (95% Muslim, 25% deletion rate):
```
tmc_lean    = 0.45 + 0.95 × 0.40 = 0.83
excess_lean = 0.83 − 0.50 = 0.33
bjp_factor  = (1 − (0.95 − 0.20) × 1.25).clip(0.25) = 0.25   ← Congress fight, not BJP
delta_p     = 0.25 × 0.33 × 0.25 = 0.021   → smaller drop (vs 0.083 in old model)
```

Without this conditioning, the old model overstated BJP's gain in Muslim-majority seats
where Congress, not BJP, is the actual beneficiary.
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

    st.markdown("#### IPAC Shutdown (TMC disadvantage) — Added April 2026")
    st.markdown("""
**Source:** IPAC (I-PAC, Indian Political Action Committee — Prashant Kishore's firm) shut down
operations **mid-election, April 2026**. Founder **Vinesh Chandel** arrested; **Rishiraj Singh**
summoned. Mamata Banerjee personally visited Pratik Jain's home after the January 2026 ED raid
— demonstrating how irreplaceable IPAC was to TMC's campaign infrastructure.
*(The Red Mic, April 2026; India Today; The Wire)*

**What IPAC ran for TMC:**
- Booth-level committee planning across all 294 constituencies
- Social media strategy and real-time counter-messaging
- Rally logistics and crowd management
- *Didi Ke Bolo* direct feedback loop → informed targeted scheme delivery
- *Duare Sarkar* coordination (the schemes that reversed 2019 damage in 2021)

**Why this matters more than it sounds:**
IPAC's 2021 strategy is precisely what turned TMC's 2019 catastrophe (BJP ahead in 121 segments)
into a 213-seat victory. They identified that Mamata's brand was still strong but TMC workers
had lost connection with voters — and fixed it through direct outreach programs. Without IPAC,
TMC reverts to relying on its cadre network, which has shown signs of corruption and disconnect.

| Region | IPAC shutdown effect on TMC win prob | Reasoning |
|--------|--------------------------------------|-----------|
| North Bengal | −1.0pp | BJP already structurally strong; IPAC less decisive |
| Jangalmahal | −1.2pp | IPAC handled tribal booth targeting in this belt |
| Medinipur | −1.8pp | Swing seats where micro-management is decisive |
| **Urban Kolkata** | **−2.5pp** | IPAC's biggest operation: urban coordination, social media, Mamata brand management |
| **South Bengal Rural** | **−2.2pp** | IPAC's rural booth network was TMC's backbone in minority + rural belt |

Uncertainty: **±40%** (some IPAC workers may continue informally; TMC state apparatus compensates partially).
""")

    st.markdown("#### Net Effect on Forecast — All Three Factors")
    st.markdown("""
All three effects combined. IPAC shutdown flips Urban Kolkata from TMC-positive to slightly negative.

| Region | RSS Δp | CM face Δp | IPAC Δp | **Net Δp** | Seats |
|--------|---------|------------|---------|-----------|-------|
| North Bengal | −0.030 | +0.010 | −0.010 | **−0.030** | 54 |
| Jangalmahal | −0.020 | +0.010 | −0.012 | **−0.022** | 25 |
| Medinipur | −0.015 | +0.015 | −0.018 | **−0.018** | 27 |
| **Urban Kolkata** | −0.012 | +0.030 | **−0.025** | **−0.007** | 68 |
| **South Bengal Rural** | −0.018 | +0.020 | **−0.022** | **−0.020** | 120 |

**Key shift vs. previous model:** Urban Kolkata flipped from **+0.018** (net TMC gain) to **−0.007**
(slight net loss) once IPAC shutdown is added. The CM face vacuum advantage (+3.0pp) is nearly wiped
out by IPAC's loss (−2.5pp). South Bengal Rural went from near-neutral (+0.002) to −0.020 — the
largest absolute change given that region's 120-seat weight.

**LEFT_OTHERS_FIXED raised to 20** (from 15): Congress wins 4–6 seats in Malda/Murshidabad
where SIR's BJP-competitiveness conditioning now correctly identifies that deleted Muslim votes
there flow to Congress, not BJP.

Current forecast with all three factors: **TMC p50 ≈ 189 seats, BJP p50 ≈ 85 seats,
P(TMC majority) = 72%** — before any news signals have been applied.
""")

    st.divider()

    # ── Step 3: Bayesian Update ──
    st.header("Step 3 — Daily Bayesian Update (Turnout + News)")
    st.markdown("""
**How Phase turnout and today's headlines move the forecast — and why source quality determines how much.**

Each day the pipeline runs two parallel signal streams, then blends them before the Bayesian update:

**Stream A — Phase turnout (Tier 1 ECI):**
AC/district turnout from ECI VTR App, CEO WB, and Form 17C booth agents.
Turnout swing vs 2021 is converted to a regional signal:
```
signal = (turnout_current − turnout_2021) / 15pp   [capped at ±1.0]

BJP-competitive areas (North Bengal, Jangalmahal):  BJP signal = swing × 0.6
TMC strongholds (Urban Kolkata, South Bengal Rural): TMC signal = swing × 0.4 (inverted)
```
Phase 1 (April 23): overall 93.19% post-scrutiny — ~+13pp swing vs 2021 (≈76.9%).
Cooch Behar 96.2% → strong BJP-belt signal. Kalimpong 83.04% → modest GJM/hill signal.

**Stream B — News signals (Tier 2–6):**
RSS articles → Claude filter (Tier 1–3 only, ~$0.08/run) → regional scores −1 to +1.
Tier 4+ (Google News, GDELT) auto-scored by keyword, never sent to Claude.

**Blend before update:**
```
combined_signal = 0.7 × news_signal + 0.3 × turnout_signal
```

**Critically: not all sources are equal.** TAU (observation noise) varies by source tier.
A low TAU means the model trusts the signal more and moves further.
""")

    st.markdown("#### Source Hierarchy & TAU Values")
    st.markdown("""
| Source tier | Examples | TAU (noise) | Effective news weight* | Role in model |
|-------------|----------|------------|----------------------|---------------|
| **Tier 1 — Official** | ECI/CEO WB turnout, Form 20 | **0.40** | **~90%** | Near ground truth — moves model strongly |
| **Tier 2 — Quality news** | Indian Express, The Hindu, NDTV, TOI | **0.80** | **~69%** | Reliable reporting — significant weight |
| **Tier 3 — Regional** | ABP Ananda, Zee 24 Ghanta | **1.10** | **~54%** | Good signal, some bias — medium weight |
| **Tier 4 — Aggregated** | Google News RSS, GDELT | **1.50** | **~39%** | Noisy aggregation — low weight |
| **Tier 5 — Markets** | Polymarket | **2.00** | **~26%** | Sentiment only — ~5% of total model |
| **Tier 6 — Social** | YouTube, WhatsApp | **2.50** | **~19%** | Very weak signal, near-ignored |

*Weight = 1 / (1 + τ²/σ²) where σ_prior ≈ 1.20. Plus tier multiplier on credibility (Tier 1 ×2.0, Tier 6 ×0.2).
""")

    st.markdown("#### Source weights per your framework")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("""
| Factor | Framework weight |
|--------|----------------|
| Historical vote base | **35%** |
| Turnout + SIR | **25%** |
| Candidate / local | **15%** |
| BJP/RSS organization | **10%** |
| Congress/Left leakage | **7%** |
| Polls / markets | **5%** |
| News / social momentum | **3%** |
""")
    with col_w2:
        st.markdown("""
**How the model implements this:**
- Steps 1–2b (prior + SIR + org factors) encode the top **67%** structurally
- Tier 1 ECI data (τ=0.40) represents turnout/SIR live updates (**25%**)
- Tier 2 news moves the model modestly (**~3% of news weight**)
- Polymarket enters as statewide weak observation (**~5% market weight**)
- Tier 6 social (τ=2.50) barely registers — correct, given its reliability

The prior's structural weight (~92%) means a single day of BJP-favoring
headlines **cannot** collapse the forecast.
""")

    with st.expander("Worked example — Urban Kolkata, Tier 2 signal = +0.35 vs Tier 6 signal = +0.35", expanded=False):
        st.markdown("""
**Prior** (from Steps 1–2b):
```
μ_prior = +1.97   (logit of ~88% TMC win rate after org factor adjustment)
σ_prior =  1.20
```

**Same signal (+0.35), different source tiers:**
```
obs = μ_prior + 0.35 × 0.30 = +2.061   (same for both)

Tier 2 (NDTV)  τ=0.80:
  precision_prior = 1/1.44 = 0.694
  precision_obs   = 1/0.64 = 1.563
  μ_post = (0.694×1.97 + 1.563×2.061) / 2.257 = +2.027   → 88.0% → 88.5% (+0.5pp)

Tier 6 (YouTube) τ=2.50:
  precision_obs   = 1/6.25 = 0.160
  μ_post = (0.694×1.97 + 0.160×2.061) / 0.854 = +1.987   → 87.9% (+0.1pp)
```

**Result:** Same signal from a Tier 2 source moves the model **5× more** than from Tier 6.
ECI official data (τ=0.40) would move it **~15× more** than YouTube.
""")

    st.markdown("#### Polymarket as statewide weak signal")
    st.markdown("""
Polymarket WB election odds are fetched daily and enter the model as a **statewide observation**
applied to all 294 constituencies simultaneously (τ=2.00, ~5% effective weight).

```
tmc_market_prob → logit → obs_market   (e.g. 70% odds → logit = +0.847)
Applied to all regions with precision_market = 1/4.0 = 0.25

If prior μ=+0.5 and market says logit=+0.847:
  μ_post ≈ 0.5 + 0.25/(0.694+0.25) × (0.847-0.5) ≈ +0.592
  TMC win prob: 62% → 64%  (small nudge statewide)
```

This respects the framework instruction: *"Use Polymarket as sentiment, not ground truth."*
Market prices reflect trader expectations, not booth-level evidence. The model weights it
at ~5% accordingly.
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

            mkt_manual = _market_data()
            forecast_m, _ = run_full_pipeline(manual_signals, market_data=mkt_manual)
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

    st.divider()

    # ── Phase Turnout Entry ─────────────────────────────────────────────────
    st.header("Phase Turnout Data Entry")
    st.caption(
        "Enter AC-wise or district-wise turnout from ECI VTR app, CEO WB, or post-scrutiny press notes. "
        "**Data is never deleted** — new entries update existing rows. "
        "Source type determines confidence: official_scrutiny > official_vtr > media > manual."
    )

    with st.expander("📊 Enter district/AC turnout", expanded=False):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            t_phase    = st.selectbox("Phase", [1, 2], key="t_phase")
            t_district = st.text_input("District", key="t_district", placeholder="Cooch Behar")
            t_ac_name  = st.text_input("AC Name (or district if AC unavailable)", key="t_ac_name", placeholder="Cooch Behar")
            t_ac_no    = st.number_input("AC No (optional)", min_value=0, value=0, key="t_ac_no")
        with col_t2:
            t_region   = st.selectbox("Region", ["north_bengal","jangalmahal","medinipur","urban_kolkata","south_bengal_rural"], key="t_region")
            t_initial  = st.number_input("Initial turnout % (VTR app)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t_initial")
            t_scrutiny = st.number_input("Post-scrutiny turnout % (RO revised)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t_scrutiny")
            t_2021     = st.number_input("2021 turnout % (historical)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t_2021")
        with col_t3:
            t_male_pct = st.number_input("Male turnout %", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t_male")
            t_fem_pct  = st.number_input("Female turnout %", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t_female")
            t_src_type = st.selectbox("Source type", ["official_scrutiny","official_vtr","media","manual"], key="t_src_type")
            t_src      = st.text_input("Source name", key="t_src", placeholder="newsonair / ECI press note")
            t_notes    = st.text_input("Notes", key="t_notes", placeholder="optional context")

        if st.button("💾 Save Turnout Entry", key="save_turnout"):
            from pipeline.db import upsert_phase_turnout
            row = {
                "phase": t_phase, "district": t_district, "ac_no": t_ac_no or None,
                "ac_name": t_ac_name, "region": t_region,
                "initial_turnout_percent": t_initial or None,
                "post_scrutiny_turnout_percent": t_scrutiny or None,
                "turnout_2021": t_2021 or None,
                "male_turnout_percent": t_male_pct or None,
                "female_turnout_percent": t_fem_pct or None,
                "source": t_src or "manual", "source_type": t_src_type,
                "confidence_score": {"official_scrutiny": 0.95, "official_vtr": 0.85, "media": 0.70, "manual": 0.60}[t_src_type],
                "notes": t_notes or None,
            }
            conn2 = _db()
            upsert_phase_turnout(conn2, [row])
            st.success(f"Saved: {t_ac_name} Phase {t_phase} turnout.")
            st.cache_data.clear()

    # ── Form 17C Booth Entry ─────────────────────────────────────────────────
    st.header("Form 17C — Booth-level Votes Polled")
    st.caption(
        "Enter booth-level data from Form 17C Part I (votes recorded in EVM, handed to polling agents at close of poll). "
        "If from a single party agent, set Source Party and confidence will be capped at 0.65 pending cross-check."
    )
    with st.expander("🗳️ Enter Form 17C booth data", expanded=False):
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            b_phase   = st.selectbox("Phase", [1, 2], key="b_phase")
            b_ac_name = st.text_input("AC Name", key="b_ac_name", placeholder="Cooch Behar")
            b_ac_no   = st.number_input("AC No", min_value=0, value=0, key="b_ac_no")
            b_booth   = st.number_input("Booth No", min_value=1, value=1, key="b_booth")
            b_ps_name = st.text_input("Polling Station Name", key="b_ps_name")
        with col_b2:
            b_electors = st.number_input("Total electors on roll", min_value=0, value=0, key="b_electors")
            b_polled   = st.number_input("Total votes polled (Form 17C)", min_value=0, value=0, key="b_polled")
            b_male     = st.number_input("Male votes", min_value=0, value=0, key="b_male")
            b_female   = st.number_input("Female votes", min_value=0, value=0, key="b_female")
        with col_b3:
            b_src_type = st.selectbox("Source type", ["form17c","official_vtr","party_agent","media"], key="b_src_type")
            b_party    = st.selectbox("Source party", ["official","TMC","BJP","INC","CPM","other"], key="b_party")
            b_t2021    = st.number_input("2021 turnout % at this booth (if known)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="b_t2021")
            b_notes    = st.text_input("Notes", key="b_notes")

        st.warning(
            "⚠️ Form 17C from a single party should be treated as unverified (confidence 0.5–0.65) "
            "until cross-checked against another party's or official figures."
        )

        if st.button("💾 Save Form 17C Entry", key="save_form17c"):
            from pipeline.db import upsert_booth_turnout
            polled = b_polled or None
            electors = b_electors or None
            turnout_pct = round(100.0 * b_polled / b_electors, 2) if b_electors and b_polled else None
            conf = 0.90 if b_party == "official" else 0.65 if b_src_type == "form17c" else 0.50
            row = {
                "phase": b_phase, "ac_no": b_ac_no or None, "ac_name": b_ac_name,
                "booth_no": b_booth, "polling_station_name": b_ps_name,
                "total_electors": electors, "votes_polled": polled,
                "male_votes": b_male or None, "female_votes": b_female or None,
                "turnout_percent": turnout_pct,
                "turnout_vs_2021": round(turnout_pct - b_t2021, 2) if turnout_pct and b_t2021 else None,
                "source": f"Form 17C ({b_party})", "source_type": b_src_type,
                "source_party": b_party, "confidence_score": conf,
                "notes": b_notes or None,
            }
            conn3 = _db()
            upsert_booth_turnout(conn3, [row])
            st.success(f"Saved: {b_ac_name} Booth {b_booth} — {turnout_pct:.1f}% turnout" if turnout_pct else "Saved.")
            st.cache_data.clear()
