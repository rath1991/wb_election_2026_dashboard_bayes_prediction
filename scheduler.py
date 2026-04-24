"""
Daily pipeline runner for WB 2026 elections dashboard.
Usage:
  python scheduler.py --once    # run once and exit (used by Render cron)
  python scheduler.py           # run immediately then schedule daily at 07:00 IST
"""
import argparse
import sys
from datetime import date
from dotenv import load_dotenv

load_dotenv()

from pipeline.fetchers import fetch_all
from pipeline.filter_extract import filter_and_extract, aggregate_regional_signals, extract_bjp_conditions
from pipeline.bayesian import run_full_pipeline
from pipeline.db import (
    get_conn,
    upsert_forecast,
    insert_articles,
    upsert_regional_signals,
    upsert_bjp_conditions,
    get_latest_forecast,
)


def run_pipeline():
    print(f"\n{'='*50}")
    print(f"WB 2026 Pipeline — {date.today()}")
    print(f"{'='*50}")

    client = get_conn()

    # 1. Fetch
    print("\n[1/5] Fetching news and social data...")
    articles = fetch_all(days_back=1)
    print(f"      {len(articles)} articles fetched")

    if not articles:
        print("      No articles found. Check API keys and network. Skipping.")
        return

    # 2. Claude filter + signal extraction
    print("\n[2/5] Running Claude analysis (noise filter + signal extraction)...")
    enriched = filter_and_extract(articles)
    signal_count = sum(1 for a in enriched if not a.get("is_noise"))
    noise_count = len(enriched) - signal_count
    print(f"      {signal_count} signal articles, {noise_count} noise filtered out")

    # 3. Store articles
    print("\n[3/5] Storing articles to database...")
    insert_articles(client, enriched)

    # 4. Aggregate signals
    print("\n[4/5] Aggregating regional signals and BJP conditions...")
    signals = aggregate_regional_signals(enriched)
    for region, sig in signals.items():
        print(f"      {region:25s}: {sig['signal_strength']:+.3f} ({sig['article_count']} articles)")
    upsert_regional_signals(client, date.today(), signals)

    conditions = extract_bjp_conditions(enriched)
    upsert_bjp_conditions(client, date.today(), conditions)
    met = [k for k, v in conditions.items() if v["status"] == "green"]
    print(f"      BJP conditions met: {len(met)}/7 — {', '.join(met) if met else 'none'}")

    # 5. Bayesian model update
    print("\n[5/5] Running Bayesian model...")
    prev = get_latest_forecast(client)
    prev_tmc_p50 = prev["tmc_p50"] if prev else None

    forecast, _ = run_full_pipeline(signals)
    forecast["prev_tmc_p50"] = prev_tmc_p50
    upsert_forecast(client, date.today(), forecast)

    delta = ""
    if prev_tmc_p50:
        d = forecast["tmc_p50"] - prev_tmc_p50
        delta = f"  (Δ {'+' if d >= 0 else ''}{d} vs yesterday)"

    print(f"\n{'─'*50}")
    print(f"FORECAST {date.today()}")
    print(f"  TMC:  {forecast['tmc_p5']}–{forecast['tmc_p95']}  (median {forecast['tmc_p50']}){delta}")
    print(f"  BJP:  {forecast['bjp_p5']}–{forecast['bjp_p95']}  (median {forecast['bjp_p50']})")
    print(f"  P(TMC majority): {forecast['p_tmc_win']:.1%}")
    print(f"  P(Hung):         {forecast['p_hung']:.1%}")
    print(f"  P(BJP majority): {forecast['p_bjp_win']:.1%}")
    print(f"  SIR uncertainty band: ±{forecast['sir_uncertainty_band']//2} seats")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(description="WB 2026 Elections — Daily Pipeline")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for Render cron)")
    args = parser.parse_args()

    if args.once:
        try:
            run_pipeline()
        except Exception as e:
            print(f"Pipeline failed: {e}")
            sys.exit(1)
        return

    # Interactive / persistent mode
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    @scheduler.scheduled_job("cron", hour=7, minute=0)
    def daily():
        try:
            run_pipeline()
        except Exception as e:
            print(f"Scheduled pipeline error: {e}")

    print("Scheduler started. Running immediately, then daily at 07:00 IST.")
    run_pipeline()  # Run once immediately
    scheduler.start()


if __name__ == "__main__":
    main()
