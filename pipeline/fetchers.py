"""
Fetchers for WB 2026 election intelligence.

Source hierarchy (per user framework):
  Tier 1 — Official: ECI, CEO WB, Form 20, affidavits
  Tier 2 — Tier-1 news: Indian Express, The Hindu, Reuters, PTI, Telegraph India,
            Times of India, Hindustan Times, NDTV
  Tier 3 — Regional: Anandabazar Patrika, Bartaman, TV9 Bangla, ABP Ananda
  Tier 4 — Aggregated: Google News RSS, GDELT
  Tier 5 — Markets / sentiment: Polymarket
  Tier 6 — Social / YouTube (weakest signal)

Each article carries source_tier so Claude can weight credibility accordingly.
"""
import os
import time
import requests
import feedparser
import trafilatura
from datetime import date, timedelta
from urllib.parse import urlencode

# ── Source registry ───────────────────────────────────────────────────────────
# (name, rss_url, tier, language)
RSS_SOURCES = [
    # Tier 2 — national English
    ("Indian Express",    "https://indianexpress.com/section/india/feed/",              2, "en"),
    ("The Hindu",         "https://www.thehindu.com/news/national/feeder/default.rss",  2, "en"),
    ("NDTV",              "https://feeds.feedburner.com/ndtvnews-india-news",            2, "en"),
    ("Hindustan Times",   "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml", 2, "en"),
    ("Times of India",    "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms", 2, "en"),
    ("Telegraph India",   "https://www.telegraphindia.com/rss-feed/india.xml",          2, "en"),
    ("The Wire",          "https://thewire.in/rss",                                     2, "en"),
    ("Scroll",            "https://scroll.in/feed",                                     2, "en"),
    # Tier 3 — Bengali / regional
    ("ABP Ananda",        "https://bengali.abplive.com/rss.xml",                        3, "bn"),
    ("Zee 24 Ghanta",     "https://zeenews.india.com/bengali/rss/bengali.xml",          3, "bn"),
]

# Google News RSS queries — free, no key, aggregates hundreds of outlets
GOOGLE_NEWS_QUERIES_EN = [
    "West Bengal election 2026 TMC BJP",
    "West Bengal voter deletion SIR 2026",
    "Mamata Banerjee BJP Bengal election",
    "West Bengal assembly election turnout phase",
    "Sunil Bansal BJP Bengal RSS",
    "BJP CM face Bengal Suvendu Sukanta",
    "Bengal election constituency result 2026",
]
GOOGLE_NEWS_QUERIES_BN = [
    "পশ্চিমবঙ্গ নির্বাচন ২০২৬",
    "পশ্চিমবঙ্গ ভোটার তালিকা মুছে",
    "তৃণমূল বিজেপি ভোট",
]

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search?{params}"

# ECI / CEO WB official pages (scraped for turnout/SIR updates)
ECI_SOURCES = [
    ("CEO WB Turnout",    "https://ceowestbengal.wb.gov.in/", 1, "en"),
    ("ECI Voters Portal", "https://voters.eci.gov.in/",       1, "en"),
]

# Polymarket WB election market
POLYMARKET_URL = "https://polymarket.com/event/west-bengal-legislative-assembly-election-winner"

WB_POLITICAL_YOUTUBE_CHANNELS = [
    "UCU0uiUUuW7E1vw3VbZyGHiA",  # ABP Ananda
    "UCH_lkEBSvmq3hHGlQJvnfAA",  # TV9 Bangla
    "UCCeJHiMvs2XBTAXE_vgWj3g",  # Zee 24 Ghanta
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ── Core helpers ─────────────────────────────────────────────────────────────

def _fetch_body(url: str, timeout: int = 15) -> str:
    """Fetch full article body using trafilatura. Falls back to empty string."""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            return (text or "")[:1200]
    except Exception:
        pass
    return ""


def _parse_rss(source_name: str, rss_url: str, tier: int, lang: str,
               days_back: int = 2) -> list[dict]:
    """Parse an RSS feed and return article dicts."""
    cutoff = date.today() - timedelta(days=days_back)
    articles = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:30]:
            title = entry.get("title", "").strip()
            link  = entry.get("link", "").strip()
            summary = entry.get("summary", entry.get("description", ""))[:600]
            if not title or not link:
                continue
            # Filter to WB-relevant only (broad keywords)
            combined = (title + " " + summary).lower()
            if not any(kw in combined for kw in [
                "bengal", "bengali", "tmc", "mamata", "bjp",
                "kolkata", "কলকাতা", "বাংলা", "তৃণমূল", "বিজেপি",
                "election", "নির্বাচন", "voter", "ভোট",
            ]):
                continue
            articles.append({
                "source": source_name,
                "source_tier": tier,
                "url": link,
                "headline": title,
                "body_snippet": summary,
                "language": lang,
                "fetch_date": str(date.today()),
                "date": str(date.today()),
            })
    except Exception as e:
        print(f"RSS error [{source_name}]: {e}")
    return articles


def fetch_rss_all(days_back: int = 2) -> list[dict]:
    """Fetch from all registered RSS sources."""
    articles = []
    for name, url, tier, lang in RSS_SOURCES:
        batch = _parse_rss(name, url, tier, lang, days_back)
        articles.extend(batch)
        print(f"  RSS [{name}]: {len(batch)} articles")
        time.sleep(0.5)
    return articles


def fetch_google_news(days_back: int = 2) -> list[dict]:
    """Query Google News RSS for WB election terms."""
    articles = []
    for q in GOOGLE_NEWS_QUERIES_EN:
        params = urlencode({"q": q, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        url = GOOGLE_NEWS_RSS_BASE.format(params=params)
        batch = _parse_rss("Google News", url, 4, "en", days_back)
        articles.extend(batch)
        time.sleep(0.3)
    for q in GOOGLE_NEWS_QUERIES_BN:
        params = urlencode({"q": q, "hl": "bn", "gl": "IN", "ceid": "IN:bn"})
        url = GOOGLE_NEWS_RSS_BASE.format(params=params)
        batch = _parse_rss("Google News", url, 4, "bn", days_back)
        articles.extend(batch)
        time.sleep(0.3)
    print(f"  Google News: {len(articles)} articles total")
    return articles


def fetch_article_bodies(articles: list[dict], max_fetch: int = 40) -> list[dict]:
    """
    Enrich top articles with full body text via trafilatura.
    Only fetches for Tier 1–3 sources (official + quality news).
    Limits to max_fetch to avoid hammering servers.
    """
    fetched = 0
    for a in articles:
        if fetched >= max_fetch:
            break
        tier = a.get("source_tier", 4)
        if tier > 3:
            continue
        if a.get("body_full"):
            continue
        url = a.get("url", "")
        if not url:
            continue
        body = _fetch_body(url)
        if body:
            a["body_snippet"] = body[:1200]
            a["body_full"] = True
            fetched += 1
        time.sleep(0.8)  # be respectful
    print(f"  Full body fetched for {fetched} Tier 1–3 articles")
    return articles


def fetch_gdelt(query: str, days_back: int = 1) -> list[dict]:
    """GDELT fallback — free, no key, broad coverage."""
    since = (date.today() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 50,
        "startdatetime": since,
        "format": "json",
        "sort": "DateDesc",
    }
    try:
        r = requests.get(url, params=params, timeout=20, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        return [
            {
                "source": a.get("domain", ""),
                "source_tier": 4,
                "url": a.get("url", ""),
                "headline": a.get("title", ""),
                "body_snippet": a.get("title", ""),
                "language": a.get("language", "en"),
                "fetch_date": str(date.today()),
                "date": str(date.today()),
            }
            for a in data.get("articles", [])
        ]
    except Exception as e:
        print(f"GDELT error [{query[:40]}]: {e}")
        return []


def fetch_polymarket() -> dict | None:
    """
    Scrape Polymarket WB election odds.
    Returns {tmc_probability, bjp_probability, volume_usd, source} or None.
    Polymarket exposes a public API for market data.
    """
    try:
        # Polymarket gamma API (public, no auth required)
        api_url = "https://gamma-api.polymarket.com/events?tag=india&limit=50"
        r = requests.get(api_url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        events = r.json()
        for event in events:
            title = (event.get("title") or "").lower()
            if "west bengal" in title or "bengal" in title and "assembly" in title:
                markets = event.get("markets", [])
                tmc_p = bjp_p = volume = None
                for m in markets:
                    outcome = (m.get("outcomePrices") or [])
                    q = (m.get("question") or "").lower()
                    if "tmc" in q or "trinamool" in q:
                        tmc_p = float(outcome[0]) if outcome else None
                    elif "bjp" in q or "bharatiya" in q:
                        bjp_p = float(outcome[0]) if outcome else None
                    vol = m.get("volume")
                    if vol:
                        volume = (volume or 0) + float(vol)
                if tmc_p is not None or bjp_p is not None:
                    print(f"  Polymarket: TMC={tmc_p}, BJP={bjp_p}, Vol=${volume:,.0f}" if volume else "")
                    return {
                        "tmc_probability": tmc_p,
                        "bjp_probability": bjp_p,
                        "volume_usd": volume,
                        "source": "Polymarket",
                        "date": str(date.today()),
                    }
    except Exception as e:
        print(f"Polymarket fetch error: {e}")
    return None


def fetch_youtube_comments(channel_ids: list[str], max_per_channel: int = 20) -> list[dict]:
    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        return []
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)
        results = []
        for channel_id in channel_ids[:5]:
            try:
                search_resp = youtube.search().list(
                    channelId=channel_id, part="id,snippet",
                    maxResults=3, order="date", type="video",
                    q="নির্বাচন OR election OR BJP OR TMC",
                ).execute()
                for item in search_resp.get("items", []):
                    video_id = item["id"].get("videoId", "")
                    if not video_id:
                        continue
                    title = item["snippet"]["title"]
                    try:
                        comments_resp = youtube.commentThreads().list(
                            videoId=video_id, part="snippet",
                            maxResults=max_per_channel, order="relevance",
                        ).execute()
                        for thread in comments_resp.get("items", []):
                            comment = thread["snippet"]["topLevelComment"]["snippet"]
                            results.append({
                                "source": f"YouTube:{channel_id}",
                                "source_tier": 6,
                                "url": f"https://youtube.com/watch?v={video_id}",
                                "headline": title,
                                "body_snippet": comment.get("textOriginal", "")[:400],
                                "language": "bn_or_en",
                                "fetch_date": str(date.today()),
                                "date": str(date.today()),
                            })
                    except Exception:
                        pass
            except Exception as e:
                print(f"YouTube channel {channel_id} error: {e}")
        return results
    except Exception as e:
        print(f"YouTube init error: {e}")
        return []


# ── ECI Voter Turnout + Phase 1 seed data ────────────────────────────────────

# Phase 1 post-scrutiny district turnout — sourced from newsonair.gov.in + ECI press notes
# April 23, 2026 polling. Updated after RO scrutiny.
PHASE1_DISTRICT_TURNOUT_SEED = [
    # (district, region, post_scrutiny_pct, turnout_2021_est, notes)
    ("Cooch Behar",    "north_bengal",       96.2, 83.0, "Highest in Phase 1; BJP-competitive Matua/Rajvanshi belt"),
    ("Alipurduar",     "north_bengal",       94.1, 82.0, "Forest belt; BJP stronghold"),
    ("Jalpaiguri",     "north_bengal",       93.8, 81.5, "Tea garden seats; swing territory"),
    ("Darjeeling",     "north_bengal",       89.2, 78.0, "Hill + plains split; GJM factor"),
    ("Kalimpong",      "north_bengal",       83.0, 74.0, "Lowest in Phase 1; GJM stronghold, GNLF competition"),
    ("Malda",          "north_bengal",       92.5, 80.5, "High Muslim concentration; TMC vs Congress fight"),
    ("Murshidabad",    "south_bengal_rural", 93.5, 81.0, "Muslim-majority; TMC vs Congress/ISF fight"),
    ("Uttar Dinajpur", "north_bengal",       91.8, 79.5, "High Muslim share; BJP non-competitive here"),
    ("Dakshin Dinajpur","north_bengal",      93.1, 80.0, "Mixed; BJP has presence in Hindu seats"),
    ("Birbhum",        "south_bengal_rural", 93.6, 81.5, "TMC stronghold; Anubrata Mondal territory"),
    ("Paschim Burdwan","jangalmahal",        92.4, 80.0, "Asansol; coal belt; BJP industrial vote"),
    ("Bankura",        "jangalmahal",        94.0, 82.0, "Tribal belt; BJP competitive"),
    ("Purulia",        "jangalmahal",        93.2, 81.0, "BJP stronghold; SC/ST concentration"),
    ("Jhargram",       "jangalmahal",        93.8, 82.5, "Jangalmahal heartland; BJP held 2021"),
]

# 2026 Phase 1 overall: 93.19% post-scrutiny (newsonair.gov.in)
PHASE1_OVERALL_TURNOUT = 93.19
# 2021 WB overall turnout: ~76.9% (ECI)
WB_2021_OVERALL_TURNOUT = 76.9


def fetch_eci_vtr(phase: int = 1) -> list[dict]:
    """
    Fetch AC-wise turnout from ECI Voter Turnout App.
    Falls back gracefully if API is unavailable — uses seed data.
    ECI VTR API: non-statutory, gives approximate 2-hourly trends.
    After scrutiny, RO/ARO updates booth-wise counts.
    """
    rows = []

    # Try ECI resultsapi endpoint (structure may change)
    try:
        url = f"https://resultsapi.eci.gov.in/ResultWebService/TurnoutResult?StateCode=S24&Phase={phase}"
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0 WB-Election-Dashboard/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("lstTurnout", []):
                rows.append({
                    "phase": phase,
                    "district": item.get("districtName", ""),
                    "ac_no": item.get("acNo"),
                    "ac_name": item.get("acName", ""),
                    "region": _infer_region(item.get("districtName", "")),
                    "total_electors": item.get("totalElectors"),
                    "initial_turnout_percent": item.get("turnoutPercent"),
                    "votes_polled": item.get("totalVotesCast"),
                    "male_turnout_percent": item.get("maleTurnout"),
                    "female_turnout_percent": item.get("femaleTurnout"),
                    "source": "ECI_VTR_API",
                    "source_type": "official_vtr",
                    "confidence_score": 0.85,
                })
            print(f"  ECI VTR API: {len(rows)} AC records for Phase {phase}")
            return rows
    except Exception as e:
        print(f"  ECI VTR API unavailable: {e} — using seed data")

    # Fallback: use seed data (post-scrutiny district-level)
    for district, region, pct, t2021, notes in PHASE1_DISTRICT_TURNOUT_SEED:
        rows.append({
            "phase": phase,
            "district": district,
            "ac_no": None,
            "ac_name": district,  # district-level entry when AC not available
            "region": region,
            "total_electors": None,
            "initial_turnout_percent": None,
            "post_scrutiny_turnout_percent": pct,
            "votes_polled": None,
            "turnout_2021": t2021,
            "source": "newsonair/ECI_press_note",
            "source_type": "official_scrutiny",
            "confidence_score": 0.92,
            "notes": notes,
        })
    print(f"  Phase {phase} seed data: {len(rows)} district records loaded")
    return rows


def _infer_region(district: str) -> str:
    d = district.lower()
    if any(x in d for x in ["cooch", "alipurduar", "jalpaiguri", "darjeeling",
                              "kalimpong", "malda", "dinajpur", "siliguri"]):
        return "north_bengal"
    if any(x in d for x in ["purulia", "bankura", "jhargram", "paschim burdwan", "birbhum"]):
        return "jangalmahal"
    if any(x in d for x in ["medinipur", "midnapore", "purba burdwan"]):
        return "medinipur"
    if any(x in d for x in ["kolkata", "howrah", "hooghly"]):
        return "urban_kolkata"
    return "south_bengal_rural"


def compute_turnout_regional_signals(turnout_rows: list[dict]) -> dict:
    """
    Convert phase turnout into regional signals for the Bayesian update.

    Signal logic:
    - turnout_swing_vs_2021 > 0 in BJP-competitive area → positive bjp signal
    - turnout_swing_vs_2021 > 0 in TMC stronghold → positive tmc signal
    - Very high swing (>15pp) suggests strong mobilization on one side

    Returns: {region: {"turnout_signal": float ∈ [-1,+1], "avg_swing": float, "district_count": int}}
    """
    from collections import defaultdict
    accum = defaultdict(list)

    BJP_COMPETITIVE = {"north_bengal", "jangalmahal"}
    TMC_STRONGHOLD  = {"urban_kolkata", "south_bengal_rural", "medinipur"}

    for r in turnout_rows:
        region = r.get("region")
        swing  = r.get("turnout_swing_vs_2021")
        if not region or swing is None:
            continue
        accum[region].append(swing)

    signals = {}
    for region, swings in accum.items():
        avg_swing = sum(swings) / len(swings)
        # Normalise: 15pp swing = strong signal (±1.0)
        raw = avg_swing / 15.0
        raw = max(-1.0, min(1.0, raw))

        if region in BJP_COMPETITIVE:
            # Higher turnout in BJP belts = BJP enthusiasm (or ECI-driven mobilisation)
            signal = raw * 0.6   # BJP +ve, muted (could also be counter-TMC)
        elif region in TMC_STRONGHOLD:
            # Higher turnout in TMC belts = TMC counter-mobilisation from SIR anger
            signal = raw * (-0.4)  # Slightly TMC -ve (more TMC voters = less BJP gain)
        else:
            signal = 0.0

        signals[region] = {
            "turnout_signal": round(signal, 3),
            "avg_swing_vs_2021": round(avg_swing, 2),
            "district_count": len(swings),
        }
    return signals


def fetch_all(days_back: int = 2) -> list[dict]:
    """
    Full fetch across all source tiers.
    Returns deduplicated list of articles sorted by source_tier (best first).
    """
    articles = []

    print("Fetching Tier 2–3: RSS from quality news outlets...")
    articles.extend(fetch_rss_all(days_back))

    print("Fetching Tier 4: Google News RSS...")
    articles.extend(fetch_google_news(days_back))

    print("Fetching Tier 4: GDELT fallback...")
    for q in ["West Bengal election TMC BJP 2026", "West Bengal SIR voter deletion"]:
        articles.extend(fetch_gdelt(q, days_back))

    print("Fetching Tier 6: YouTube...")
    articles.extend(fetch_youtube_comments(WB_POLITICAL_YOUTUBE_CHANNELS))

    # Deduplicate by URL
    seen: set[str] = set()
    unique = []
    for a in articles:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(a)
        elif not url:
            unique.append(a)

    # Sort by tier (Tier 1 first) so Claude sees best sources first
    unique.sort(key=lambda x: x.get("source_tier", 9))

    print(f"Fetching full body text for Tier 1–3 articles...")
    unique = fetch_article_bodies(unique, max_fetch=40)

    tier_counts = {}
    for a in unique:
        t = a.get("source_tier", 9)
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Total {len(unique)} unique articles: {tier_counts}")
    return unique
