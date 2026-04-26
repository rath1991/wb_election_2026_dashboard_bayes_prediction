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
