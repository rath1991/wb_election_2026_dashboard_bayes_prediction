import os
import requests
from datetime import date, timedelta

WB_QUERIES_EN = [
    "West Bengal election TMC BJP",
    "West Bengal assembly election voter",
    "Mamata Banerjee BJP Bengal election",
    "West Bengal SIR voter deletion",
    "North Bengal Jangalmahal election",
    "Bengal election 2026 results turnout",
]
WB_QUERIES_BN = [
    "পশ্চিমবঙ্গ নির্বাচন",
    "পশ্চিমবঙ্গ ভোটার তালিকা",
    "তৃণমূল বিজেপি",
]

WB_POLITICAL_YOUTUBE_CHANNELS = [
    # ABP Ananda (major Bengali news channel)
    "UCU0uiUUuW7E1vw3VbZyGHiA",
    # TV9 Bangla
    "UCH_lkEBSvmq3hHGlQJvnfAA",
    # Zee 24 Ghanta
    "UCCeJHiMvs2XBTAXE_vgWj3g",
]


def fetch_gdelt(query: str, days_back: int = 1) -> list[dict]:
    since = (date.today() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 75,
        "startdatetime": since,
        "format": "json",
        "sort": "DateDesc",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return [
            {
                "source": a.get("domain", ""),
                "url": a.get("url", ""),
                "headline": a.get("title", ""),
                "body_snippet": a.get("title", ""),
                "language": a.get("language", "eng"),
                "fetch_date": str(date.today()),
            }
            for a in data.get("articles", [])
        ]
    except Exception as e:
        print(f"GDELT error [{query[:40]}]: {e}")
        return []


def fetch_newsapi(query: str, days_back: int = 1) -> list[dict]:
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        return []
    url = "http://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "keyword": query,
        "keywordSearchMode": "simple",
        "lang": ["eng", "ben"],
        "dateStart": str(date.today() - timedelta(days=max(days_back, 3))),
        "dateEnd": str(date.today()),
        "articlesCount": 50,
        "articlesSortBy": "date",
        "apiKey": api_key,
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", {}).get("results", [])
        return [
            {
                "source": (a.get("source") or {}).get("title", ""),
                "url": a.get("url", ""),
                "headline": a.get("title", ""),
                "body_snippet": (a.get("body") or a.get("title", ""))[:600],
                "language": a.get("lang", "eng"),
                "fetch_date": str(date.today()),
                "date": (a.get("date") or str(date.today())),
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        print(f"NewsAPI.ai error [{query[:40]}]: {e}")
        return []


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
                    channelId=channel_id,
                    part="id,snippet",
                    maxResults=3,
                    order="date",
                    type="video",
                    q="নির্বাচন OR election OR BJP OR TMC",
                ).execute()

                for item in search_resp.get("items", []):
                    video_id = item["id"].get("videoId", "")
                    if not video_id:
                        continue
                    title = item["snippet"]["title"]

                    try:
                        comments_resp = youtube.commentThreads().list(
                            videoId=video_id,
                            part="snippet",
                            maxResults=max_per_channel,
                            order="relevance",
                        ).execute()
                        for thread in comments_resp.get("items", []):
                            comment = thread["snippet"]["topLevelComment"]["snippet"]
                            results.append({
                                "source": f"YouTube:{channel_id}",
                                "url": f"https://youtube.com/watch?v={video_id}",
                                "headline": title,
                                "body_snippet": comment.get("textOriginal", "")[:400],
                                "language": "bn_or_en",
                                "fetch_date": str(date.today()),
                            })
                    except Exception:
                        pass
            except Exception as e:
                print(f"YouTube channel {channel_id} error: {e}")

        return results
    except Exception as e:
        print(f"YouTube init error: {e}")
        return []


def fetch_all(days_back: int = 1) -> list[dict]:
    articles = []

    for q in WB_QUERIES_EN:
        articles.extend(fetch_gdelt(q, days_back))
        articles.extend(fetch_newsapi(q, days_back))

    for q in WB_QUERIES_BN:
        articles.extend(fetch_gdelt(q, days_back))

    articles.extend(fetch_youtube_comments(WB_POLITICAL_YOUTUBE_CHANNELS))

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique = []
    for a in articles:
        url = a.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(a)
        elif not url:
            unique.append(a)  # keep URL-less items (e.g. YT comments with same video URL)

    print(f"Fetched {len(unique)} unique articles (from {len(articles)} total)")
    return unique
