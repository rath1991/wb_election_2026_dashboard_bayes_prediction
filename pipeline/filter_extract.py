"""
Uses Claude API (claude-sonnet-4-6) with prompt caching to:
1. Filter noise from fetched articles
2. Extract per-region sentiment signals
3. Score BJP win pathway conditions
"""
import os
import json
import anthropic
from datetime import date

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_KEY"])
    return _client


SYSTEM_PROMPT = """You are a non-partisan political analyst specializing in West Bengal (India) assembly elections 2026.

Context:
- TMC (Trinamool Congress) is the incumbent ruling party under CM Mamata Banerjee.
- BJP is the main opposition, trying hard to win WB for the first time.
- SIR (Special Summary Revision): 91 lakh voter names deleted from rolls, disproportionately in Muslim/minority areas, which are TMC strongholds. This is a major structural issue.
- Key regions: north_bengal (BJP competitive), jangalmahal (BJP's 2019 stronghold, weakened in 2024), medinipur (mixed), urban_kolkata (TMC stronghold), south_bengal_rural (TMC dominant, heavily affected by SIR).

SOURCE TIER SYSTEM — each article includes a source_tier field. Use it to anchor credibility:
  Tier 1 = Official ECI/CEO WB data — treat as ground truth (credibility 0.95–1.0)
  Tier 2 = Tier-1 national news (Indian Express, The Hindu, NDTV, Reuters, TOI, Telegraph India) — high credibility (0.75–0.90)
  Tier 3 = Regional Bengali news (ABP Ananda, Zee 24 Ghanta, Bartaman) — medium-high (0.60–0.80)
  Tier 4 = Aggregated (Google News, GDELT) — medium (0.50–0.70), verify claim before scoring high
  Tier 5 = Prediction markets — sentiment only, not factual (credibility 0.30–0.50)
  Tier 6 = Social/YouTube — treat as weak signal, high noise (credibility 0.20–0.45)

IMPORTANT: A claim is only as credible as its source tier. Do not give a Tier 6 YouTube comment the same score as a Tier 2 Indian Express report even if the content sounds authoritative.

For each article/comment in the input JSON array, return an object with:

1. "is_noise": true/false
   Mark TRUE if: pure party propaganda/PR, sensationalist without factual content, completely unrelated to WB election, duplicate framing, unverifiable anonymous rumor.

2. "credibility_score": 0.0–1.0
   Anchor to source_tier ranges above. Adjust within range based on content quality.
   High: ECI official data, established Tier 2 outlets with named sources and data
   Medium: Tier 3 regional outlets, Tier 2 opinion pieces
   Low: Tier 4–6, anonymous sources, unverified claims

3. "region_tags": list from ["north_bengal", "jangalmahal", "medinipur", "urban_kolkata", "south_bengal_rural", "statewide"]
   Which WB regions does this article provide signal for?

4. "signal_tags": object with scores in [-1.0, +1.0] — score ONLY based on explicit content, do not infer:
   - "tmc_momentum": +1 = strong TMC surge evidence, -1 = TMC collapse evidence
   - "bjp_momentum": +1 = strong BJP surge evidence, -1 = BJP collapse evidence
   - "sir_impact": +1 = SIR effectively suppressing TMC votes, -1 = SIR impact being reversed/challenged
   - "minority_consolidation": +1 = minorities consolidating behind TMC, -1 = minorities fragmenting
   - "turnout_signal": +1 = high broad-based turnout (structurally favors TMC), -1 = signs of suppression/low turnout in TMC areas
   - "rss_mobilization_signal": +1 = strong RSS/BJP organizational activity (shakha expansion, booth management, voter-awareness meetings, Sunil Bansal/Bhupendra Yadav involvement), -1 = BJP organizational failure/disarray
   - "bjp_leadership_signal": +1 = BJP announced credible CM face or strong unified leadership, -1 = leadership vacuum, factionalism (Suvendu vs Sukanta vs Dilip Ghosh), or negative news about BJP leadership

5. "bjp_conditions_hit": list of BJP win conditions evidenced in this article:
   ["north_bengal_sweep_35plus", "jangalmahal_hold_18plus", "medinipur_majority", "urban_kolkata_gain_10plus", "minority_fragmentation", "sir_voter_suppression_effective", "anti_incumbency_national", "rss_organizational_mobilization", "bjp_clear_cm_face"]

Return a valid JSON array, one object per input item, in the same order. No markdown, no explanation."""


def filter_and_extract(articles: list[dict]) -> list[dict]:
    if not articles:
        return []

    client = get_client()
    results = []
    batch_size = 20

    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        batch_input = [
            {
                "index": j,
                "source": a.get("source", ""),
                "source_tier": a.get("source_tier", 4),
                "headline": a.get("headline", "")[:200],
                "snippet": a.get("body_snippet", "")[:600],
            }
            for j, a in enumerate(batch)
        ]

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=6000,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Analyze these {len(batch)} articles and return a JSON array:\n\n"
                            + json.dumps(batch_input, ensure_ascii=False)
                        ),
                    }
                ],
            )

            raw = response.content[0].text.strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                for j, item in enumerate(parsed):
                    if j < len(batch):
                        batch[j].update({
                            "is_noise": bool(item.get("is_noise", True)),
                            "credibility_score": float(item.get("credibility_score", 0.3)),
                            "region_tags": item.get("region_tags", []),
                            "signal_tags": item.get("signal_tags", {}),
                            "bjp_conditions_hit": item.get("bjp_conditions_hit", []),
                            "date": str(date.today()),
                        })
                        results.append(batch[j])
            else:
                _mark_noise(batch, results)

            # Log cache usage if available
            if hasattr(response, "usage"):
                u = response.usage
                print(f"  Batch {i//batch_size+1}: cache_read={getattr(u,'cache_read_input_tokens',0)}, input={u.input_tokens}, output={u.output_tokens}")

        except Exception as e:
            print(f"Claude API error (batch {i//batch_size+1}): {e}")
            _mark_noise(batch, results)

    return results


def _mark_noise(batch: list[dict], results: list[dict]):
    for a in batch:
        a.update({
            "is_noise": True,
            "credibility_score": 0.3,
            "region_tags": [],
            "signal_tags": {},
            "bjp_conditions_hit": [],
            "date": str(date.today()),
        })
        results.append(a)


REGIONS = ["north_bengal", "jangalmahal", "medinipur", "urban_kolkata", "south_bengal_rural"]


def aggregate_regional_signals(enriched: list[dict]) -> dict:
    """
    Aggregate per-article signals into per-region signal_strength ∈ [-1, +1].
    Net signal = (tmc_momentum - bjp_momentum + 0.5*minority_consolidation + 0.3*turnout_signal
                  - 0.4*rss_mobilization_signal - 0.3*bjp_leadership_signal) / 2.5
    Weighted by credibility. Also returns top articles per region for citations.
    """
    accum = {r: {"weighted_sum": 0.0, "weight_total": 0.0, "article_count": 0,
                 "tier_sum": 0.0, "tier_count": 0, "top_articles": []} for r in REGIONS}

    # Source tier multiplier: higher-tier sources get amplified weight
    TIER_WEIGHT = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.6, 5: 0.3, 6: 0.2}

    for article in enriched:
        if article.get("is_noise"):
            continue
        tags = article.get("signal_tags", {})
        cred = float(article.get("credibility_score", 0.3))
        tier = int(article.get("source_tier", 4))
        tier_mult = TIER_WEIGHT.get(tier, 0.5)
        region_tags = article.get("region_tags", [])

        net = (
            tags.get("tmc_momentum", 0.0)
            - tags.get("bjp_momentum", 0.0)
            + 0.5 * tags.get("minority_consolidation", 0.0)
            + 0.3 * tags.get("turnout_signal", 0.0)
            - 0.4 * tags.get("rss_mobilization_signal", 0.0)
            - 0.3 * tags.get("bjp_leadership_signal", 0.0)
        ) / 2.5

        effective_regions = [r for r in region_tags if r in REGIONS]
        if not effective_regions:
            effective_regions = REGIONS  # statewide

        for region in effective_regions:
            effective_weight = cred * tier_mult
            accum[region]["weighted_sum"] += net * effective_weight
            accum[region]["weight_total"] += effective_weight
            accum[region]["article_count"] += 1
            accum[region]["tier_sum"] += tier
            accum[region]["tier_count"] += 1
            if len(accum[region]["top_articles"]) < 5:
                accum[region]["top_articles"].append({
                    "headline": article.get("headline", "")[:120],
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "credibility": round(cred, 2),
                    "net_signal": round(net, 3),
                })

    signals = {}
    for region in REGIONS:
        a = accum[region]
        if a["weight_total"] > 0:
            strength = max(-1.0, min(1.0, a["weighted_sum"] / a["weight_total"]))
        else:
            strength = 0.0
        # Sort top articles by abs(net_signal) descending so most impactful comes first
        top = sorted(a["top_articles"], key=lambda x: abs(x["net_signal"]), reverse=True)
        avg_tier = round(a["tier_sum"] / a["tier_count"], 2) if a["tier_count"] > 0 else 4.0
        signals[region] = {
            "signal_strength": round(strength, 4),
            "article_count": a["article_count"],
            "avg_source_tier": avg_tier,
            "top_articles": top,
        }

    return signals


def extract_bjp_conditions(enriched: list[dict]) -> dict:
    CONDITIONS = [
        "north_bengal_sweep_35plus",
        "jangalmahal_hold_18plus",
        "medinipur_majority",
        "urban_kolkata_gain_10plus",
        "minority_fragmentation",
        "sir_voter_suppression_effective",
        "anti_incumbency_national",
        "rss_organizational_mobilization",
        "bjp_clear_cm_face",
    ]

    evidence: dict = {c: {"hits": 0, "snippets": []} for c in CONDITIONS}

    for article in enriched:
        if article.get("is_noise"):
            continue
        cred = float(article.get("credibility_score", 0.3))
        for cond in article.get("bjp_conditions_hit", []):
            if cond in evidence:
                evidence[cond]["hits"] += 1
                if len(evidence[cond]["snippets"]) < 3:
                    evidence[cond]["snippets"].append({
                        "headline": article.get("headline", "")[:100],
                        "source": article.get("source", ""),
                        "url": article.get("url", ""),
                        "credibility": round(cred, 2),
                    })

    results = {}
    for cond, data in evidence.items():
        h = data["hits"]
        if h == 0:
            status, confidence = "red", 0.05
        elif h <= 2:
            status, confidence = "yellow", 0.30
        else:
            status, confidence = "green", min(0.75, 0.35 + h * 0.04)

        results[cond] = {
            "status": status,
            "confidence": round(confidence, 3),
            "evidence_json": json.dumps(data["snippets"]),
        }

    return results
