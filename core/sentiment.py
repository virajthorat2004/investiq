"""
core/sentiment.py
Sentiment analysis on financial news headlines.
Uses a lightweight rule-based approach + optional transformer model.
"""

import re
from typing import Optional


# Financial-domain positive/negative word lists
POSITIVE_WORDS = {
    "surge", "soar", "rally", "gain", "rise", "jump", "strong", "growth",
    "profit", "beat", "outperform", "upgrade", "buy", "bullish", "record",
    "high", "positive", "boost", "expand", "recovery", "win", "success",
    "dividend", "revenue", "earnings", "exceed", "upside", "momentum",
    "robust", "solid", "optimistic", "breakthrough", "milestone", "award",
    "partnership", "acquisition", "innovation", "launch", "approve", "approved",
}

NEGATIVE_WORDS = {
    "fall", "drop", "decline", "loss", "crash", "plunge", "weak", "concern",
    "risk", "sell", "bearish", "downgrade", "cut", "miss", "disappoint",
    "low", "negative", "debt", "fraud", "probe", "investigation", "penalty",
    "fine", "layoff", "resign", "exit", "default", "warning", "downside",
    "volatile", "uncertain", "pressure", "slowdown", "contraction", "dispute",
    "lawsuit", "regulation", "ban", "block", "reject", "fail", "failed",
}


def analyze_sentiment(text: str) -> dict:
    """
    Analyzes sentiment of a single text (headline or paragraph).
    Returns: score (-1 to +1), label (Positive/Negative/Neutral), confidence.
    """
    if not text:
        return {"score": 0, "label": "Neutral", "confidence": 0.5}

    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))

    pos_hits = len(words & POSITIVE_WORDS)
    neg_hits = len(words & NEGATIVE_WORDS)
    total_hits = pos_hits + neg_hits

    if total_hits == 0:
        return {"score": 0.0, "label": "Neutral", "confidence": 0.5}

    score = (pos_hits - neg_hits) / total_hits  # -1 to +1
    confidence = min(0.5 + (total_hits * 0.08), 0.95)

    if score > 0.1:
        label = "Positive"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "score": round(score, 3),
        "label": label,
        "confidence": round(confidence, 2),
        "pos_words": pos_hits,
        "neg_words": neg_hits,
    }


def analyze_articles_sentiment(articles: list[dict]) -> dict:
    """
    Runs sentiment on a list of news articles.
    Returns per-article scores + aggregated overall sentiment.
    """
    if not articles:
        return {
            "overall": "Neutral",
            "overall_score": 0.0,
            "articles": [],
            "distribution": {"Positive": 0, "Negative": 0, "Neutral": 0},
            "summary": "No news available to analyze.",
        }

    results = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        sentiment = analyze_sentiment(text)
        results.append({
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "published_at": article.get("published_at", ""),
            "url": article.get("url", ""),
            "sentiment": sentiment,
        })

    # Aggregate
    scores = [r["sentiment"]["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in results:
        distribution[r["sentiment"]["label"]] += 1

    # Determine overall mood
    if avg_score > 0.15:
        overall = "Bullish 📈"
    elif avg_score < -0.15:
        overall = "Bearish 📉"
    else:
        overall = "Neutral ➡️"

    # Human-readable summary
    pos = distribution["Positive"]
    neg = distribution["Negative"]
    neu = distribution["Neutral"]
    total = len(results)
    summary = (
        f"Out of {total} recent articles, {pos} are positive, "
        f"{neg} are negative, and {neu} are neutral. "
        f"Overall market sentiment is {overall.lower()} with a score of {avg_score:.2f}."
    )

    return {
        "overall": overall,
        "overall_score": round(avg_score, 3),
        "articles": results,
        "distribution": distribution,
        "summary": summary,
    }


def get_sentiment_color(label: str) -> str:
    """Returns hex color for a sentiment label — used in UI."""
    colors = {
        "Positive": "#22c55e",
        "Negative": "#ef4444",
        "Neutral": "#94a3b8",
    }
    return colors.get(label, "#94a3b8")


def get_sentiment_emoji(label: str) -> str:
    emojis = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
    return emojis.get(label, "⚪")
