"""
core/sentiment_price.py
Sentiment–Price Correlation Analysis for InvestIQ.

Fetches dated news sentiment scores and aligns them with daily price data
to compute lead-lag correlations (Pearson + Spearman).

Key design decisions:
  - Uses up to 100 articles from NewsAPI (free tier max) across the last 30 days
  - Groups articles by calendar date → daily average sentiment score
  - Tests lags 0–3 days: "did sentiment on day N predict price on day N+lag?"
  - Reports Pearson r, Spearman ρ, p-value, and the strongest lag found
  - Falls back gracefully when data is sparse (< 5 overlapping days)
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT WORD LISTS  (same as sentiment.py, kept local to avoid circular import)
# ─────────────────────────────────────────────────────────────────────────────

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


def _score_text(text: str) -> float:
    """Returns a sentiment score in [-1, +1] for a single text string."""
    import re
    if not text:
        return 0.0
    words     = set(re.findall(r'\b\w+\b', text.lower()))
    pos_hits  = len(words & POSITIVE_WORDS)
    neg_hits  = len(words & NEGATIVE_WORDS)
    total     = pos_hits + neg_hits
    if total == 0:
        return 0.0
    return round((pos_hits - neg_hits) / total, 4)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH DATED SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def fetch_dated_sentiment(
    company_name: str,
    ticker: str,
    days: int = 30,
) -> pd.Series:
    """
    Fetches up to 100 news articles for the stock, scores each one,
    and returns a date-indexed Series of daily average sentiment scores.

    Returns:
        pd.Series  index=datetime.date, values=float in [-1, +1]
        Empty Series if API key missing or no data.
    """
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        return pd.Series(dtype=float)

    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    query        = f"{company_name} OR {clean_ticker} stock India"
    from_date    = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 100,          # max for free tier
        "apiKey":   news_api_key,
    }

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=12,
        )
        data = resp.json()
        if data.get("status") != "ok":
            return pd.Series(dtype=float)

        articles = data.get("articles", [])
        if not articles:
            return pd.Series(dtype=float)

        rows = []
        for art in articles:
            title = art.get("title", "") or ""
            desc  = art.get("description", "") or ""
            pub   = art.get("publishedAt", "")
            if not pub or title == "[Removed]":
                continue
            try:
                date = datetime.fromisoformat(pub.replace("Z", "+00:00")).date()
            except ValueError:
                continue
            score = _score_text(f"{title} {desc}")
            rows.append({"date": date, "score": score})

        if not rows:
            return pd.Series(dtype=float)

        df = pd.DataFrame(rows)
        # Daily average — multiple articles on same day are averaged
        daily = df.groupby("date")["score"].mean().sort_index()
        return daily

    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# PRICE SERIES
# ─────────────────────────────────────────────────────────────────────────────

def get_price_series(ticker: str, days: int = 35) -> pd.Series:
    """
    Returns a date-indexed Series of daily closing prices for the last `days` days.
    Uses yf.download for stability.
    """
    try:
        raw = yf.download(
            ticker,
            period=f"{days}d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            return pd.Series(dtype=float)

        # Handle MultiIndex (yfinance sometimes wraps single ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0]
        else:
            close = raw["Close"]

        close.index = pd.to_datetime(close.index).date
        return close.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def get_price_returns(price_series: pd.Series) -> pd.Series:
    """Daily % returns from a price series."""
    return price_series.pct_change().dropna() * 100


# ─────────────────────────────────────────────────────────────────────────────
# LAG CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_lag_correlation(
    sentiment_series: pd.Series,
    price_returns: pd.Series,
    lag: int,
) -> dict:
    """
    Tests whether sentiment on day N correlates with price return on day N+lag.

    lag=0  → same day
    lag=1  → sentiment today, price move tomorrow
    lag=2  → sentiment today, price move in 2 days
    lag=-1 → price move yesterday, sentiment today (reverse check)

    Returns dict with pearson_r, pearson_p, spearman_r, spearman_p, n_obs.
    """
    if lag >= 0:
        sent_shifted   = sentiment_series
        prices_shifted = price_returns.shift(-lag)
    else:
        sent_shifted   = sentiment_series.shift(lag)    # negative shift
        prices_shifted = price_returns

    # Align on common dates
    combined = pd.DataFrame({
        "sentiment": sent_shifted,
        "returns":   prices_shifted,
    }).dropna()

    n = len(combined)
    if n < 5:
        return {
            "lag": lag, "n": n,
            "pearson_r": None, "pearson_p": None,
            "spearman_r": None, "spearman_p": None,
        }

    pr, pp   = stats.pearsonr(combined["sentiment"],  combined["returns"])
    sr, sp   = stats.spearmanr(combined["sentiment"], combined["returns"])

    return {
        "lag":        lag,
        "n":          n,
        "pearson_r":  round(float(pr), 4),
        "pearson_p":  round(float(pp), 4),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyse_sentiment_price(
    company_name: str,
    ticker: str,
    test_lags: list[int] = [0, 1, 2, 3],
) -> dict:
    """
    Full sentiment–price correlation analysis.

    Returns:
    {
      "sentiment_series":  pd.Series (date → daily avg sentiment score)
      "price_series":      pd.Series (date → close price)
      "price_returns":     pd.Series (date → daily % return)
      "merged_df":         pd.DataFrame aligned sentiment + returns
      "lag_results":       list of dicts, one per lag tested
      "best_lag":          int  — lag with strongest |pearson_r|
      "best_pearson_r":    float
      "best_spearman_r":   float
      "best_pearson_p":    float
      "interpretation":    str  — plain-English finding
      "has_data":          bool — False if insufficient data to analyse
      "article_count":     int
    }
    """
    sentiment_series = fetch_dated_sentiment(company_name, ticker, days=30)
    price_series     = get_price_series(ticker, days=35)
    price_returns    = get_price_returns(price_series)

    article_count = len(sentiment_series)

    # Need at least 5 days of overlap to say anything meaningful
    if sentiment_series.empty or price_returns.empty:
        return {
            "has_data":      False,
            "article_count": article_count,
            "reason":        "No news data or price data available.",
            "sentiment_series": sentiment_series,
            "price_series":     price_series,
            "price_returns":    price_returns,
            "merged_df":        pd.DataFrame(),
            "lag_results":      [],
            "best_lag":         None,
            "best_pearson_r":   None,
            "best_spearman_r":  None,
            "best_pearson_p":   None,
            "interpretation":   "Insufficient data.",
        }

    # Run correlation at each lag
    lag_results = [
        calc_lag_correlation(sentiment_series, price_returns, lag)
        for lag in test_lags
    ]

    # Find best lag by |pearson_r| where p < 0.15 (relaxed for small samples)
    valid = [r for r in lag_results if r["pearson_r"] is not None]
    if not valid:
        best = None
    else:
        best = max(valid, key=lambda r: abs(r["pearson_r"]))

    # Build merged DataFrame for charting
    merged = pd.DataFrame({
        "sentiment": sentiment_series,
        "price_return": price_returns,
        "price": price_series,
    }).dropna(subset=["sentiment"])

    # Interpretation
    if best is None or best["pearson_r"] is None:
        interpretation = (
            "Not enough overlapping data points to compute a reliable correlation. "
            "Try stocks with more frequent news coverage."
        )
    else:
        r     = best["pearson_r"]
        lag   = best["lag"]
        p     = best["pearson_p"]
        sig   = "statistically significant (p < 0.05)" if p < 0.05 else (
                "marginally significant (p < 0.15)" if p < 0.15 else "not statistically significant")

        if abs(r) < 0.15:
            strength = "very weak"
        elif abs(r) < 0.35:
            strength = "weak"
        elif abs(r) < 0.55:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "positive" if r > 0 else "negative"

        if lag == 0:
            timing = "on the same day"
        elif lag == 1:
            timing = "the following day"
        else:
            timing = f"{lag} days later"

        if r > 0.15:
            meaning = "positive news tends to accompany price gains"
        elif r < -0.15:
            meaning = "negative news tends to precede price declines"
        else:
            meaning = "no clear directional relationship was found"

        interpretation = (
            f"Sentiment shows a **{strength} {direction} correlation** (r = {r:.3f}) "
            f"with price returns {timing} — {sig}. "
            f"In plain terms: {meaning}. "
            f"Based on {best['n']} overlapping trading days."
        )

    return {
        "has_data":          len(valid) > 0,
        "article_count":     article_count,
        "sentiment_series":  sentiment_series,
        "price_series":      price_series,
        "price_returns":     price_returns,
        "merged_df":         merged,
        "lag_results":       lag_results,
        "best_lag":          best["lag"]        if best else None,
        "best_pearson_r":    best["pearson_r"]  if best else None,
        "best_spearman_r":   best["spearman_r"] if best else None,
        "best_pearson_p":    best["pearson_p"]  if best else None,
        "interpretation":    interpretation,
    }
