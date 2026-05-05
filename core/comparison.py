"""
core/comparison.py
Fetches and compares two stocks side by side.
"""

import pandas as pd
from core.stock_data import get_stock_info, get_historical_data, get_financials_summary
from core.news_fetcher import fetch_stock_news
from core.sentiment import analyze_articles_sentiment


def get_comparison_data(ticker1: str, ticker2: str, period: str = "6mo") -> dict:
    """
    Fetches all data for both stocks needed for comparison.
    Returns a dict with info, history, financials, sentiment for each.
    """
    def fetch_one(ticker):
        info = get_stock_info(ticker)
        hist = get_historical_data(ticker, period)
        financials = get_financials_summary(ticker)
        articles = fetch_stock_news(info.get("name", ticker), ticker, max_articles=5)
        sentiment = analyze_articles_sentiment(articles)
        return {
            "info": info,
            "hist": hist,
            "financials": financials,
            "sentiment": sentiment,
        }

    stock1 = fetch_one(ticker1)
    stock2 = fetch_one(ticker2)
    return {"stock1": stock1, "stock2": stock2}


def normalize_prices(hist1: pd.DataFrame, hist2: pd.DataFrame) -> tuple:
    """
    Normalizes both price series to 100 at start date.
    So we can compare % performance on same chart.
    """
    if hist1.empty or hist2.empty:
        return hist1, hist2

    norm1 = (hist1["close"] / hist1["close"].iloc[0]) * 100
    norm2 = (hist2["close"] / hist2["close"].iloc[0]) * 100
    return norm1, norm2


def compare_metrics(stock1: dict, stock2: dict) -> list[dict]:
    """
    Returns a list of metric comparisons between two stocks.
    Each item has: metric name, value1, value2, winner (1 or 2 or None).
    """
    info1 = stock1["info"]
    info2 = stock2["info"]
    fin1 = stock1["financials"]
    fin2 = stock2["financials"]

    def winner(v1, v2, higher_is_better=True):
        if v1 is None or v2 is None:
            return None
        if higher_is_better:
            return 1 if v1 > v2 else 2
        else:
            return 1 if v1 < v2 else 2

    metrics = [
        {
            "metric": "Today's Change",
            "v1": f"{info1.get('change_pct', 0):+.2f}%",
            "v2": f"{info2.get('change_pct', 0):+.2f}%",
            "winner": winner(info1.get("change_pct"), info2.get("change_pct")),
        },
        {
            "metric": "P/E Ratio",
            "v1": f"{info1.get('pe_ratio', 'N/A')}",
            "v2": f"{info2.get('pe_ratio', 'N/A')}",
            "winner": winner(info1.get("pe_ratio"), info2.get("pe_ratio"), higher_is_better=False),
        },
        {
            "metric": "Profit Margin",
            "v1": f"{fin1.get('profit_margin', 0)*100:.1f}%" if fin1.get("profit_margin") else "N/A",
            "v2": f"{fin2.get('profit_margin', 0)*100:.1f}%" if fin2.get("profit_margin") else "N/A",
            "winner": winner(fin1.get("profit_margin"), fin2.get("profit_margin")),
        },
        {
            "metric": "Revenue Growth",
            "v1": f"{fin1.get('revenue_growth', 0)*100:.1f}%" if fin1.get("revenue_growth") else "N/A",
            "v2": f"{fin2.get('revenue_growth', 0)*100:.1f}%" if fin2.get("revenue_growth") else "N/A",
            "winner": winner(fin1.get("revenue_growth"), fin2.get("revenue_growth")),
        },
        {
            "metric": "Return on Equity",
            "v1": f"{fin1.get('return_on_equity', 0)*100:.1f}%" if fin1.get("return_on_equity") else "N/A",
            "v2": f"{fin2.get('return_on_equity', 0)*100:.1f}%" if fin2.get("return_on_equity") else "N/A",
            "winner": winner(fin1.get("return_on_equity"), fin2.get("return_on_equity")),
        },
        {
            "metric": "Debt to Equity",
            "v1": f"{fin1.get('debt_to_equity', 'N/A')}",
            "v2": f"{fin2.get('debt_to_equity', 'N/A')}",
            "winner": winner(fin1.get("debt_to_equity"), fin2.get("debt_to_equity"), higher_is_better=False),
        },
        {
            "metric": "News Sentiment",
            "v1": stock1["sentiment"].get("overall", "N/A"),
            "v2": stock2["sentiment"].get("overall", "N/A"),
            "winner": None,
        },
    ]
    return metrics
