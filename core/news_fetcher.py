"""
core/news_fetcher.py
Fetches live financial news for any stock using NewsAPI.
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/everything"


def fetch_stock_news(company_name: str, ticker: str, max_articles: int = 10) -> list[dict]:
    """
    Fetches recent news articles for a given company.
    Returns list of dicts with title, description, url, publishedAt, source.
    """
    if not NEWS_API_KEY:
        return _get_fallback_news(company_name)

    # Search by company name — remove .NS suffix for cleaner results
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    query = f"{company_name} OR {clean_ticker} stock India"

    # Only last 7 days (free tier limitation)
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            return _get_fallback_news(company_name)

        articles = []
        for article in data.get("articles", []):
            if not article.get("title") or article["title"] == "[Removed]":
                continue
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", "") or "",
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "content": article.get("content", "") or article.get("description", "") or "",
            })

        return articles if articles else _get_fallback_news(company_name)

    except Exception as e:
        return _get_fallback_news(company_name)


def format_news_for_rag(articles: list[dict]) -> str:
    """
    Converts articles list into a single text blob for the RAG engine to embed.
    Each article gets a numbered header for traceability.
    """
    if not articles:
        return "No recent news found."

    text_chunks = []
    for i, article in enumerate(articles, 1):
        chunk = f"""
--- Article {i} ---
Source: {article['source']}
Published: {article['published_at'][:10] if article['published_at'] else 'Unknown'}
Title: {article['title']}
Content: {article['description']}
"""
        text_chunks.append(chunk.strip())

    return "\n\n".join(text_chunks)


def _get_fallback_news(company_name: str) -> list[dict]:
    """
    Returns placeholder data if API key is missing or rate limited.
    Prevents app from crashing.
    """
    return [
        {
            "title": f"Market update: {company_name} — Live data requires NewsAPI key",
            "description": "Please add your NEWS_API_KEY to the .env file to see live news.",
            "url": "https://newsapi.org",
            "published_at": datetime.now().isoformat(),
            "source": "InvestIQ",
            "content": "No live news available. Configure your NEWS_API_KEY.",
        }
    ]
