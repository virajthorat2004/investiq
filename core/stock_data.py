"""
core/stock_data.py
Fetches live NSE/BSE stock data using yfinance.
v3 — Triple-layer caching to defeat Yahoo Finance rate limiting:
  Layer 1: Streamlit memory cache (5 min TTL) — survives reruns
  Layer 2: Disk cache fallback — survives Yahoo outages
  Layer 3: Retry with jitter — handles transient failures
"""

import yfinance as yf
import pandas as pd
import time
import random
import json
import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st


# ── Disk cache setup ────────────────────────────────────────────────────────
CACHE_DIR = Path(".yahoo_cache")
CACHE_DIR.mkdir(exist_ok=True)
DISK_CACHE_MAX_AGE_HOURS = 24  # Serve stale data up to 24h old if Yahoo is down


def _disk_cache_path(key: str, kind: str) -> Path:
    """Returns disk cache file path for a given key."""
    safe_key = key.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{kind}_{safe_key}.pkl"


def _save_to_disk(key: str, kind: str, data) -> None:
    """Saves successful fetch to disk for fallback use."""
    try:
        path = _disk_cache_path(key, kind)
        with open(path, "wb") as f:
            pickle.dump({"timestamp": datetime.now(), "data": data}, f)
    except Exception:
        pass  # Disk cache is best-effort, never crash on save fail


def _load_from_disk(key: str, kind: str):
    """Loads from disk cache if exists and not too old. Returns (data, age_hours) or (None, None)."""
    try:
        path = _disk_cache_path(key, kind)
        if not path.exists():
            return None, None
        with open(path, "rb") as f:
            payload = pickle.load(f)
        age = datetime.now() - payload["timestamp"]
        age_hours = age.total_seconds() / 3600
        if age_hours > DISK_CACHE_MAX_AGE_HOURS:
            return None, None
        return payload["data"], round(age_hours, 1)
    except Exception:
        return None, None


def clean_ticker(ticker: str) -> str:
    """Auto-formats ticker to NSE format."""
    ticker = ticker.strip().upper()
    if "." in ticker:
        return ticker
    return f"{ticker}.NS"


# Popular Indian stocks — shown in the sidebar dropdown
POPULAR_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Larsen & Toubro": "LT.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Titan": "TITAN.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "Power Grid": "POWERGRID.NS",
    "Nestle India": "NESTLEIND.NS",
}


def _is_valid_info(info: dict) -> bool:
    """Yahoo returns near-empty dict when rate-limited. Detect that."""
    if not info or not isinstance(info, dict):
        return False
    if len(info) <= 5:
        return False
    has_price = (
        info.get("currentPrice") is not None
        or info.get("regularMarketPrice") is not None
        or info.get("previousClose") is not None
    )
    return has_price


def _get_price_from_fast_info(ticker_obj):
    """Uses yfinance fast_info — lightweight endpoint, less rate-limited."""
    try:
        fi = ticker_obj.fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        prev_close = getattr(fi, "previous_close", None)
        market_cap = getattr(fi, "market_cap", None)
        high_52w = getattr(fi, "year_high", None)
        low_52w = getattr(fi, "year_low", None)

        if price and price > 0:
            prev_close = prev_close or price
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            return {
                "current_price": round(price, 2),
                "previous_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "market_cap": market_cap or 0,
                "52w_high": round(high_52w, 2) if high_52w else None,
                "52w_low": round(low_52w, 2) if low_52w else None,
            }
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1: Streamlit memory cache (5 min)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_info(ticker: str, retries: int = 3, delay: int = 5) -> dict:
    """
    Returns key info about a stock.
    Cached for 5 min. Falls back to disk cache if Yahoo is down.
    """
    result = _fetch_stock_info_uncached(ticker, retries, delay)

    # If success, save to disk for future fallback
    if "error" not in result and result.get("current_price", 0) > 0:
        _save_to_disk(ticker, "info", result)
        return result

    # If failed, try disk cache
    cached, age_hours = _load_from_disk(ticker, "info")
    if cached:
        cached["_stale"] = True
        cached["_stale_hours"] = age_hours
        return cached

    return result  # Return error dict if no fallback available


def _fetch_stock_info_uncached(ticker: str, retries: int = 3, delay: int = 5) -> dict:
    """The actual fetching logic (uncached). Called by cached wrapper."""
    last_error = ""

    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)

            # Step 1: Try fast_info first
            fast_data = _get_price_from_fast_info(stock)

            # Step 2: Try .info for metadata
            info = {}
            try:
                info = stock.info
            except Exception:
                pass

            # Step 3: fast_info worked — use it
            if fast_data:
                current_price = fast_data["current_price"]
                prev_close = fast_data["previous_close"]
                name = info.get("longName") or info.get("shortName")
                if not name:
                    name = ticker.replace(".NS", "").replace(".BO", "")

                return {
                    "name": name,
                    "ticker": ticker,
                    "current_price": current_price,
                    "previous_close": prev_close,
                    "change": fast_data["change"],
                    "change_pct": fast_data["change_pct"],
                    "market_cap": fast_data["market_cap"] or info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", None),
                    "52w_high": fast_data["52w_high"] or info.get("fiftyTwoWeekHigh", None),
                    "52w_low": fast_data["52w_low"] or info.get("fiftyTwoWeekLow", None),
                    "volume": info.get("volume", 0),
                    "avg_volume": info.get("averageVolume", 0),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "summary": info.get("longBusinessSummary", "No description available."),
                }

            # Step 4: fast_info failed — try .info directly
            if _is_valid_info(info):
                current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                prev_close = info.get("previousClose", current_price)
                change = current_price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0

                return {
                    "name": info.get("longName") or info.get("shortName") or ticker,
                    "ticker": ticker,
                    "current_price": round(current_price, 2),
                    "previous_close": round(prev_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", None),
                    "52w_high": info.get("fiftyTwoWeekHigh", None),
                    "52w_low": info.get("fiftyTwoWeekLow", None),
                    "volume": info.get("volume", 0),
                    "avg_volume": info.get("averageVolume", 0),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "summary": info.get("longBusinessSummary", "No description available."),
                }

            last_error = "Both fast_info and .info returned empty (Yahoo rate limit)"
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, 3))
            continue

        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, 2))
            continue

    return {
        "error": f"Could not fetch data for {ticker} after {retries} attempts. Reason: {last_error}",
        "ticker": ticker,
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_historical_data(ticker: str, period: str = "6mo", retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """
    Returns OHLCV historical price data. Cached + disk fallback.
    """
    df = _fetch_historical_uncached(ticker, period, retries, delay)

    if not df.empty:
        _save_to_disk(f"{ticker}_{period}", "hist", df)
        return df

    # Fallback to disk
    cached, age_hours = _load_from_disk(f"{ticker}_{period}", "hist")
    if cached is not None and not cached.empty:
        return cached

    return pd.DataFrame()


def _fetch_historical_uncached(ticker: str, period: str, retries: int, delay: int) -> pd.DataFrame:
    """Uncached historical data fetch."""
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                if attempt < retries - 1:
                    time.sleep(delay + random.uniform(0, 2))
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            return df
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, 2))
            continue

    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_financials_summary(ticker: str, retries: int = 3, delay: int = 5) -> dict:
    """
    Returns revenue, earnings summary. Cached for 10 min (financials change slowly).
    """
    result = _fetch_financials_uncached(ticker, retries, delay)

    if result:
        _save_to_disk(ticker, "fin", result)
        return result

    cached, _ = _load_from_disk(ticker, "fin")
    if cached:
        return cached

    return {}


def _fetch_financials_uncached(ticker: str, retries: int, delay: int) -> dict:
    """Uncached financials fetch."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not _is_valid_info(info):
                if attempt < retries - 1:
                    time.sleep(delay + random.uniform(0, 2))
                continue

            return {
                "revenue": info.get("totalRevenue", None),
                "gross_profit": info.get("grossProfits", None),
                "ebitda": info.get("ebitda", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "return_on_equity": info.get("returnOnEquity", None),
                "profit_margin": info.get("profitMargins", None),
                "earnings_growth": info.get("earningsGrowth", None),
                "revenue_growth": info.get("revenueGrowth", None),
            }
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, 2))
            continue

    return {}


def format_market_cap(value: int) -> str:
    """Converts raw market cap number to readable format (₹ Cr / ₹ L Cr)."""
    if not value:
        return "N/A"
    cr = value / 1e7
    if cr >= 1_00_000:
        return f"₹{cr/1_00_000:.2f} L Cr"
    return f"₹{cr:,.0f} Cr"
