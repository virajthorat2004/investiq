"""
core/market_overview.py
Nifty 50, Sensex live data + top gainers/losers of the day.
v2 — All functions cached + fast_info used for less rate-limit risk.
"""
import yfinance as yf
import pandas as pd
import streamlit as st
import time
import random


INDICES = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
}

NIFTY50_STOCKS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
    "INFY.NS","SBIN.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "KOTAKBANK.NS","HCLTECH.NS","MARUTI.NS","AXISBANK.NS","BAJFINANCE.NS",
    "ASIANPAINT.NS","TITAN.NS","SUNPHARMA.NS","TATAMOTORS.NS","WIPRO.NS",
    "ULTRACEMCO.NS","NESTLEIND.NS","TECHM.NS","POWERGRID.NS","NTPC.NS",
    "ONGC.NS","COALINDIA.NS","TATASTEEL.NS","ADANIENT.NS","DIVISLAB.NS",
]


def _get_index_via_fast_info(ticker: str):
    """Try fast_info first for indices."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        prev = getattr(fi, "previous_close", None)
        if price and prev:
            return price, prev
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=180, show_spinner=False)
def get_index_data() -> list[dict]:
    """
    Fetches live data for major Indian indices. Cached for 3 min.
    """
    results = []
    for name, ticker in INDICES.items():
        try:
            curr, prev = _get_index_via_fast_info(ticker)

            # Fallback to .info if fast_info failed
            if not curr:
                t = yf.Ticker(ticker)
                info = t.info
                prev = info.get("previousClose", 0)
                curr = info.get("regularMarketPrice") or info.get("currentPrice", 0)
                if not curr:
                    hist = t.history(period="2d")
                    if not hist.empty:
                        curr = hist["Close"].iloc[-1]
                        prev = hist["Close"].iloc[-2] if len(hist) > 1 else curr

            change = curr - prev
            change_pct = (change / prev * 100) if prev else 0
            results.append({
                "name": name,
                "ticker": ticker,
                "value": round(curr, 2) if curr else 0,
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
            })
        except Exception:
            results.append({
                "name": name,
                "ticker": ticker,
                "value": 0,
                "change": 0,
                "change_pct": 0,
            })
    return results


@st.cache_data(ttl=600, show_spinner=False)
def get_top_gainers_losers(n: int = 5) -> dict:
    """
    Fetches top N gainers and losers from Nifty 50 stocks.
    Cached for 10 min. Uses fast_info to reduce rate limit risk.
    """
    movers = []
    for ticker in NIFTY50_STOCKS:
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
            prev = getattr(fi, "previous_close", None)
            if price and prev and price > 0 and prev > 0:
                change_pct = (price - prev) / prev * 100
                name = ticker.replace(".NS", "").replace(".BO", "")
                movers.append({
                    "ticker": ticker,
                    "name": name[:20],
                    "price": round(price, 2),
                    "change_pct": round(change_pct, 2),
                })
            # Tiny delay to be polite to Yahoo
            time.sleep(random.uniform(0.1, 0.25))
        except Exception:
            continue

    movers.sort(key=lambda x: x["change_pct"], reverse=True)
    return {
        "gainers": movers[:n],
        "losers": movers[-n:][::-1],
    }


@st.cache_data(ttl=180, show_spinner=False)
def get_nifty_history(period: str = "1mo") -> pd.DataFrame:
    """
    Returns Nifty 50 historical data for chart. Cached for 3 min.
    """
    try:
        df = yf.Ticker("^NSEI").history(period=period)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()
