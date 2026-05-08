"""
core/sector_heatmap.py
NSE sector performance heatmap using sector ETF proxies.
v2 — Now cached for 10 min + uses fast_info for speed & lower rate-limit risk.
"""
import yfinance as yf
import pandas as pd
import streamlit as st
import time
import random


SECTORS = {
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"],
    "Pharma": ["SUNPHARMA.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Finance": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIGI.NS"],
    "Metal": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "SAIL.NS", "VEDL.NS"],
}


def _get_change_fast(ticker: str):
    """Uses fast_info — much less rate-limited than .info. Returns (price, change_pct, name) or None."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        prev = getattr(fi, "previous_close", None)
        if price and prev and price > 0 and prev > 0:
            change_pct = (price - prev) / prev * 100
            name = ticker.replace(".NS", "").replace(".BO", "")
            return round(price, 2), round(change_pct, 2), name
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def get_sector_performance() -> pd.DataFrame:
    """
    Fetches today's % change for each sector by averaging its top stocks.
    Cached for 10 min — sector data doesn't shift dramatically by minute.
    """
    rows = []
    for sector, tickers in SECTORS.items():
        changes = []
        for t in tickers:
            result = _get_change_fast(t)
            if result:
                _, change_pct, _ = result
                changes.append(change_pct)
            # Tiny delay between requests to be polite to Yahoo
            time.sleep(random.uniform(0.1, 0.3))

        avg = round(sum(changes) / len(changes), 2) if changes else 0
        rows.append({"sector": sector, "change_pct": avg, "stocks": len(changes)})

    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def get_top_movers(n: int = 5) -> dict:
    """
    Returns top N gainers and losers from all sector stocks.
    Cached for 10 min.
    """
    all_stocks = []
    for tickers in SECTORS.values():
        for t in tickers:
            result = _get_change_fast(t)
            if result:
                price, change_pct, name = result
                all_stocks.append({
                    "ticker": t,
                    "name": name[:20],
                    "price": price,
                    "change_pct": change_pct,
                })
            time.sleep(random.uniform(0.1, 0.3))

    all_stocks.sort(key=lambda x: x["change_pct"], reverse=True)
    return {
        "gainers": all_stocks[:n],
        "losers": all_stocks[-n:][::-1],
    }
