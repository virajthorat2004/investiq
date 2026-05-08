"""
core/market_overview.py
Nifty 50, Sensex live data + top gainers/losers of the day.
Fixed: Added @st.cache_data(ttl=180) to get_index_data() and get_nifty_history()
"""
import yfinance as yf
import pandas as pd
import streamlit as st


INDICES = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
}

NIFTY50_STOCKS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
    "INFOSYS.NS","SBIN.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "KOTAKBANK.NS","HCLTECH.NS","MARUTI.NS","AXISBANK.NS","BAJFINANCE.NS",
    "ASIANPAINT.NS","TITAN.NS","SUNPHARMA.NS","TATAMOTORS.NS","WIPRO.NS",
    "ULTRACEMCO.NS","NESTLEIND.NS","TECHM.NS","POWERGRID.NS","NTPC.NS",
    "ONGC.NS","COALINDIA.NS","TATASTEEL.NS","ADANIENT.NS","DIVISLAB.NS",
]


@st.cache_data(ttl=180)
def get_index_data() -> list[dict]:
    """
    Fetches live data for major Indian indices.
    Cached for 3 minutes to avoid hammering Yahoo Finance on every rerun.
    """
    results = []
    for name, ticker in INDICES.items():
        try:
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
                "value": round(curr, 2),
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


def get_top_gainers_losers(n: int = 5) -> dict:
    """Fetches top N gainers and losers from Nifty 50 stocks."""
    movers = []
    for ticker in NIFTY50_STOCKS:
        try:
            stock = yf.Ticker(ticker)
            # Use history instead of info — more reliable
            hist = stock.history(period="2d")
            if len(hist) < 2:
                continue
            prev = hist["Close"].iloc[-2]
            curr = hist["Close"].iloc[-1]
            name = stock.info.get("shortName", ticker.replace(".NS",""))
            if prev and curr and prev > 0:
                change_pct = (curr - prev) / prev * 100
                movers.append({
                    "ticker": ticker,
                    "name": name[:20],
                    "price": round(curr, 2),
                    "change_pct": round(change_pct, 2),
                })
        except Exception:
            continue

    movers.sort(key=lambda x: x["change_pct"], reverse=True)
    return {
        "gainers": [m for m in movers if m["change_pct"] > 0][:n] or movers[:n],
        "losers": [m for m in movers if m["change_pct"] < 0][-n:][::-1] or movers[-n:][::-1],
    }

@st.cache_data(ttl=180)
def get_nifty_history(period: str = "1mo") -> pd.DataFrame:
    """
    Returns Nifty 50 historical data for chart.
    Cached for 3 minutes — chart data doesn't change second by second.
    """
    try:
        df = yf.Ticker("^NSEI").history(period=period)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()
