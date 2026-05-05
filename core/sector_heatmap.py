"""
core/sector_heatmap.py
NSE sector performance heatmap using sector ETF proxies.
"""
import yfinance as yf
import pandas as pd


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


def get_sector_performance() -> pd.DataFrame:
    """
    Fetches today's % change for each sector by averaging its top stocks.
    Returns DataFrame with columns: sector, change_pct, color.
    """
    rows = []
    for sector, tickers in SECTORS.items():
        changes = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                prev = info.get("previousClose", 0)
                curr = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                if prev and curr:
                    changes.append((curr - prev) / prev * 100)
            except Exception:
                continue
        avg = round(sum(changes) / len(changes), 2) if changes else 0
        rows.append({"sector": sector, "change_pct": avg, "stocks": len(changes)})

    return pd.DataFrame(rows)


def get_top_movers(n: int = 5) -> dict:
    """Returns top N gainers and losers from all sector stocks."""
    all_stocks = []
    for tickers in SECTORS.values():
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                prev = info.get("previousClose", 0)
                curr = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                name = info.get("shortName", t)
                if prev and curr:
                    change = (curr - prev) / prev * 100
                    all_stocks.append({"ticker": t, "name": name, "price": curr, "change_pct": round(change, 2)})
            except Exception:
                continue

    all_stocks.sort(key=lambda x: x["change_pct"], reverse=True)
    return {
        "gainers": all_stocks[:n],
        "losers": all_stocks[-n:][::-1],
    }
