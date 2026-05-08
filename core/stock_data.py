"""
core/stock_data.py
Fetches live NSE/BSE stock data using yfinance.
Fixed v2: 
  - Uses fast_info as primary source (much faster, less rate-limited)
  - Falls back to .info only if fast_info is missing price
  - Longer delay between retries (5s instead of 3s)
  - Added jitter to avoid thundering herd on retries
"""

import yfinance as yf
import pandas as pd
import time
import random
from datetime import datetime, timedelta


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
    """
    Yahoo Finance returns a near-empty dict when rate-limited or blocked.
    A valid response always has more than 5 keys with real data.
    """
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


def _get_price_from_fast_info(ticker_obj) -> dict | None:
    """
    Uses yfinance fast_info — a lightweight endpoint that's much less
    rate-limited than .info. Returns price data dict or None.
    """
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


def get_stock_info(ticker: str, retries: int = 3, delay: int = 5) -> dict:
    """
    Returns key info about a stock: name, price, change, market cap etc.
    
    Strategy:
      1. Try fast_info first (lightweight, less rate-limited) for price data
      2. Try .info for name, PE, sector, summary
      3. If .info is rate-limited, use fast_info data + history fallback
    """
    last_error = ""

    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)

            # Step 1: Try fast_info for price (much more reliable)
            fast_data = _get_price_from_fast_info(stock)

            # Step 2: Try .info for metadata
            info = {}
            try:
                info = stock.info
            except Exception:
                pass

            # Step 3: If fast_info got us a price, we can proceed even if .info failed
            if fast_data:
                current_price = fast_data["current_price"]
                prev_close = fast_data["previous_close"]

                # Get name from info if available, else fall back to history
                name = info.get("longName") or info.get("shortName")
                if not name:
                    # Last resort: derive a display name from ticker
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

            # Step 4: fast_info failed too — try .info directly
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

            # Both failed — rate limited. Wait with jitter and retry.
            last_error = "Both fast_info and .info returned empty (Yahoo Finance rate limit)"
            if attempt < retries - 1:
                wait = delay + random.uniform(0, 3)  # jitter prevents thundering herd
                time.sleep(wait)
            continue

        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                wait = delay + random.uniform(0, 2)
                time.sleep(wait)
            continue

    return {
        "error": f"Could not fetch data for {ticker} after {retries} attempts. Reason: {last_error}",
        "ticker": ticker,
    }


def get_historical_data(ticker: str, period: str = "6mo", retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """
    Returns OHLCV historical price data.
    period options: 1mo, 3mo, 6mo, 1y, 2y
    Uses download() instead of .history() — more reliable, fewer rate limits.
    """
    for attempt in range(retries):
        try:
            # yf.download is more stable than ticker.history for single tickers
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

            if df.empty:
                if attempt < retries - 1:
                    time.sleep(delay + random.uniform(0, 2))
                continue

            # yf.download returns MultiIndex columns when downloading single ticker
            # Flatten if needed
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


def get_financials_summary(ticker: str, retries: int = 3, delay: int = 5) -> dict:
    """
    Returns revenue, earnings summary for fundamental context.
    Retries up to 3 times if Yahoo Finance returns empty data.
    """
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
