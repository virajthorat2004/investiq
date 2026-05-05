"""
core/stock_data.py
Fetches live NSE/BSE stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def clean_ticker(ticker: str) -> str:
    """Auto-formats ticker to NSE format."""
    ticker = ticker.strip().upper()
    # Already has exchange suffix
    if "." in ticker:
        return ticker
    # Add .NS by default for Indian stocks
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


def get_stock_info(ticker: str) -> dict:
    """
    Returns key info about a stock: name, price, change, market cap etc.
    ticker: NSE format e.g. 'RELIANCE.NS'
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev_close = info.get("previousClose", current_price)
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            "name": info.get("longName", ticker),
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
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_historical_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Returns OHLCV historical price data.
    period options: 1mo, 3mo, 6mo, 1y, 2y
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    except Exception as e:
        return pd.DataFrame()


def get_financials_summary(ticker: str) -> dict:
    """
    Returns revenue, earnings summary for fundamental context.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
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
        return {}


def format_market_cap(value: int) -> str:
    """Converts raw market cap number to readable format (₹ Cr / ₹ L Cr)."""
    if not value:
        return "N/A"
    cr = value / 1e7  # convert to crores
    if cr >= 1_00_000:
        return f"₹{cr/1_00_000:.2f} L Cr"
    return f"₹{cr:,.0f} Cr"
