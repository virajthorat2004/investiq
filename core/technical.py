"""
core/technical.py
Calculates RSI, MACD, and Bollinger Bands from historical price data.
"""

import pandas as pd
import numpy as np


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates RSI (Relative Strength Index)."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates MACD line, signal line, and histogram."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0):
    """Calculates Bollinger Bands — upper, middle (SMA), lower."""
    middle = df["close"].rolling(window=period).mean()
    std_dev = df["close"].rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower


def get_rsi_signal(rsi_value: float) -> dict:
    """Returns human-readable RSI signal."""
    if rsi_value is None or pd.isna(rsi_value):
        return {"signal": "N/A", "color": "#94a3b8", "desc": "Not enough data"}
    if rsi_value >= 70:
        return {"signal": "Overbought", "color": "#ef4444", "desc": f"RSI {rsi_value:.1f} — Stock may be overvalued, possible pullback"}
    elif rsi_value <= 30:
        return {"signal": "Oversold", "color": "#22c55e", "desc": f"RSI {rsi_value:.1f} — Stock may be undervalued, possible bounce"}
    else:
        return {"signal": "Neutral", "color": "#94a3b8", "desc": f"RSI {rsi_value:.1f} — No extreme signal"}


def get_macd_signal(macd: float, signal: float) -> dict:
    """Returns human-readable MACD signal."""
    if macd is None or pd.isna(macd):
        return {"signal": "N/A", "color": "#94a3b8", "desc": "Not enough data"}
    if macd > signal:
        return {"signal": "Bullish", "color": "#22c55e", "desc": "MACD above signal — bullish momentum"}
    else:
        return {"signal": "Bearish", "color": "#ef4444", "desc": "MACD below signal — bearish momentum"}


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all indicators to the dataframe and returns it."""
    df = df.copy()
    df["rsi"] = calculate_rsi(df)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(df)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    return df
