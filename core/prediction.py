"""
core/prediction.py
ML-based price forecasting using Linear Regression + confidence intervals.
Predicts next 7, 14, or 30 days of stock price.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


def predict_prices(hist_df: pd.DataFrame, days_ahead: int = 14) -> dict:
    """
    Predicts future stock prices using Linear Regression on features.
    Returns predicted dates, prices, upper/lower confidence bands.
    """
    if hist_df.empty or len(hist_df) < 30:
        return {"error": "Not enough historical data for prediction (need 30+ days)"}

    df = hist_df.copy()

    # ── Feature Engineering ──────────────────────────────────────────────────
    df["day_num"] = np.arange(len(df))
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df = df.dropna()

    if len(df) < 20:
        return {"error": "Not enough data after feature engineering"}

    # ── Train model ──────────────────────────────────────────────────────────
    features = ["day_num", "ma5", "ma10", "ma20", "volatility"]
    X = df[features].values
    y = df["close"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # ── Score ─────────────────────────────────────────────────────────────────
    train_score = model.score(X_scaled, y)

    # ── Predict future ────────────────────────────────────────────────────────
    last_day = df["day_num"].iloc[-1]
    last_ma5 = df["ma5"].iloc[-1]
    last_ma10 = df["ma10"].iloc[-1]
    last_ma20 = df["ma20"].iloc[-1]
    last_vol = df["volatility"].iloc[-1]

    future_dates = []
    future_prices = []

    # Step through each future day
    simulated_close = df["close"].tolist()

    for i in range(1, days_ahead + 1):
        future_day = last_day + i
        sim_series = pd.Series(simulated_close)
        ma5 = sim_series.tail(5).mean()
        ma10 = sim_series.tail(10).mean()
        ma20 = sim_series.tail(20).mean()
        vol = sim_series.pct_change().tail(10).std()

        X_future = np.array([[future_day, ma5, ma10, ma20, vol]])
        X_future_scaled = scaler.transform(X_future)
        pred = model.predict(X_future_scaled)[0]

        # Add small noise to avoid perfectly straight line
        noise = np.random.normal(0, last_vol * pred * 0.3)
        pred_with_noise = pred + noise

        future_date = df.index[-1] + timedelta(days=i)
        # Skip weekends
        while future_date.weekday() >= 5:
            future_date += timedelta(days=1)

        future_dates.append(future_date)
        future_prices.append(round(pred_with_noise, 2))
        simulated_close.append(pred_with_noise)

    # ── Confidence bands ──────────────────────────────────────────────────────
    residuals = y - model.predict(X_scaled)
    std_residual = np.std(residuals)

    # Bands widen over time (more uncertainty further out)
    upper_band = [p + std_residual * (1 + i * 0.08) for i, p in enumerate(future_prices)]
    lower_band = [p - std_residual * (1 + i * 0.08) for i, p in enumerate(future_prices)]

    # ── Summary stats ─────────────────────────────────────────────────────────
    current_price = df["close"].iloc[-1]
    predicted_end = future_prices[-1]
    expected_change = ((predicted_end - current_price) / current_price) * 100

    if expected_change > 2:
        outlook = f"📈 Bullish — model expects +{expected_change:.1f}% over {days_ahead} days"
        outlook_color = "#22c55e"
    elif expected_change < -2:
        outlook = f"📉 Bearish — model expects {expected_change:.1f}% over {days_ahead} days"
        outlook_color = "#ef4444"
    else:
        outlook = f"➡️ Sideways — model expects {expected_change:.1f}% over {days_ahead} days"
        outlook_color = "#94a3b8"

    return {
        "dates": future_dates,
        "prices": future_prices,
        "upper": upper_band,
        "lower": lower_band,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_end, 2),
        "expected_change": round(expected_change, 2),
        "outlook": outlook,
        "outlook_color": outlook_color,
        "model_score": round(train_score, 3),
        "days_ahead": days_ahead,
    }
