"""
core/prediction.py  —  InvestIQ v5.0
─────────────────────────────────────────────────────────────────────────────
ML Price Forecasting: Facebook Prophet  vs  Linear Regression Baseline

Architecture
────────────
• Feature Engineering  : RSI-14, MACD signal, Bollinger %B, Volume Δ%,
                         MA-5/10/20, daily returns, 10-day volatility
• Baseline Model       : Ridge Regression (regularised LR) on engineered features
• Primary Model        : Facebook Prophet with external regressors
                         (RSI, MACD, Bollinger %B, volume change)
• Evaluation           : 80/20 temporal train-test split → MAE, RMSE, R², MAPE
• Forecast horizon     : 7 / 14 / 30 trading days forward

Why Prophet for Indian Markets?
────────────────────────────────
  - Handles NSE/BSE trading gaps (weekends, Diwali, Republic Day) natively
  - Robust to outliers from budget/earnings events common in Indian markets
  - Citable peer-reviewed method (Taylor & Letham, 2018, The American Statistician)
  - Trains in <2 s on Streamlit Cloud — no GPU required
  - External regressors let us inject technical indicators as features

Publication metrics produced
─────────────────────────────
  model_metrics = {
      "lr":      { mae, rmse, r2, mape }   # baseline
      "prophet": { mae, rmse, r2, mape }   # primary
  }
  These are the numbers that go in Table II of your IEEE paper.
"""

import warnings
warnings.filterwarnings("ignore")          # suppress Prophet / Stan noise

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd_signal(series: pd.Series) -> pd.Series:
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal          # MACD histogram (macd − signal)


def _compute_bollinger_pct(series: pd.Series, period: int = 20) -> pd.Series:
    """Bollinger %B: 0 = at lower band, 1 = at upper band."""
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - (ma - 2 * std)) / (4 * std + 1e-9)


def build_features(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : hist_df with columns [open, high, low, close, volume]  (lowercase)
             index  = DatetimeIndex
    Output : DataFrame with all engineered features, NaNs dropped.
    """
    df = hist_df.copy()
    df = df.sort_index()

    c = df["close"]
    v = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    # Trend / momentum
    df["ma5"]         = c.rolling(5).mean()
    df["ma10"]        = c.rolling(10).mean()
    df["ma20"]        = c.rolling(20).mean()
    df["returns"]     = c.pct_change()
    df["volatility"]  = df["returns"].rolling(10).std()

    # Technical indicators
    df["rsi"]         = _compute_rsi(c, 14)
    df["macd_hist"]   = _compute_macd_signal(c)
    df["bollinger_b"] = _compute_bollinger_pct(c, 20)

    # Volume
    df["vol_change"]  = v.pct_change().clip(-1, 5)   # cap extreme spikes

    # Time index (needed by Ridge baseline)
    df["day_num"]     = np.arange(len(df))

    df = df.dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "day_num", "ma5", "ma10", "ma20",
    "volatility", "rsi", "macd_hist", "bollinger_b", "vol_change",
]


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE — guard against zeros
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        "mae":  round(float(mae),  2),
        "rmse": round(float(rmse), 2),
        "r2":   round(float(r2),   4),
        "mape": round(float(mape), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE  :  Ridge Regression
# ─────────────────────────────────────────────────────────────────────────────

def _run_ridge(df: pd.DataFrame, days_ahead: int) -> dict:
    """
    Trains Ridge on 80 % of data, evaluates on 20 %, then re-trains on all
    data and forecasts `days_ahead` trading days forward.
    """
    split = int(len(df) * 0.8)
    if split < 20:
        return {"error": "Insufficient data for Ridge baseline (need 100+ rows)"}

    X = df[FEATURE_COLS].values
    y = df["close"].values

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred_test = model.predict(X_test_s)

    metrics = _eval_metrics(y_test, y_pred_test)

    # Re-train on full data for forecasting
    model.fit(scaler.fit_transform(X), y)

    # Roll forward day-by-day
    simulated_close  = df["close"].tolist()
    simulated_vol    = df["volume"].tolist() if "volume" in df.columns else [1e6] * len(df)
    last_day         = df["day_num"].iloc[-1]
    last_date        = df.index[-1]

    future_dates, future_prices = [], []
    for i in range(1, days_ahead + 1):
        sim = pd.Series(simulated_close)
        sv  = pd.Series(simulated_vol)

        ma5   = sim.tail(5).mean()
        ma10  = sim.tail(10).mean()
        ma20  = sim.tail(20).mean()
        ret   = sim.pct_change()
        vol   = ret.tail(10).std()
        rsi   = float(_compute_rsi(sim, min(14, len(sim))).iloc[-1]) if len(sim) >= 5 else 50.0
        macd  = float(_compute_macd_signal(sim).iloc[-1])            if len(sim) >= 26 else 0.0
        bb    = float(_compute_bollinger_pct(sim, min(20, len(sim))).iloc[-1]) if len(sim) >= 5 else 0.5
        vc    = float(sv.pct_change().iloc[-1]) if len(sv) > 1 else 0.0

        X_f  = np.array([[last_day + i, ma5, ma10, ma20, vol, rsi, macd, bb, vc]])
        pred = model.predict(scaler.transform(X_f))[0]

        # Small noise proportional to recent volatility (avoids flat line)
        noise = np.random.normal(0, abs(pred) * vol * 0.25)
        pred  = max(pred + noise, 1.0)

        future_date = last_date + timedelta(days=i)
        while future_date.weekday() >= 5:
            future_date += timedelta(days=1)

        future_dates.append(future_date)
        future_prices.append(round(pred, 2))
        simulated_close.append(pred)
        simulated_vol.append(simulated_vol[-1])

    # Confidence bands from residual std
    residuals   = y - model.predict(scaler.transform(X))
    std_res     = np.std(residuals)
    upper = [p + std_res * (1 + k * 0.07) for k, p in enumerate(future_prices)]
    lower = [p - std_res * (1 + k * 0.07) for k, p in enumerate(future_prices)]

    return {
        "model":    "Ridge Regression",
        "dates":    future_dates,
        "prices":   future_prices,
        "upper":    upper,
        "lower":    lower,
        "metrics":  metrics,
        "test_actual":    y_test.tolist(),
        "test_predicted": y_pred_test.tolist(),
        "test_dates":     df.index[split:].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY MODEL  :  Facebook Prophet
# ─────────────────────────────────────────────────────────────────────────────

def _run_prophet(df: pd.DataFrame, days_ahead: int) -> dict:
    """
    Trains Prophet with RSI, MACD, Bollinger %B, volume_change as extra
    regressors. Evaluates on the last 20 % of data (temporal holdout).
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {"error": "prophet not installed — run: pip install prophet"}

    split = int(len(df) * 0.8)
    if split < 30:
        return {"error": "Insufficient data for Prophet (need 150+ rows after feature engineering)"}

    # Prophet expects ds (datetime) + y (target) columns
    prop_df = df.reset_index()[["Date", "close", "rsi", "macd_hist", "bollinger_b", "vol_change"]].copy()
    prop_df.columns = ["ds", "y", "rsi", "macd_hist", "bollinger_b", "vol_change"]
    prop_df["ds"] = pd.to_datetime(prop_df["ds"])

    train_df = prop_df.iloc[:split].copy()
    test_df  = prop_df.iloc[split:].copy()

    model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        seasonality_mode    = "multiplicative",   # better for trending financial series
        changepoint_prior_scale = 0.15,           # allow faster trend changes
        interval_width      = 0.80,
    )
    # Add Indian market custom seasonalities
    model.add_seasonality(name="monthly", period=30.5,  fourier_order=5)
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=3)

    # External regressors (technical indicators)
    for reg in ["rsi", "macd_hist", "bollinger_b", "vol_change"]:
        model.add_regressor(reg)

    model.fit(train_df)

    # ── Evaluate on test set ──────────────────────────────────────────────
    test_forecast = model.predict(test_df[["ds", "rsi", "macd_hist", "bollinger_b", "vol_change"]])
    y_test      = test_df["y"].values
    y_pred_test = test_forecast["yhat"].values
    metrics     = _eval_metrics(y_test, y_pred_test)

    # ── Re-train on full data ─────────────────────────────────────────────
    full_model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        seasonality_mode    = "multiplicative",
        changepoint_prior_scale = 0.15,
        interval_width      = 0.80,
    )
    full_model.add_seasonality(name="monthly",   period=30.5,  fourier_order=5)
    full_model.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
    for reg in ["rsi", "macd_hist", "bollinger_b", "vol_change"]:
        full_model.add_regressor(reg)
    full_model.fit(prop_df)

    # ── Build future DataFrame ────────────────────────────────────────────
    # Prophet's make_future_dataframe uses calendar days; we filter to weekdays
    future_cal = full_model.make_future_dataframe(periods=days_ahead * 2)
    future_cal = future_cal[future_cal["ds"].dt.weekday < 5]
    future_cal = future_cal.tail(days_ahead)   # keep exactly days_ahead trading days

    # Fill regressors for future rows with last known values (persistence forecast)
    last_row = prop_df.iloc[-1]
    for reg in ["rsi", "macd_hist", "bollinger_b", "vol_change"]:
        future_cal[reg] = last_row[reg]

    forecast = full_model.predict(future_cal)
    forecast_future = forecast.tail(days_ahead)

    future_dates  = forecast_future["ds"].tolist()
    future_prices = [round(float(p), 2) for p in forecast_future["yhat"]]
    upper         = [round(float(p), 2) for p in forecast_future["yhat_upper"]]
    lower         = [round(float(p), 2) for p in forecast_future["yhat_lower"]]

    return {
        "model":   "Facebook Prophet",
        "dates":   future_dates,
        "prices":  future_prices,
        "upper":   upper,
        "lower":   lower,
        "metrics": metrics,
        "test_actual":    y_test.tolist(),
        "test_predicted": y_pred_test.tolist(),
        "test_dates":     test_df["ds"].tolist(),
        "components":     forecast[["ds", "trend", "weekly", "yearly"]].tail(days_ahead).to_dict("records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  —  called from app.py
# ─────────────────────────────────────────────────────────────────────────────

def predict_prices(hist_df: pd.DataFrame, days_ahead: int = 14) -> dict:
    """
    Main entry point.  Runs both Ridge (baseline) and Prophet (primary),
    returns combined results with evaluation metrics for both.

    Parameters
    ──────────
    hist_df    : DataFrame with columns [open, high, low, close, volume]
                 and a DatetimeIndex.  At least 150 rows recommended.
    days_ahead : int — 7, 14, or 30

    Returns
    ───────
    {
      "prophet"      : { dates, prices, upper, lower, metrics, test_actual,
                         test_predicted, test_dates, components }
      "ridge"        : { dates, prices, upper, lower, metrics, test_actual,
                         test_predicted, test_dates }
      "features_df"  : pd.DataFrame — engineered features (for display)
      "current_price": float
      "days_ahead"   : int
      "outlook"      : str   — plain-English direction from Prophet
      "outlook_color": str   — hex colour
      "model_metrics": dict  — { "ridge": {...}, "prophet": {...} }
                               MAE / RMSE / R² / MAPE on 20% holdout
                               → goes straight into IEEE paper Table II
    }
    """
    if hist_df.empty or len(hist_df) < 60:
        return {"error": "Need at least 60 days of history for prediction."}

    # Normalise column names to lowercase
    hist_df = hist_df.copy()
    hist_df.columns = [c.lower() for c in hist_df.columns]

    # Build features
    df = build_features(hist_df)
    if len(df) < 40:
        return {"error": "Not enough data after feature engineering."}

    # Ensure DatetimeIndex has a name "Date" for Prophet
    df.index.name = "Date"

    current_price = float(df["close"].iloc[-1])

    # ── Run both models ────────────────────────────────────────────────────
    ridge_result   = _run_ridge(df, days_ahead)
    prophet_result = _run_prophet(df, days_ahead)

    # ── Outlook from Prophet (fallback to Ridge if Prophet failed) ─────────
    primary = prophet_result if "prices" in prophet_result else ridge_result
    if "prices" not in primary:
        return {"error": primary.get("error", "Both models failed.")}

    predicted_end   = primary["prices"][-1]
    expected_change = (predicted_end - current_price) / current_price * 100

    if expected_change > 2:
        outlook       = f"📈 Bullish — model expects +{expected_change:.1f}% over {days_ahead} days"
        outlook_color = "#22c55e"
    elif expected_change < -2:
        outlook       = f"📉 Bearish — model expects {expected_change:.1f}% over {days_ahead} days"
        outlook_color = "#ef4444"
    else:
        outlook       = f"➡️ Sideways — model expects {expected_change:.1f}% over {days_ahead} days"
        outlook_color = "#94a3b8"

    # ── Assemble model_metrics for paper ──────────────────────────────────
    model_metrics = {}
    if "metrics" in ridge_result:
        model_metrics["ridge"] = ridge_result["metrics"]
    if "metrics" in prophet_result:
        model_metrics["prophet"] = prophet_result["metrics"]

    return {
        "prophet":       prophet_result,
        "ridge":         ridge_result,
        "features_df":   df,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_end, 2),
        "expected_change": round(expected_change, 2),
        "days_ahead":    days_ahead,
        "outlook":       outlook,
        "outlook_color": outlook_color,
        "model_metrics": model_metrics,
    }
