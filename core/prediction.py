"""
core/prediction.py  —  InvestIQ v5.1
─────────────────────────────────────────────────────────────────────────────
ML Price Forecasting: Gradient Boosting  vs  Ridge Regression (Baseline)

Why Gradient Boosting instead of Prophet:
  - Prophet requires Stan/cmdstanpy which is incompatible with Python 3.14
  - GradientBoostingRegressor (scikit-learn) runs on any Python version
  - Captures non-linear interactions between RSI, MACD, Bollinger %B, volume
  - Ensemble of 200 shallow trees → naturally handles financial noise
  - Citable: Friedman (2001), "Greedy Function Approximation: A Gradient
    Boosting Machine", Annals of Statistics — peer-reviewed foundation

Feature Engineering (9 features):
  RSI-14, MACD histogram, Bollinger %B, Volume Δ%,
  MA-5/10/20, daily returns, 10-day volatility

Evaluation:
  80/20 temporal train-test split → MAE, RMSE, R², MAPE on holdout
  Both models evaluated — results go straight into IEEE paper Table II
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
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
    return macd - signal


def _compute_bollinger_pct(series: pd.Series, period: int = 20) -> pd.Series:
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - (ma - 2 * std)) / (4 * std + 1e-9)


def build_features(hist_df: pd.DataFrame) -> pd.DataFrame:
    df = hist_df.copy().sort_index()
    c  = df["close"]
    v  = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    df["ma5"]         = c.rolling(5).mean()
    df["ma10"]        = c.rolling(10).mean()
    df["ma20"]        = c.rolling(20).mean()
    df["returns"]     = c.pct_change()
    df["volatility"]  = df["returns"].rolling(10).std()
    df["rsi"]         = _compute_rsi(c, 14)
    df["macd_hist"]   = _compute_macd_signal(c)
    df["bollinger_b"] = _compute_bollinger_pct(c, 20)
    df["vol_change"]  = v.pct_change().clip(-1, 5)
    df["day_num"]     = np.arange(len(df))

    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN LIST  (imported by app.py for feature importance chart)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "day_num", "ma5", "ma10", "ma20",
    "volatility", "rsi", "macd_hist", "bollinger_b", "vol_change",
]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        "mae":  round(float(mae),  2),
        "rmse": round(float(rmse), 2),
        "r2":   round(float(r2),   4),
        "mape": round(float(mape), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FORECAST ROLL-FORWARD
# ─────────────────────────────────────────────────────────────────────────────

def _roll_forward(df: pd.DataFrame, model, scaler: StandardScaler,
                  days_ahead: int) -> tuple:
    simulated_close = df["close"].tolist()
    simulated_vol   = df["volume"].tolist() if "volume" in df.columns else [1_000_000] * len(df)
    last_day        = df["day_num"].iloc[-1]
    last_date       = df.index[-1]
    last_vol_std    = float(df["volatility"].iloc[-1])

    future_dates, future_prices = [], []

    for i in range(1, days_ahead + 1):
        sim  = pd.Series(simulated_close)
        sv   = pd.Series(simulated_vol)
        n    = len(sim)

        ma5  = float(sim.tail(5).mean())
        ma10 = float(sim.tail(10).mean())
        ma20 = float(sim.tail(20).mean())
        ret  = sim.pct_change()
        vlt  = float(ret.tail(10).std()) if n >= 10 else last_vol_std
        rsi  = float(_compute_rsi(sim, min(14, n)).iloc[-1]) if n >= 5 else 50.0
        macd = float(_compute_macd_signal(sim).iloc[-1])     if n >= 26 else 0.0
        bb   = float(_compute_bollinger_pct(sim, min(20, n)).iloc[-1]) if n >= 5 else 0.5
        vc   = float(sv.pct_change().iloc[-1]) if len(sv) > 1 else 0.0

        X_f  = np.array([[last_day + i, ma5, ma10, ma20, vlt, rsi, macd, bb, vc]])
        pred = float(model.predict(scaler.transform(X_f))[0])

        noise = np.random.normal(0, abs(pred) * last_vol_std * 0.25)
        pred  = max(pred + noise, 1.0)

        future_date = last_date + timedelta(days=i)
        while future_date.weekday() >= 5:
            future_date += timedelta(days=1)

        future_dates.append(future_date)
        future_prices.append(round(pred, 2))
        simulated_close.append(pred)
        simulated_vol.append(simulated_vol[-1])

    return future_dates, future_prices


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE  :  Ridge Regression
# ─────────────────────────────────────────────────────────────────────────────

def _run_ridge(df: pd.DataFrame, days_ahead: int) -> dict:
    split = int(len(df) * 0.8)
    if split < 20:
        return {"error": "Not enough data for Ridge baseline."}

    X = df[FEATURE_COLS].values
    y = df["close"].values

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X[:split])
    X_te_s  = scaler.transform(X[split:])

    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y[:split])
    y_pred_test = model.predict(X_te_s)
    metrics     = _eval_metrics(y[split:], y_pred_test)

    model.fit(scaler.fit_transform(X), y)
    future_dates, future_prices = _roll_forward(df, model, scaler, days_ahead)

    residuals = y - model.predict(scaler.transform(X))
    std_res   = float(np.std(residuals))
    upper = [p + std_res * (1 + k * 0.07) for k, p in enumerate(future_prices)]
    lower = [p - std_res * (1 + k * 0.07) for k, p in enumerate(future_prices)]

    return {
        "model":          "Ridge Regression",
        "dates":          future_dates,
        "prices":         future_prices,
        "upper":          upper,
        "lower":          lower,
        "metrics":        metrics,
        "test_actual":    y[split:].tolist(),
        "test_predicted": y_pred_test.tolist(),
        "test_dates":     df.index[split:].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY MODEL  :  Gradient Boosting Regressor
# ─────────────────────────────────────────────────────────────────────────────

def _run_gradient_boosting(df: pd.DataFrame, days_ahead: int) -> dict:
    split = int(len(df) * 0.8)
    if split < 20:
        return {"error": "Not enough data for Gradient Boosting."}

    X = df[FEATURE_COLS].values
    y = df["close"].values

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X[:split])
    X_te_s  = scaler.transform(X[split:])

    model = GradientBoostingRegressor(
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        min_samples_split = 5,
        random_state      = 42,
    )
    model.fit(X_tr_s, y[:split])
    y_pred_test = model.predict(X_te_s)
    metrics     = _eval_metrics(y[split:], y_pred_test)

    model.fit(scaler.fit_transform(X), y)
    future_dates, future_prices = _roll_forward(df, model, scaler, days_ahead)

    residuals = y - model.predict(scaler.transform(X))
    std_res   = float(np.std(residuals))
    upper = [p + std_res * (1 + k * 0.07) for k, p in enumerate(future_prices)]
    lower = [p - std_res * (1 + k * 0.07) for k, p in enumerate(future_prices)]

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    return {
        "model":               "Gradient Boosting (GBR)",
        "dates":               future_dates,
        "prices":              future_prices,
        "upper":               upper,
        "lower":               lower,
        "metrics":             metrics,
        "test_actual":         y[split:].tolist(),
        "test_predicted":      y_pred_test.tolist(),
        "test_dates":          df.index[split:].tolist(),
        "feature_importances": importances,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def predict_prices(hist_df: pd.DataFrame, days_ahead: int = 14) -> dict:
    """
    Runs Ridge (baseline) + GradientBoosting (primary).
    Safe to call from any Python version — no Stan/Prophet dependency.
    """
    try:
        if hist_df is None or hist_df.empty or len(hist_df) < 60:
            return {"error": "Need at least 60 days of history for prediction."}

        df_raw = hist_df.copy()
        df_raw.columns = [c.lower() for c in df_raw.columns]

        df = build_features(df_raw)
        if len(df) < 40:
            return {"error": "Not enough data after feature engineering (need 60+ trading days)."}

        current_price = float(df["close"].iloc[-1])

        ridge_result = _run_ridge(df, days_ahead)
        gbr_result   = _run_gradient_boosting(df, days_ahead)

        primary = gbr_result if "prices" in gbr_result else ridge_result
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

        model_metrics = {}
        if "metrics" in ridge_result:
            model_metrics["ridge"] = ridge_result["metrics"]
        if "metrics" in gbr_result:
            model_metrics["gbr"] = gbr_result["metrics"]

        return {
            "gbr":             gbr_result,
            "ridge":           ridge_result,
            "features_df":     df,
            "current_price":   round(current_price, 2),
            "predicted_price": round(predicted_end, 2),
            "expected_change": round(expected_change, 2),
            "days_ahead":      days_ahead,
            "outlook":         outlook,
            "outlook_color":   outlook_color,
            "model_metrics":   model_metrics,
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
