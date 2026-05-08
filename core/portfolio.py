"""
core/portfolio.py  v2.0
AI Portfolio Analyser — upload CSV of holdings, get full AI analysis.

New in v2.0:
  - get_portfolio_hist()      — fetches 1y daily returns for each holding + Nifty 50
  - calc_sharpe_ratio()       — annualised Sharpe (252-day, risk-free = 7% Indian T-bill)
  - calc_portfolio_beta()     — portfolio-weighted beta vs Nifty 50
  - calc_correlation_matrix() — pairwise price correlation between holdings
  - rebalancing_hints()       — weight-based + beta-based plain-English suggestions
  All new metrics are returned inside analyse_portfolio() under the key "metrics".
"""

import pandas as pd
import numpy as np
import io
import yfinance as yf

from core.stock_data import get_stock_info
from core.news_fetcher import fetch_stock_news
from core.sentiment import analyze_articles_sentiment


SAMPLE_CSV = """stock,ticker,qty,buy_price
Reliance Industries,RELIANCE.NS,10,2400
TCS,TCS.NS,5,3500
Infosys,INFY.NS,8,1450
HDFC Bank,HDFCBANK.NS,12,1600
Wipro,WIPRO.NS,20,450
"""

# Indian risk-free rate (approx 10-yr G-Sec / T-bill yield)
RISK_FREE_ANNUAL = 0.07
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / 252


# ─────────────────────────────────────────────────────────────────────────────
# CSV PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_portfolio_csv(file_content: str) -> pd.DataFrame:
    """Parses uploaded CSV into a clean DataFrame."""
    try:
        df = pd.read_csv(io.StringIO(file_content))
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"ticker", "qty", "buy_price"}
        if not required.issubset(set(df.columns)):
            return None
        df["qty"]       = pd.to_numeric(df["qty"],       errors="coerce")
        df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
        df = df.dropna(subset=["ticker", "qty", "buy_price"])
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA (for risk metrics)
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_hist(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Downloads adjusted-close history for all holdings + Nifty 50 benchmark.
    Returns a DataFrame of daily % returns, columns = tickers + "^NSEI".
    Missing/short tickers are silently dropped.
    """
    all_tickers = list(set(tickers)) + ["^NSEI"]
    try:
        raw = yf.download(
            all_tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # yfinance returns MultiIndex when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw

        # Keep only columns that have enough data (>60 rows)
        prices = prices.dropna(axis=1, thresh=60)
        returns = prices.pct_change().dropna(how="all")
        return returns
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SHARPE RATIO
# ─────────────────────────────────────────────────────────────────────────────

def calc_sharpe_ratio(
    holdings: list[dict],
    returns_df: pd.DataFrame,
) -> float | None:
    """
    Computes annualised Sharpe Ratio for the portfolio.

    weights = current_value weight of each holding.
    Uses daily portfolio returns → annualises with √252.
    Risk-free rate = 7% p.a. (Indian T-bill proxy).

    Returns None if data is insufficient.
    """
    if returns_df.empty:
        return None

    tickers      = [h["ticker"] for h in holdings]
    values       = [h["current_value"] for h in holdings]
    total_value  = sum(values)

    if total_value == 0:
        return None

    # Filter to tickers that exist in returns_df
    valid = [(t, v) for t, v in zip(tickers, values) if t in returns_df.columns]
    if not valid:
        return None

    valid_tickers, valid_values = zip(*valid)
    weights = np.array(valid_values, dtype=float)
    weights /= weights.sum()

    port_returns = returns_df[list(valid_tickers)].dropna().values @ weights  # daily portfolio return series

    excess        = port_returns - RISK_FREE_DAILY
    avg_excess    = np.mean(excess)
    std_excess    = np.std(excess, ddof=1)

    if std_excess == 0:
        return None

    sharpe = (avg_excess / std_excess) * np.sqrt(252)
    return round(float(sharpe), 3)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO BETA
# ─────────────────────────────────────────────────────────────────────────────

def calc_stock_beta(
    ticker: str,
    returns_df: pd.DataFrame,
    benchmark: str = "^NSEI",
) -> float | None:
    """OLS beta of a single stock vs the benchmark."""
    if ticker not in returns_df.columns or benchmark not in returns_df.columns:
        return None

    df = returns_df[[ticker, benchmark]].dropna()
    if len(df) < 30:
        return None

    cov_matrix = np.cov(df[ticker].values, df[benchmark].values)
    var_market  = cov_matrix[1, 1]
    if var_market == 0:
        return None

    beta = cov_matrix[0, 1] / var_market
    return round(float(beta), 3)


def calc_portfolio_beta(
    holdings: list[dict],
    returns_df: pd.DataFrame,
    benchmark: str = "^NSEI",
) -> dict:
    """
    Returns:
      portfolio_beta  — value-weighted average beta of all holdings
      stock_betas     — {ticker: beta} for each holding
    """
    stock_betas   = {}
    total_value   = sum(h["current_value"] for h in holdings)
    weighted_beta = 0.0
    covered_value = 0.0

    for h in holdings:
        b = calc_stock_beta(h["ticker"], returns_df, benchmark)
        stock_betas[h["ticker"]] = b
        if b is not None and total_value > 0:
            weight        = h["current_value"] / total_value
            weighted_beta += b * weight
            covered_value += h["current_value"]

    portfolio_beta = round(weighted_beta, 3) if covered_value > 0 else None
    return {
        "portfolio_beta": portfolio_beta,
        "stock_betas": stock_betas,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def calc_correlation_matrix(
    tickers: list[str],
    returns_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Returns a pairwise Pearson correlation DataFrame for all tickers
    present in returns_df.  Excludes the benchmark column.
    """
    valid = [t for t in tickers if t in returns_df.columns]
    if len(valid) < 2:
        return None

    corr = returns_df[valid].dropna().corr()
    return corr.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# REBALANCING HINTS
# ─────────────────────────────────────────────────────────────────────────────

def rebalancing_hints(
    holdings: list[dict],
    portfolio_beta: float | None,
    stock_betas: dict,
) -> list[str]:
    """
    Generates plain-English rebalancing suggestions based on:
      - position weight concentration (>30% = overweight)
      - individual stock beta (>1.5 = high risk, <0.5 = very defensive)
      - P&L: large losses combined with high beta = reduce signal
    Returns a list of hint strings (empty if nothing notable).
    """
    hints    = []
    total_v  = sum(h["current_value"] for h in holdings)
    if total_v == 0:
        return hints

    for h in holdings:
        weight = h["current_value"] / total_v * 100
        beta   = stock_betas.get(h["ticker"])
        name   = h["name"][:20]

        if weight > 30:
            hints.append(
                f"⚖️ **{name}** is {weight:.1f}% of portfolio — consider trimming to reduce concentration risk."
            )
        if beta is not None and beta > 1.5 and h["pnl_pct"] < -5:
            hints.append(
                f"⚠️ **{name}** has high beta ({beta:.2f}) and is down {h['pnl_pct']:.1f}% — elevated downside risk."
            )
        if beta is not None and beta < 0.4 and weight < 5:
            hints.append(
                f"🛡️ **{name}** is a very defensive stock (β={beta:.2f}) but only {weight:.1f}% of portfolio — increasing allocation could reduce volatility."
            )

    if portfolio_beta is not None:
        if portfolio_beta > 1.3:
            hints.append(
                f"📈 Portfolio beta is **{portfolio_beta:.2f}** — more volatile than Nifty 50. "
                "Consider adding defensive stocks (FMCG, pharma) to dampen swings."
            )
        elif portfolio_beta < 0.7:
            hints.append(
                f"🛡️ Portfolio beta is **{portfolio_beta:.2f}** — more defensive than Nifty 50. "
                "You may be leaving upside on the table in bull markets."
            )

    if not hints:
        hints.append("✅ Portfolio looks reasonably balanced. No major rebalancing flags.")

    return hints


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyse_portfolio(df: pd.DataFrame) -> dict:
    """
    For each holding: fetches live price, calculates P&L, gets sentiment.
    Now also computes Sharpe, Beta, Correlation and rebalancing hints.
    Returns full portfolio analysis dict including a "metrics" sub-dict.
    """
    holdings        = []
    total_invested  = 0
    total_current   = 0

    for _, row in df.iterrows():
        ticker     = str(row["ticker"]).strip()
        qty        = float(row["qty"])
        buy_price  = float(row["buy_price"])
        stock_name = str(row.get("stock", ticker))

        info = get_stock_info(ticker)
        if "error" in info:
            current_price = buy_price
            name          = stock_name
        else:
            current_price = info.get("current_price", buy_price)
            name          = info.get("name", stock_name)

        invested    = qty * buy_price
        current_val = qty * current_price
        pnl         = current_val - invested
        pnl_pct     = (pnl / invested * 100) if invested else 0

        # Quick sentiment
        articles  = fetch_stock_news(name, ticker, max_articles=3)
        sentiment = analyze_articles_sentiment(articles)

        holdings.append({
            "name":            name,
            "ticker":          ticker,
            "qty":             qty,
            "buy_price":       buy_price,
            "current_price":   current_price,
            "invested":        round(invested, 2),
            "current_value":   round(current_val, 2),
            "pnl":             round(pnl, 2),
            "pnl_pct":         round(pnl_pct, 2),
            "sentiment":       sentiment.get("overall", "Neutral"),
            "sentiment_score": sentiment.get("overall_score", 0),
        })

        total_invested += invested
        total_current  += current_val

    total_pnl     = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0

    # ── Portfolio health score (0-100) ────────────────────────────────────
    winners      = sum(1 for h in holdings if h["pnl"] > 0)
    win_rate     = winners / len(holdings) if holdings else 0
    avg_sentiment = sum(h["sentiment_score"] for h in holdings) / len(holdings) if holdings else 0
    pnl_score     = min(max((total_pnl_pct + 20) / 40 * 50, 0), 50)
    sentiment_score = min(max((avg_sentiment + 1) / 2 * 30, 0), 30)
    win_score     = win_rate * 20
    health_score  = round(pnl_score + sentiment_score + win_score)

    if health_score >= 70:
        health_label = "Healthy 💚";    health_color = "#22c55e"
    elif health_score >= 45:
        health_label = "Moderate 🟡";  health_color = "#f59e0b"
    else:
        health_label = "Needs Attention 🔴"; health_color = "#ef4444"

    # ── Risk Metrics ──────────────────────────────────────────────────────
    tickers     = [h["ticker"] for h in holdings]
    returns_df  = get_portfolio_hist(tickers, period="1y")

    sharpe      = calc_sharpe_ratio(holdings, returns_df)
    beta_data   = calc_portfolio_beta(holdings, returns_df)
    corr_matrix = calc_correlation_matrix(tickers, returns_df)
    hints       = rebalancing_hints(
                      holdings,
                      beta_data["portfolio_beta"],
                      beta_data["stock_betas"],
                  )

    # Add weight + beta to each holding for display
    total_v = sum(h["current_value"] for h in holdings)
    for h in holdings:
        h["weight_pct"] = round(h["current_value"] / total_v * 100, 1) if total_v else 0
        h["beta"]       = beta_data["stock_betas"].get(h["ticker"])

    return {
        "holdings":       holdings,
        "total_invested": round(total_invested, 2),
        "total_current":  round(total_current, 2),
        "total_pnl":      round(total_pnl, 2),
        "total_pnl_pct":  round(total_pnl_pct, 2),
        "health_score":   health_score,
        "health_label":   health_label,
        "health_color":   health_color,
        "winners":        winners,
        "losers":         len(holdings) - winners,
        "metrics": {
            "sharpe_ratio":    sharpe,
            "portfolio_beta":  beta_data["portfolio_beta"],
            "stock_betas":     beta_data["stock_betas"],
            "corr_matrix":     corr_matrix,
            "rebalancing":     hints,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT  (now includes risk metrics)
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_prompt(analysis: dict) -> str:
    """Builds a rich prompt for the LLM to generate portfolio advice."""
    m     = analysis.get("metrics", {})
    sharpe = m.get("sharpe_ratio")
    beta   = m.get("portfolio_beta")

    lines = []
    for h in analysis["holdings"]:
        beta_str  = f"β={h['beta']:.2f}" if h.get("beta") is not None else "β=N/A"
        lines.append(
            f"- {h['name']} ({h['ticker']}): "
            f"Qty={h['qty']}, Bought@₹{h['buy_price']}, "
            f"Now@₹{h['current_price']}, P&L={h['pnl_pct']:+.1f}%, "
            f"Weight={h['weight_pct']:.1f}%, {beta_str}, "
            f"Sentiment={h['sentiment']}"
        )

    holdings_text  = "\n".join(lines)
    sharpe_text    = f"{sharpe:.3f}" if sharpe is not None else "N/A"
    beta_text      = f"{beta:.3f}"   if beta   is not None else "N/A"

    return f"""You are an expert Indian stock market portfolio advisor.

PORTFOLIO SUMMARY:
Total Invested: ₹{analysis['total_invested']:,.0f}
Current Value: ₹{analysis['total_current']:,.0f}
Total P&L: ₹{analysis['total_pnl']:+,.0f} ({analysis['total_pnl_pct']:+.1f}%)
Portfolio Health Score: {analysis['health_score']}/100 ({analysis['health_label']})
Winners: {analysis['winners']} | Losers: {analysis['losers']}
Annualised Sharpe Ratio: {sharpe_text}
Portfolio Beta vs Nifty 50: {beta_text}

INDIVIDUAL HOLDINGS:
{holdings_text}

Please provide:
1. Overall portfolio assessment (2-3 sentences, mention risk-adjusted return if Sharpe is available)
2. Top 2 holdings to HOLD or ADD (with brief reason)
3. Top 1-2 holdings to REVIEW or REDUCE (with brief reason)
4. One key risk to watch (consider beta if above 1.2)
5. One actionable recommendation

Keep response concise and practical. Use ₹ for prices. End with a disclaimer about SEBI advice."""
