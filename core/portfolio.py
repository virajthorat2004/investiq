"""
core/portfolio.py
AI Portfolio Analyser — upload CSV of holdings, get full AI analysis.
"""
import pandas as pd
import io
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


def parse_portfolio_csv(file_content: str) -> pd.DataFrame:
    """Parses uploaded CSV into a clean DataFrame."""
    try:
        df = pd.read_csv(io.StringIO(file_content))
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"ticker", "qty", "buy_price"}
        if not required.issubset(set(df.columns)):
            return None
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
        df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
        df = df.dropna(subset=["ticker", "qty", "buy_price"])
        return df
    except Exception as e:
        return None


def analyse_portfolio(df: pd.DataFrame) -> dict:
    """
    For each holding: fetches live price, calculates P&L,
    gets sentiment. Returns full portfolio analysis.
    """
    holdings = []
    total_invested = 0
    total_current = 0

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        qty = float(row["qty"])
        buy_price = float(row["buy_price"])
        stock_name = str(row.get("stock", ticker))

        info = get_stock_info(ticker)
        if "error" in info:
            current_price = buy_price
            name = stock_name
        else:
            current_price = info.get("current_price", buy_price)
            name = info.get("name", stock_name)

        invested = qty * buy_price
        current_val = qty * current_price
        pnl = current_val - invested
        pnl_pct = (pnl / invested * 100) if invested else 0

        # Quick sentiment
        articles = fetch_stock_news(name, ticker, max_articles=3)
        sentiment = analyze_articles_sentiment(articles)

        holdings.append({
            "name": name,
            "ticker": ticker,
            "qty": qty,
            "buy_price": buy_price,
            "current_price": current_price,
            "invested": round(invested, 2),
            "current_value": round(current_val, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "sentiment": sentiment.get("overall", "Neutral"),
            "sentiment_score": sentiment.get("overall_score", 0),
        })

        total_invested += invested
        total_current += current_val

    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0

    # Portfolio health score (0-100)
    winners = sum(1 for h in holdings if h["pnl"] > 0)
    win_rate = winners / len(holdings) if holdings else 0
    avg_sentiment = sum(h["sentiment_score"] for h in holdings) / len(holdings) if holdings else 0
    pnl_score = min(max((total_pnl_pct + 20) / 40 * 50, 0), 50)
    sentiment_score = min(max((avg_sentiment + 1) / 2 * 30, 0), 30)
    win_score = win_rate * 20
    health_score = round(pnl_score + sentiment_score + win_score)

    if health_score >= 70:
        health_label = "Healthy 💚"
        health_color = "#22c55e"
    elif health_score >= 45:
        health_label = "Moderate 🟡"
        health_color = "#f59e0b"
    else:
        health_label = "Needs Attention 🔴"
        health_color = "#ef4444"

    return {
        "holdings": holdings,
        "total_invested": round(total_invested, 2),
        "total_current": round(total_current, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "health_score": health_score,
        "health_label": health_label,
        "health_color": health_color,
        "winners": winners,
        "losers": len(holdings) - winners,
    }


def build_portfolio_prompt(analysis: dict) -> str:
    """Builds a rich prompt for the LLM to generate portfolio advice."""
    h = analysis["holdings"]
    lines = []
    for holding in h:
        lines.append(
            f"- {holding['name']} ({holding['ticker']}): "
            f"Qty={holding['qty']}, Bought@₹{holding['buy_price']}, "
            f"Now@₹{holding['current_price']}, "
            f"P&L={holding['pnl_pct']:+.1f}%, "
            f"News Sentiment={holding['sentiment']}"
        )

    holdings_text = "\n".join(lines)

    return f"""You are an expert Indian stock market portfolio advisor.

PORTFOLIO SUMMARY:
Total Invested: ₹{analysis['total_invested']:,.0f}
Current Value: ₹{analysis['total_current']:,.0f}
Total P&L: ₹{analysis['total_pnl']:+,.0f} ({analysis['total_pnl_pct']:+.1f}%)
Portfolio Health Score: {analysis['health_score']}/100 ({analysis['health_label']})
Winners: {analysis['winners']} | Losers: {analysis['losers']}

INDIVIDUAL HOLDINGS:
{holdings_text}

Please provide:
1. Overall portfolio assessment (2-3 sentences)
2. Top 2 holdings to HOLD or ADD (with brief reason)
3. Top 1-2 holdings to REVIEW or REDUCE (with brief reason)
4. One key risk to watch
5. One actionable recommendation

Keep response concise and practical. Use ₹ for prices. End with a disclaimer about SEBI advice."""
