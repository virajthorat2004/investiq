"""
core/pdf_report.py
Generates a professional 1-page PDF research report for any stock.
Uses fpdf2 library.
"""
from fpdf import FPDF
from datetime import datetime
import io
import re


def sanitize(text: str) -> str:
    """Removes all non-Latin-1 characters (emojis, arrows, symbols) that Helvetica can't render."""
    if not text:
        return ""
    # Remove emojis and special unicode symbols
    text = re.sub(r'[^\x00-\xFF]', '', str(text))
    # Also remove common problematic chars
    text = text.replace('\u2192', '->').replace('\u2190', '<-').replace('\u2022', '-')
    return text.strip()


class StockReportPDF(FPDF):
    def header(self):
        self.set_fill_color(15, 17, 23)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(26, 29, 46)
        self.rect(0, 0, 210, 22, "F")
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.set_y(6)
        self.cell(0, 10, "InvestIQ | AI Research Report", align="C")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(136, 146, 176)
        self.set_y(14)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}  |  For research purposes only", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(74, 85, 104)
        self.cell(0, 10, "Disclaimer: This report is AI-generated for research only. Not SEBI-registered financial advice. Consult a qualified advisor before investing.", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(102, 126, 234)
        self.set_fill_color(26, 29, 46)
        self.cell(0, 8, f"  {sanitize(title)}", ln=True, fill=True)
        self.ln(1)

    def kv_row(self, label: str, value: str, value_color=None):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(136, 146, 176)
        self.cell(55, 7, sanitize(label))
        self.set_font("Helvetica", "B", 9)
        if value_color:
            self.set_text_color(*value_color)
        else:
            self.set_text_color(226, 232, 240)
        self.cell(0, 7, sanitize(value), ln=True)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(148, 163, 184)
        self.multi_cell(0, 5, sanitize(text))
        self.ln(2)


def generate_stock_report(
    stock_info: dict,
    financials: dict,
    sentiment_data: dict,
    technical_signals: dict,
    prediction: dict,
    ai_outlook: str,
) -> bytes:
    """
    Generates a full PDF research report.
    Returns PDF as bytes for Streamlit download button.
    """
    pdf = StockReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    name = stock_info.get("name", "Unknown")
    ticker = stock_info.get("ticker", "")
    price = stock_info.get("current_price", 0)
    change_pct = stock_info.get("change_pct", 0)
    sector = stock_info.get("sector", "N/A")

    # ── Company Header ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, sanitize(name), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(136, 146, 176)
    pdf.cell(0, 6, f"{sanitize(ticker)}  |  {sanitize(sector)}", ln=True)
    pdf.ln(4)

    # ── Price Banner ──────────────────────────────────────────────────────
    pdf.set_fill_color(26, 29, 46)
    pdf.set_draw_color(45, 49, 84)
    pdf.rect(10, pdf.get_y(), 190, 18, "FD")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(14)
    pdf.cell(60, 18, f"Rs {price}")
    clr = (34, 197, 94) if change_pct >= 0 else (239, 68, 68)
    arr = "+" if change_pct >= 0 else ""
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*clr)
    pdf.cell(40, 18, f"{arr}{change_pct:.2f}% today")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(136, 146, 176)
    sentiment_overall = sanitize(sentiment_data.get("overall", "Neutral"))
    pdf.cell(0, 18, f"Sentiment: {sentiment_overall}")
    pdf.ln(22)

    # ── Two column layout ─────────────────────────────────────────────────
    col_x = [10, 110]
    start_y = pdf.get_y()

    # Left column — Key Metrics
    pdf.set_xy(col_x[0], start_y)
    pdf.section_title("Key Metrics")
    pe = stock_info.get("pe_ratio")
    pdf.kv_row("P/E Ratio:", f"{pe:.1f}" if pe else "N/A")
    pdf.kv_row("52W High:", f"Rs {stock_info.get('52w_high', 'N/A')}")
    pdf.kv_row("52W Low:", f"Rs {stock_info.get('52w_low', 'N/A')}")
    mc = stock_info.get("market_cap", 0)
    mc_str = f"Rs {mc/1e7:,.0f} Cr" if mc else "N/A"
    pdf.kv_row("Market Cap:", mc_str)

    pm = financials.get("profit_margin")
    roe = financials.get("return_on_equity")
    de = financials.get("debt_to_equity")
    rg = financials.get("revenue_growth")

    pdf.kv_row("Profit Margin:", f"{pm*100:.1f}%" if pm else "N/A")
    pdf.kv_row("Return on Equity:", f"{roe*100:.1f}%" if roe else "N/A")
    pdf.kv_row("Debt/Equity:", f"{de:.2f}" if de else "N/A")
    pdf.kv_row("Revenue Growth:", f"{rg*100:.1f}%" if rg else "N/A", (34,197,94) if rg and rg>0 else (239,68,68))

    mid_y = pdf.get_y()

    # Right column — Technical Signals
    pdf.set_xy(col_x[1], start_y)
    pdf.section_title("Technical Signals")
    pdf.set_xy(col_x[1], pdf.get_y())

    rsi_val = technical_signals.get("rsi", "N/A")
    rsi_sig = technical_signals.get("rsi_signal", "Neutral")
    macd_sig = technical_signals.get("macd_signal", "N/A")
    bb_sig = technical_signals.get("bb_signal", "N/A")

    rsi_clr = (239,68,68) if rsi_sig=="Overbought" else ((34,197,94) if rsi_sig=="Oversold" else None)
    pdf.set_xy(col_x[1], pdf.get_y())
    pdf.kv_row("RSI (14):", f"{rsi_val} — {rsi_sig}", rsi_clr)
    pdf.set_xy(col_x[1], pdf.get_y())
    macd_clr = (34,197,94) if macd_sig=="Bullish" else (239,68,68)
    pdf.kv_row("MACD:", macd_sig, macd_clr)
    pdf.set_xy(col_x[1], pdf.get_y())
    pdf.kv_row("Bollinger Bands:", bb_sig)

    # Prediction
    if prediction and "error" not in prediction:
        pdf.set_xy(col_x[1], pdf.get_y() + 2)
        pdf.section_title("ML Price Forecast")
        pdf.set_xy(col_x[1], pdf.get_y())
        days = prediction.get("days_ahead", 14)
        pred_price = prediction.get("predicted_price", "N/A")
        exp_change = prediction.get("expected_change", 0)
        pclr = (34,197,94) if exp_change >= 0 else (239,68,68)
        pdf.kv_row(f"{days}-Day Target:", f"Rs {pred_price}", pclr)
        pdf.set_xy(col_x[1], pdf.get_y())
        pdf.kv_row("Expected Change:", f"{exp_change:+.1f}%", pclr)
        pdf.set_xy(col_x[1], pdf.get_y())
        pdf.kv_row("Model R²:", f"{prediction.get('model_score', 0):.3f}")

    pdf.set_y(max(mid_y, pdf.get_y()) + 4)

    # ── Sentiment Analysis ────────────────────────────────────────────────
    pdf.section_title("News Sentiment Analysis")
    dist = sentiment_data.get("distribution", {})
    score = sentiment_data.get("overall_score", 0)
    summary = sentiment_data.get("summary", "No sentiment data available.")
    pos = dist.get("Positive", 0)
    neg = dist.get("Negative", 0)
    neu = dist.get("Neutral", 0)
    pdf.kv_row("Overall:", f"{sentiment_overall}  (Score: {score:.2f})")
    pdf.kv_row("Distribution:", f"Positive: {pos}  |  Negative: {neg}  |  Neutral: {neu}")
    pdf.body_text(summary)

    # ── Company Description ───────────────────────────────────────────────
    pdf.section_title("Company Overview")
    desc = stock_info.get("summary", "No description available.")
    pdf.body_text(desc[:400] + ("..." if len(desc) > 400 else ""))

    # ── AI Outlook ────────────────────────────────────────────────────────
    if ai_outlook:
        pdf.section_title("AI Analyst Outlook")
        clean_outlook = ai_outlook.replace("**", "").replace("*", "").replace("#", "")
        pdf.body_text(clean_outlook[:600] + ("..." if len(clean_outlook) > 600 else ""))

    # Return as bytes
    return bytes(pdf.output())
