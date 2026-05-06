"""
app.py - InvestIQ Full Version v4.0
Features: Market Overview, Stock Analysis, Technicals, Compare, Prediction, Watchlist, Sector Heatmap, Portfolio Analyser, PDF Report
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from core.stock_data import get_stock_info, get_historical_data, get_financials_summary, POPULAR_STOCKS, format_market_cap
from core.news_fetcher import fetch_stock_news, format_news_for_rag
from core.sentiment import analyze_articles_sentiment, get_sentiment_emoji, analyze_sentiment
from core.rag_engine import rag_engine
from core.technical import add_all_indicators, get_rsi_signal, get_macd_signal
from core.comparison import get_comparison_data, normalize_prices, compare_metrics
from core.prediction import predict_prices
from core.watchlist import init_watchlist, add_to_watchlist, remove_from_watchlist, get_watchlist_data
from core.portfolio import analyse_portfolio, parse_portfolio_csv, build_portfolio_prompt, SAMPLE_CSV
from core.sector_heatmap import get_sector_performance, get_top_movers
from core.market_overview import get_index_data, get_top_gainers_losers, get_nifty_history
from core.pdf_report import generate_stock_report

st.set_page_config(page_title="InvestIQ — AI Stock Analyst", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .metric-card { background: linear-gradient(135deg, #1a1d2e, #252840); border: 1px solid #2d3154; border-radius: 12px; padding: 16px 20px; text-align: center; margin-bottom: 8px; }
    .metric-value { font-size: 24px; font-weight: 700; color: #ffffff; margin: 4px 0; }
    .metric-label { font-size: 12px; color: #8892b0; text-transform: uppercase; letter-spacing: 0.8px; }
    .news-card { background: #1a1d2e; border: 1px solid #2d3154; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
    .chat-msg-user { background: #1e3a5f; border-radius: 10px 10px 2px 10px; padding: 10px 14px; margin: 6px 0; color: #e2e8f0; font-size: 14px; }
    .chat-msg-bot { background: #1a1d2e; border: 1px solid #2d3154; border-radius: 10px 10px 10px 2px; padding: 10px 14px; margin: 6px 0; color: #e2e8f0; font-size: 14px; }
    .app-title { font-size: 36px; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1d2e; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [aria-selected="true"] { color: #ffffff; }
    section[data-testid="stSidebar"] { background: #13151f; }
    .holding-row { background: #1a1d2e; border: 1px solid #2d3154; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# Init session state
init_watchlist()
for key, val in [("chat_history", []), ("stock_data_loaded", False), ("current_stock_info", {}), ("current_articles", []), ("sentiment_data", {}), ("hist_df", pd.DataFrame()), ("financials", {})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 InvestIQ")
    st.markdown("*AI Research Analyst for Indian Markets*")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Market Overview",
        "🔍 Stock Analysis",
        "🔔 Watchlist",
        "🏭 Sector Heatmap",
        "🧠 Portfolio Analyser",
    ])

    st.divider()
    st.markdown('<p style="color:#8892b0; font-size:11px; text-align:center;">Built by Viraj Thorat<br>InvestIQ v3.0</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 0: MARKET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Market Overview":
    st.markdown('<div class="app-title">🏠 Market Overview</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#8892b0">Live pulse of Indian stock markets</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        ov_period = st.select_slider("Nifty Chart Period", options=["1mo","3mo","6mo","1y"], value="3mo")
        if st.button("🔄 Refresh Data", use_container_width='stretch'):
            st.cache_data.clear()
            st.rerun()

    # Index cards
    with st.spinner("Fetching market data..."):
        indices = get_index_data()

    cols = st.columns(len(indices))
    for col, idx in zip(cols, indices):
        clr = "#22c55e" if idx["change_pct"] >= 0 else "#ef4444"
        arr = "▲" if idx["change_pct"] >= 0 else "▼"
        col.markdown(f'''<div class="metric-card">
            <div class="metric-label">{idx["name"]}</div>
            <div class="metric-value" style="font-size:20px">{idx["value"]:,.2f}</div>
            <div style="color:{clr};font-size:13px">{arr} {idx["change"]:+.2f} ({idx["change_pct"]:+.2f}%)</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Nifty chart
    nifty_hist = get_nifty_history(ov_period)
    if not nifty_hist.empty:
        st.markdown("### 📈 Nifty 50 — Price Chart")
        fig_nifty = go.Figure()
        fig_nifty.add_trace(go.Scatter(
            x=nifty_hist.index, y=nifty_hist["Close"],
            line=dict(color="#667eea", width=2),
            fill="tozeroy", fillcolor="rgba(102,126,234,0.08)",
            name="Nifty 50"
        ))
        fig_nifty.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e2230"),
            yaxis=dict(gridcolor="#1e2230", title="Index Value"),
            height=300, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig_nifty, use_container_width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # Top gainers / losers
    st.markdown("### 🏆 Today's Top Movers (Nifty 50)")
    st.markdown('<p style="color:#8892b0;font-size:12px">Takes ~20 seconds to fetch all stocks</p>', unsafe_allow_html=True)

    if st.button("📊 Load Top Gainers & Losers", type="primary"):
        with st.spinner("Scanning Nifty 50 stocks..."):
            movers = get_top_gainers_losers(5)

        gc, lc = st.columns(2)
        with gc:
            st.markdown("#### 🟢 Top Gainers")
            for g in movers["gainers"]:
                st.markdown(f'''<div style="background:#1a1d2e;border:1px solid #2d3154;border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;justify-content:space-between">
                    <div>
                        <div style="color:#e2e8f0;font-size:13px;font-weight:500">{g["name"]}</div>
                        <div style="color:#8892b0;font-size:11px">₹{g["price"]}</div>
                    </div>
                    <div style="color:#22c55e;font-size:14px;font-weight:600;align-self:center">▲ {g["change_pct"]:+.2f}%</div>
                </div>''', unsafe_allow_html=True)

        with lc:
            st.markdown("#### 🔴 Top Losers")
            for l in movers["losers"]:
                st.markdown(f'''<div style="background:#1a1d2e;border:1px solid #2d3154;border-radius:8px;padding:10px 14px;margin-bottom:6px">
                    <div style="display:flex;justify-content:space-between">
                        <div>
                            <div style="color:#e2e8f0;font-size:13px;font-weight:500">{l["name"]}</div>
                            <div style="color:#8892b0;font-size:11px">₹{l["price"]}</div>
                        </div>
                        <div style="color:#ef4444;font-size:14px;font-weight:600">▼ {l["change_pct"]:.2f}%</div>
                    </div>
                </div>''', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: STOCK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Stock Analysis":

    with st.sidebar:
        st.markdown("### 🔍 Select Stock")
        mode = st.radio("Select Mode", ["Popular Stocks", "Custom Ticker"], horizontal=True, label_visibility="collapsed")
        if mode == "Popular Stocks":
            stock_name = st.selectbox("Company", list(POPULAR_STOCKS.keys()))
            ticker = POPULAR_STOCKS[stock_name]
        else:
            raw = st.text_input("NSE Ticker (e.g. HCLTECH)", value="HCLTECH.NS")
            ticker = raw.strip().upper()
            if "." not in ticker:
                ticker += ".NS"
            st.caption(f"Using: `{ticker}`")

        period = st.select_slider("Chart Period", options=["1mo","2mo","3mo","4mo","5mo","6mo","9mo","1y","18mo","2y"], value="6mo")

        load_btn = st.button("🚀 Analyse Stock", use_container_width='stretch', type="primary")

        if st.session_state.stock_data_loaded:
            info_s = st.session_state.current_stock_info
            if st.button(f"➕ Add to Watchlist", use_container_width='stretch'):
                add_to_watchlist(ticker)
                st.success("Added!")

        st.divider()
        st.markdown("### 💬 Quick Questions")
        for q in ["Why did this stock move today?", "What is the overall sentiment?", "Should I be concerned about risks?", "Give me a buy/hold/sell outlook"]:
            if st.button(q, use_container_width='stretch', key=f"qq_{q}"):
                st.session_state.quick_question = q

    @st.cache_data(ttl=300)
    def load_all_data(ticker, period):
        stock_info = get_stock_info(ticker)
        hist_df = get_historical_data(ticker, period)
        financials = get_financials_summary(ticker)
        articles = fetch_stock_news(stock_info.get("name", ticker), ticker, max_articles=10)
        sentiment = analyze_articles_sentiment(articles)
        news_text = format_news_for_rag(articles)
        return stock_info, hist_df, financials, articles, sentiment, news_text

    if load_btn:
        with st.spinner(f"Loading {ticker}..."):
            stock_info, hist_df, financials, articles, sentiment, news_text = load_all_data(ticker, period)
            if "error" in stock_info or stock_info.get("current_price", 0) == 0:
                st.error("⚠️ Could not load stock data right now.")
                st.info("💡 Yahoo Finance may be busy. Wait 2-3 minutes and try again.")
                st.stop()
            else:
                st.session_state.current_stock_info = stock_info
                st.session_state.hist_df = hist_df
                st.session_state.financials = financials
                st.session_state.current_articles = articles
                st.session_state.sentiment_data = sentiment
                st.session_state.stock_data_loaded = True
                st.session_state.chat_history = []
                rag_success = rag_engine.build_knowledge_base(news_text, stock_info, financials)
                if rag_success:
                    st.success("✅ AI Knowledge base ready!")

    st.markdown('<div class="app-title">InvestIQ 📊</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#8892b0">AI-Powered Research Analyst for Indian Stock Markets</p>', unsafe_allow_html=True)

    if not st.session_state.stock_data_loaded:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown("### 📈 Live Data\nReal-time NSE/BSE prices, charts & fundamentals")
        c2.markdown("### 🧠 AI Chat\nAsk anything — LLaMA 3 + RAG answers from real news")
        c3.markdown("### 📊 Technicals\nRSI, MACD, Bollinger Bands + ML Price Forecast")
        st.info("👈 Select a stock and click **Analyse Stock** to begin.")
    else:
        info = st.session_state.current_stock_info
        hist_df = st.session_state.hist_df
        financials = st.session_state.financials
        articles = st.session_state.current_articles
        sentiment = st.session_state.sentiment_data

        # KPI row
        ch_cls = "color:#22c55e" if info["change"] >= 0 else "color:#ef4444"
        arr = "▲" if info["change"] >= 0 else "▼"
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Price</div><div class="metric-value">₹{info["current_price"]}</div><div style="{ch_cls}; font-size:13px">{arr} {info["change_pct"]:+.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value" style="font-size:17px">{format_market_cap(info["market_cap"])}</div><div style="color:#8892b0;font-size:12px">{info.get("sector","N/A")}</div></div>', unsafe_allow_html=True)
        pe = info.get("pe_ratio")
        c3.markdown(f'<div class="metric-card"><div class="metric-label">P/E Ratio</div><div class="metric-value">{f"{pe:.1f}" if pe else "N/A"}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">52W High/Low</div><div class="metric-value" style="font-size:16px">₹{info.get("52w_high","N/A")}</div><div style="color:#ef4444;font-size:12px">₹{info.get("52w_low","N/A")}</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="metric-card"><div class="metric-label">Sentiment</div><div class="metric-value" style="font-size:18px">{sentiment.get("overall","Neutral")}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["📈 Chart","📊 Technicals","🆚 Compare","🔮 Prediction","📰 News","💬 AI Chat"])

        # PDF Download button
        with st.expander("📄 Download Research Report (PDF)"):
            if st.button("🖨️ Generate PDF Report", type="primary"):
                with st.spinner("Generating report..."):
                    try:
                        df_ta = add_all_indicators(hist_df) if not hist_df.empty else pd.DataFrame()
                        rsi_v = float(df_ta["rsi"].dropna().iloc[-1]) if not df_ta.empty and not df_ta["rsi"].dropna().empty else None
                        macd_v = df_ta["macd"].dropna().iloc[-1] if not df_ta.empty and not df_ta["macd"].dropna().empty else None
                        sig_v = df_ta["macd_signal"].dropna().iloc[-1] if not df_ta.empty and not df_ta["macd_signal"].dropna().empty else None
                        bb_u = df_ta["bb_upper"].dropna().iloc[-1] if not df_ta.empty and not df_ta["bb_upper"].dropna().empty else None
                        bb_l = df_ta["bb_lower"].dropna().iloc[-1] if not df_ta.empty and not df_ta["bb_lower"].dropna().empty else None
                        rsi_sig = get_rsi_signal(rsi_v)
                        macd_sig = get_macd_signal(macd_v, sig_v)
                        curr_p = info["current_price"]
                        if bb_u and bb_l:
                            bb_label = "Above Upper" if curr_p > bb_u else ("Below Lower" if curr_p < bb_l else "Inside Bands")
                        else:
                            bb_label = "N/A"
                        tech_signals = {
                            "rsi": f"{rsi_v:.1f}" if rsi_v else "N/A",
                            "rsi_signal": rsi_sig["signal"],
                            "macd_signal": macd_sig["signal"],
                            "bb_signal": bb_label,
                        }
                        pred = predict_prices(hist_df, days_ahead=14) if not hist_df.empty else {}
                        ai_outlook = rag_engine.ask("Give a brief investment outlook for this stock in 3-4 sentences.") if rag_engine.vectorstore else ""
                        pdf_bytes = generate_stock_report(info, financials, sentiment, tech_signals, pred, ai_outlook)
                        st.download_button(
                            label="⬇ Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"InvestIQ_{info.get('ticker','stock').replace('.NS','')}_Report.pdf",
                            mime="application/pdf",
                        )
                        st.success("✅ Report ready! Click Download above.")
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")

        with tab1:
            if not hist_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist_df.index, open=hist_df["open"], high=hist_df["high"], low=hist_df["low"], close=hist_df["close"], increasing_line_color="#22c55e", decreasing_line_color="#ef4444", name="OHLC"))
                fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["close"].rolling(20).mean(), line=dict(color="#667eea", width=1.5), name="MA20"))
                fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230", title="Price (₹)"), xaxis_rangeslider_visible=False, height=450, margin=dict(l=0,r=0,t=20,b=0), legend=dict(bgcolor="#1a1d2e"))
                st.plotly_chart(fig, use_container_width='stretch')
                clrs = ["#22c55e" if c>=o else "#ef4444" for c,o in zip(hist_df["close"],hist_df["open"])]
                vf = go.Figure(go.Bar(x=hist_df.index, y=hist_df["volume"], marker_color=clrs))
                vf.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), height=160, margin=dict(l=0,r=0,t=5,b=0), yaxis=dict(gridcolor="#1e2230",title="Volume"), xaxis=dict(gridcolor="#1e2230"))
                st.plotly_chart(vf, use_container_width='stretch')

        with tab2:
            if not hist_df.empty:
                df_ta = add_all_indicators(hist_df)
                rsi_v = df_ta["rsi"].dropna().iloc[-1] if not df_ta["rsi"].dropna().empty else None
                macd_v = df_ta["macd"].dropna().iloc[-1] if not df_ta["macd"].dropna().empty else None
                sig_v = df_ta["macd_signal"].dropna().iloc[-1] if not df_ta["macd_signal"].dropna().empty else None
                bb_u = df_ta["bb_upper"].dropna().iloc[-1] if not df_ta["bb_upper"].dropna().empty else None
                bb_l = df_ta["bb_lower"].dropna().iloc[-1] if not df_ta["bb_lower"].dropna().empty else None
                rsi_s = get_rsi_signal(rsi_v)
                macd_s = get_macd_signal(macd_v, sig_v)
                curr = info["current_price"]
                if bb_u and bb_l:
                    bbl, bbc = ("Above Upper 🔴","#ef4444") if curr>bb_u else (("Below Lower 🟢","#22c55e") if curr<bb_l else ("Inside Bands ✓","#94a3b8"))
                else: bbl, bbc = "N/A","#94a3b8"

                c1,c2,c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><div class="metric-label">RSI (14)</div><div class="metric-value" style="color:{rsi_s["color"]}">{f"{rsi_v:.1f}" if rsi_v else "N/A"}</div><div style="color:{rsi_s["color"]};font-size:13px">{rsi_s["signal"]}</div><div style="color:#8892b0;font-size:11px">{rsi_s["desc"]}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="metric-label">MACD</div><div class="metric-value" style="color:{macd_s["color"]};font-size:20px">{macd_s["signal"]}</div><div style="color:{macd_s["color"]};font-size:12px">{macd_s["desc"]}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="metric-label">Bollinger Bands</div><div class="metric-value" style="color:{bbc};font-size:15px">{bbl}</div><div style="color:#8892b0;font-size:11px">U:₹{f"{bb_u:.1f}" if bb_u else "N/A"} L:₹{f"{bb_l:.1f}" if bb_l else "N/A"}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                fb = go.Figure()
                fb.add_trace(go.Scatter(x=df_ta.index, y=df_ta["bb_upper"], line=dict(color="#667eea",width=1,dash="dash"), name="Upper"))
                fb.add_trace(go.Scatter(x=df_ta.index, y=df_ta["bb_lower"], line=dict(color="#667eea",width=1,dash="dash"), name="Lower", fill="tonexty", fillcolor="rgba(102,126,234,0.08)"))
                fb.add_trace(go.Scatter(x=df_ta.index, y=df_ta["bb_middle"], line=dict(color="#94a3b8",width=1), name="SMA20"))
                fb.add_trace(go.Scatter(x=df_ta.index, y=df_ta["close"], line=dict(color="#ffffff",width=2), name="Close"))
                fb.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230",title="Price (₹)"), height=280, margin=dict(l=0,r=0,t=10,b=0), legend=dict(bgcolor="#1a1d2e"))
                st.plotly_chart(fb, use_container_width='stretch')

                fr = go.Figure()
                fr.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0)
                fr.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.08)", line_width=0)
                fr.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought 70")
                fr.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="Oversold 30")
                fr.add_trace(go.Scatter(x=df_ta.index, y=df_ta["rsi"], line=dict(color="#f59e0b",width=2), name="RSI"))
                fr.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230"), yaxis=dict(range=[0,100],gridcolor="#1e2230"), height=200, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fr, use_container_width='stretch')

                fm = go.Figure()
                fm.add_trace(go.Scatter(x=df_ta.index, y=df_ta["macd"], line=dict(color="#667eea",width=2), name="MACD"))
                fm.add_trace(go.Scatter(x=df_ta.index, y=df_ta["macd_signal"], line=dict(color="#f59e0b",width=2), name="Signal"))
                hc = ["#22c55e" if v>=0 else "#ef4444" for v in df_ta["macd_hist"].fillna(0)]
                fm.add_trace(go.Bar(x=df_ta.index, y=df_ta["macd_hist"], marker_color=hc, name="Histogram"))
                fm.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230"), height=200, margin=dict(l=0,r=0,t=10,b=0), legend=dict(bgcolor="#1a1d2e"))
                st.plotly_chart(fm, use_container_width='stretch')

        with tab3:
            st.markdown("### 🆚 Compare Two Stocks")
            cc1,cc2,cc3 = st.columns([2,2,1])
            t1 = cc1.text_input("Stock 1", value="TCS.NS", key="c1")
            t2 = cc2.text_input("Stock 2", value="INFY.NS", key="c2")
            with cc3:
                st.markdown("<br>", unsafe_allow_html=True)
                cbtn = st.button("Compare", type="primary", use_container_width='stretch')
            if cbtn:
                with st.spinner("Fetching..."):
                    cd = get_comparison_data(t1, t2, period)
                s1,s2 = cd["stock1"],cd["stock2"]
                n1,n2 = s1["info"].get("name",t1), s2["info"].get("name",t2)
                rc1,rc2 = st.columns(2)
                for col,s,n in [(rc1,s1,n1),(rc2,s2,n2)]:
                    p=s["info"].get("current_price",0); ch=s["info"].get("change_pct",0)
                    clr="#22c55e" if ch>=0 else "#ef4444"
                    col.markdown(f'<div class="metric-card"><div class="metric-label">{n}</div><div class="metric-value">₹{p}</div><div style="color:{clr}">{ch:+.2f}%</div></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                nm1,nm2 = normalize_prices(s1["hist"],s2["hist"])
                if not s1["hist"].empty and not s2["hist"].empty:
                    fc = go.Figure()
                    fc.add_trace(go.Scatter(x=s1["hist"].index, y=nm1, line=dict(color="#667eea",width=2), name=n1))
                    fc.add_trace(go.Scatter(x=s2["hist"].index, y=nm2, line=dict(color="#f59e0b",width=2), name=n2))
                    fc.add_hline(y=100, line_dash="dash", line_color="#4a5568")
                    fc.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230",title="Performance (Base=100)"), height=350, margin=dict(l=0,r=0,t=10,b=0), legend=dict(bgcolor="#1a1d2e"))
                    st.plotly_chart(fc, use_container_width='stretch')
                metrics = compare_metrics(s1,s2)
                hc1,hc2,hc3 = st.columns([2,1,1])
                hc1.markdown('<div style="color:#8892b0;font-size:12px;font-weight:500">METRIC</div>', unsafe_allow_html=True)
                hc2.markdown(f'<div style="color:#667eea;font-size:12px;font-weight:500">{n1[:20]}</div>', unsafe_allow_html=True)
                hc3.markdown(f'<div style="color:#f59e0b;font-size:12px;font-weight:500">{n2[:20]}</div>', unsafe_allow_html=True)
                for m in metrics:
                    mc1,mc2,mc3 = st.columns([2,1,1])
                    mc1.markdown(f'<div style="color:#94a3b8;font-size:13px;padding:4px 0;border-top:0.5px solid #1e2230">{m["metric"]}</div>', unsafe_allow_html=True)
                    s1s = "color:#22c55e;font-weight:600" if m["winner"]==1 else "color:#e2e8f0"
                    s2s = "color:#22c55e;font-weight:600" if m["winner"]==2 else "color:#e2e8f0"
                    mc2.markdown(f'<div style="{s1s};font-size:13px;padding:4px 0;border-top:0.5px solid #1e2230">{m["v1"]} {"✓" if m["winner"]==1 else ""}</div>', unsafe_allow_html=True)
                    mc3.markdown(f'<div style="{s2s};font-size:13px;padding:4px 0;border-top:0.5px solid #1e2230">{m["v2"]} {"✓" if m["winner"]==2 else ""}</div>', unsafe_allow_html=True)
            else:
                st.info("Enter two tickers above and click Compare.")

        with tab4:
            st.markdown(f"### 🔮 ML Price Prediction — {info['name']}")
            pred_days = st.radio("Forecast", [7,14,30], horizontal=True, format_func=lambda x: f"{x} days")
            if not hist_df.empty:
                with st.spinner("Running model..."):
                    pred = predict_prices(hist_df, days_ahead=pred_days)
                if "error" in pred:
                    st.warning(pred["error"])
                else:
                    pc1,pc2,pc3 = st.columns(3)
                    pc1.markdown(f'<div class="metric-card"><div class="metric-label">Current Price</div><div class="metric-value">₹{pred["current_price"]}</div></div>', unsafe_allow_html=True)
                    pclr = "#22c55e" if pred["expected_change"]>=0 else "#ef4444"
                    pc2.markdown(f'<div class="metric-card"><div class="metric-label">Predicted in {pred_days}d</div><div class="metric-value">₹{pred["predicted_price"]}</div><div style="color:{pclr};font-size:13px">{pred["expected_change"]:+.1f}%</div></div>', unsafe_allow_html=True)
                    pc3.markdown(f'<div class="metric-card"><div class="metric-label">Model R²</div><div class="metric-value">{pred["model_score"]:.3f}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="background:#1a1d2e;border:1px solid #2d3154;border-radius:10px;padding:12px 16px;margin:10px 0;color:{pred["outlook_color"]};font-size:14px;font-weight:500">{pred["outlook"]}</div>', unsafe_allow_html=True)
                    fp = go.Figure()
                    fp.add_trace(go.Scatter(x=hist_df.index, y=hist_df["close"], line=dict(color="#667eea",width=2), name="Historical"))
                    fp.add_trace(go.Scatter(x=pred["dates"]+pred["dates"][::-1], y=pred["upper"]+pred["lower"][::-1], fill="toself", fillcolor="rgba(245,158,11,0.1)", line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))
                    fp.add_trace(go.Scatter(x=pred["dates"], y=pred["prices"], line=dict(color="#f59e0b",width=2,dash="dot"), name=f"{pred_days}d Forecast"))
                    fp.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230",title="Price (₹)"), height=400, margin=dict(l=0,r=0,t=20,b=0), legend=dict(bgcolor="#1a1d2e"))
                    st.plotly_chart(fp, use_container_width='stretch')
                    st.markdown('<p style="color:#4a5568;font-size:11px">⚠️ ML forecasts are based on historical patterns only. Not financial advice.</p>', unsafe_allow_html=True)

        with tab5:
            cl,cr = st.columns([3,2])
            with cl:
                st.markdown("### 📰 Latest News")
                for art in articles[:8]:
                    s = analyze_sentiment(f"{art.get('title','')} {art.get('description','')}")
                    pub = art.get("published_at","")[:10]
                    st.markdown(f'<div class="news-card"><div style="font-size:14px;font-weight:600;color:#e2e8f0">{get_sentiment_emoji(s["label"])} {art.get("title","")}</div><div style="font-size:11px;color:#8892b0;margin-top:4px">{art.get("source","")} · {pub} · {s["label"]}</div><div style="font-size:12px;color:#94a3b8;margin-top:6px">{art.get("description","")[:140]}...</div></div>', unsafe_allow_html=True)
                    if art.get("url") and "newsapi.org" not in art["url"]:
                        st.markdown(f"[Read ↗]({art['url']})")
            with cr:
                st.markdown("### 🎯 Sentiment")
                dist = sentiment.get("distribution",{})
                if sum(dist.values())>0:
                    pf = go.Figure(go.Pie(labels=list(dist.keys()), values=list(dist.values()), hole=0.6, marker_colors=["#22c55e","#ef4444","#94a3b8"]))
                    pf.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), height=230, margin=dict(l=0,r=0,t=10,b=0), legend=dict(bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(pf, use_container_width='stretch')
                st.markdown(f'<div class="metric-card"><div class="metric-label">Overall</div><div class="metric-value">{sentiment.get("overall","Neutral")}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background:#1a1d2e;border:1px solid #2d3154;border-radius:10px;padding:12px;margin-top:8px;color:#94a3b8;font-size:13px;line-height:1.7">{sentiment.get("summary","")}</div>', unsafe_allow_html=True)

        with tab6:
            st.markdown(f"### 💬 Ask Anything About {info['name']}")
            st.markdown('<p style="color:#8892b0;font-size:13px">Powered by LLaMA 3 + RAG — grounded in real news & fundamentals</p>', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                css = "chat-msg-user" if msg["role"]=="user" else "chat-msg-bot"
                icon = "🧑" if msg["role"]=="user" else "🤖"
                st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
            if hasattr(st.session_state, "quick_question"):
                user_input = st.session_state.quick_question
                del st.session_state.quick_question
            else:
                user_input = None
            with st.form("cf", clear_on_submit=True):
                fi1,fi2 = st.columns([5,1])
                typed = fi1.text_input("", placeholder="Ask anything about this stock...", label_visibility="collapsed")
                send = fi2.form_submit_button("Send", use_container_width='stretch')
            if send and typed: user_input = typed
            if user_input:
                st.session_state.chat_history.append({"role":"user","content":user_input})
                with st.spinner("Thinking..."):
                    response = rag_engine.ask(user_input)
                st.session_state.chat_history.append({"role":"assistant","content":response})
                st.rerun()
            st.markdown('<p style="color:#4a5568;font-size:11px">⚠️ Not SEBI-registered financial advice.</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: WATCHLIST
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔔 Watchlist":
    st.markdown('<div class="app-title">🔔 Watchlist</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#8892b0">Track your favourite stocks in one place</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ➕ Add Stock")
        new_t = st.text_input("Ticker (e.g. WIPRO.NS)")
        if st.button("Add to Watchlist", use_container_width='stretch'):
            if new_t:
                t = new_t.strip().upper()
                if "." not in t: t += ".NS"
                add_to_watchlist(t)
                st.success(f"Added {t}!")

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add stocks from the sidebar or click '➕ Add to Watchlist' on the Stock Analysis page.")
    else:
        with st.spinner("Fetching live prices..."):
            wdata = get_watchlist_data()

        if wdata:
            header = st.columns([3,1,1,1,1,1])
            for col, label in zip(header, ["Company","Price","Change","Market Cap","P/E","Action"]):
                col.markdown(f'<div style="color:#8892b0;font-size:12px;font-weight:500;padding:6px 0;border-bottom:1px solid #2d3154">{label}</div>', unsafe_allow_html=True)

            for w in wdata:
                ch = w.get("change_pct", 0)
                clr = "#22c55e" if ch >= 0 else "#ef4444"
                arr = "▲" if ch >= 0 else "▼"
                pe = w.get("pe_ratio")
                c1,c2,c3,c4,c5,c6 = st.columns([3,1,1,1,1,1])
                c1.markdown(f'<div style="color:#e2e8f0;font-size:13px;padding:8px 0">{w.get("name","")[:30]}</div>', unsafe_allow_html=True)
                c2.markdown(f'<div style="color:#ffffff;font-size:13px;padding:8px 0;font-weight:500">₹{w.get("current_price","")}</div>', unsafe_allow_html=True)
                c3.markdown(f'<div style="color:{clr};font-size:13px;padding:8px 0">{arr} {ch:+.2f}%</div>', unsafe_allow_html=True)
                c4.markdown(f'<div style="color:#94a3b8;font-size:13px;padding:8px 0">{format_market_cap(w.get("market_cap",0))}</div>', unsafe_allow_html=True)
                c5.markdown(f'<div style="color:#94a3b8;font-size:13px;padding:8px 0">{f"{pe:.1f}" if pe else "N/A"}</div>', unsafe_allow_html=True)
                if c6.button("✕", key=f"rm_{w['ticker']}"):
                    remove_from_watchlist(w["ticker"])
                    st.rerun()

        if st.button("🔄 Refresh Prices", use_container_width='content'):
            st.cache_data.clear()
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: SECTOR HEATMAP
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Sector Heatmap":
    st.markdown('<div class="app-title">🏭 NSE Sector Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#8892b0">Today\'s performance across Indian market sectors</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Fetching sector data... (takes ~20 seconds)"):
        sector_df = get_sector_performance()

    if not sector_df.empty:
        # Treemap
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in sector_df["change_pct"]]
        labels = [f"{row['sector']}<br>{row['change_pct']:+.2f}%" for _, row in sector_df.iterrows()]

        fig_tree = go.Figure(go.Treemap(
            labels=labels,
            parents=[""] * len(sector_df),
            values=[max(abs(v), 0.1) for v in sector_df["change_pct"]],
            marker=dict(colors=sector_df["change_pct"].tolist(), colorscale=[[0,"#ef4444"],[0.5,"#1a1d2e"],[1,"#22c55e"]], cmid=0, showscale=True, colorbar=dict(title=dict(text="% Change", font=dict(color="#e2e8f0")), tickfont=dict(color="#e2e8f0"))),
            textfont=dict(size=16, color="#ffffff"),
        ))
        fig_tree.update_layout(paper_bgcolor="#0f1117", margin=dict(l=0,r=0,t=10,b=0), height=450)
        st.plotly_chart(fig_tree, use_container_width='stretch')

        # Bar chart
        sector_df_sorted = sector_df.sort_values("change_pct", ascending=True)
        bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in sector_df_sorted["change_pct"]]
        fig_bar = go.Figure(go.Bar(
            x=sector_df_sorted["change_pct"],
            y=sector_df_sorted["sector"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.2f}%" for v in sector_df_sorted["change_pct"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_bar.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230", title="% Change Today"), yaxis=dict(gridcolor="#1e2230"), height=320, margin=dict(l=0,r=60,t=10,b=0))
        st.plotly_chart(fig_bar, use_container_width='stretch')

        # Top/bottom sectors
        best = sector_df.loc[sector_df["change_pct"].idxmax()]
        worst = sector_df.loc[sector_df["change_pct"].idxmin()]
        c1,c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Best Sector Today</div><div class="metric-value" style="color:#22c55e">{best["sector"]}</div><div style="color:#22c55e;font-size:14px">{best["change_pct"]:+.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Worst Sector Today</div><div class="metric-value" style="color:#ef4444">{worst["sector"]}</div><div style="color:#ef4444;font-size:14px">{worst["change_pct"]:+.2f}%</div></div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh Sector Data"):
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: PORTFOLIO ANALYSER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Portfolio Analyser":
    st.markdown('<div class="app-title">🧠 AI Portfolio Analyser</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#8892b0">Upload your holdings — get live P&L, sentiment analysis & AI advice</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📥 Download Sample CSV")
        st.download_button("⬇ Download Sample CSV", data=SAMPLE_CSV, file_name="sample_portfolio.csv", mime="text/csv")
        st.markdown("**CSV Format Required:**")
        st.code("stock, ticker, qty, buy_price")

    col_up, col_info = st.columns([2,1])
    with col_up:
        uploaded = st.file_uploader("Upload your portfolio CSV", type=["csv"])
    with col_info:
        st.markdown("""
        <div style="background:#1a1d2e;border:1px solid #2d3154;border-radius:10px;padding:14px;margin-top:8px">
        <div style="color:#e2e8f0;font-size:13px;font-weight:500">Required columns:</div>
        <div style="color:#94a3b8;font-size:12px;margin-top:6px;line-height:1.8">
        • <b>stock</b> — Company name<br>
        • <b>ticker</b> — NSE ticker (e.g. TCS.NS)<br>
        • <b>qty</b> — Number of shares<br>
        • <b>buy_price</b> — Your purchase price (₹)
        </div>
        </div>""", unsafe_allow_html=True)

    if uploaded:
        content = uploaded.read().decode("utf-8")
        df_port = parse_portfolio_csv(content)

        if df_port is None:
            st.error("Invalid CSV format. Please check columns: stock, ticker, qty, buy_price")
        else:
            with st.spinner("Analysing your portfolio — fetching live prices & news..."):
                analysis = analyse_portfolio(df_port)

            # Summary KPIs
            pnl_clr = "#22c55e" if analysis["total_pnl"] >= 0 else "#ef4444"
            pnl_arr = "▲" if analysis["total_pnl"] >= 0 else "▼"
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="metric-label">Total Invested</div><div class="metric-value" style="font-size:18px">₹{analysis["total_invested"]:,.0f}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-label">Current Value</div><div class="metric-value" style="font-size:18px">₹{analysis["total_current"]:,.0f}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-label">Total P&L</div><div class="metric-value" style="color:{pnl_clr};font-size:18px">{pnl_arr} ₹{abs(analysis["total_pnl"]):,.0f}</div><div style="color:{pnl_clr};font-size:13px">{analysis["total_pnl_pct"]:+.2f}%</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="metric-label">Portfolio Health</div><div class="metric-value" style="color:{analysis["health_color"]};font-size:18px">{analysis["health_score"]}/100</div><div style="color:{analysis["health_color"]};font-size:13px">{analysis["health_label"]}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Holdings breakdown
            st.markdown("### 📋 Holdings Breakdown")
            hdr = st.columns([3,1,1,1,1,1,1])
            for col, lbl in zip(hdr, ["Stock","Qty","Buy Price","Current","Invested","P&L","Sentiment"]):
                col.markdown(f'<div style="color:#8892b0;font-size:11px;font-weight:500;border-bottom:1px solid #2d3154;padding-bottom:6px">{lbl}</div>', unsafe_allow_html=True)

            for h in analysis["holdings"]:
                pclr = "#22c55e" if h["pnl"] >= 0 else "#ef4444"
                sc = {"Bullish 📈":"#22c55e","Bearish 📉":"#ef4444"}.get(h["sentiment"],"#94a3b8")
                cols = st.columns([3,1,1,1,1,1,1])
                cols[0].markdown(f'<div style="color:#e2e8f0;font-size:13px;padding:6px 0">{h["name"][:25]}</div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div style="color:#94a3b8;font-size:13px;padding:6px 0">{h["qty"]:.0f}</div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div style="color:#94a3b8;font-size:13px;padding:6px 0">₹{h["buy_price"]:.0f}</div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div style="color:#ffffff;font-size:13px;padding:6px 0;font-weight:500">₹{h["current_price"]:.0f}</div>', unsafe_allow_html=True)
                cols[4].markdown(f'<div style="color:#94a3b8;font-size:13px;padding:6px 0">₹{h["invested"]:,.0f}</div>', unsafe_allow_html=True)
                cols[5].markdown(f'<div style="color:{pclr};font-size:13px;padding:6px 0;font-weight:500">{h["pnl_pct"]:+.1f}%</div>', unsafe_allow_html=True)
                cols[6].markdown(f'<div style="color:{sc};font-size:12px;padding:6px 0">{h["sentiment"]}</div>', unsafe_allow_html=True)

            # P&L waterfall chart
            st.markdown("<br>")
            st.markdown("### 📊 P&L by Stock")
            names = [h["name"][:15] for h in analysis["holdings"]]
            pnls = [h["pnl_pct"] for h in analysis["holdings"]]
            bar_c = ["#22c55e" if p >= 0 else "#ef4444" for p in pnls]
            fig_pnl = go.Figure(go.Bar(x=names, y=pnls, marker_color=bar_c, text=[f"{p:+.1f}%" for p in pnls], textposition="outside", textfont=dict(color="#e2e8f0")))
            fig_pnl.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230", title="P&L %"), height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_pnl, use_container_width='stretch')

            # AI Analysis
            st.markdown("### 🤖 AI Portfolio Advice")
            if st.button("🧠 Generate AI Analysis", type="primary"):
                with st.spinner("AI is analysing your portfolio..."):
                    prompt = build_portfolio_prompt(analysis)
                    ai_response = rag_engine.llm.invoke(prompt) if rag_engine.llm else None
                    if ai_response:
                        advice = ai_response.content if hasattr(ai_response, "content") else str(ai_response)
                        st.markdown(f'<div style="background:#1a1d2e;border:1px solid #2d3154;border-radius:12px;padding:20px;color:#e2e8f0;font-size:14px;line-height:1.8">{advice}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Configure GROQ_API_KEY to enable AI portfolio advice.")

            st.markdown('<p style="color:#4a5568;font-size:11px;margin-top:10px">⚠️ AI analysis is for research only. Not SEBI-registered financial advice. Consult a qualified advisor before investing.</p>', unsafe_allow_html=True)
    else:
        st.info("👆 Upload your portfolio CSV to get started. Download the sample CSV from the sidebar to see the required format.")
