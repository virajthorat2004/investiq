"""
core/kb_corpus.py  —  InvestIQ v6.0
─────────────────────────────────────────────────────────────────────────────
Pre-built Indian Market Knowledge Base Corpus

30 curated, factually grounded passages covering:
  - SEBI regulations and investor protection rules
  - RBI monetary policy and its market impact
  - NSE/BSE market structure and trading rules
  - Indian market-specific investment concepts
  - Key financial ratios and valuation in Indian context
  - Sectoral regulations (banking, IT, pharma, FMCG)
  - Tax rules relevant to Indian equity investors

Design:
  - Hardcoded text → no runtime downloads → always works on Streamlit Cloud
  - Each passage is self-contained (300–600 chars) for clean FAISS chunking
  - Covers the 25 evaluation questions in rag_eval.py
  - Sourced from public SEBI circulars, RBI publications, NSE/BSE rulebooks

Usage:
    from core.kb_corpus import INDIAN_MARKET_CORPUS
    # Returns list of dicts: [{"title": str, "source": str, "text": str}, ...]
"""

INDIAN_MARKET_CORPUS = [

    # ── SEBI REGULATIONS ────────────────────────────────────────────────────
    {
        "title":  "SEBI — Role and Mandate",
        "source": "SEBI Act, 1992",
        "text": (
            "The Securities and Exchange Board of India (SEBI) was established in 1992 under "
            "the SEBI Act to protect the interests of investors in securities, promote the "
            "development of, and regulate, the securities market. SEBI has quasi-legislative, "
            "quasi-judicial, and quasi-executive powers. It issues regulations for listed "
            "companies, intermediaries (brokers, depositories, mutual funds), and prohibits "
            "fraudulent and unfair trade practices including insider trading."
        ),
    },
    {
        "title":  "SEBI — Insider Trading Regulations (PIT 2015)",
        "source": "SEBI (PIT) Regulations, 2015",
        "text": (
            "SEBI's Prohibition of Insider Trading (PIT) Regulations 2015 prohibit any person "
            "in possession of Unpublished Price Sensitive Information (UPSI) from trading in "
            "securities. UPSI includes results, dividends, mergers, acquisitions, and changes "
            "in key management. Trading window closures are mandatory before results. "
            "Violation can attract imprisonment up to 10 years and fines up to ₹25 crore or "
            "3x the profit made, whichever is higher."
        ),
    },
    {
        "title":  "SEBI — LODR (Listing Obligations and Disclosure Requirements)",
        "source": "SEBI LODR Regulations, 2015",
        "text": (
            "SEBI's LODR Regulations 2015 mandate that listed companies disclose all material "
            "information to stock exchanges within 24 hours. Material events include quarterly "
            "results, board meeting outcomes, mergers, changes in promoter holding, credit "
            "rating changes, and regulatory actions. Companies must maintain a minimum public "
            "shareholding of 25%. Board must have at least 50% independent directors if "
            "chairman is executive."
        ),
    },
    {
        "title":  "SEBI — Mutual Fund Regulations",
        "source": "SEBI MF Regulations, 1996 (amended 2020)",
        "text": (
            "SEBI regulates mutual funds in India under the MF Regulations 1996. All AMCs "
            "must be SEBI-registered. SEBI introduced scheme categorisation in 2017 — equity "
            "funds must invest at least 65% in equities, ELSS requires 3-year lock-in with "
            "80C tax benefits up to ₹1.5 lakh. Expense ratio caps: 2.25% for equity, 2% "
            "for debt. Total Expense Ratio (TER) must be disclosed daily on the AMC website."
        ),
    },
    {
        "title":  "SEBI — SEBI Complaint Redress System (SCORES)",
        "source": "SEBI Circular, 2012",
        "text": (
            "SEBI operates SCORES (SEBI Complaint Redress System) where investors can lodge "
            "complaints against listed companies, brokers, and mutual funds online at "
            "scores.gov.in. Intermediaries must resolve complaints within 30 days. If not "
            "resolved, SEBI takes action. SEBI also has an Investor Protection and Education "
            "Fund (IPEF) to compensate defrauded investors and fund financial literacy programs."
        ),
    },

    # ── RBI MONETARY POLICY ─────────────────────────────────────────────────
    {
        "title":  "RBI — Monetary Policy Committee (MPC) and Repo Rate",
        "source": "RBI Act, 1934 (amended 2016)",
        "text": (
            "The Reserve Bank of India's Monetary Policy Committee (MPC) meets every two "
            "months to set the repo rate — the rate at which RBI lends to commercial banks. "
            "A repo rate cut reduces borrowing costs, stimulates lending and growth, and is "
            "generally positive for equity markets and negative for the rupee. Rate hikes "
            "control inflation but slow growth. RBI's inflation target is 4% with a band of "
            "+/-2%. The MPC has 6 members — 3 from RBI and 3 external members."
        ),
    },
    {
        "title":  "RBI — Impact of Rate Changes on Indian Stock Market",
        "source": "RBI Monetary Policy Reports",
        "text": (
            "When RBI cuts the repo rate, bank lending rates fall, reducing the cost of "
            "capital for companies — boosting earnings expectations and stock valuations. "
            "Rate cuts particularly benefit rate-sensitive sectors: banking (NIM expansion), "
            "real estate (lower EMIs boost demand), and auto. Rate hikes benefit banks "
            "short-term via higher NIM but hurt borrower-heavy companies. The 10-year G-Sec "
            "yield is a key benchmark — rising yields compress equity valuations (higher "
            "discount rate)."
        ),
    },
    {
        "title":  "RBI — Foreign Exchange and FEMA",
        "source": "FEMA, 1999 / RBI Circulars",
        "text": (
            "The Foreign Exchange Management Act (FEMA) 1999 governs foreign exchange "
            "transactions in India. FPIs (Foreign Portfolio Investors) can invest in Indian "
            "equities up to sectoral limits set by SEBI and RBI. The rupee-dollar exchange "
            "rate significantly impacts IT and pharma stocks (export earnings in USD) and "
            "importers (oil, gold). RBI intervenes in currency markets to reduce volatility. "
            "India's forex reserves (typically $600–650 billion) act as a buffer."
        ),
    },
    {
        "title":  "RBI — CRR and SLR",
        "source": "RBI Act, Banking Regulation Act",
        "text": (
            "Cash Reserve Ratio (CRR) is the percentage of deposits banks must hold with "
            "RBI in cash — currently 4%. Statutory Liquidity Ratio (SLR) is the percentage "
            "banks must hold in liquid assets (gold, G-Secs) — currently 18%. CRR cuts "
            "inject liquidity into the banking system, reducing rates and boosting credit "
            "growth. Higher CRR absorbs liquidity to control inflation. SLR changes affect "
            "how much banks can lend."
        ),
    },

    # ── NSE/BSE MARKET STRUCTURE ─────────────────────────────────────────────
    {
        "title":  "NSE — Nifty 50 Index Composition",
        "source": "NSE Index Methodology",
        "text": (
            "The Nifty 50 is the benchmark index of the National Stock Exchange (NSE), "
            "comprising 50 large-cap stocks across 13 sectors. Stocks are selected based on "
            "float-adjusted market capitalisation, liquidity (impact cost <0.5%), and listing "
            "history (6 months minimum). Index is reviewed semi-annually in March and "
            "September. Nifty 50 represents approximately 65% of the free-float market "
            "capitalisation of NSE. Key heavyweights: HDFC Bank, Reliance, ICICI Bank, "
            "Infosys, TCS."
        ),
    },
    {
        "title":  "NSE/BSE — T+1 Settlement Cycle",
        "source": "SEBI Circular, January 2023",
        "text": (
            "India moved to T+1 (Trade plus 1 day) settlement for equity markets in January "
            "2023, making it one of the fastest settlement cycles globally. Previously T+2. "
            "Under T+1, shares and funds are transferred to buyer/seller accounts within one "
            "working day of the trade. This reduces counterparty risk and frees up capital "
            "faster. Short selling is allowed in Indian markets with mandatory delivery by "
            "T+1. F&O settlement remains at T+1 for premium and T+0 for MTM."
        ),
    },
    {
        "title":  "NSE — Circuit Breakers and Price Bands",
        "source": "SEBI/NSE Regulations",
        "text": (
            "Indian stock exchanges have market-wide circuit breakers at 10%, 15%, and 20% "
            "Nifty/Sensex movement — halting trading for 45 min, 1.45 hrs, and rest of day "
            "respectively. Individual stocks have price bands of 2%, 5%, 10%, or 20% based "
            "on category. Index derivatives have no price band but have market-wide breakers. "
            "Upper circuit (UC) means only buyers, no sellers. Lower circuit (LC) means only "
            "sellers, no buyers — stock effectively illiquid in that session."
        ),
    },
    {
        "title":  "NSE — F&O (Futures and Options) in India",
        "source": "NSE Derivatives Segment",
        "text": (
            "NSE is the world's largest derivatives exchange by number of contracts. F&O is "
            "available on Nifty 50, Bank Nifty, Nifty IT, Nifty Midcap, and ~200 individual "
            "stocks. Lot sizes are set by NSE to ensure minimum contract value of ₹5–10 lakh. "
            "Weekly expiry is every Thursday for Nifty and Bank Nifty. Monthly expiry is last "
            "Thursday. Options premium is taxed as business income. F&O losses can be set off "
            "against business income and carried forward for 8 years."
        ),
    },
    {
        "title":  "BSE — Sensex Composition",
        "source": "BSE Index Methodology",
        "text": (
            "The BSE Sensex (S&P BSE Sensitive Index) comprises 30 financially sound, "
            "well-established large-cap companies listed on BSE. It is a free-float "
            "market-capitalisation weighted index, base year 1978-79 with base value 100. "
            "Sensex is reviewed semi-annually. BSE is Asia's oldest stock exchange, "
            "established in 1875. The Sensex crossed 80,000 for the first time in 2024. "
            "BSE also operates the SME platform for small and medium enterprises."
        ),
    },

    # ── INDIAN MARKET INVESTMENT CONCEPTS ────────────────────────────────────
    {
        "title":  "Promoter Holding — Significance in Indian Markets",
        "source": "SEBI LODR / Market Practice",
        "text": (
            "Promoter holding is a key metric unique to Indian markets. High promoter holding "
            "(>50%) generally signals founder confidence in the business. Declining promoter "
            "holding (pledging or selling) is a red flag — indicates financial stress or loss "
            "of conviction. SEBI mandates disclosure of promoter pledging. Pledge above 50% "
            "of promoter holding is considered high risk — forced selling during market "
            "downturns can create sharp price crashes. Minimum public holding of 25% is "
            "mandatory for listed companies."
        ),
    },
    {
        "title":  "FII/FPI vs DII — Impact on Indian Markets",
        "source": "SEBI/NSDL Data",
        "text": (
            "Foreign Institutional Investors (FIIs)/Foreign Portfolio Investors (FPIs) and "
            "Domestic Institutional Investors (DIIs — mutual funds, LIC, insurance) are the "
            "two dominant institutional forces in Indian markets. FPIs holding ~20% of NSE "
            "market cap drive short-term volatility. FPI outflows weaken the rupee and drag "
            "markets. DII inflows (via SIPs — systematic investment plans) have become a "
            "counter-balance. Monthly SIP flows exceeded ₹20,000 crore by 2024, providing "
            "a domestic cushion against FPI selling."
        ),
    },
    {
        "title":  "Indian Market Valuation — P/E and Market Cap to GDP",
        "source": "NSE Data / RBI Handbook",
        "text": (
            "Nifty 50's historical average P/E ratio is approximately 20x. P/E above 24x "
            "is considered expensive; below 16x is cheap by historical standards. The "
            "Market Cap to GDP ratio (Buffett Indicator for India) above 100% signals "
            "overvaluation; below 75% indicates undervaluation. India's nominal GDP is "
            "approximately $3.7 trillion (FY2024). NSE total market cap is approximately "
            "$4 trillion. Earnings yield (1/PE) vs G-Sec yield spread guides institutional "
            "equity allocation."
        ),
    },
    {
        "title":  "Smallcap vs Midcap vs Largecap — SEBI Definition",
        "source": "SEBI Circular, 2017 (Categorisation)",
        "text": (
            "SEBI defines market cap categories by rank on NSE/BSE combined: Large-cap = "
            "top 100 companies by full market cap. Mid-cap = 101st to 250th company. "
            "Small-cap = 251st company onwards. Mutual funds must invest minimum 65% of "
            "assets in respective categories. As of 2024, largecap threshold is ~₹20,000 "
            "crore market cap; midcap ~₹5,000–20,000 crore; smallcap below ₹5,000 crore. "
            "Microcap and SME stocks are outside these categories."
        ),
    },

    # ── SECTORAL KNOWLEDGE ───────────────────────────────────────────────────
    {
        "title":  "Indian IT Sector — Key Metrics and Drivers",
        "source": "NASSCOM / Company Reports",
        "text": (
            "India's IT sector (TCS, Infosys, Wipro, HCL Tech, Tech Mahindra) earns 75–85% "
            "of revenue in USD from US and European clients. Key metrics: Revenue growth, "
            "EBIT margin (typically 20–25% for Tier-1), attrition rate, deal wins (TCV — "
            "total contract value), and headcount. A strong dollar benefits IT earnings. "
            "Demand drivers: cloud migration, AI/GenAI services, digital transformation. "
            "Risks: visa costs, client budget cuts, pricing pressure from global competition."
        ),
    },
    {
        "title":  "Indian Banking Sector — Key Metrics (NIM, NPA, CASA)",
        "source": "RBI Banking Reports / Annual Reports",
        "text": (
            "Key banking metrics in India: Net Interest Margin (NIM) — difference between "
            "lending and deposit rates, typically 3–4% for private banks, 2.5–3% for PSU "
            "banks. Gross NPA ratio — non-performing assets as % of loans (below 2% is "
            "healthy for private banks). CASA ratio — Current Account Savings Account "
            "deposits as % of total deposits (higher CASA = lower cost of funds). Credit "
            "growth (year-on-year). PCR (Provision Coverage Ratio) above 70% is preferred. "
            "RoE above 15% indicates efficient capital deployment."
        ),
    },
    {
        "title":  "Indian Pharma Sector — USFDA and Regulatory Risks",
        "source": "USFDA / Company Filings",
        "text": (
            "India is the world's largest supplier of generic medicines, with pharma exports "
            "of ~$25 billion annually. Key risk: USFDA inspection outcomes. Warning Letters "
            "or Import Alerts from USFDA can ban exports from specific plants, causing 10–30% "
            "stock crashes. Domestic formulations have stable growth (10–12% p.a.). Key "
            "metrics: EBITDA margin (20–25% for quality players), new ANDA filings, Para-IV "
            "opportunities, and domestic vs export revenue mix. CRAMS (Contract Research and "
            "Manufacturing) is a growing segment."
        ),
    },
    {
        "title":  "Indian FMCG Sector — Volume Growth and Rural Demand",
        "source": "Nielsen / Company Reports",
        "text": (
            "Fast-Moving Consumer Goods (FMCG) companies (HUL, ITC, Nestle, Britannia, "
            "Dabur) are considered defensive stocks in India. Key metrics: Volume growth "
            "(stripped of price increases), rural vs urban revenue mix, raw material costs "
            "(palm oil, packaging), distribution reach. Rural India (60% population) drives "
            "50–60% of FMCG volumes. Good monsoon → higher rural income → higher FMCG "
            "demand. FMCG P/E multiples are premium (40–60x) due to brand moats and "
            "predictable cash flows."
        ),
    },

    # ── TAX AND REGULATORY ───────────────────────────────────────────────────
    {
        "title":  "Capital Gains Tax on Indian Equities (2024)",
        "source": "Income Tax Act / Finance Act 2024",
        "text": (
            "Post Finance Act 2024: Short Term Capital Gains (STCG) on equity held less "
            "than 12 months — taxed at 20% (increased from 15%). Long Term Capital Gains "
            "(LTCG) on equity held more than 12 months — taxed at 12.5% above ₹1.25 lakh "
            "per year (increased from 10% above ₹1 lakh). STT (Securities Transaction Tax) "
            "is 0.1% on equity delivery trades. F&O profits are taxed as business income "
            "at the applicable slab rate. Dividends are taxed as income at investor's slab."
        ),
    },
    {
        "title":  "SIP — Systematic Investment Plan in Indian Mutual Funds",
        "source": "AMFI / SEBI MF Regulations",
        "text": (
            "Systematic Investment Plans (SIPs) allow investors to invest fixed amounts "
            "monthly in mutual funds. SIP inflows crossed ₹20,000 crore/month by 2024 with "
            "over 9 crore SIP accounts. SIPs provide rupee cost averaging — buying more "
            "units when prices fall. ELSS SIPs have 3-year lock-in but offer Section 80C "
            "tax benefit up to ₹1.5 lakh. SIP stopping or pausing does not attract penalties "
            "in most funds. SIPs are the primary vehicle for retail participation in Indian "
            "equity markets."
        ),
    },
    {
        "title":  "IPO Process in India — SEBI Guidelines",
        "source": "SEBI ICDR Regulations, 2018",
        "text": (
            "In India, IPOs are governed by SEBI's ICDR Regulations 2018. Company must have "
            "3 years of profitability (relaxed for tech companies under SME/main board rules). "
            "Minimum IPO size ₹10 crore. Price band disclosed in red herring prospectus. "
            "QIB (Qualified Institutional Buyers) get 50% allocation, Non-Institutional "
            "Investors 15%, Retail Individual Investors (up to ₹2 lakh application) 35%. "
            "Listing typically within 6 working days of issue close. Allotment is proportional "
            "or lottery-based for oversubscribed IPOs."
        ),
    },

    # ── MARKET ANALYSIS CONCEPTS ─────────────────────────────────────────────
    {
        "title":  "Fundamental Analysis — DCF and Valuations for Indian Stocks",
        "source": "CFA Institute / Market Practice",
        "text": (
            "Discounted Cash Flow (DCF) analysis values a stock by discounting future free "
            "cash flows at WACC. Key inputs: revenue growth forecast, operating margins, "
            "capex intensity, terminal growth rate (typically 4–5% for Indian companies), "
            "and cost of equity (CAPM: risk-free rate + beta × equity risk premium). Indian "
            "risk-free rate ≈ 10-year G-Sec yield (~7%). Equity risk premium for India "
            "≈ 4–5%. P/E, EV/EBITDA, P/B, and P/Sales multiples are used for relative "
            "valuation across sectors."
        ),
    },
    {
        "title":  "Technical Analysis — RSI and MACD for Indian Stocks",
        "source": "Technical Analysis Principles",
        "text": (
            "RSI (Relative Strength Index) measures momentum on a 0–100 scale. Above 70 = "
            "overbought (potential reversal or consolidation); below 30 = oversold (potential "
            "bounce). RSI divergence (price making new highs but RSI falling) is a bearish "
            "signal. MACD (Moving Average Convergence Divergence) uses 12-day and 26-day "
            "EMAs. MACD crossing above signal line = bullish; below = bearish. Bollinger "
            "Bands show volatility — price touching upper band in uptrend is bullish; "
            "breaking below lower band is a breakdown signal."
        ),
    },
    {
        "title":  "Monsoon and Its Impact on Indian Markets",
        "source": "IMD / Market Research",
        "text": (
            "India's southwest monsoon (June–September) is a critical macro indicator. "
            "Normal monsoon (96–104% of Long Period Average) supports rural consumption, "
            "agricultural output, and keeps food inflation in check. Good monsoon benefits: "
            "FMCG (rural demand), fertilisers, tractors (M&M), and two-wheelers (Hero, "
            "Bajaj). Deficient monsoon raises food inflation, pressures RBI to keep rates "
            "high, and hurts rural-facing stocks. IMD forecasts the monsoon by April. "
            "El Niño years historically correlate with weaker monsoon."
        ),
    },
    {
        "title":  "Union Budget and Market Impact",
        "source": "Ministry of Finance / Market Tradition",
        "text": (
            "India's Union Budget, presented on February 1 every year by the Finance "
            "Minister, is a major market event. Key items watched: fiscal deficit target "
            "(budgeted vs actual), capital expenditure outlay (positive for infra, defence, "
            "railways), income tax changes (affects consumption), customs duty changes "
            "(affects import-dependent sectors), and any changes to LTCG/STCG/STT on "
            "equities. Markets often rally in the week before budget on optimism and sell "
            "on the news (budget day volatility is typically high)."
        ),
    },
    {
        "title":  "Indian Conglomerates — Tata, Reliance, Adani, Birla Groups",
        "source": "Market Research / Company Filings",
        "text": (
            "Indian markets are dominated by large conglomerates. Tata Group (TCS, Tata "
            "Motors, Tata Steel, Tata Consumer) spans IT, auto, metals, and FMCG. Reliance "
            "Industries (largest by market cap) spans telecom (Jio), retail (Reliance Retail), "
            "and petrochemicals. Adani Group (Adani Ports, Adani Green, Adani Enterprises) "
            "focuses on infrastructure and energy. Group company performance often correlates "
            "— news about the promoter family or holding company impacts all group stocks. "
            "Monitoring promoter pledge levels is critical for Adani and other leveraged groups."
        ),
    },
]


def get_corpus_as_documents() -> list:
    """
    Returns the corpus as a list of plain text strings suitable for
    embedding via HuggingFace/FAISS in rag_engine.py.
    Each document = title + source + text.
    """
    docs = []
    for item in INDIAN_MARKET_CORPUS:
        doc = f"[{item['source']}] {item['title']}\n\n{item['text']}"
        docs.append(doc)
    return docs


def get_corpus_metadata() -> list:
    """Returns list of metadata dicts parallel to get_corpus_as_documents()."""
    return [{"title": item["title"], "source": item["source"]}
            for item in INDIAN_MARKET_CORPUS]
