"""
eval/rag_eval.py  —  InvestIQ v6.0
─────────────────────────────────────────────────────────────────────────────
RAG Evaluation Script — Generates IEEE Paper Table III

Compares:
  A) Plain LLaMA 3 (no retrieval, no context)
  B) KB-RAG (LLaMA 3 + 30-passage Indian market knowledge base)

Evaluation method:
  - 25 Indian market questions spanning SEBI, RBI, NSE/BSE, sectors, tax
  - Each answer scored 0/1/2 by keyword matching against ground-truth
    key facts (automated proxy for human evaluation)
  - Final table: Question | Plain LLM Score | RAG Score | Winner
  - Saves to eval/rag_eval_results.csv (paste into IEEE paper Table III)

Usage:
    cd /path/to/investiq
    export GROQ_API_KEY=your_key_here
    python eval/rag_eval.py

Runtime: ~3–5 minutes (25 questions × 2 models × Groq API calls)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import time

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# 25 EVALUATION QUESTIONS + GROUND TRUTH KEY FACTS
# ─────────────────────────────────────────────────────────────────────────────
# Scoring: answer gets 1 point per key_fact found (case-insensitive substring)
# Max score per question = len(key_facts), normalised to 0–2 scale.

EVAL_QUESTIONS = [
    # SEBI (5 questions)
    {
        "id": 1, "category": "SEBI",
        "question": "What is SEBI and what are its main powers?",
        "key_facts": ["1992", "investor", "quasi-legislative", "securities market"],
    },
    {
        "id": 2, "category": "SEBI",
        "question": "What is insider trading and what are the penalties under SEBI regulations?",
        "key_facts": ["UPSI", "unpublished price sensitive", "10 years", "25 crore"],
    },
    {
        "id": 3, "category": "SEBI",
        "question": "What are SEBI LODR regulations and what must companies disclose?",
        "key_facts": ["24 hours", "material", "25%", "independent directors"],
    },
    {
        "id": 4, "category": "SEBI",
        "question": "How are mutual funds regulated in India and what are expense ratio limits?",
        "key_facts": ["2.25%", "65%", "80C", "AMC", "SEBI"],
    },
    {
        "id": 5, "category": "SEBI",
        "question": "How can an investor file a complaint against a broker in India?",
        "key_facts": ["SCORES", "30 days", "scores.gov.in", "IPEF"],
    },

    # RBI (5 questions)
    {
        "id": 6, "category": "RBI",
        "question": "What is the repo rate and how does it affect the stock market?",
        "key_facts": ["repo rate", "RBI", "lending", "inflation", "equity"],
    },
    {
        "id": 7, "category": "RBI",
        "question": "Which sectors benefit most from an RBI rate cut?",
        "key_facts": ["banking", "real estate", "auto", "NIM", "rate-sensitive"],
    },
    {
        "id": 8, "category": "RBI",
        "question": "What is CRR and SLR and how do they affect banking stocks?",
        "key_facts": ["Cash Reserve Ratio", "Statutory Liquidity Ratio", "4%", "18%", "liquidity"],
    },
    {
        "id": 9, "category": "RBI",
        "question": "How does a weak Indian rupee affect IT and pharma stocks?",
        "key_facts": ["USD", "export", "rupee", "earnings", "dollar"],
    },
    {
        "id": 10, "category": "RBI",
        "question": "What is the RBI's inflation target?",
        "key_facts": ["4%", "2%", "band", "MPC", "inflation target"],
    },

    # NSE/BSE Market Structure (5 questions)
    {
        "id": 11, "category": "Market Structure",
        "question": "What is the Nifty 50 and how is it composed?",
        "key_facts": ["50", "13 sectors", "market capitalisation", "semi-annually", "NSE"],
    },
    {
        "id": 12, "category": "Market Structure",
        "question": "What is T+1 settlement and when did India adopt it?",
        "key_facts": ["T+1", "2023", "one day", "settlement", "counterparty"],
    },
    {
        "id": 13, "category": "Market Structure",
        "question": "How do circuit breakers work in Indian stock markets?",
        "key_facts": ["10%", "15%", "20%", "Nifty", "halt"],
    },
    {
        "id": 14, "category": "Market Structure",
        "question": "What is promoter holding and why does it matter for Indian investors?",
        "key_facts": ["promoter", "pledge", "25%", "red flag", "confidence"],
    },
    {
        "id": 15, "category": "Market Structure",
        "question": "What is the difference between FII/FPI and DII in Indian markets?",
        "key_facts": ["FPI", "DII", "SIP", "mutual fund", "rupee"],
    },

    # Sectors (5 questions)
    {
        "id": 16, "category": "Sectors",
        "question": "What are the key metrics to evaluate Indian IT companies like TCS and Infosys?",
        "key_facts": ["USD", "EBIT margin", "attrition", "deal wins", "TCV"],
    },
    {
        "id": 17, "category": "Sectors",
        "question": "What is NIM and NPA and why are they important for Indian banking stocks?",
        "key_facts": ["Net Interest Margin", "Non-performing", "CASA", "PCR", "credit"],
    },
    {
        "id": 18, "category": "Sectors",
        "question": "What is the biggest risk for Indian pharma stocks?",
        "key_facts": ["USFDA", "Warning Letter", "Import Alert", "generic", "export"],
    },
    {
        "id": 19, "category": "Sectors",
        "question": "How does the monsoon affect Indian stock markets?",
        "key_facts": ["monsoon", "rural", "FMCG", "inflation", "IMD"],
    },
    {
        "id": 20, "category": "Sectors",
        "question": "What is the significance of the Union Budget for Indian stock markets?",
        "key_facts": ["February", "fiscal deficit", "capital expenditure", "LTCG", "STT"],
    },

    # Tax & Valuations (5 questions)
    {
        "id": 21, "category": "Tax",
        "question": "What is the capital gains tax on equity in India after the 2024 Finance Act?",
        "key_facts": ["20%", "12.5%", "12 months", "LTCG", "STCG"],
    },
    {
        "id": 22, "category": "Tax",
        "question": "How are F&O profits taxed in India?",
        "key_facts": ["business income", "slab", "F&O", "carry forward", "8 years"],
    },
    {
        "id": 23, "category": "Tax",
        "question": "What is SIP and why is it popular in India?",
        "key_facts": ["systematic", "monthly", "rupee cost averaging", "80C", "SIP"],
    },
    {
        "id": 24, "category": "Valuations",
        "question": "What Nifty P/E ratio is considered expensive or cheap historically?",
        "key_facts": ["20x", "24x", "16x", "P/E", "Nifty"],
    },
    {
        "id": 25, "category": "Valuations",
        "question": "How are largecap, midcap, and smallcap stocks defined in India by SEBI?",
        "key_facts": ["top 100", "101", "250", "251", "market cap"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_answer(answer: str, key_facts: list) -> tuple:
    """
    Returns (raw_score, normalised_0_to_2).
    Checks how many key_facts appear in the answer (case-insensitive).
    """
    answer_lower = answer.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    raw  = hits
    norm = round(min(2.0, (hits / max(len(key_facts), 1)) * 2), 2)
    return raw, norm


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "="*70)
    print("  InvestIQ RAG Evaluation — IEEE Paper Table III")
    print("  Comparing: Plain LLaMA 3  vs  KB-RAG (Indian Market Corpus)")
    print("="*70 + "\n")

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not set.")
        print("   Export it: export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    # Initialise RAG engine
    print("Loading RAG engine and KB corpus...")
    from core.rag_engine import rag_engine
    if not rag_engine._kb_loaded:
        print("❌ ERROR: KB corpus not loaded. Check core/kb_corpus.py exists.")
        sys.exit(1)
    if not rag_engine.llm:
        print("❌ ERROR: LLM not initialised. Check GROQ_API_KEY.")
        sys.exit(1)
    print(f"✅ KB loaded: {len(rag_engine._kb_docs)} passages")
    print(f"✅ LLM ready: llama3-8b-8192\n")

    results = []
    total_plain = 0.0
    total_rag   = 0.0

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i:02d}/25] {q['category']}: {q['question'][:60]}...")

        # Plain LLaMA (no RAG)
        plain_answer = rag_engine.ask_plain_llm(q["question"])
        time.sleep(0.5)   # avoid Groq rate limit

        # KB-RAG
        rag_answer = rag_engine.ask_kb_only(q["question"])
        time.sleep(0.5)

        # Score both
        _, plain_norm = score_answer(plain_answer, q["key_facts"])
        _, rag_norm   = score_answer(rag_answer,   q["key_facts"])

        winner = "RAG" if rag_norm > plain_norm else ("TIE" if rag_norm == plain_norm else "Plain LLM")

        total_plain += plain_norm
        total_rag   += rag_norm

        results.append({
            "id":            q["id"],
            "category":      q["category"],
            "question":      q["question"],
            "plain_score":   plain_norm,
            "rag_score":     rag_norm,
            "winner":        winner,
            "plain_answer":  plain_answer[:200].replace("\n", " "),
            "rag_answer":    rag_answer[:200].replace("\n", " "),
        })

        status = "✅ RAG wins" if winner == "RAG" else ("🔴 Plain wins" if winner == "Plain LLM" else "⚪ Tie")
        print(f"   Plain: {plain_norm:.2f} | RAG: {rag_norm:.2f} | {status}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n     = len(results)
    avg_p = total_plain / n
    avg_r = total_rag   / n
    rag_wins   = sum(1 for r in results if r["winner"] == "RAG")
    plain_wins = sum(1 for r in results if r["winner"] == "Plain LLM")
    ties       = sum(1 for r in results if r["winner"] == "TIE")
    improvement = ((avg_r - avg_p) / max(avg_p, 0.01)) * 100

    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"  Questions evaluated  : {n}")
    print(f"  Plain LLaMA avg score: {avg_p:.3f} / 2.00")
    print(f"  KB-RAG avg score     : {avg_r:.3f} / 2.00")
    print(f"  RAG improvement      : +{improvement:.1f}%")
    print(f"  RAG wins             : {rag_wins}/25")
    print(f"  Plain LLM wins       : {plain_wins}/25")
    print(f"  Ties                 : {ties}/25")
    print("="*70)

    # ── Category breakdown ────────────────────────────────────────────────────
    print("\n  CATEGORY BREAKDOWN:")
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_r = [r for r in results if r["category"] == cat]
        cp = sum(r["plain_score"] for r in cat_r) / len(cat_r)
        cr = sum(r["rag_score"]   for r in cat_r) / len(cat_r)
        print(f"  {cat:<20} Plain: {cp:.2f}  RAG: {cr:.2f}  Δ: {cr-cp:+.2f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs("eval", exist_ok=True)
    csv_path = "eval/rag_eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "category", "question",
                      "plain_score", "rag_score", "winner",
                      "plain_answer", "rag_answer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

        # Summary rows
        writer.writerow({})
        writer.writerow({"id": "AVERAGE", "category": "",
                         "question": "OVERALL AVERAGE",
                         "plain_score": round(avg_p, 3),
                         "rag_score": round(avg_r, 3),
                         "winner": f"RAG +{improvement:.1f}%"})

    print(f"\n✅ Results saved to: {csv_path}")
    print("   → Use this table as Table III in your IEEE paper.\n")

    return results, avg_p, avg_r


if __name__ == "__main__":
    run_evaluation()
