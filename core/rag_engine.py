"""
core/rag_engine.py  —  InvestIQ v6.0
─────────────────────────────────────────────────────────────────────────────
RAG Engine with Pre-Built Indian Market Knowledge Base

Architecture:
  1. Pre-built KB corpus (30 passages from kb_corpus.py) — loaded at startup,
     always available, covers SEBI/RBI/NSE/BSE/tax/sector knowledge
  2. Live news + stock fundamentals — added per stock when user loads a ticker
  3. Both merged into a single FAISS index for each query

This means the AI can now answer:
  - "What does SEBI say about insider trading?" → KB corpus
  - "What is TCS's current revenue growth?" → live news
  - "Should I buy TCS given RBI policy?" → both combined

Query routing:
  - All queries search the combined index (KB + live)
  - KB passages are retrieved alongside news for context
"""

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGEngine:

    def __init__(self):
        self.llm         = None
        self.embeddings  = None
        self.vectorstore = None
        self._kb_docs    = []      # pre-built KB documents (loaded once)
        self._kb_loaded  = False
        self._init_llm()
        self._init_embeddings()
        self._load_kb_corpus()     # load KB at startup

    # ── LLM ──────────────────────────────────────────────────────────────────
    def _init_llm(self):
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                self.llm = ChatGroq(
                    model_name  = "llama3-8b-8192",
                    temperature = 0.1,
                    max_tokens  = 1024,
                )
            except Exception:
                self.llm = None

    # ── Embeddings ────────────────────────────────────────────────────────────
    def _init_embeddings(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name      = "sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs    = {"device": "cpu"},
                encode_kwargs   = {"normalize_embeddings": True},
            )
        except Exception:
            self.embeddings = None

    # ── Load Pre-Built KB Corpus ──────────────────────────────────────────────
    def _load_kb_corpus(self):
        """
        Loads the 30-passage Indian market knowledge base from kb_corpus.py
        and stores as LangChain Document objects.  Called once at startup.
        """
        try:
            from core.kb_corpus import get_corpus_as_documents, get_corpus_metadata
            texts    = get_corpus_as_documents()
            metadata = get_corpus_metadata()
            self._kb_docs = [
                Document(
                    page_content = text,
                    metadata     = {**meta, "source_type": "knowledge_base"},
                )
                for text, meta in zip(texts, metadata)
            ]
            self._kb_loaded = True
        except Exception:
            self._kb_docs  = []
            self._kb_loaded = False

    # ── Build Knowledge Base (called when user loads a stock) ─────────────────
    def build_knowledge_base(self, news_text: str, stock_info: dict,
                             financials: dict) -> bool:
        """
        Merges pre-built KB corpus with live news and stock fundamentals
        into a single FAISS index.
        """
        if not self.embeddings:
            return False

        try:
            # ── Live content ────────────────────────────────────────────────
            live_text = self._build_live_context(news_text, stock_info, financials)
            splitter  = RecursiveCharacterTextSplitter(
                chunk_size    = 500,
                chunk_overlap = 50,
            )
            live_chunks = splitter.split_text(live_text)
            live_docs   = [
                Document(
                    page_content = chunk,
                    metadata     = {"source_type": "live_news",
                                    "ticker": stock_info.get("ticker", "")},
                )
                for chunk in live_chunks
            ]

            # ── Merge KB + live ─────────────────────────────────────────────
            all_docs = self._kb_docs + live_docs

            if not all_docs:
                return False

            self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
            return True

        except Exception:
            return False

    def _build_live_context(self, news_text: str, stock_info: dict,
                            financials: dict) -> str:
        """Assembles the live stock context string."""
        lines = [
            f"Stock: {stock_info.get('name', 'N/A')} ({stock_info.get('ticker', '')})",
            f"Sector: {stock_info.get('sector', 'N/A')}",
            f"Current Price: ₹{stock_info.get('current_price', 'N/A')}",
            f"Market Cap: {stock_info.get('market_cap', 'N/A')}",
            f"P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}",
            f"52-Week High: ₹{stock_info.get('52w_high', 'N/A')}",
            f"52-Week Low: ₹{stock_info.get('52w_low', 'N/A')}",
            f"Change Today: {stock_info.get('change_pct', 0):+.2f}%",
            "",
            "--- Financials ---",
            f"Revenue: {financials.get('revenue', 'N/A')}",
            f"Net Profit: {financials.get('net_income', 'N/A')}",
            f"Debt to Equity: {financials.get('debt_to_equity', 'N/A')}",
            f"ROE: {financials.get('roe', 'N/A')}",
            "",
            "--- Recent News ---",
            news_text or "No recent news available.",
        ]
        return "\n".join(lines)

    # ── Query ─────────────────────────────────────────────────────────────────
    def ask(self, question: str) -> str:
        """
        Answers a question using:
          1. FAISS retrieval (KB + live docs if vectorstore exists)
          2. LLaMA 3 via Groq for generation
        Falls back gracefully if components are unavailable.
        """
        if not self.llm:
            return "⚠️ AI chat requires a GROQ_API_KEY environment variable."

        # ── Retrieve context ─────────────────────────────────────────────────
        context = ""
        if self.vectorstore:
            try:
                docs    = self.vectorstore.similarity_search(question, k=5)
                context = "\n\n".join(d.page_content for d in docs)
            except Exception:
                context = ""

        # ── Build prompt ─────────────────────────────────────────────────────
        if context:
            prompt = (
                "You are InvestIQ, an expert AI research analyst for Indian stock markets. "
                "Answer the question using the context below. "
                "Be specific, factual, and concise (3–5 sentences). "
                "If the context doesn't contain enough information, say so clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
        else:
            prompt = (
                "You are InvestIQ, an expert AI research analyst for Indian stock markets. "
                "Answer the following question based on your knowledge of Indian markets, "
                "SEBI regulations, RBI policy, and financial analysis. "
                "Be specific, factual, and concise.\n\n"
                f"Question: {question}\n\nAnswer:"
            )

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return f"⚠️ AI response error: {str(e)}"

    # ── KB-only query (used for RAG evaluation) ───────────────────────────────
    def ask_kb_only(self, question: str) -> str:
        """
        Answers using ONLY the pre-built KB corpus — no live news.
        Used in rag_eval.py to compare KB-RAG vs plain LLaMA.
        """
        if not self.llm:
            return "NO_LLM"
        if not self.embeddings or not self._kb_docs:
            return "NO_KB"

        try:
            kb_vs = FAISS.from_documents(self._kb_docs, self.embeddings)
            docs  = kb_vs.similarity_search(question, k=4)
            ctx   = "\n\n".join(d.page_content for d in docs)

            prompt = (
                "You are InvestIQ, an expert AI analyst for Indian stock markets. "
                "Answer using ONLY the context below. Be factual and concise.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            resp = self.llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            return f"ERROR: {e}"

    # ── Plain LLaMA (no RAG — baseline for evaluation) ───────────────────────
    def ask_plain_llm(self, question: str) -> str:
        """
        Answers using plain LLaMA 3 with NO retrieval — baseline for comparison.
        Used in rag_eval.py.
        """
        if not self.llm:
            return "NO_LLM"
        try:
            prompt = (
                "You are a financial analyst. Answer the following question about "
                "Indian stock markets concisely and factually.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            resp = self.llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            return f"ERROR: {e}"


# Singleton instance
rag_engine = RAGEngine()
