"""
core/rag_engine.py - Updated for LangChain v0.2+
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

import os
try:
    import streamlit as st
    for k, v in st.secrets.items():
        os.environ.setdefault(k, v)
except Exception:
    pass # local dev uses .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ANALYST_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are InvestIQ, an expert AI financial analyst specializing in Indian stock markets (NSE/BSE).

You have access to the latest news and data about the stock the user is asking about.
Answer the user's question based on the context provided below.

RULES:
- Be specific, factual, and concise
- Use rupee symbol for prices when relevant
- If the context does not have enough info, say so honestly
- Mention key risks if relevant
- End with a 1-line investment outlook when asked about buy/sell/hold
- Never give definitive financial advice, always mention to consult a SEBI-registered advisor

CONTEXT (latest news and data):
{context}

USER QUESTION: {question}

ANALYST RESPONSE:"""
)


class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.current_stock = None

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if GROQ_API_KEY:
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=GROQ_API_KEY,
                temperature=0.3,
                max_tokens=300,
            )
        else:
            self.llm = None

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )

    def build_knowledge_base(self, news_text: str, stock_info: dict, financials: dict) -> bool:
        try:
            fundamentals_text = self._format_fundamentals(stock_info, financials)
            full_context = f"{fundamentals_text}\n\n--- RECENT NEWS ---\n\n{news_text}"
            chunks = self.splitter.split_text(full_context)
            if not chunks:
                return False
            self.vectorstore = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
            self.current_stock = stock_info.get("name", "Unknown")
            return True
        except Exception as e:
            print(f"RAG build error: {e}")
            return False

    def ask(self, question: str) -> str:
        if not self.llm:
            return "Gemini API key not configured. Please add GEMINI_API_KEY to your .env file."
        if not self.vectorstore:
            return "Please select a stock first to build the knowledge base."
    
        import time
        import re
        
        for attempt in range(3):  # try 3 times automatically
            try:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
                docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                context = context[:800]
                chain = ANALYST_PROMPT | self.llm | StrOutputParser()
                result = chain.invoke({"context": context, "question": question})
                return result
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    seconds = re.search(r'(\d+)s\'', err)
                    wait = int(seconds.group(1)) + 2 if seconds else 15
                    time.sleep(wait)  # auto wait and retry
                    continue
                return f"Error: {err}"
    
        return "⏳ Gemini is busy right now. Please wait 1 minute and ask again."

    def _format_fundamentals(self, stock_info: dict, financials: dict) -> str:
        name = stock_info.get("name", "Unknown")
        price = stock_info.get("current_price", "N/A")
        change_pct = stock_info.get("change_pct", 0)
        pe = stock_info.get("pe_ratio", "N/A")
        high_52w = stock_info.get("52w_high", "N/A")
        low_52w = stock_info.get("52w_low", "N/A")
        sector = stock_info.get("sector", "N/A")
        summary = stock_info.get("summary", "")
        profit_margin = financials.get("profit_margin")
        roe = financials.get("return_on_equity")
        debt_equity = financials.get("debt_to_equity")
        rev_growth = financials.get("revenue_growth")

        return f"""STOCK FUNDAMENTALS - {name}
Current Price: {price} ({change_pct:+.2f}% today)
Sector: {sector}
P/E Ratio: {pe}
52-Week High: {high_52w} | 52-Week Low: {low_52w}
Profit Margin: {f'{profit_margin*100:.1f}%' if profit_margin else 'N/A'}
Return on Equity: {f'{roe*100:.1f}%' if roe else 'N/A'}
Debt to Equity: {debt_equity or 'N/A'}
Revenue Growth: {f'{rev_growth*100:.1f}%' if rev_growth else 'N/A'}
Business: {summary[:500]}"""


rag_engine = RAGEngine()
