"""
core/watchlist.py
Watchlist management using Streamlit session state.
"""
from core.stock_data import get_stock_info, POPULAR_STOCKS


def init_watchlist():
    import streamlit as st
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []


def add_to_watchlist(ticker: str):
    import streamlit as st
    if ticker and ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)


def remove_from_watchlist(ticker: str):
    import streamlit as st
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)


def get_watchlist_data() -> list[dict]:
    import streamlit as st
    results = []
    for ticker in st.session_state.watchlist:
        info = get_stock_info(ticker)
        if "error" not in info:
            results.append(info)
    return results
