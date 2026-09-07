"""Handels-Agenten — Streamlit Web-UI"""
import queue
import threading
from datetime import date, timedelta

import streamlit as st

from tradingagents.dataflows.google_news import get_global_news_google, get_news_google
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

st.set_page_config(page_title="Handels-Agenten", page_icon="📈", layout="wide")
st.title("Handels-Agenten")
st.caption("Multi-Agent LLM Handelsanalyse")

with st.sidebar:
    st.header("Einstellungen")
    ticker = st.text_input("Ticker-Symbol", value="NVDA", help="z.B. NVDA, AAPL, TSLA")
    analysis_date = st.date_input(
        "Analyse-Datum",
        value=date.today() - timedelta(days=1),
        max_value=date.today() - timedelta(days=1),
    )
    st.divider()
    st.caption("LLM-Anbieter und API-Keys werden per Umgebungsvariablen konfiguriert.")

st.markdown(f"**Ticker:** `{ticker}` &nbsp;|&nbsp; **Datum:** `{analysis_date}`")

# === Aktuelle Nachrichten (keyless Google News Scraper) ===
st.subheader("📰 Aktuelle Nachrichten")
news_col1, news_col2 = st.columns(2)
with news_col1:
    if st.button("🔍 Ticker-Nachrichten laden", disabled=not ticker.strip()):
        with st.spinner(f"Scrape aktuelle Nachrichten für {ticker.strip().upper()} ..."):
            news = get_news_google(
                ticker.strip().upper(),
                str(analysis_date - timedelta(days=7)),
                str(analysis_date),
            )
        st.session_state["ticker_news"] = news
with news_col2:
    if st.button("🌍 Globale Nachrichten laden"):
        with st.spinner("Scrape globale Marktnachrichten ..."):
            news = get_global_news_google(str(analysis_date), look_back_days=7)
        st.session_state["global_news"] = news

if st.session_state.get("ticker_news"):
    with st.expander("🔍 Ticker-Nachrichten", expanded=True):
        st.markdown(st.session_state["ticker_news"])
if st.session_state.get("global_news"):
    with st.expander("🌍 Globale Marktnachrichten", expanded=False):
        st.markdown(st.session_state["global_news"])

st.divider()

if st.button("Analyse starten", type="primary", disabled=not ticker.strip()):
    result_q: queue.Queue = queue.Queue()

    def run_analysis():
        try:
            config = DEFAULT_CONFIG.copy()
            ta = TradingAgentsGraph(debug=False, config=config)
            final_state, decision = ta.propagate(ticker.strip().upper(), str(analysis_date))
            result_q.put(("ok", decision, final_state))
        except Exception as exc:
            result_q.put(("error", str(exc)))

    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    with st.spinner(f"Analysiere {ticker.upper()} für {analysis_date} ..."):
        thread.join(timeout=300)

    if not result_q.empty():
        kind, *payload = result_q.get()
        if kind == "ok":
            decision, final_state = payload
            st.success("Analyse abgeschlossen")
            st.subheader("Handelsentscheidung")
            st.write(decision)

            news_report = final_state.get("news_report", "")
            if news_report:
                with st.expander("📰 News Analyst Report", expanded=False):
                    st.markdown(news_report)
        else:
            st.error(f"Fehler: {payload[0]}")
    else:
        st.error("Zeitüberschreitung (5 min). Bitte erneut versuchen.")
