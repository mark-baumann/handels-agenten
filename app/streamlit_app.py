"""Handels-Agenten — Streamlit Web-UI"""
import os
import queue
import threading
from datetime import date, timedelta

import streamlit as st

from tradingagents.dataflows.google_news import get_global_news_google, get_news_google
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS
from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

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
    provider = st.selectbox("LLM-Provider", list(MODEL_OPTIONS.keys()), format_func=lambda p: {
        "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google Gemini",
        "xai": "xAI", "deepseek": "DeepSeek", "qwen": "Qwen",
        "qwen-cn": "Qwen CN", "glm": "GLM", "glm-cn": "GLM CN",
        "minimax": "MiniMax", "minimax-cn": "MiniMax CN",
        "openrouter": "OpenRouter", "ollama": "Ollama / Ollama Cloud",
        "openai_compatible": "Eigener OpenAI-kompatibler Endpoint",
        "mistral": "Mistral", "kimi": "Kimi", "groq": "Groq",
        "nvidia": "NVIDIA NIM", "bedrock": "AWS Bedrock", "azure": "Azure OpenAI",
    }.get(p, p))
    provider_spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    default_endpoint = provider_spec.base_url if provider_spec else ""
    if provider == "ollama":
        ollama_mode = st.radio(
            "Ollama-Ziel",
            ["Ollama Cloud", "Lokales Ollama"],
            horizontal=True,
            help="Ollama Cloud nutzt https://ollama.com/v1 und einen Ollama API-Key.",
        )
        default_endpoint = (
            "https://ollama.com/v1"
            if ollama_mode == "Ollama Cloud"
            else "http://localhost:11434/v1"
        )
    endpoint = st.text_input(
        "API Endpoint (optional)",
        value=default_endpoint if provider == "ollama" else "",
        placeholder=default_endpoint or "z.B. https://example.com/v1",
        help=(
            "Für Ollama Cloud https://ollama.com/v1 eintragen. "
            "Lokal funktioniert http://localhost:11434/v1."
        ) if provider == "ollama" else "Leer lassen für den Provider-Standard.",
    ).strip()
    api_env = get_api_key_env(provider)
    optional_api_env = "OLLAMA_API_KEY" if provider == "ollama" else api_env
    api_key = st.text_input(
        f"{optional_api_env} (optional)" if optional_api_env else "API-Key",
        value="",
        type="password",
        help=(
            "Für Ollama Cloud erforderlich, lokal leer lassen. Wird nur für diese "
            "Sitzung verwendet."
            if provider == "ollama"
            else "Wird nur für diese Sitzung verwendet."
        ) if optional_api_env else "AWS Bedrock verwendet die AWS Credential Chain.",
    )
    if optional_api_env and api_key:
        os.environ[optional_api_env] = api_key
    model_options = MODEL_OPTIONS[provider]["deep"]
    model_ids = [model_id for _, model_id in model_options]
    deep_model = st.selectbox("Deep-Think-Modell", model_ids)
    if deep_model == "custom":
        deep_model = st.text_input("Eigenes Deep-Think-Modell", value="")
    quick_model = st.selectbox(
        "Quick-Think-Modell", [model_id for _, model_id in MODEL_OPTIONS[provider]["quick"]]
    )
    if quick_model == "custom":
        quick_model = st.text_input("Eigenes Quick-Think-Modell", value="")

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
            config["llm_provider"] = provider
            config["backend_url"] = endpoint or None
            config["deep_think_llm"] = deep_model
            config["quick_think_llm"] = quick_model
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
