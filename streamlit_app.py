"""Browser UI for running a TradingAgents analysis.

Start it from the repository root with ``streamlit run streamlit_app.py``.
API keys are deliberately read from the environment/.env file and are never
entered into or stored by this interface.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from cli.utils import provider_default_url
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

ANALYST_OPTIONS = {
    "Markt & Technik": "market",
    "Social Sentiment": "social",
    "Nachrichten & Makro": "news",
    "Fundamentaldaten": "fundamentals",
}


def _model_picker(label: str, provider: str, mode: str, default: str) -> str:
    """Render a catalog picker and return either a catalog or custom model ID."""
    options = MODEL_OPTIONS[provider][mode]
    values = [value for _, value in options]
    selected = st.selectbox(
        label,
        values,
        index=values.index(default) if default in values else 0,
        format_func=lambda value: next(
            (display for display, model in options if model == value), value
        ),
    )
    if selected == "custom":
        return st.text_input(f"Eigene Modell-ID für {label}", placeholder="model-id").strip()
    return selected


def _render_report(state: dict) -> None:
    """Show all available agent reports without assuming every analyst ran."""
    st.subheader("Ergebnis")
    st.success(f"Handelssignal: **{st.session_state.decision}**")

    reports = [
        ("Marktanalyse", state.get("market_report")),
        ("Social Sentiment", state.get("sentiment_report")),
        ("Nachrichtenanalyse", state.get("news_report")),
        ("Fundamentalanalyse", state.get("fundamentals_report")),
        ("Research-Team", state.get("investment_plan")),
        ("Trading-Plan", state.get("trader_investment_plan")),
        ("Finale Portfolioentscheidung", state.get("final_trade_decision")),
    ]
    present_reports = [(title, report) for title, report in reports if report]
    tabs = st.tabs([title for title, _ in present_reports])
    for tab, (_title, report) in zip(tabs, present_reports, strict=True):
        with tab:
            st.markdown(report)

    st.download_button(
        "Finale Entscheidung herunterladen",
        data=state.get("final_trade_decision", ""),
        file_name="tradingagents_entscheidung.md",
        mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(page_title="TradingAgents", page_icon="📈", layout="wide")
    st.title("📈 TradingAgents")
    st.caption("Multi-Agent-Analyse für Forschungszwecke – keine Anlageberatung.")

    with st.sidebar:
        st.header("Konfiguration")
        provider = st.selectbox(
            "LLM-Anbieter",
            list(MODEL_OPTIONS),
            index=list(MODEL_OPTIONS).index(DEFAULT_CONFIG["llm_provider"])
            if DEFAULT_CONFIG["llm_provider"] in MODEL_OPTIONS
            else 0,
        )
        st.caption("API-Schlüssel werden aus `.env` bzw. Umgebungsvariablen gelesen.")

    with st.form("analysis_form"):
        left, right = st.columns(2)
        with left:
            ticker = st.text_input("Ticker", value="NVDA", help="Zum Beispiel AAPL, 0700.HK oder BTC-USD.")
            trade_date = st.date_input("Analysedatum", value=date.today(), max_value=date.today())
            asset_type = st.selectbox(
                "Asset-Typ",
                ["stock", "crypto"],
                format_func=lambda value: "Aktie" if value == "stock" else "Krypto",
            )
        with right:
            selected_labels = st.multiselect(
                "Analyse-Teams",
                list(ANALYST_OPTIONS),
                default=list(ANALYST_OPTIONS),
            )
            output_language = st.selectbox("Sprache der Berichte", ["Deutsch", "English"], index=0)
            checkpoint_enabled = st.checkbox("Unterbrochene Analyse fortsetzen", value=False)

        with st.expander("Erweiterte Einstellungen"):
            deep_model = _model_picker("Modell für ausführliches Denken", provider, "deep", DEFAULT_CONFIG["deep_think_llm"])
            quick_model = _model_picker("Modell für schnelle Aufgaben", provider, "quick", DEFAULT_CONFIG["quick_think_llm"])
            backend_url = st.text_input("Backend-URL (optional)", value=provider_default_url(provider) or "")
            max_debate_rounds = st.slider("Research-Debatten", 1, 5, DEFAULT_CONFIG["max_debate_rounds"])
            max_risk_rounds = st.slider("Risiko-Debatten", 1, 5, DEFAULT_CONFIG["max_risk_discuss_rounds"])

        submitted = st.form_submit_button("Analyse starten", type="primary", use_container_width=True)

    if submitted:
        ticker = ticker.strip().upper()
        selected_analysts = [ANALYST_OPTIONS[label] for label in selected_labels]
        if not ticker:
            st.error("Bitte gib einen Ticker ein.")
        elif not selected_analysts:
            st.error("Bitte wähle mindestens ein Analyse-Team aus.")
        elif not deep_model or not quick_model:
            st.error("Bitte gib für beide Modellrollen eine Modell-ID ein.")
        else:
            config = DEFAULT_CONFIG.copy()
            config.update(
                {
                    "llm_provider": provider,
                    "deep_think_llm": deep_model,
                    "quick_think_llm": quick_model,
                    "backend_url": backend_url.strip() or None,
                    "output_language": output_language,
                    "max_debate_rounds": max_debate_rounds,
                    "max_risk_discuss_rounds": max_risk_rounds,
                    "checkpoint_enabled": checkpoint_enabled,
                }
            )
            try:
                with st.spinner("Die Agenten sammeln Daten, diskutieren und erstellen eine Entscheidung …"):
                    graph = TradingAgentsGraph(selected_analysts=selected_analysts, config=config)
                    state, decision = graph.propagate(ticker, trade_date.isoformat(), asset_type=asset_type)
                    graph.save_reports(state, ticker)
                st.session_state.state = state
                st.session_state.decision = decision
            except Exception as exc:
                st.exception(exc)

    if "state" in st.session_state:
        _render_report(st.session_state.state)


if __name__ == "__main__":
    main()
