"""Handels-Agenten — Streamlit Web-UI"""
import os
import queue
import threading
from contextlib import nullcontext
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Lade .env Config (Ollama Cloud, etc.)
load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.llm_clients.api_key_env import get_api_key_env  # noqa: E402
from tradingagents.llm_clients.model_catalog import get_model_options  # noqa: E402
from tradingagents.portfolio import PortfolioStore, load_portfolio_price_history  # noqa: E402
from tradingagents.reporting import write_report_tree  # noqa: E402

st.set_page_config(page_title="Handels-Agenten", page_icon="📈", layout="wide")
st.title("Handels-Agenten")
st.caption("Dein Portfolio und KI-gestützte Aktienanalysen")

# The OpenAI SDK discovers credentials through the process environment. Since a
# Streamlit process can serve multiple users, serialize temporary UI-key use so
# one session's key cannot be used by another concurrent analysis.
_openai_api_key_lock = threading.Lock()


# Provider und Modelle werden über die Umgebung (.env) konfiguriert. Der
# OpenAI-Schlüssel kann für die laufende Browser-Sitzung auch direkt in der UI
# hinterlegt werden; er wird dabei nicht in eine Datei geschrieben.
def _save_analysis_report(final_state: dict, ticker: str, analysis_date: date) -> Path:
    """Persistiert jeden UI-Lauf in einem datierten Report-Ordner."""
    ticker_safe = ticker.strip().upper().replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (
        Path(DEFAULT_CONFIG["results_dir"])
        / "streamlit_sessions"
        / ticker_safe
        / str(analysis_date)
        / stamp
    )
    return write_report_tree(final_state, ticker_safe, report_dir).parent


def _portfolio_store() -> PortfolioStore:
    return PortfolioStore(Path(DEFAULT_CONFIG["data_cache_dir"]) / "portfolio.sqlite3")


llm_provider = DEFAULT_CONFIG["llm_provider"]
deep_model = DEFAULT_CONFIG["deep_think_llm"]
quick_model = DEFAULT_CONFIG["quick_think_llm"]
endpoint = DEFAULT_CONFIG.get("backend_url")


def api_key_required(provider: str) -> str | None:
    """Env-var name eines Pflicht-Keys, oder None wenn kein Key nötig ist."""
    if provider in ("ollama", "bedrock", "openai_compatible"):
        return None
    return get_api_key_env(provider)


required_api_env = api_key_required(llm_provider)
openai_api_key = ""

with st.sidebar:
    st.header("Einstellungen")
    st.divider()
    st.subheader("🤖 LLM (aus .env)")
    st.markdown(
        f"- **Provider:** `{llm_provider}`\n"
        f"- **Endpoint:** `{endpoint or 'Provider-Standard'}`"
    )

    if llm_provider == "openai":
        available_models = [model_id for _, model_id in get_model_options("openai", "deep")]
        default_deep_index = available_models.index(deep_model) if deep_model in available_models else 0
        default_quick_index = available_models.index(quick_model) if quick_model in available_models else 0
        st.caption("Modell für die Analyse auswählen (vollständiger OpenAI-Modellkatalog).")
        deep_model = st.selectbox(
            "Deep-Think Modell", available_models, index=default_deep_index, key="deep_model_select"
        )
        quick_model = st.selectbox(
            "Quick-Think Modell", available_models, index=default_quick_index, key="quick_model_select"
        )
        st.info(
            "TradingAgents benötigt Chat-Modelle. Bild-, Audio-, Embedding-, "
            "Moderations- und TTS-Modelle sind auswählbar, können aber keine Analyse ausführen."
        )
    else:
        st.markdown(f"- **Deep-Think:** `{deep_model}`\n- **Quick-Think:** `{quick_model}`")

    if llm_provider == "openai":
        st.divider()
        st.subheader("🔑 ChatGPT / OpenAI API")
        openai_api_key = st.text_input(
            "OpenAI API-Key",
            type="password",
            key="openai_api_key_input",
            help="Wird nur für diese laufende Server-Sitzung verwendet und nicht gespeichert.",
        ).strip()

    api_key_available = openai_api_key or os.environ.get(required_api_env or "")
    api_key_missing = bool(required_api_env) and not api_key_available
    if api_key_missing:
        if llm_provider == "openai":
            st.error("Bitte gib oben einen OpenAI API-Key ein oder setze `OPENAI_API_KEY` in `.env`.")
        else:
            st.error(f"API-Key fehlt: `{required_api_env}`. Bitte in der `.env`-Datei setzen.")
    elif required_api_env:
        st.caption(f"🔑 `{required_api_env}` gesetzt")

store = _portfolio_store()
portfolio_tab, analysis_tab, reports_tab = st.tabs(
    ["📊 Portfolio", "🔎 Analyse", "🗂️ Berichte"]
)

with portfolio_tab:
    st.subheader("Dein Portfolio")
    st.caption("Bestände und Analysen werden dauerhaft lokal in SQLite gespeichert.")

    with st.form("add_holding_form", clear_on_submit=True):
        ticker_col, shares_col, cost_col, submit_col = st.columns([2, 1, 1, 1])
        with ticker_col:
            holding_symbol = st.text_input("Ticker", placeholder="z. B. AAPL oder SAP.DE")
        with shares_col:
            holding_shares = st.number_input(
                "Stückzahl", min_value=0.000001, value=1.0, step=1.0, format="%.6f"
            )
        with cost_col:
            average_cost = st.number_input(
                "Kaufpreis je Aktie (optional)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.4f",
                help="In der Notierung des jeweiligen Wertpapiers; 0 = nicht hinterlegt.",
            )
        with submit_col:
            st.write("")
            add_holding = st.form_submit_button("Hinzufügen", type="primary")
        if add_holding:
            try:
                store.add_holding(
                    holding_symbol,
                    holding_shares,
                    average_cost if average_cost > 0 else None,
                )
                st.success(f"{holding_symbol.strip().upper()} wurde gespeichert.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    holdings = store.list_holdings()
    if not holdings:
        st.info("Noch keine Positionen. Füge oben ein Wertpapier hinzu.")
    else:
        st.metric("Positionen", len(holdings))
        with st.spinner("Lade Kursverläufe …"):
            price_history, latest_prices, price_errors = load_portfolio_price_history(holdings)

        if not price_history.empty:
            st.markdown("**Kursentwicklung je Position**")
            st.caption("Auf 100 normiert, um unterschiedliche Aktienkurse vergleichbar zu machen.")
            st.line_chart(price_history)
        for symbol, error in price_errors.items():
            st.warning(f"{symbol}: Kursdaten konnten nicht geladen werden ({error})")

        st.markdown("**Bestände**")
        for holding in holdings:
            symbol = holding["symbol"]
            latest = latest_prices.get(symbol)
            current_return = (
                f"{(latest / holding['average_cost'] - 1) * 100:+.2f}%"
                if latest is not None and holding["average_cost"]
                else "—"
            )
            current_price = f"{latest:,.2f}" if latest is not None else "—"
            col_symbol, col_shares, col_cost, col_price, col_return, col_action = st.columns(
                [1.2, 1, 1.2, 1, 1, 1.5]
            )
            col_symbol.write(f"**{symbol}**")
            col_shares.write(f"{holding['shares']:,.6g} Stk.")
            col_cost.write(
                f"Kauf: {holding['average_cost']:,.2f}"
                if holding["average_cost"]
                else "Kauf: —"
            )
            col_price.write(f"Kurs: {current_price}")
            col_return.write(current_return)
            with col_action:
                if st.button("Analysieren", key=f"analyze_{symbol}", disabled=api_key_missing):
                    st.session_state["analysis_ticker"] = symbol
                    st.session_state["run_portfolio_analysis"] = True
                    st.rerun()
            remove_col, _ = st.columns([1.5, 8.5])
            with remove_col:
                if st.button("Entfernen", key=f"remove_{symbol}"):
                    store.remove_holding(symbol)
                    st.rerun()

with analysis_tab:
    holdings = store.list_holdings()
    held_symbols = [holding["symbol"] for holding in holdings]
    default_symbol = st.session_state.get("analysis_ticker", held_symbols[0] if held_symbols else "NVDA")
    if held_symbols:
        ticker = st.selectbox(
            "Portfolio-Aktie", held_symbols,
            index=held_symbols.index(default_symbol) if default_symbol in held_symbols else 0,
        )
    else:
        ticker = st.text_input("Ticker-Symbol", value=default_symbol, help="z. B. NVDA, AAPL, TSLA")
    analysis_date = st.date_input(
        "Analyse-Datum",
        value=date.today() - timedelta(days=1),
        max_value=date.today() - timedelta(days=1),
        key="analysis_date",
    )

    requested_analysis = st.button(
        "Analyse starten",
        type="primary",
        disabled=not ticker.strip() or api_key_missing,
        key="start_analysis",
    )
    run_portfolio_analysis = st.session_state.pop("run_portfolio_analysis", False)
    if requested_analysis or run_portfolio_analysis:
        ticker = ticker.strip().upper()
        result_q: queue.Queue = queue.Queue()

        def run_analysis():
            try:
                key_lock = _openai_api_key_lock if llm_provider == "openai" else nullcontext()
                with key_lock:
                    previous_openai_api_key = os.environ.get("OPENAI_API_KEY")
                    try:
                        if openai_api_key:
                            os.environ["OPENAI_API_KEY"] = openai_api_key
                        config = DEFAULT_CONFIG.copy()
                        config["llm_provider"] = llm_provider
                        config["backend_url"] = endpoint or None
                        config["deep_think_llm"] = deep_model
                        config["quick_think_llm"] = quick_model
                        ta = TradingAgentsGraph(debug=False, config=config)
                        final_state, decision = ta.propagate(ticker, str(analysis_date))
                        result_q.put(("ok", decision, final_state))
                    finally:
                        if openai_api_key:
                            if previous_openai_api_key is None:
                                os.environ.pop("OPENAI_API_KEY", None)
                            else:
                                os.environ["OPENAI_API_KEY"] = previous_openai_api_key
            except Exception as exc:
                result_q.put(("error", str(exc)))

        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()
        with st.spinner(f"Analysiere {ticker} für {analysis_date} …"):
            thread.join(timeout=300)

        if not result_q.empty():
            kind, *payload = result_q.get()
            if kind == "ok":
                decision, final_state = payload
                try:
                    report_dir = _save_analysis_report(final_state, ticker, analysis_date)
                    report_markdown = (report_dir / "complete_report.md").read_text(encoding="utf-8")
                    report_id = store.save_report(
                        ticker, str(analysis_date), str(decision), report_markdown
                    )
                except Exception as exc:
                    st.error(f"Analyse abgeschlossen, der Bericht konnte nicht gespeichert werden: {exc}")
                else:
                    st.success(f"Analyse abgeschlossen und dauerhaft gespeichert (Bericht #{report_id}).")
                    st.subheader("Handelsentscheidung")
                    st.write(decision)
                    st.markdown(report_markdown)
            else:
                st.error(f"Fehler: {payload[0]}")
        else:
            st.error("Zeitüberschreitung (5 min). Bitte erneut versuchen.")

with reports_tab:
    st.subheader("Gespeicherte Analyseberichte")
    reports = store.list_reports()
    if not reports:
        st.info("Nach der ersten Analyse erscheinen die Berichte hier dauerhaft.")
    else:
        selected_report = st.selectbox(
            "Bericht auswählen",
            reports,
            format_func=lambda report: (
                f"#{report['id']} · {report['symbol']} · {report['analysis_date']} "
                f"· {report['created_at'][:19].replace('T', ' ')}"
            ),
        )
        full_report = store.get_report(selected_report["id"])
        if full_report is None:
            st.error("Der ausgewählte Bericht ist nicht mehr verfügbar.")
        else:
            st.markdown(f"**Entscheidung:** {full_report['decision']}")
            st.markdown(full_report["report_markdown"])
