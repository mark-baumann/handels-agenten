"""Handels-Agenten — Streamlit Web-UI"""
import inspect
import os
import queue
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load the project .env before configuration imports, independent of the CWD.
# Exported server variables take precedence over the file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.llm_clients.api_key_env import get_api_key_env  # noqa: E402
from tradingagents.llm_clients.model_catalog import get_model_options  # noqa: E402
from tradingagents.portfolio import PortfolioStore, load_portfolio_price_history  # noqa: E402
from tradingagents.reporting import write_report_tree  # noqa: E402

st.set_page_config(page_title="Handels-Agenten", page_icon="📈", layout="wide")

# === Theme & Styling ========================================================

_THEME_LIGHT = {
    "card_border": "#e3e7ef",
    "text": "#1b2437",
    "muted": "#667085",
    "up": "#059669",
    "up_soft": "#e7f7f0",
    "down": "#dc2626",
    "down_soft": "#fdeaea",
    "chip_bg": "#f2f4f9",
}

_THEME_DARK = {
    "card_border": "#313a4d",
    "text": "#e7ebf5",
    "muted": "#98a2b8",
    "up": "#34d399",
    "up_soft": "#12362b",
    "down": "#f87171",
    "down_soft": "#3d1f22",
    "chip_bg": "#262d3f",
}

_CSS = """
<style>
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #6d28d9 55%, #2563eb 100%);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 4px;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.28);
    }
    .hero-title {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        line-height: 1.15;
    }
    .hero-sub {
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.95rem;
        margin-top: 3px;
    }
    .chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 6px 6px 2px 0;
        background: @@chip_bg@@;
        color: @@muted@@;
        border: 1px solid @@card_border@@;
    }
    .chip-ok { background: @@up_soft@@; color: @@up@@; border-color: transparent; }
    .chip-bad { background: @@down_soft@@; color: @@down@@; border-color: transparent; }
    .sym { font-size: 1.1rem; font-weight: 700; color: @@text@@; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        vertical-align: middle;
        margin-left: 6px;
    }
    .badge-up { background: @@up_soft@@; color: @@up@@; }
    .badge-down { background: @@down_soft@@; color: @@down@@; }
    .badge-flat { background: @@chip_bg@@; color: @@muted@@; }
    .stat { display: flex; flex-direction: column; gap: 2px; margin-top: 6px; }
    .stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: @@muted@@;
        font-weight: 600;
    }
    .stat-value { font-size: 1.02rem; font-weight: 650; color: @@text@@; }
    .footer { text-align: center; color: @@muted@@; font-size: 0.78rem; padding: 4px 0 12px; }
</style>
"""


def _active_theme() -> dict:
    base = (st.get_option("theme.base") or "light").lower()
    return _THEME_DARK if base == "dark" else _THEME_LIGHT


def _apply_theme_css() -> None:
    css = _CSS
    for key, value in _active_theme().items():
        css = css.replace(f"@@{key}@@", value)
    st.markdown(css, unsafe_allow_html=True)


def _chip(label: str, kind: str = "neutral") -> str:
    suffix = "" if kind == "neutral" else f" chip-{kind}"
    return f'<span class="chip{suffix}">{label}</span>'


def _hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📈 Handels-Agenten</div>
            <div class="hero-sub">Dein Portfolio und KI-gestützte Aktienanalysen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _return_badge(return_value: float | None) -> str:
    if return_value is None:
        return '<span class="badge badge-flat">—</span>'
    percent = return_value * 100
    kind = "badge-up" if percent >= 0 else "badge-down"
    return f'<span class="badge {kind}">{percent:+.2f}%</span>'


def _decision_tone(decision: str) -> str:
    text = decision.lower()
    if any(word in text for word in ("buy", "long", "overweight")):
        return "buy"
    if any(word in text for word in ("sell", "short", "underweight")):
        return "sell"
    return "hold"


HAS_BORDERED_CONTAINERS = "border" in inspect.signature(st.container).parameters


def _box():
    if HAS_BORDERED_CONTAINERS:
        return st.container(border=True)
    return st.container()


def _stat(label: str, value: str) -> None:
    st.markdown(
        f'<div class="stat"><span class="stat-label">{label}</span>'
        f'<span class="stat-value">{value}</span></div>',
        unsafe_allow_html=True,
    )


_apply_theme_css()

# Provider, model defaults and credentials come from the server environment.
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
api_key_available = os.environ.get(required_api_env or "", "").strip()
api_key_missing = bool(required_api_env) and not api_key_available

_hero()
status_chips = (
    _chip(f"🤖 {llm_provider}")
    + _chip(f"🧠 {deep_model}")
    + _chip(f"⚡ {quick_model}")
)
if required_api_env:
    status_chips += (
        _chip("🔑 API-Key fehlt", "bad")
        if api_key_missing
        else _chip("🔑 API-Key gesetzt", "ok")
    )
else:
    status_chips += _chip("🔑 Kein API-Key nötig", "ok")
st.markdown(status_chips, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Einstellungen")
    st.divider()
    st.markdown("#### 🤖 LLM (aus .env)")
    st.markdown(
        f"**Provider:** `{llm_provider}`  \n"
        f"**Endpoint:** `{endpoint or 'Provider-Standard'}`"
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
        st.markdown(
            f"**Deep-Think:** `{deep_model}`  \n**Quick-Think:** `{quick_model}`"
        )

    if api_key_missing:
        st.error(
            f"API-Key fehlt: `{required_api_env}`. "
            "Bitte serverseitig in `.env` setzen und die Anwendung neu starten."
        )
    elif required_api_env:
        st.success(f"🔑 `{required_api_env}` gesetzt")

    st.divider()
    st.caption("Konfiguration wird zentral über die Server-`.env` bereitgestellt.")

store = _portfolio_store()
portfolio_tab, analysis_tab, reports_tab = st.tabs(
    ["📊 Portfolio", "🔎 Analyse", "🗂️ Berichte"]
)

with portfolio_tab:
    st.subheader("Dein Portfolio")
    st.caption("Bestände und Analysen werden dauerhaft lokal in SQLite gespeichert.")

    with st.form("add_holding_form", clear_on_submit=True):
        st.markdown("**Position hinzufügen**")
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
        with st.spinner("Lade Kursverläufe …"):
            price_history, latest_prices, price_errors = load_portfolio_price_history(holdings)

        valued_symbols = [h["symbol"] for h in holdings if h["symbol"] in latest_prices]
        total_value = sum(
            holding["shares"] * latest_prices[holding["symbol"]]
            for holding in holdings
            if holding["symbol"] in latest_prices
        )
        returns = [
            latest_prices[holding["symbol"]] / holding["average_cost"] - 1
            for holding in holdings
            if holding["symbol"] in latest_prices and holding["average_cost"]
        ]
        average_return = sum(returns) / len(returns) if returns else None

        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Positionen", len(holdings))
        kpi_col2.metric(
            "Gesamtwert",
            f"{total_value:,.2f}" if valued_symbols else "—",
        )
        kpi_col3.metric(
            "Ø Rendite",
            f"{average_return * 100:+.2f}%" if average_return is not None else "—",
        )

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
                latest / holding["average_cost"] - 1
                if latest is not None and holding["average_cost"]
                else None
            )
            current_price = f"{latest:,.2f}" if latest is not None else "—"
            cost_text = (
                f"{holding['average_cost']:,.2f}" if holding["average_cost"] else "—"
            )

            with _box():
                head_col, action_col = st.columns([3.2, 1.0])
                with head_col:
                    st.markdown(
                        f'<span class="sym">{symbol}</span>{_return_badge(current_return)}',
                        unsafe_allow_html=True,
                    )
                    stat_cols = st.columns(3)
                    with stat_cols[0]:
                        _stat("Stückzahl", f"{holding['shares']:,.6g}")
                    with stat_cols[1]:
                        _stat("Kauf", cost_text)
                    with stat_cols[2]:
                        _stat("Kurs", current_price)
                with action_col:
                    if st.button(
                        "🔎 Analysieren",
                        key=f"analyze_{symbol}",
                        disabled=api_key_missing,
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state["analysis_ticker"] = symbol
                        st.session_state["run_portfolio_analysis"] = True
                        st.rerun()
                    if st.button(
                        "🗑️ Entfernen",
                        key=f"remove_{symbol}",
                        use_container_width=True,
                    ):
                        store.remove_holding(symbol)
                        st.rerun()

with analysis_tab:
    st.subheader("KI-Analyse starten")
    st.caption(
        "Multi-Agenten-Analyse mit Analysten, Research-Debatte, Trader und Risikomanagement."
    )

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
                config = DEFAULT_CONFIG.copy()
                config["llm_provider"] = llm_provider
                config["backend_url"] = endpoint or None
                config["deep_think_llm"] = deep_model
                config["quick_think_llm"] = quick_model
                ta = TradingAgentsGraph(debug=False, config=config)
                final_state, decision = ta.propagate(ticker, str(analysis_date))
                result_q.put(("ok", decision, final_state))
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
                    tone = _decision_tone(str(decision))
                    if tone == "buy":
                        st.success(f"**Handelsentscheidung**  \n{decision}")
                    elif tone == "sell":
                        st.error(f"**Handelsentscheidung**  \n{decision}")
                    else:
                        st.info(f"**Handelsentscheidung**  \n{decision}")
                    with _box():
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
            tone = _decision_tone(full_report["decision"])
            if tone == "buy":
                st.success(f"**Entscheidung:** {full_report['decision']}")
            elif tone == "sell":
                st.error(f"**Entscheidung:** {full_report['decision']}")
            else:
                st.info(f"**Entscheidung:** {full_report['decision']}")

            download_col, _ = st.columns([1, 3])
            with download_col:
                st.download_button(
                    "⬇️ Markdown herunterladen",
                    full_report["report_markdown"],
                    file_name=(
                        f"{full_report['symbol']}_{full_report['analysis_date']}"
                        "_bericht.md"
                    ),
                    mime="text/markdown",
                    use_container_width=True,
                )
            with _box():
                st.markdown(full_report["report_markdown"])

st.divider()
st.markdown(
    '<div class="footer">⚠️ Nur zu Forschungszwecken — keine Anlageberatung. '
    "Kursdaten: Yahoo Finance.</div>",
    unsafe_allow_html=True,
)
