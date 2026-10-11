"""Handels-Agenten — Streamlit Web-UI"""
import inspect
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load the project .env before configuration imports, independent of the CWD.
# Exported server variables take precedence over the file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from tradingagents.daily_reports import (  # noqa: E402
    DailyReportScheduler,
    missing_api_key,
    previous_trading_day,
    reports_enabled,
    schedule_time,
    schedule_zone,
)
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.duel import (  # noqa: E402
    PLAYERS,
    DuelStore,
    EurQuotes,
    daily_duel_job,
    duel_enabled,
    max_analyses,
    max_picks,
    run_ai_turn,
)
from tradingagents.live_progress import AnalysisJob, stage_progress  # noqa: E402
from tradingagents.llm_clients.api_key_env import get_api_key_env  # noqa: E402
from tradingagents.news_impact import news_with_impact  # noqa: E402
from tradingagents.portfolio import PortfolioStore, load_portfolio_price_history  # noqa: E402
from tradingagents.ui_settings import (  # noqa: E402
    default_model,
    displayed_endpoint,
    load_llm_settings,
    model_choices,
    provider_choices,
    resolve_endpoint,
    run_config,
    save_llm_settings,
    selected_provider,
    stored_model,
)

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


def _portfolio_store() -> PortfolioStore:
    return PortfolioStore(Path(DEFAULT_CONFIG["data_cache_dir"]) / "portfolio.sqlite3")


# Credentials and endpoints come from the server environment; provider and
# models are chosen in the sidebar and persisted next to the portfolio data.
_LLM_SETTINGS_PATH = Path(DEFAULT_CONFIG["data_cache_dir"]) / "llm_settings.json"
_CUSTOM_MODEL = "Eigene Modell-ID …"


def _saved_run_config() -> dict:
    """Config from the persisted sidebar selection (used by the scheduler)."""
    return run_config(load_llm_settings(_LLM_SETTINGS_PATH), DEFAULT_CONFIG)


@st.cache_resource
def _daily_scheduler() -> DailyReportScheduler | None:
    """One scheduler thread per server process, shared by all sessions."""
    if not reports_enabled():
        return None
    jobs = []
    if duel_enabled():
        jobs.append(daily_duel_job(_portfolio_store(), _duel_store(), _saved_run_config))
    return DailyReportScheduler(_portfolio_store(), _saved_run_config, extra_jobs=jobs).start()


def _duel_store() -> DuelStore:
    return DuelStore(Path(DEFAULT_CONFIG["data_cache_dir"]) / "portfolio.sqlite3")


@st.cache_resource
def _duel_runner() -> dict:
    """Process-wide handle for a manually started AI turn."""
    return {"thread": None}


@st.cache_data(ttl=300, show_spinner=False)
def _eur_price(symbol: str) -> float:
    return EurQuotes()(symbol)


def _start_ai_turn(turn_date: date) -> bool:
    runner = _duel_runner()
    thread = runner["thread"]
    if thread is not None and thread.is_alive():
        return False

    def play():
        run_ai_turn(_portfolio_store(), _duel_store(), _saved_run_config, turn_date)

    runner["thread"] = threading.Thread(target=play, name="duel-turn", daemon=True)
    runner["thread"].start()
    return True


@st.cache_data(ttl=600, show_spinner=False)
def _price_history(symbols: tuple[str, ...]):
    return load_portfolio_price_history([{"symbol": symbol} for symbol in symbols])


@st.cache_data(ttl=900, show_spinner=False)
def _news(symbol: str, provider: str, model: str, rate: bool, _prices, _config):
    store = _portfolio_store()
    return news_with_impact(store, symbol, _config if rate else None, _prices)


def _model_select(label: str, provider: str, mode: str, settings: dict) -> str:
    """Modellauswahl aus dem Katalog des Providers oder als freie Modell-ID."""
    current = stored_model(settings, provider, mode) or default_model(provider, mode, DEFAULT_CONFIG)
    options = model_choices(provider, mode, current, default_model(provider, mode, DEFAULT_CONFIG))
    choice = st.selectbox(
        label,
        [*options, _CUSTOM_MODEL],
        index=options.index(current) if current in options else 0,
        key=f"{mode}_model_select_{provider}",
    )
    if choice != _CUSTOM_MODEL:
        return choice
    return st.text_input(
        f"{label} (Modell-ID)",
        key=f"{mode}_model_custom_{provider}",
        placeholder="z. B. gpt-oss:120b",
    ).strip()


def _decision_box(label: str, decision: str) -> None:
    tone = _decision_tone(decision)
    text = f"**{label}**  \n{decision}"
    if tone == "buy":
        st.success(text)
    elif tone == "sell":
        st.error(text)
    else:
        st.info(text)


_DECISION_ICON = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}
_DIRECTION_LABEL = {"up": "↑ steigt", "down": "↓ fällt", "neutral": "→ neutral"}


def _render_event(event) -> None:
    if event.kind == "tool":
        st.markdown(f"{event.title}  \n`{event.body}`" if event.body else event.title)
    elif event.kind == "data":
        with st.expander(event.title, expanded=False):
            st.text(event.body)
    elif event.kind == "report":
        with st.expander(event.title, expanded=event.title.startswith("✅")):
            st.markdown(event.body)
    else:
        with st.chat_message("assistant", avatar=event.title.split(" ", 1)[0]):
            st.markdown(f"**{event.title.split(' ', 1)[-1]}**")
            st.markdown(event.body)


def _render_job(job: AnalysisJob) -> None:
    """Show a running or finished analysis; streams while the agents work."""
    st.markdown(f"#### {job.symbol} · {job.analysis_date}")
    progress = st.progress(0.0)
    status_line = st.empty()
    feed = st.container()
    shown = 0
    while True:
        running = job.running
        events, state = job.snapshot()
        fraction, upcoming = stage_progress(state)
        progress.progress(fraction if running else 1.0)
        if running:
            status_line.caption(
                f"⏳ Läuft seit {int(job.elapsed())} s · nächster Schritt: {upcoming} · "
                f"{len(events)} Ereignisse"
            )
        with feed:
            for event in events[shown:]:
                _render_event(event)
        shown = len(events)
        if not running:
            break
        time.sleep(0.7)

    if job.error:
        status_line.empty()
        st.error(f"Analyse fehlgeschlagen: {job.error}")
        return
    report_id, decision, report_markdown = job.result
    status_line.caption(f"✅ Fertig nach {int(job.elapsed())} s · Bericht #{report_id} gespeichert")
    _decision_box("Handelsentscheidung", decision)
    with st.expander("📄 Vollständiger Bericht", expanded=False):
        st.markdown(report_markdown)


llm_settings = load_llm_settings(_LLM_SETTINGS_PATH)
providers = provider_choices(DEFAULT_CONFIG)
default_provider = selected_provider(llm_settings, DEFAULT_CONFIG)

with st.sidebar:
    st.markdown("### ⚙️ Einstellungen")
    st.divider()
    st.markdown("#### 🤖 LLM")
    llm_provider = st.selectbox(
        "Provider",
        list(providers),
        index=list(providers).index(default_provider),
        format_func=providers.get,
        key="llm_provider_select",
    )
    deep_model = _model_select("Deep-Think Modell", llm_provider, "deep", llm_settings)
    quick_model = _model_select("Quick-Think Modell", llm_provider, "quick", llm_settings)
    if llm_provider == "ollama_cloud":
        st.caption(
            "gpt-oss:120b läuft im kostenlosen Ollama-Plan (mit Nutzungslimits). "
            "Key: https://ollama.com/settings/keys → `OLLAMA_API_KEY`."
        )
    elif llm_provider == "openai":
        st.caption(
            "TradingAgents benötigt Chat-Modelle. Bild-, Audio-, Embedding-, "
            "Moderations- und TTS-Modelle können keine Analyse ausführen."
        )
    endpoint = resolve_endpoint(llm_provider, DEFAULT_CONFIG)
    st.caption(f"Endpoint: `{displayed_endpoint(llm_provider, DEFAULT_CONFIG)}`")

    required_api_env = get_api_key_env(llm_provider)
    missing_key_env = missing_api_key(llm_provider)
    api_key_missing = bool(missing_key_env)
    model_missing = not deep_model or not quick_model

    if api_key_missing:
        st.error(
            f"API-Key fehlt: `{missing_key_env}`. "
            "Bitte serverseitig in `.env` setzen und die Anwendung neu starten."
        )
    elif required_api_env:
        st.success(f"🔑 `{required_api_env}` gesetzt")
    if model_missing:
        st.warning("Bitte für beide Modelle eine Modell-ID angeben.")

    st.divider()
    st.markdown("#### 🕖 Tägliche Berichte")
    scheduler = _daily_scheduler()
    if scheduler is None:
        st.caption("Deaktiviert (`TRADINGAGENTS_DAILY_REPORTS=false`).")
    else:
        st.caption(
            f"Automatisch für alle Portfolio-Aktien, täglich ab "
            f"{schedule_time().strftime('%H:%M')} ({schedule_zone().key}). "
            f"Nächster Start: {scheduler.next_run().strftime('%d.%m. %H:%M')}"
        )
    st.divider()
    st.caption("API-Keys und Endpoints werden zentral über die Server-`.env` bereitgestellt.")

if not model_missing:
    selected_settings = {
        "provider": llm_provider,
        "models": {
            **(llm_settings.get("models") if isinstance(llm_settings.get("models"), dict) else {}),
            llm_provider: {"deep": deep_model, "quick": quick_model},
        },
    }
    if selected_settings != llm_settings:
        try:
            save_llm_settings(_LLM_SETTINGS_PATH, selected_settings)
        except OSError as exc:
            st.sidebar.warning(f"LLM-Auswahl konnte nicht gespeichert werden: {exc}")

ui_config = {
    **DEFAULT_CONFIG,
    "llm_provider": llm_provider,
    "backend_url": endpoint or None,
    "deep_think_llm": deep_model,
    "quick_think_llm": quick_model,
}
llm_ready = not api_key_missing and not model_missing

_hero()
status_chips = (
    _chip(f"🤖 {providers[llm_provider]}")
    + _chip(f"🧠 {deep_model or '–'}")
    + _chip(f"⚡ {quick_model or '–'}")
)
if required_api_env:
    status_chips += (
        _chip("🔑 API-Key fehlt", "bad")
        if api_key_missing
        else _chip("🔑 API-Key gesetzt", "ok")
    )
else:
    status_chips += _chip("🔑 Kein API-Key nötig", "ok")
status_chips += _chip("🕖 Tagesberichte aktiv", "ok") if scheduler else _chip("🕖 Tagesberichte aus")
st.markdown(status_chips, unsafe_allow_html=True)
running_notice = st.empty()

store = _portfolio_store()
holdings = store.list_holdings()
held_symbols = [holding["symbol"] for holding in holdings]
if held_symbols:
    price_history, latest_prices, price_errors = _price_history(tuple(held_symbols))
else:
    price_history, latest_prices, price_errors = None, {}, {}
latest_reports = store.latest_reports()

portfolio_tab, duel_tab, news_tab, analysis_tab, reports_tab = st.tabs(
    ["📊 Portfolio", "⚔️ Ich vs. KI", "📰 Nachrichten", "🔎 Analyse", "🗂️ Berichte"]
)

with portfolio_tab:
    st.subheader("Dein Portfolio")
    st.caption(
        "Bestände und Analysen werden dauerhaft lokal in SQLite gespeichert. "
        "Für jede Position entsteht automatisch jeden Tag ein neuer Bericht."
    )

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
                _price_history.clear()
                st.success(f"{holding_symbol.strip().upper()} wurde gespeichert.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    if not holdings:
        st.info("Noch keine Positionen. Füge oben ein Wertpapier hinzu.")
    else:
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

        if price_history is not None and not price_history.empty:
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
            last_report = latest_reports.get(symbol)
            recommendation = (
                f"{_DECISION_ICON[_decision_tone(last_report['decision'])]} "
                f"{last_report['decision'][:20]} · {last_report['analysis_date']}"
                if last_report
                else "—"
            )

            with _box():
                head_col, action_col = st.columns([3.2, 1.0])
                with head_col:
                    st.markdown(
                        f'<span class="sym">{symbol}</span>{_return_badge(current_return)}',
                        unsafe_allow_html=True,
                    )
                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        _stat("Stückzahl", f"{holding['shares']:,.6g}")
                    with stat_cols[1]:
                        _stat("Kauf", cost_text)
                    with stat_cols[2]:
                        _stat("Kurs", current_price)
                    with stat_cols[3]:
                        _stat("KI-Empfehlung", recommendation)
                with action_col:
                    if st.button(
                        "🔎 Jetzt analysieren",
                        key=f"analyze_{symbol}",
                        disabled=not llm_ready,
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
                        _price_history.clear()
                        st.rerun()

with news_tab:
    st.subheader("Nachrichten zu deinen Aktien")
    st.caption(
        "Aktuelle Meldungen von Yahoo Finance. Die KI schätzt je Meldung, ob der Kurs eher "
        "steigt oder fällt; daneben steht, wie sich die Aktie seit der Meldung tatsächlich "
        "entwickelt hat. Bewertungen werden gespeichert und nicht erneut berechnet."
    )
    if not held_symbols:
        st.info("Füge im Portfolio Positionen hinzu, um hier Nachrichten zu sehen.")
    else:
        if not llm_ready:
            st.warning("Ohne API-Key/Modell werden die Meldungen ohne KI-Einschätzung angezeigt.")
        if st.button("🔄 Nachrichten aktualisieren", key="refresh_news"):
            _news.clear()
        for symbol in held_symbols:
            prices = (
                price_history[symbol]
                if price_history is not None and symbol in getattr(price_history, "columns", [])
                else None
            )
            with st.spinner(f"Lade Nachrichten für {symbol} …"):
                try:
                    articles, rating_error = _news(
                        symbol, llm_provider, quick_model, llm_ready, prices, ui_config
                    )
                except Exception as exc:
                    st.warning(f"{symbol}: Nachrichten konnten nicht geladen werden ({exc})")
                    continue
            ups = sum(1 for a in articles if (a["rating"] or {}).get("direction") == "up")
            downs = sum(1 for a in articles if (a["rating"] or {}).get("direction") == "down")
            with _box():
                st.markdown(
                    f'<span class="sym">{symbol}</span>'
                    f'<span class="badge badge-up">↑ {ups}</span>'
                    f'<span class="badge badge-down">↓ {downs}</span>',
                    unsafe_allow_html=True,
                )
                if rating_error:
                    st.caption(f"KI-Bewertung fehlgeschlagen: {rating_error}")
                if not articles:
                    st.caption("Keine aktuellen Meldungen gefunden.")
                    continue
                rows = []
                for article in articles:
                    rating = article["rating"] or {}
                    rows.append({
                        "Datum": article["pub_date"].strftime("%d.%m.%Y %H:%M")
                        if article["pub_date"] else "—",
                        "Meldung": article["title"],
                        "Link": article["link"] or None,
                        "Quelle": article["publisher"],
                        "KI-Einschätzung": _DIRECTION_LABEL.get(rating.get("direction"), "—"),
                        "Sicherheit": rating.get("confidence"),
                        "Begründung": rating.get("reason", ""),
                        "Kurs seit Meldung": (
                            article["reaction"] * 100 if article["reaction"] is not None else None
                        ),
                    })
                st.dataframe(
                    rows,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Link": st.column_config.LinkColumn("Link", display_text="öffnen"),
                        "Sicherheit": st.column_config.ProgressColumn(
                            "Sicherheit", min_value=0, max_value=100, format="%d%%"
                        ),
                        "Kurs seit Meldung": st.column_config.NumberColumn(
                            "Kurs seit Meldung", format="%+.2f%%"
                        ),
                    },
                )

with analysis_tab:
    st.subheader("KI-Analyse")
    st.caption(
        "Multi-Agenten-Analyse mit Analysten, Research-Debatte, Trader und Risikomanagement. "
        "Was die Agenten denken, erscheint live, während sie arbeiten. Der Bericht wird "
        "automatisch gespeichert, auch wenn du die Seite verlässt."
    )

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

    job: AnalysisJob | None = st.session_state.get("analysis_job")
    job_running = job is not None and job.running
    requested_analysis = st.button(
        "Analyse starten",
        type="primary",
        disabled=not ticker.strip() or not llm_ready or job_running,
        key="start_analysis",
    )
    run_portfolio_analysis = st.session_state.pop("run_portfolio_analysis", False)
    if (requested_analysis or run_portfolio_analysis) and not job_running:
        job_symbol = (
            st.session_state.get("analysis_ticker", ticker) if run_portfolio_analysis else ticker
        )
        job = AnalysisJob(store, job_symbol.strip().upper(), str(analysis_date), ui_config)
        st.session_state["analysis_job"] = job
    # Filled at the end of the script, so streaming never blocks the other tabs.
    job_area = st.container()

with reports_tab:
    st.subheader("Analyseberichte")

    if latest_reports:
        st.markdown("**Letzte Empfehlung je Aktie**")
        st.dataframe(
            [
                {
                    "Aktie": report["symbol"],
                    "Signal": _DECISION_ICON[_decision_tone(report["decision"])],
                    "Entscheidung": report["decision"],
                    "Analyse-Datum": report["analysis_date"],
                    "Erstellt": report["created_at"][:16].replace("T", " "),
                }
                for report in sorted(latest_reports.values(), key=lambda r: r["symbol"])
            ],
            hide_index=True,
            use_container_width=True,
        )

    daily_runs = store.list_daily_runs(limit=30)
    with st.expander(f"🕖 Automatische Tagesläufe ({len(daily_runs)})", expanded=False):
        if scheduler is not None:
            st.caption(f"Nächster Start: {scheduler.next_run().strftime('%d.%m.%Y %H:%M')}")
        if not daily_runs:
            st.caption("Noch keine automatischen Läufe.")
        else:
            status_label = {"done": "✅ fertig", "running": "⏳ läuft", "error": "❌ Fehler"}
            st.dataframe(
                [
                    {
                        "Aktie": run["symbol"],
                        "Analyse-Datum": run["analysis_date"],
                        "Status": status_label.get(run["status"], run["status"]),
                        "Versuche": run["attempts"],
                        "Bericht": f"#{run['report_id']}" if run["report_id"] else "",
                        "Fehler": run["error"] or "",
                        "Aktualisiert": run["updated_at"][:16].replace("T", " "),
                    }
                    for run in daily_runs
                ],
                hide_index=True,
                use_container_width=True,
            )

    reports = store.list_reports()
    if not reports:
        st.info("Nach der ersten Analyse erscheinen die Berichte hier dauerhaft.")
    else:
        report_symbols = sorted({report["symbol"] for report in reports})
        symbol_filter = st.selectbox("Aktie", ["Alle", *report_symbols], key="report_symbol_filter")
        filtered = [r for r in reports if symbol_filter == "Alle" or r["symbol"] == symbol_filter]
        selected_report = st.selectbox(
            "Bericht auswählen",
            filtered,
            format_func=lambda report: (
                f"{_DECISION_ICON[_decision_tone(report['decision'])]} #{report['id']} · "
                f"{report['symbol']} · {report['analysis_date']} "
                f"· {report['created_at'][:16].replace('T', ' ')}"
            ),
        )
        full_report = store.get_report(selected_report["id"])
        if full_report is None:
            st.error("Der ausgewählte Bericht ist nicht mehr verfügbar.")
        else:
            _decision_box("Entscheidung", full_report["decision"])

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

with duel_tab:
    duel = _duel_store()
    st.subheader("⚔️ Ich vs. KI")
    st.caption(
        "Beide starten mit demselben Spielgeld und handeln virtuell zu echten Kursen (in €). "
        "Du handelst frei; die KI spielt jeden Tag automatisch einen Zug: Sie sucht sich "
        "Kandidaten über Yahoo-Screener, lässt die komplette Multi-Agenten-Analyse laufen und "
        "handelt nach dem Rating des Portfolio-Managers. Jeder KI-Trade ist begründet."
    )

    def _quote(symbol: str) -> float:
        return _eur_price(symbol)

    standings = {player: duel.valuation(player, _quote) for player in PLAYERS}
    me_val, ai_val = standings["me"], standings["ai"]
    lead = me_val["total"] - ai_val["total"]
    if abs(lead) < 0.01:
        st.info("🤝 Gleichstand.")
    elif lead > 0:
        st.success(f"🏆 Du führst mit {lead:,.2f} € vor der KI.")
    else:
        st.error(f"🤖 Die KI führt mit {-lead:,.2f} € vor dir.")

    score_me, score_ai = st.columns(2)
    score_me.metric(
        "🧑 Ich — Depotwert", f"{me_val['total']:,.2f} €", f"{me_val['return'] * 100:+.2f}%"
    )
    score_ai.metric(
        "🤖 KI — Depotwert", f"{ai_val['total']:,.2f} €", f"{ai_val['return'] * 100:+.2f}%"
    )

    history = duel.equity_history()
    if history:
        frame = pd.DataFrame(history).pivot(index="day", columns="player", values="value")
        frame = frame.rename(columns=PLAYERS)
        frame.index = pd.to_datetime(frame.index)
        st.markdown("**Depotwert im Verlauf**")
        st.line_chart(frame)

    def _position_rows(valuation: dict) -> list[dict]:
        return [
            {
                "Aktie": p["symbol"],
                "Stück": round(p["shares"], 4),
                "Kurs €": round(p["price"], 2),
                "Wert €": round(p["value"], 2),
                "G/V €": round(p["pnl"], 2),
                "G/V %": round(p["pnl"] / p["cost"] * 100, 2) if p["cost"] else None,
            }
            for p in valuation["positions"]
        ]

    me_col, ai_col = st.columns(2)
    with me_col, _box():
        st.markdown("**🧑 Mein Depot**")
        st.caption(f"Spielgeld: {me_val['cash']:,.2f} €")
        with st.form("duel_trade_form", clear_on_submit=True):
            trade_symbol = st.text_input("Ticker", placeholder="z. B. NVDA, SAP.DE")
            side_col, amount_col = st.columns(2)
            with side_col:
                trade_side = st.radio("Aktion", ["Kaufen", "Verkaufen"], horizontal=True)
            with amount_col:
                trade_amount = st.number_input(
                    "Betrag in €", min_value=0.0, value=1000.0, step=100.0
                )
            sell_all = st.checkbox("Ganze Position verkaufen")
            submitted = st.form_submit_button("Order ausführen", type="primary")
        if submitted:
            try:
                symbol = trade_symbol.strip().upper()
                price = _eur_price(symbol)
                if trade_side == "Kaufen":
                    shares = trade_amount / price
                    duel.trade("me", symbol, "buy", shares, price)
                else:
                    held = next(
                        (p["shares"] for p in me_val["positions"] if p["symbol"] == symbol), 0.0
                    )
                    shares = held if sell_all else min(held, trade_amount / price)
                    duel.trade("me", symbol, "sell", shares, price)
                duel.record_equity(_quote)
                st.success(
                    f"{trade_side}: {shares:,.4f} × {symbol} zu {price:,.2f} € "
                    f"= {shares * price:,.2f} €"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Order abgelehnt: {exc}")
        if me_val["positions"]:
            st.dataframe(_position_rows(me_val), hide_index=True, use_container_width=True)
        else:
            st.caption("Noch keine Positionen.")
    with ai_col, _box():
        st.markdown("**🤖 Depot der KI**")
        st.caption(f"Spielgeld: {ai_val['cash']:,.2f} €")
        if ai_val["positions"]:
            st.dataframe(_position_rows(ai_val), hide_index=True, use_container_width=True)
        else:
            st.caption("Die KI hat noch nicht gehandelt.")

    st.markdown("#### 🤖 Züge der KI")
    turn_date = previous_trading_day(date.today())
    runner_thread = _duel_runner()["thread"]
    turn_running = runner_thread is not None and runner_thread.is_alive()
    info_col, button_col = st.columns([3, 1])
    with info_col:
        if scheduler is not None and duel_enabled():
            st.caption(
                f"Automatisch täglich ab {schedule_time().strftime('%H:%M')} "
                f"({schedule_zone().key}); nächster Start "
                f"{scheduler.next_run().strftime('%d.%m. %H:%M')}. "
                f"Pro Zug bis zu {max_picks()} neue Kandidaten, maximal {max_analyses()} Analysen."
            )
        else:
            st.caption("Automatische KI-Züge sind deaktiviert.")
    with button_col:
        if st.button(
            "▶️ KI-Zug jetzt",
            disabled=not llm_ready or turn_running,
            use_container_width=True,
            help=f"Spielt den Zug für den Handelstag {turn_date} (falls noch nicht gespielt).",
        ) and _start_ai_turn(turn_date):
            st.toast("Die KI denkt nach … das dauert einige Minuten.")
            time.sleep(1)
            st.rerun()

    turns = duel.turns(limit=10)
    if not turns:
        st.caption("Noch kein KI-Zug gespielt.")
    for index, turn in enumerate(turns):
        status = {"done": "✅", "running": "⏳", "error": "❌"}.get(turn["status"], turn["status"])
        analyses = turn["analyses"] or []
        trades_made = sum(1 for a in analyses if a.get("trade"))
        with st.expander(
            f"{status} Zug für {turn['turn_date']} · {len(analyses)} Analysen · {trades_made} Trades",
            expanded=index == 0,
        ):
            if turn["error"]:
                st.error(turn["error"])
            scouting = turn["scouting"] or {}
            if scouting.get("picks"):
                st.markdown(
                    f"**🔭 Scouting** — {len(scouting.get('candidates', []))} Kandidaten aus "
                    "den Yahoo-Screenern, davon ausgewählt:"
                )
                st.dataframe(
                    [{"Aktie": p["symbol"], "Warum interessant": p["reason"]} for p in scouting["picks"]],
                    hide_index=True,
                    use_container_width=True,
                )
            if scouting.get("candidates"):
                with st.popover("Alle Kandidaten") if hasattr(st, "popover") else st.container():
                    st.dataframe(
                        [
                            {
                                "Aktie": c["symbol"],
                                "Name": c["name"],
                                "Heute %": c.get("change_pct"),
                                "52W %": c.get("change_52w_pct"),
                                "KGV": c.get("pe"),
                                "Screener": ", ".join(c.get("screens", [])),
                            }
                            for c in scouting["candidates"]
                        ],
                        hide_index=True,
                    )
            for analysis in analyses:
                symbol = analysis["symbol"]
                if analysis.get("error"):
                    st.warning(f"{symbol}: Analyse fehlgeschlagen ({analysis['error']})")
                    continue
                rating = analysis.get("rating", "—")
                trade = analysis.get("trade")
                action = (
                    f"{'🟢 kauft' if trade['side'] == 'buy' else '🔴 verkauft'} "
                    f"{trade['shares']:,.4f} Stück für {trade['amount']:,.2f} €"
                    if trade
                    else "hält still"
                )
                st.markdown(
                    f"{_DECISION_ICON[_decision_tone(rating)]} **{symbol}** · Rating "
                    f"**{rating}** → {action}"
                )
                report_id = analysis.get("report_id")
                if report_id and st.checkbox(
                    f"Warum? Vollständige Analyse zu {symbol} zeigen",
                    key=f"duel_report_{turn['turn_date']}_{symbol}",
                ):
                    report = store.get_report(report_id)
                    if report:
                        with _box():
                            st.markdown(report["report_markdown"])
            if turn["status"] == "running":
                st.caption("⏳ Zug läuft — Seite neu laden, um den Fortschritt zu sehen.")

    st.markdown("#### 📜 Alle Trades")
    all_trades = duel.trades(limit=200)
    if not all_trades:
        st.caption("Noch keine Trades.")
    else:
        st.dataframe(
            [
                {
                    "Zeit": t["created_at"][:16].replace("T", " "),
                    "Spieler": PLAYERS[t["player"]],
                    "Aktie": t["symbol"],
                    "Aktion": "Kauf" if t["side"] == "buy" else "Verkauf",
                    "Stück": round(t["shares"], 4),
                    "Kurs €": round(t["price"], 2),
                    "Betrag €": round(t["amount"], 2),
                    "Rating": t["rating"] or "",
                    "Begründung (Portfolio-Manager)": (t["reason"] or "")[:300],
                }
                for t in all_trades
            ],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("🔄 Neues Spiel starten"):
        st.caption("Setzt beide Depots auf das Startkapital zurück und löscht alle Trades.")
        confirm_reset = st.checkbox("Ja, ich will wirklich neu anfangen", key="duel_reset_confirm")
        if st.button("Spiel zurücksetzen", disabled=not confirm_reset):
            duel.reset()
            st.rerun()


if job is not None and job.running:
    running_notice.info(
        f"⏳ Analyse von **{job.symbol}** läuft — die Agenten-Gedanken erscheinen live "
        "im Tab „🔎 Analyse“."
    )

st.divider()
st.markdown(
    '<div class="footer">⚠️ Nur zu Forschungszwecken — keine Anlageberatung. '
    "Kurs- und Nachrichtendaten: Yahoo Finance.</div>",
    unsafe_allow_html=True,
)

if job is not None:
    with job_area:
        _render_job(job)
    running_notice.empty()
