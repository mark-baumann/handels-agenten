"""Automatic daily analysis reports for every portfolio holding.

The dashboard starts one :class:`DailyReportScheduler` per server process. Once
per day (after ``TRADINGAGENTS_DAILY_REPORT_TIME`` in
``TRADINGAGENTS_DAILY_REPORT_TZ``) it analyses every holding for the previous
trading day and stores the report, so nobody has to click "Analyse starten".
Runs are claimed in SQLite, so a restart catches up on missed reports without
ever running the same symbol/date twice.

Run ``python -m tradingagents.daily_reports`` to process due reports once, e.g.
from cron instead of the in-process scheduler.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.portfolio import PortfolioStore
from tradingagents.reporting import write_report_tree

logger = logging.getLogger(__name__)

_KEYLESS_PROVIDERS = ("ollama", "bedrock", "openai_compatible")


def reports_enabled() -> bool:
    return os.environ.get("TRADINGAGENTS_DAILY_REPORTS", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def schedule_time() -> time:
    raw = os.environ.get("TRADINGAGENTS_DAILY_REPORT_TIME", "07:00").strip()
    try:
        return time.fromisoformat(raw)
    except ValueError:
        logger.warning("Invalid TRADINGAGENTS_DAILY_REPORT_TIME %r, using 07:00", raw)
        return time(7, 0)


def schedule_zone() -> ZoneInfo:
    name = os.environ.get("TRADINGAGENTS_DAILY_REPORT_TZ", "Europe/Berlin").strip()
    try:
        return ZoneInfo(name)
    except Exception:
        logger.warning("Invalid TRADINGAGENTS_DAILY_REPORT_TZ %r, using Europe/Berlin", name)
        return ZoneInfo("Europe/Berlin")


def previous_trading_day(today: date) -> date:
    """Last weekday before ``today`` (Monday → Friday)."""
    day = today - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def missing_api_key(provider: str) -> str | None:
    """Env-var name of a required but unset API key, else None."""
    if provider in _KEYLESS_PROVIDERS:
        return None
    env = get_api_key_env(provider)
    if env and not os.environ.get(env, "").strip():
        return env
    return None


def run_and_store_analysis(
    store: PortfolioStore,
    symbol: str,
    analysis_date: str,
    config: dict,
    on_chunk: Callable[[dict], None] | None = None,
    session_dir: str = "streamlit_sessions",
) -> tuple[int, str, str]:
    """Run the agent graph, persist the report; return (report_id, decision, markdown)."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph(debug=False, config=config)
    final_state, decision = graph.propagate(symbol, analysis_date, on_chunk=on_chunk)

    ticker_safe = symbol.strip().upper().replace("/", "_")
    report_dir = (
        Path(config["results_dir"])
        / session_dir
        / ticker_safe
        / analysis_date
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    markdown = write_report_tree(final_state, ticker_safe, report_dir).read_text(encoding="utf-8")
    report_id = store.save_report(symbol, analysis_date, str(decision), markdown)
    return report_id, str(decision), markdown


def run_due_reports(
    store: PortfolioStore,
    config_factory: Callable[[], dict],
    analysis_date: date,
    analyze: Callable[..., tuple[int, str, str]] = run_and_store_analysis,
    stop: threading.Event | None = None,
) -> list[str]:
    """Analyse every favorite and holding that has no report for ``analysis_date`` yet.

    Symbols are processed one after another to stay within free-tier rate
    limits. Returns the symbols that got a new report.
    """
    day = analysis_date.isoformat()
    done = []
    for symbol in store.tracked_symbols():
        if stop is not None and stop.is_set():
            break
        if store.has_report(symbol, day):
            continue
        config = config_factory()
        key_env = missing_api_key(config["llm_provider"])
        if key_env:
            logger.warning("Daily reports skipped: %s is not set", key_env)
            break
        if not store.claim_daily_run(symbol, day):
            continue
        logger.info("Daily report: analysing %s for %s", symbol, day)
        try:
            report_id, _, _ = analyze(store, symbol, day, config, session_dir="daily_reports")
        except Exception as exc:
            logger.exception("Daily report for %s failed", symbol)
            store.finish_daily_run(symbol, day, error=str(exc)[:500])
        else:
            store.finish_daily_run(symbol, day, report_id=report_id)
            done.append(symbol)
    return done


class DailyReportScheduler:
    """Background thread that keeps the daily reports up to date."""

    def __init__(
        self,
        store: PortfolioStore,
        config_factory: Callable[[], dict],
        check_interval: float = 300,
        extra_jobs: list[Callable[[date, threading.Event], object]] | None = None,
    ):
        self.store = store
        self.config_factory = config_factory
        self.check_interval = check_interval
        # Further daily jobs (e.g. the AI's duel turn), run after the reports.
        self.extra_jobs = list(extra_jobs or [])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="daily-reports", daemon=True)

    def start(self) -> DailyReportScheduler:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def next_run(self, now: datetime | None = None) -> datetime:
        """Next scheduled start in the schedule's time zone."""
        zone = schedule_zone()
        now = now.astimezone(zone) if now else datetime.now(zone)
        start = datetime.combine(now.date(), schedule_time(), tzinfo=zone)
        return start if now < start else start + timedelta(days=1)

    def due_date(self, now: datetime | None = None) -> date | None:
        """Analysis date to process now, or None before today's start time."""
        zone = schedule_zone()
        now = now.astimezone(zone) if now else datetime.now(zone)
        if now.time() < schedule_time():
            return None
        return previous_trading_day(now.date())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                due = self.due_date()
                if due is not None:
                    run_due_reports(self.store, self.config_factory, due, stop=self._stop)
                    for job in self.extra_jobs:
                        try:
                            job(due, self._stop)
                        except Exception:
                            logger.exception("Daily job %r failed", job)
            except Exception:
                logger.exception("Daily report scheduler iteration failed")
            self._stop.wait(self.check_interval)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ui_settings import load_llm_settings, run_config

    cache = Path(DEFAULT_CONFIG["data_cache_dir"])
    store = PortfolioStore(cache / "portfolio.sqlite3")
    done = run_due_reports(
        store,
        lambda: run_config(load_llm_settings(cache / "llm_settings.json"), DEFAULT_CONFIG),
        previous_trading_day(datetime.now(schedule_zone()).date()),
    )
    print(f"Neue Berichte: {', '.join(done) or 'keine'}")


if __name__ == "__main__":
    main()
