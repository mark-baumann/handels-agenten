"""Tests for the automatic daily portfolio reports."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from tradingagents.daily_reports import (
    DailyReportScheduler,
    previous_trading_day,
    run_due_reports,
)
from tradingagents.portfolio import PortfolioStore

pytestmark = pytest.mark.unit

_CONFIG = {"llm_provider": "openai", "deep_think_llm": "d", "quick_think_llm": "q"}


def _store(tmp_path, *symbols):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    for symbol in symbols:
        store.add_holding(symbol, 1)
    return store


def _fake_analyze(calls, fail=()):
    def analyze(store, symbol, day, config, session_dir):
        calls.append(symbol)
        if symbol in fail:
            raise RuntimeError("rate limited")
        return store.save_report(symbol, day, "Buy", f"# {symbol}"), "Buy", f"# {symbol}"
    return analyze


def test_previous_trading_day_skips_weekends():
    assert previous_trading_day(date(2026, 10, 13)) == date(2026, 10, 12)  # Tue → Mon
    assert previous_trading_day(date(2026, 10, 12)) == date(2026, 10, 9)   # Mon → Fri
    assert previous_trading_day(date(2026, 10, 11)) == date(2026, 10, 9)   # Sun → Fri


def test_run_due_reports_analyses_each_holding_once(tmp_path):
    store = _store(tmp_path, "AAPL", "SAP.DE")
    calls = []
    done = run_due_reports(store, lambda: _CONFIG, date(2026, 10, 9), analyze=_fake_analyze(calls))
    assert sorted(done) == ["AAPL", "SAP.DE"]
    # A second pass the same day finds the reports and does nothing.
    assert run_due_reports(store, lambda: _CONFIG, date(2026, 10, 9), analyze=_fake_analyze(calls)) == []
    assert sorted(calls) == ["AAPL", "SAP.DE"]
    assert {run["status"] for run in store.list_daily_runs()} == {"done"}
    assert store.latest_reports()["AAPL"]["analysis_date"] == "2026-10-09"


def test_favorites_without_holding_get_daily_reports(tmp_path):
    store = _store(tmp_path, "AAPL")
    store.add_favorite("NVDA")
    calls = []
    done = run_due_reports(store, lambda: _CONFIG, date(2026, 10, 9), analyze=_fake_analyze(calls))
    assert sorted(done) == ["AAPL", "NVDA"]


def test_failed_run_is_recorded_and_not_retried_immediately(tmp_path):
    store = _store(tmp_path, "AAPL")
    calls = []
    analyze = _fake_analyze(calls, fail={"AAPL"})
    assert run_due_reports(store, lambda: _CONFIG, date(2026, 10, 9), analyze=analyze) == []
    run = store.list_daily_runs()[0]
    assert run["status"] == "error" and "rate limited" in run["error"]
    run_due_reports(store, lambda: _CONFIG, date(2026, 10, 9), analyze=analyze)
    assert calls == ["AAPL"]  # retry waits for the back-off


def test_claim_retries_after_backoff_up_to_max_attempts(tmp_path):
    store = _store(tmp_path, "AAPL")
    no_wait = timedelta(0)
    for _ in range(3):
        assert store.claim_daily_run("AAPL", "2026-10-09", retry_after=no_wait)
        store.finish_daily_run("AAPL", "2026-10-09", error="boom")
    assert not store.claim_daily_run("AAPL", "2026-10-09", retry_after=no_wait)


def test_running_claim_blocks_until_stale(tmp_path):
    store = _store(tmp_path, "AAPL")
    assert store.claim_daily_run("AAPL", "2026-10-09")
    assert not store.claim_daily_run("AAPL", "2026-10-09")
    assert store.claim_daily_run("AAPL", "2026-10-09", stale_after=timedelta(0))


def test_missing_api_key_skips_without_claiming(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    store = _store(tmp_path, "AAPL")
    calls = []
    config = {**_CONFIG, "llm_provider": "ollama_cloud"}
    assert run_due_reports(store, lambda: config, date(2026, 10, 9), analyze=_fake_analyze(calls)) == []
    assert calls == [] and store.list_daily_runs() == []


def test_scheduler_due_date_and_next_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_DAILY_REPORT_TIME", "07:00")
    monkeypatch.setenv("TRADINGAGENTS_DAILY_REPORT_TZ", "Europe/Berlin")
    scheduler = DailyReportScheduler(_store(tmp_path), lambda: _CONFIG)
    berlin = ZoneInfo("Europe/Berlin")
    early = datetime(2026, 10, 13, 6, 59, tzinfo=berlin)
    late = datetime(2026, 10, 13, 7, 0, tzinfo=berlin)
    assert scheduler.due_date(early) is None
    assert scheduler.due_date(late) == date(2026, 10, 12)
    assert scheduler.next_run(early) == datetime(2026, 10, 13, 7, 0, tzinfo=berlin)
    assert scheduler.next_run(late) == datetime(2026, 10, 14, 7, 0, tzinfo=berlin)
    # Times in other zones are converted first.
    assert scheduler.due_date(datetime(2026, 10, 13, 4, 30, tzinfo=timezone.utc)) is None
