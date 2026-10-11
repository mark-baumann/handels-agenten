"""Persistent portfolio holdings and analysis reports for the Streamlit app."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.symbol_utils import is_yahoo_safe, normalize_symbol


class PortfolioStore:
    """SQLite-backed storage for a single shared app portfolio and its reports."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS holdings (
                    symbol TEXT PRIMARY KEY,
                    shares REAL NOT NULL CHECK (shares > 0),
                    average_cost REAL CHECK (average_cost IS NULL OR average_cost > 0),
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    report_markdown TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            # One row per scheduled (symbol, analysis_date) run; doubles as a
            # claim so a restart or a second process never runs it twice.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_runs (
                    symbol TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    report_id INTEGER,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, analysis_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS news_impact (
                    symbol TEXT NOT NULL,
                    article_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, article_key)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _symbol(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Ticker symbol must be text.")
        symbol = normalize_symbol(value)
        if not symbol or not is_yahoo_safe(symbol):
            raise ValueError("Enter a valid Yahoo Finance ticker symbol.")
        return symbol

    def add_holding(self, symbol: str, shares: float, average_cost: float | None = None) -> None:
        canonical = self._symbol(symbol)
        if not math.isfinite(shares) or shares <= 0:
            raise ValueError("Share quantity must be a positive number.")
        if average_cost is not None and (not math.isfinite(average_cost) or average_cost <= 0):
            raise ValueError("Average purchase price must be a positive number.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO holdings (symbol, shares, average_cost, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    shares = excluded.shares,
                    average_cost = excluded.average_cost,
                    updated_at = excluded.updated_at
                """,
                (canonical, shares, average_cost, datetime.now(timezone.utc).isoformat()),
            )

    def remove_holding(self, symbol: str) -> bool:
        canonical = self._symbol(symbol)
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM holdings WHERE symbol = ?", (canonical,))
            return cursor.rowcount > 0

    def list_holdings(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT symbol, shares, average_cost, updated_at FROM holdings ORDER BY symbol"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_report(
        self, symbol: str, analysis_date: str, decision: str, report_markdown: str
    ) -> int:
        canonical = self._symbol(symbol)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO reports (symbol, analysis_date, decision, report_markdown, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    canonical,
                    analysis_date,
                    decision,
                    report_markdown,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def list_reports(self, limit: int = 100) -> list[dict]:
        if limit <= 0:
            raise ValueError("Report limit must be positive.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, symbol, analysis_date, decision, created_at
                FROM reports
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_report(self, report_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, symbol, analysis_date, decision, report_markdown, created_at
                FROM reports WHERE id = ?
                """,
                (report_id,),
            ).fetchone()
        return dict(row) if row else None

    def latest_reports(self) -> dict[str, dict]:
        """Newest report per symbol (without the markdown body)."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, symbol, analysis_date, decision, created_at FROM reports r
                WHERE id = (
                    SELECT id FROM reports WHERE symbol = r.symbol
                    ORDER BY analysis_date DESC, created_at DESC, id DESC LIMIT 1
                )
                """
            ).fetchall()
        return {row["symbol"]: dict(row) for row in rows}

    def has_report(self, symbol: str, analysis_date: str) -> bool:
        canonical = self._symbol(symbol)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM reports WHERE symbol = ? AND analysis_date = ? LIMIT 1",
                (canonical, analysis_date),
            ).fetchone()
        return row is not None

    # --- Scheduled daily runs ---------------------------------------------

    def claim_daily_run(
        self,
        symbol: str,
        analysis_date: str,
        max_attempts: int = 3,
        retry_after: timedelta = timedelta(minutes=30),
        stale_after: timedelta = timedelta(hours=2),
    ) -> bool:
        """Atomically mark a daily run as started; False if it must not run now.

        A run is claimable when it never ran, failed fewer than ``max_attempts``
        times and the last failure is older than ``retry_after``, or when a
        "running" claim is older than ``stale_after`` (crashed process).
        """
        canonical = self._symbol(symbol)
        now = datetime.now(timezone.utc)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, attempts, updated_at FROM daily_runs "
                "WHERE symbol = ? AND analysis_date = ?",
                (canonical, analysis_date),
            ).fetchone()
            if row is not None:
                age = now - datetime.fromisoformat(row["updated_at"])
                claimable = (
                    (row["status"] == "error" and row["attempts"] < max_attempts and age >= retry_after)
                    or (row["status"] == "running" and age >= stale_after)
                )
                if not claimable:
                    connection.rollback()
                    return False
            connection.execute(
                """
                INSERT INTO daily_runs (symbol, analysis_date, status, attempts, updated_at)
                VALUES (?, ?, 'running', 1, ?)
                ON CONFLICT(symbol, analysis_date) DO UPDATE SET
                    status = 'running', attempts = attempts + 1, error = NULL,
                    updated_at = excluded.updated_at
                """,
                (canonical, analysis_date, now.isoformat()),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def finish_daily_run(
        self, symbol: str, analysis_date: str, report_id: int | None = None, error: str | None = None
    ) -> None:
        canonical = self._symbol(symbol)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE daily_runs SET status = ?, report_id = ?, error = ?, updated_at = ?
                WHERE symbol = ? AND analysis_date = ?
                """,
                (
                    "error" if error else "done",
                    report_id,
                    error,
                    datetime.now(timezone.utc).isoformat(),
                    canonical,
                    analysis_date,
                ),
            )

    def list_daily_runs(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT symbol, analysis_date, status, attempts, report_id, error, updated_at
                FROM daily_runs ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    # --- News impact cache ------------------------------------------------

    def get_news_impacts(self, symbol: str) -> dict[str, dict]:
        canonical = self._symbol(symbol)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT article_key, payload FROM news_impact WHERE symbol = ?", (canonical,)
            ).fetchall()
        return {row["article_key"]: json.loads(row["payload"]) for row in rows}

    def save_news_impacts(self, symbol: str, impacts: dict[str, dict]) -> None:
        canonical = self._symbol(symbol)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO news_impact (symbol, article_key, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol, article_key) DO UPDATE SET payload = excluded.payload
                """,
                [(canonical, key, json.dumps(value), now) for key, value in impacts.items()],
            )


def load_portfolio_price_history(
    holdings: list[dict], period: str = "1y"
) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    """Return indexed price history, latest prices, and explicit symbol errors."""
    series_by_symbol = {}
    latest_prices = {}
    errors = {}
    for holding in holdings:
        symbol = holding["symbol"]
        try:
            history = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if history.empty or "Close" not in history:
                errors[symbol] = "No price history was returned."
                continue
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if close.empty:
                errors[symbol] = "No valid closing prices were returned."
                continue
            close.index = pd.to_datetime(close.index).tz_localize(None)
            series_by_symbol[symbol] = close / close.iloc[0] * 100
            latest_prices[symbol] = float(close.iloc[-1])
        except Exception as exc:
            errors[symbol] = str(exc)

    if not series_by_symbol:
        return pd.DataFrame(), latest_prices, errors
    return pd.concat(series_by_symbol, axis=1).sort_index(), latest_prices, errors
