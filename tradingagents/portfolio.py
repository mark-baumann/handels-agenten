"""Persistent portfolio holdings and analysis reports for the Streamlit app."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, datetime.now(timezone.utc).isoformat()),
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
