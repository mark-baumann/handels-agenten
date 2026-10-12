""""Ich vs. KI": a paper-trading duel between the user and the agents.

Both players start with the same virtual cash (in EUR) and trade at real
market prices. The user trades by hand in the dashboard; the AI takes one turn
per day from the scheduler:

1. **Scouting** — Yahoo Finance screeners (most active, gainers, losers,
   undervalued, growth tech) produce candidates; the quick-think model picks
   the most promising ones and says why.
2. **Analysis** — the full TradingAgents pipeline runs for every AI holding and
   every pick (capped by a daily budget). Reports are shared with the rest of
   the dashboard, so a symbol already analysed that day costs nothing extra.
3. **Trading** — each 5-tier rating maps to a target weight; the resulting
   trades are stored with the report that justifies them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from tradingagents.dataflows.symbol_utils import is_yahoo_safe, normalize_symbol

logger = logging.getLogger(__name__)

PLAYERS = {"me": "Ich", "ai": "KI"}

# Target share of total equity per rating. ``None`` keeps the position as is;
# Underweight halves it.
TARGET_WEIGHTS = {"Buy": 0.20, "Overweight": 0.10, "Hold": None, "Underweight": "half", "Sell": 0.0}
MIN_TRADE_EUR = 50.0

SCREENERS = (
    "most_actives",
    "day_gainers",
    "day_losers",
    "undervalued_large_caps",
    "growth_technology_stocks",
)


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        logger.warning("Invalid %s, using %s", name, default)
        return default


def start_cash() -> float:
    return _env_number("TRADINGAGENTS_DUEL_START_CASH", 10_000)


def max_analyses() -> int:
    return int(_env_number("TRADINGAGENTS_DUEL_MAX_ANALYSES", 5))


def max_picks() -> int:
    return int(_env_number("TRADINGAGENTS_DUEL_PICKS", 3))


def duel_enabled() -> bool:
    return os.environ.get("TRADINGAGENTS_DUEL", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise ValueError("Ticker symbol must be text.")
    canonical = normalize_symbol(symbol)
    if not canonical or not is_yahoo_safe(canonical):
        raise ValueError("Enter a valid Yahoo Finance ticker symbol.")
    return canonical


# --- Prices ---------------------------------------------------------------


class QuoteError(RuntimeError):
    pass


def _valid_price(price) -> bool:
    try:
        return price is not None and math.isfinite(float(price)) and float(price) > 0
    except (TypeError, ValueError):
        return False


class EurQuotes:
    """Latest price per symbol converted to EUR, cached for one instance."""

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}
        self._fx: dict[str, float] = {"EUR": 1.0}

    @staticmethod
    def _last(symbol: str) -> tuple[float, str]:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        price = currency = None
        try:
            info = ticker.fast_info
            price, currency = info["lastPrice"], info["currency"]
        except Exception as exc:  # unknown symbols raise e.g. KeyError('currentTradingPeriod')
            logger.debug("fast_info failed for %s: %s", symbol, exc)
        if not _valid_price(price):
            try:
                history = ticker.history(period="5d", auto_adjust=False)
                closes = history["Close"].dropna() if "Close" in history else []
                if len(closes):
                    price = float(closes.iloc[-1])
                    currency = currency or (ticker.history_metadata or {}).get("currency")
            except Exception as exc:
                logger.debug("history failed for %s: %s", symbol, exc)
        if not _valid_price(price):
            raise QuoteError(
                f"Kürzel „{symbol}“ bei Yahoo Finance nicht gefunden – bitte das "
                "Börsenkürzel verwenden (z. B. AAPL, NVDA, SAP.DE), nicht den Firmennamen."
            )
        return float(price), str(currency or "USD")

    def _rate(self, currency: str) -> float:
        if currency not in self._fx:
            self._fx[currency] = self._last(f"{currency}EUR=X")[0]
        return self._fx[currency]

    def __call__(self, symbol: str) -> float:
        if symbol not in self._prices:
            price, currency = self._last(symbol)
            if currency in ("GBp", "GBX"):  # London quotes in pence
                price, currency = price / 100, "GBP"
            self._prices[symbol] = price * self._rate(currency.upper())
        return self._prices[symbol]


# --- Storage --------------------------------------------------------------


class DuelStore:
    """Accounts, positions, trades, equity history and AI turns in SQLite."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS duel_accounts (
                    player TEXT PRIMARY KEY,
                    cash REAL NOT NULL,
                    start_cash REAL NOT NULL,
                    started_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS duel_positions (
                    player TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    shares REAL NOT NULL,
                    cost REAL NOT NULL,
                    PRIMARY KEY (player, symbol)
                );
                CREATE TABLE IF NOT EXISTS duel_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    rating TEXT,
                    report_id INTEGER,
                    reason TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS duel_equity (
                    player TEXT NOT NULL,
                    day TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (player, day)
                );
                CREATE TABLE IF NOT EXISTS duel_turns (
                    turn_date TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    scouting TEXT,
                    analyses TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
        self.ensure_accounts()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    # Accounts -------------------------------------------------------------

    def ensure_accounts(self) -> None:
        cash = start_cash()
        with self._connect() as connection:
            for player in PLAYERS:
                connection.execute(
                    "INSERT OR IGNORE INTO duel_accounts (player, cash, start_cash, started_at) "
                    "VALUES (?, ?, ?, ?)",
                    (player, cash, cash, _now()),
                )

    def reset(self) -> None:
        """Start a new game: both players back to the start cash."""
        with self._connect() as connection:
            for table in ("duel_accounts", "duel_positions", "duel_trades", "duel_equity", "duel_turns"):
                connection.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names
        self.ensure_accounts()

    def account(self, player: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT player, cash, start_cash, started_at FROM duel_accounts WHERE player = ?",
                (player,),
            ).fetchone()
        return dict(row)

    def positions(self, player: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT symbol, shares, cost FROM duel_positions WHERE player = ? ORDER BY symbol",
                (player,),
            ).fetchall()
        return [dict(row) for row in rows]

    # Trading ----------------------------------------------------------------

    def trade(
        self,
        player: str,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        rating: str | None = None,
        report_id: int | None = None,
        reason: str | None = None,
    ) -> int:
        """Execute a virtual trade; raises ValueError on insufficient cash/shares."""
        if player not in PLAYERS:
            raise ValueError(f"Unknown player {player!r}")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        symbol = _canonical(symbol)
        if not (math.isfinite(shares) and shares > 0 and math.isfinite(price) and price > 0):
            raise ValueError("Stückzahl und Kurs müssen positiv sein.")
        amount = shares * price
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cash = connection.execute(
                "SELECT cash FROM duel_accounts WHERE player = ?", (player,)
            ).fetchone()["cash"]
            row = connection.execute(
                "SELECT shares, cost FROM duel_positions WHERE player = ? AND symbol = ?",
                (player, symbol),
            ).fetchone()
            held, cost = (row["shares"], row["cost"]) if row else (0.0, 0.0)
            if side == "buy":
                if amount > cash + 1e-6:
                    raise ValueError(f"Nicht genug Spielgeld ({cash:,.2f} € verfügbar).")
                cash -= amount
                held, cost = held + shares, cost + amount
            else:
                if shares > held + 1e-9:
                    raise ValueError(f"Nur {held:,.6g} Stück {symbol} im Depot.")
                cash += amount
                cost = cost * (1 - shares / held) if held else 0.0
                held -= shares
            connection.execute(
                "UPDATE duel_accounts SET cash = ? WHERE player = ?", (max(cash, 0.0), player)
            )
            if held <= 1e-9:
                connection.execute(
                    "DELETE FROM duel_positions WHERE player = ? AND symbol = ?", (player, symbol)
                )
            else:
                connection.execute(
                    """
                    INSERT INTO duel_positions (player, symbol, shares, cost) VALUES (?, ?, ?, ?)
                    ON CONFLICT(player, symbol) DO UPDATE SET shares = excluded.shares,
                        cost = excluded.cost
                    """,
                    (player, symbol, held, cost),
                )
            cursor = connection.execute(
                """
                INSERT INTO duel_trades
                    (player, symbol, side, shares, price, amount, rating, report_id, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (player, symbol, side, shares, price, amount, rating, report_id, reason, _now()),
            )
            connection.commit()
            return int(cursor.lastrowid)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def trades(self, player: str | None = None, limit: int = 200) -> list[dict]:
        query = "SELECT * FROM duel_trades"
        params: tuple = ()
        if player:
            query += " WHERE player = ?"
            params = (player,)
        query += " ORDER BY id DESC LIMIT ?"
        with self._connect() as connection:
            rows = connection.execute(query, (*params, limit)).fetchall()
        return [dict(row) for row in rows]

    # Valuation --------------------------------------------------------------

    def valuation(self, player: str, quote: Callable[[str], float]) -> dict:
        """Cash, positions with current value and the total equity in EUR."""
        account = self.account(player)
        positions = []
        for position in self.positions(player):
            try:
                price = quote(position["symbol"])
            except Exception as exc:
                logger.warning("No quote for %s: %s", position["symbol"], exc)
                price = position["cost"] / position["shares"]
            value = position["shares"] * price
            positions.append({**position, "price": price, "value": value,
                              "pnl": value - position["cost"]})
        total = account["cash"] + sum(p["value"] for p in positions)
        return {
            **account,
            "positions": positions,
            "total": total,
            "return": total / account["start_cash"] - 1,
        }

    def record_equity(self, quote: Callable[[str], float], day: date | None = None) -> None:
        day_text = (day or date.today()).isoformat()
        with self._connect() as connection:
            for player in PLAYERS:
                value = self.valuation(player, quote)["total"]
                connection.execute(
                    """
                    INSERT INTO duel_equity (player, day, value) VALUES (?, ?, ?)
                    ON CONFLICT(player, day) DO UPDATE SET value = excluded.value
                    """,
                    (player, day_text, value),
                )

    def has_equity(self, day: date) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM duel_equity WHERE day = ? LIMIT 1", (day.isoformat(),)
            ).fetchone()
        return row is not None

    def equity_history(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT player, day, value FROM duel_equity ORDER BY day"
            ).fetchall()
        return [dict(row) for row in rows]

    # AI turns ---------------------------------------------------------------

    def claim_turn(
        self,
        turn_date: str,
        max_attempts: int = 3,
        retry_after: timedelta = timedelta(minutes=30),
        stale_after: timedelta = timedelta(hours=6),
    ) -> bool:
        now = datetime.now(timezone.utc)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, attempts, updated_at FROM duel_turns WHERE turn_date = ?",
                (turn_date,),
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
                INSERT INTO duel_turns (turn_date, status, attempts, updated_at)
                VALUES (?, 'running', 1, ?)
                ON CONFLICT(turn_date) DO UPDATE SET status = 'running',
                    attempts = attempts + 1, error = NULL, updated_at = excluded.updated_at
                """,
                (turn_date, now.isoformat()),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def update_turn(self, turn_date: str, **fields) -> None:
        allowed = {"status", "scouting", "analyses", "error"}
        values = {k: (json.dumps(v) if k in ("scouting", "analyses") else v)
                  for k, v in fields.items() if k in allowed}
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE duel_turns SET {assignments} WHERE turn_date = ?",  # noqa: S608
                (*values.values(), turn_date),
            )

    def turns(self, limit: int = 30) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM duel_turns ORDER BY turn_date DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            turn = dict(row)
            for key in ("scouting", "analyses"):
                turn[key] = json.loads(turn[key]) if turn[key] else None
            result.append(turn)
        return result


# --- Scouting -------------------------------------------------------------


def screener_candidates(per_screen: int = 15) -> list[dict]:
    """Equity candidates from several Yahoo screeners, de-duplicated."""
    import yfinance as yf

    candidates: dict[str, dict] = {}
    for screen in SCREENERS:
        try:
            quotes = yf.screen(screen, count=per_screen).get("quotes", [])
        except Exception as exc:
            logger.warning("Screener %s failed: %s", screen, exc)
            continue
        for quote in quotes:
            symbol = quote.get("symbol")
            if quote.get("quoteType") not in (None, "EQUITY") or not symbol or not is_yahoo_safe(symbol):
                continue
            entry = candidates.setdefault(symbol, {
                "symbol": symbol,
                "name": quote.get("shortName") or quote.get("longName") or symbol,
                "price": quote.get("regularMarketPrice"),
                "currency": quote.get("currency"),
                "change_pct": quote.get("regularMarketChangePercent"),
                "change_52w_pct": quote.get("fiftyTwoWeekChangePercent"),
                "pe": quote.get("trailingPE"),
                "market_cap": quote.get("marketCap"),
                "screens": [],
            })
            entry["screens"].append(screen)
    return list(candidates.values())


def _scout_prompt(candidates: list[dict], holdings: list[str], picks: int) -> str:
    lines = [
        "Du bist ein KI-Trader in einem Spiel gegen einen Menschen. Beide starten mit gleichem "
        "Spielgeld; wer mehr Rendite macht, gewinnt.",
        f"Wähle aus den Kandidaten bis zu {picks} Aktien, die du heute genauer analysieren "
        "lassen willst (Chancen auf Kursgewinn). Bereits gehalten: "
        + (", ".join(holdings) or "nichts") + ".",
        'Antworte NUR mit einem JSON-Array: [{"symbol": "...", "reason": "<ein Satz auf Deutsch>"}]',
        "",
        "symbol | name | kurs | änderung heute % | 52W % | KGV | marktkap. | screener",
    ]
    for c in candidates:
        def fmt(value, pattern="{:.1f}"):
            return pattern.format(value) if isinstance(value, (int, float)) else "-"
        lines.append(
            f"{c['symbol']} | {c['name']} | {fmt(c['price'], '{:.2f}')} {c.get('currency') or ''} | "
            f"{fmt(c['change_pct'])} | {fmt(c['change_52w_pct'])} | {fmt(c['pe'])} | "
            f"{fmt((c['market_cap'] or 0) / 1e9, '{:.1f}')} Mrd | {','.join(c['screens'])}"
        )
    return "\n".join(lines)


def parse_picks(text: str, allowed: set[str], limit: int) -> list[dict]:
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    picks = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if symbol in allowed and symbol not in {p["symbol"] for p in picks}:
            picks.append({"symbol": symbol, "reason": str(item.get("reason", "")).strip()})
        if len(picks) >= limit:
            break
    return picks


def llm_scout(candidates: list[dict], holdings: list[str], config: dict, picks: int) -> list[dict]:
    """Let the quick-think model choose today's analysis candidates."""
    from tradingagents.llm_clients.factory import create_llm_client

    llm = create_llm_client(
        provider=config["llm_provider"],
        model=config["quick_think_llm"],
        base_url=config.get("backend_url"),
    ).get_llm()
    answer = llm.invoke(_scout_prompt(candidates, holdings, picks))
    content = answer.content if hasattr(answer, "content") else answer
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return parse_picks(str(content), {c["symbol"] for c in candidates}, picks)


# --- Trading decisions ------------------------------------------------------


def decision_excerpt(report_markdown: str, limit: int = 1200) -> str:
    """The Portfolio Manager's final decision from a full report."""
    marker = "## V. Portfolio Manager Decision"
    text = report_markdown.split(marker, 1)[1] if marker in report_markdown else report_markdown
    text = text.replace("### Portfolio Manager", "").strip()
    return text[:limit] + (" …" if len(text) > limit else "")


def plan_trades(rating: str, shares_held: float, price: float, total_equity: float, cash: float) -> tuple[str, float] | None:
    """(side, shares) to move a position towards the rating's target, or None."""
    target = TARGET_WEIGHTS.get(rating)
    current_value = shares_held * price
    if target is None:
        return None
    if target == "half":
        sell_shares = shares_held / 2
        return ("sell", sell_shares) if sell_shares * price >= MIN_TRADE_EUR else None
    target_value = total_equity * target
    delta = target_value - current_value
    if delta > 0:
        amount = min(delta, cash)
        return ("buy", amount / price) if amount >= MIN_TRADE_EUR else None
    if target == 0.0 and shares_held > 0:
        return ("sell", shares_held)
    if -delta >= MIN_TRADE_EUR:
        return ("sell", min(shares_held, -delta / price))
    return None


def run_ai_turn(
    store,
    duel: DuelStore,
    config_factory: Callable[[], dict],
    turn_date: date,
    analyze: Callable[..., tuple[int, str, str]] | None = None,
    scout: Callable[[list[dict], list[str], dict, int], list[dict]] = llm_scout,
    candidates_fn: Callable[[], list[dict]] = screener_candidates,
    quote: Callable[[str], float] | None = None,
    stop: threading.Event | None = None,
) -> bool:
    """Play the AI's turn for ``turn_date``; False if it was not due/claimable."""
    from tradingagents.daily_reports import missing_api_key, run_and_store_analysis

    analyze = analyze or run_and_store_analysis
    quote = quote or EurQuotes()
    day = turn_date.isoformat()
    config = config_factory()
    if missing_api_key(config["llm_provider"]):
        return False
    if not duel.claim_turn(day):
        return False
    try:
        holdings = [p["symbol"] for p in duel.positions("ai")]
        candidates = candidates_fn()
        try:
            picks = scout(candidates, holdings, config, max_picks()) if candidates else []
        except Exception as exc:
            logger.warning("AI scouting failed, falling back to most active: %s", exc)
            picks = [{"symbol": c["symbol"], "reason": "Meistgehandelt (Fallback)"}
                     for c in candidates[: max_picks()]]
        duel.update_turn(day, scouting={"candidates": candidates[:40], "picks": picks})

        queue = list(dict.fromkeys([*holdings, *(p["symbol"] for p in picks)]))[: max_analyses()]
        analyses = []
        for symbol in queue:
            if stop is not None and stop.is_set():
                break
            entry = {"symbol": symbol}
            try:
                existing = store.find_report(symbol, day)
                if existing:
                    report_id, rating = existing["id"], existing["decision"]
                    markdown = existing["report_markdown"]
                else:
                    report_id, rating, markdown = analyze(
                        store, symbol, day, config, session_dir="duel"
                    )
                entry.update(report_id=report_id, rating=rating)
                entry["trade"] = _apply_rating(duel, quote, symbol, rating, report_id, markdown)
            except Exception as exc:
                logger.exception("AI analysis of %s failed", symbol)
                entry["error"] = str(exc)[:300]
            analyses.append(entry)
            duel.update_turn(day, analyses=analyses)

        duel.record_equity(quote)
        duel.update_turn(day, status="done")
        return True
    except Exception as exc:
        logger.exception("AI turn failed")
        duel.update_turn(day, status="error", error=str(exc)[:500])
        return False


def _apply_rating(duel: DuelStore, quote, symbol: str, rating: str, report_id: int, markdown: str):
    price = quote(symbol)
    valuation = duel.valuation("ai", quote)
    held = next((p["shares"] for p in valuation["positions"] if p["symbol"] == symbol), 0.0)
    plan = plan_trades(rating, held, price, valuation["total"], valuation["cash"])
    if plan is None:
        return None
    side, shares = plan
    duel.trade("ai", symbol, side, shares, price, rating=rating, report_id=report_id,
               reason=decision_excerpt(markdown))
    return {"side": side, "shares": shares, "price": price, "amount": shares * price}


def daily_duel_job(store, duel: DuelStore, config_factory: Callable[[], dict]):
    """Scheduler job: the AI's daily turn plus one equity snapshot per day."""

    def job(turn_date: date, stop: threading.Event) -> None:
        quote = EurQuotes()
        run_ai_turn(store, duel, config_factory, turn_date, quote=quote, stop=stop)
        if not duel.has_equity(date.today()):
            duel.record_equity(quote)

    return job
