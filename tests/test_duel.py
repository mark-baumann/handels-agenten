"""Tests for the "Ich vs. KI" paper-trading duel."""

from __future__ import annotations

from datetime import date

import pytest

from tradingagents import duel as duel_mod
from tradingagents.duel import DuelStore, decision_excerpt, parse_picks, plan_trades, run_ai_turn
from tradingagents.portfolio import PortfolioStore

pytestmark = pytest.mark.unit

_CONFIG = {"llm_provider": "openai", "quick_think_llm": "q"}
_PRICES = {"AAPL": 100.0, "NVDA": 50.0, "T": 20.0}


def _stores(tmp_path, monkeypatch, cash=10_000):
    monkeypatch.setenv("TRADINGAGENTS_DUEL_START_CASH", str(cash))
    path = tmp_path / "portfolio.sqlite3"
    return PortfolioStore(path), DuelStore(path)


def test_both_players_start_with_equal_cash(tmp_path, monkeypatch):
    _, duel = _stores(tmp_path, monkeypatch, cash=5000)
    assert duel.account("me")["cash"] == duel.account("ai")["cash"] == 5000


def test_trade_updates_cash_positions_and_cost(tmp_path, monkeypatch):
    _, duel = _stores(tmp_path, monkeypatch)
    duel.trade("me", "aapl", "buy", 10, 100)
    duel.trade("me", "AAPL", "buy", 10, 120)
    assert duel.account("me")["cash"] == pytest.approx(10_000 - 2200)
    assert duel.positions("me") == [{"symbol": "AAPL", "shares": 20, "cost": 2200}]
    duel.trade("me", "AAPL", "sell", 5, 130)
    position = duel.positions("me")[0]
    assert position["shares"] == 15 and position["cost"] == pytest.approx(1650)
    duel.trade("me", "AAPL", "sell", 15, 130)
    assert duel.positions("me") == []
    assert duel.account("me")["cash"] == pytest.approx(10_000 - 2200 + 20 * 130)
    assert [t["side"] for t in duel.trades("me")] == ["sell", "sell", "buy", "buy"]
    assert duel.positions("ai") == []


@pytest.mark.parametrize(
    ("side", "shares", "message"),
    [("buy", 200, "Spielgeld"), ("sell", 1, "Nur 0")],
)
def test_trade_rejects_overspending_and_short_selling(tmp_path, monkeypatch, side, shares, message):
    _, duel = _stores(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=message):
        duel.trade("me", "AAPL", side, shares, 100)
    assert duel.trades() == [] and duel.account("me")["cash"] == 10_000


def test_valuation_and_equity_history(tmp_path, monkeypatch):
    _, duel = _stores(tmp_path, monkeypatch)
    duel.trade("ai", "NVDA", "buy", 100, 40)
    valuation = duel.valuation("ai", _PRICES.__getitem__)
    assert valuation["total"] == pytest.approx(6000 + 5000)
    assert valuation["return"] == pytest.approx(0.1)
    assert valuation["positions"][0]["pnl"] == pytest.approx(1000)
    duel.record_equity(_PRICES.__getitem__, date(2026, 10, 9))
    duel.record_equity(_PRICES.__getitem__, date(2026, 10, 9))  # upsert, no duplicate
    assert {(r["player"], r["value"]) for r in duel.equity_history()} == {("me", 10_000), ("ai", 11_000)}
    assert duel.has_equity(date(2026, 10, 9))


def test_reset_starts_a_new_game(tmp_path, monkeypatch):
    _, duel = _stores(tmp_path, monkeypatch)
    duel.trade("me", "AAPL", "buy", 1, 100)
    duel.reset()
    assert duel.trades() == [] and duel.account("me")["cash"] == 10_000


@pytest.mark.parametrize(
    ("rating", "held", "expected"),
    [
        ("Buy", 0, ("buy", 20.0)),          # 20% of 10k at 100 €
        ("Overweight", 0, ("buy", 10.0)),
        ("Hold", 5, None),
        ("Underweight", 10, ("sell", 5.0)),
        ("Sell", 7, ("sell", 7.0)),
        ("Sell", 0, None),
        ("Buy", 20, None),                  # already at target
        ("Overweight", 30, ("sell", 20.0)),  # above target → trim to 10%
    ],
)
def test_plan_trades_moves_towards_rating_target(rating, held, expected):
    assert plan_trades(rating, held, 100.0, 10_000, 10_000) == expected


def test_plan_trades_respects_cash_and_minimum():
    assert plan_trades("Buy", 0, 100.0, 10_000, 500) == ("buy", 5.0)
    assert plan_trades("Buy", 0, 100.0, 10_000, 10) is None


def test_parse_picks_only_accepts_known_candidates():
    text = 'Gern: [{"symbol": "nvda", "reason": "KI-Boom"}, {"symbol": "FAKE"}, {"symbol": "NVDA"}, {"symbol": "T", "reason": "billig"}]'
    assert parse_picks(text, {"NVDA", "T"}, 5) == [
        {"symbol": "NVDA", "reason": "KI-Boom"}, {"symbol": "T", "reason": "billig"},
    ]
    assert parse_picks(text, {"NVDA", "T"}, 1) == [{"symbol": "NVDA", "reason": "KI-Boom"}]
    assert parse_picks("nope", {"NVDA"}, 3) == []


def test_decision_excerpt_extracts_portfolio_manager_section():
    md = "# R\n## IV. Risk\nx\n## V. Portfolio Manager Decision\n\n### Portfolio Manager\nRating: Buy\nWeil."
    assert decision_excerpt(md) == "Rating: Buy\nWeil."


def _fake_analyze(ratings, calls):
    def analyze(store, symbol, day, config, session_dir):
        calls.append(symbol)
        rating = ratings[symbol]
        markdown = f"## V. Portfolio Manager Decision\nRating: {rating} for {symbol}"
        return store.save_report(symbol, day, rating, markdown), rating, markdown
    return analyze


def test_ai_turn_scouts_analyses_and_trades_once(tmp_path, monkeypatch):
    store, duel = _stores(tmp_path, monkeypatch)
    calls = []
    candidates = [{"symbol": "NVDA"}, {"symbol": "T"}]
    kwargs = {
        "analyze": _fake_analyze({"NVDA": "Buy", "T": "Hold"}, calls),
        "scout": lambda c, h, cfg, n: [{"symbol": "NVDA", "reason": "r"}, {"symbol": "T", "reason": "r"}],
        "candidates_fn": lambda: candidates,
        "quote": _PRICES.__getitem__,
    }
    assert run_ai_turn(store, duel, lambda: _CONFIG, date(2026, 10, 9), **kwargs)
    assert calls == ["NVDA", "T"]
    assert duel.positions("ai") == [{"symbol": "NVDA", "shares": 40.0, "cost": 2000.0}]
    trade = duel.trades("ai")[0]
    assert trade["rating"] == "Buy" and trade["report_id"] and "Rating: Buy" in trade["reason"]
    turn = duel.turns()[0]
    assert turn["status"] == "done" and len(turn["analyses"]) == 2
    assert turn["scouting"]["picks"][0]["symbol"] == "NVDA"
    # Same day again: already played.
    assert not run_ai_turn(store, duel, lambda: _CONFIG, date(2026, 10, 9), **kwargs)
    assert calls == ["NVDA", "T"]
    assert duel.has_equity(date.today())


def test_ai_turn_reuses_existing_report_and_reviews_holdings_first(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_DUEL_MAX_ANALYSES", "1")
    store, duel = _stores(tmp_path, monkeypatch)
    duel.trade("ai", "AAPL", "buy", 10, 100)
    store.save_report("AAPL", "2026-10-09", "Sell", "## V. Portfolio Manager Decision\nRaus.")
    calls = []
    run_ai_turn(
        store, duel, lambda: _CONFIG, date(2026, 10, 9),
        analyze=_fake_analyze({}, calls),
        scout=lambda c, h, cfg, n: [{"symbol": "NVDA", "reason": "r"}],
        candidates_fn=lambda: [{"symbol": "NVDA"}],
        quote=_PRICES.__getitem__,
    )
    assert calls == []  # budget of 1 spent on the holding, whose report already existed
    assert duel.positions("ai") == []
    assert duel.trades("ai")[0]["reason"] == "Raus."


def test_ai_turn_falls_back_when_scouting_fails(tmp_path, monkeypatch):
    store, duel = _stores(tmp_path, monkeypatch)
    calls = []

    def broken_scout(*_):
        raise RuntimeError("429")

    run_ai_turn(
        store, duel, lambda: _CONFIG, date(2026, 10, 9),
        analyze=_fake_analyze({"T": "Hold"}, calls),
        scout=broken_scout,
        candidates_fn=lambda: [{"symbol": "T"}],
        quote=_PRICES.__getitem__,
    )
    assert calls == ["T"]
    assert duel.turns()[0]["scouting"]["picks"][0]["reason"].endswith("(Fallback)")


def test_ai_turn_waits_for_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    store, duel = _stores(tmp_path, monkeypatch)
    config = {"llm_provider": "ollama_cloud", "quick_think_llm": "q"}
    assert not run_ai_turn(store, duel, lambda: config, date(2026, 10, 9),
                           candidates_fn=lambda: [], quote=_PRICES.__getitem__)
    assert duel.turns() == []


def test_eur_quotes_convert_currency_and_pence(monkeypatch):
    raw = {"SAP.DE": (200.0, "EUR"), "AAPL": (100.0, "USD"), "VOD.L": (80.0, "GBp"),
           "USDEUR=X": (0.9, "EUR"), "GBPEUR=X": (1.2, "EUR")}
    monkeypatch.setattr(duel_mod.EurQuotes, "_last", staticmethod(lambda s: raw[s]))
    quotes = duel_mod.EurQuotes()
    assert quotes("SAP.DE") == 200.0
    assert quotes("AAPL") == pytest.approx(90.0)
    assert quotes("VOD.L") == pytest.approx(0.96)


class _FakeTicker:
    def __init__(self, fast=None, closes=(), currency="USD"):
        self._fast, self._closes = fast, closes
        self.history_metadata = {"currency": currency}

    @property
    def fast_info(self):
        if self._fast is None:
            raise KeyError("currentTradingPeriod")
        return self._fast

    def history(self, period, auto_adjust):
        import pandas as pd

        return pd.DataFrame({"Close": list(self._closes)})


def test_last_falls_back_to_history_when_fast_info_breaks(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda s: _FakeTicker(closes=[10.0, 12.5], currency="EUR"))
    assert duel_mod.EurQuotes._last("RHM.DE") == (12.5, "EUR")


def test_unknown_symbol_gives_readable_error(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda s: _FakeTicker())
    with pytest.raises(duel_mod.QuoteError, match="nicht gefunden.*Börsenkürzel"):
        duel_mod.EurQuotes()("apple")
