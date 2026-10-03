import pandas as pd
import pytest

from tradingagents.portfolio import PortfolioStore, load_portfolio_price_history


@pytest.mark.unit
def test_holdings_and_reports_survive_store_reopen(tmp_path):
    database = tmp_path / "portfolio.sqlite3"
    store = PortfolioStore(database)
    store.add_holding("aapl", 2.5, 150)
    store.add_holding("AAPL", 3, 160)
    report_id = store.save_report("aapl", "2026-10-01", "Hold", "# Full report")

    reopened = PortfolioStore(database)
    holdings = reopened.list_holdings()
    assert len(holdings) == 1
    assert holdings[0]["symbol"] == "AAPL"
    assert holdings[0]["shares"] == 3
    assert holdings[0]["average_cost"] == 160
    assert reopened.get_report(report_id)["report_markdown"] == "# Full report"
    assert reopened.list_reports()[0]["decision"] == "Hold"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbol", "shares", "average_cost"),
    [
        ("../../secrets", 1, None),
        ("AAPL", 0, None),
        ("AAPL", float("inf"), None),
        ("AAPL", 1, 0),
        ("AAPL", 1, float("nan")),
    ],
)
def test_rejects_invalid_holding_data(tmp_path, symbol, shares, average_cost):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    with pytest.raises(ValueError):
        store.add_holding(symbol, shares, average_cost)


@pytest.mark.unit
def test_remove_holding_is_idempotent(tmp_path):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    store.add_holding("AAPL", 1)

    assert store.remove_holding("AAPL")
    assert not store.remove_holding("AAPL")
    assert store.list_holdings() == []


@pytest.mark.unit
def test_load_portfolio_price_history_normalizes_series(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, auto_adjust):
            return pd.DataFrame(
                {"Close": [100.0, 105.0]},
                index=pd.to_datetime(["2026-09-01", "2026-10-01"]),
            )

    monkeypatch.setattr("tradingagents.portfolio.yf.Ticker", FakeTicker)
    history, latest, errors = load_portfolio_price_history([{"symbol": "AAPL"}])

    assert history["AAPL"].tolist() == [100.0, 105.0]
    assert latest == {"AAPL": 105.0}
    assert errors == {}


@pytest.mark.unit
def test_load_portfolio_price_history_reports_missing_prices(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period, auto_adjust):
            return pd.DataFrame()

    monkeypatch.setattr("tradingagents.portfolio.yf.Ticker", FakeTicker)
    history, latest, errors = load_portfolio_price_history([{"symbol": "UNKNOWN"}])

    assert history.empty
    assert latest == {}
    assert "UNKNOWN" in errors
