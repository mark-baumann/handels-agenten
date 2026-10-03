"""Report parity: the shared writer produces the report tree for the CLI and the
programmatic API alike (#1037)."""

import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree


def _state():
    return {
        "market_report": "MKT",
        "news_report": "NEWS",
        "investment_debate_state": {"judge_decision": "RM PLAN"},
        "trader_investment_plan": "TRADE",
        "risk_debate_state": {"judge_decision": "PM DECISION"},
    }


@pytest.mark.unit
def test_write_report_tree_creates_files(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    assert out.name == "complete_report.md"
    assert (tmp_path / "1_analysts" / "market.md").read_text() == "MKT"
    assert (tmp_path / "1_analysts" / "news.md").read_text() == "NEWS"
    assert (tmp_path / "2_research" / "manager.md").read_text() == "RM PLAN"
    assert (tmp_path / "3_trading" / "trader.md").read_text() == "TRADE"
    assert (tmp_path / "5_portfolio" / "decision.md").read_text() == "PM DECISION"
    complete = out.read_text()
    assert "Trading Analysis Report: AAPL" in complete
    assert "MKT" in complete and "PM DECISION" in complete


@pytest.mark.unit
def test_save_reports_explicit_path(tmp_path):
    # Unbound: with an explicit save_path, the method doesn't touch self/config.
    out = TradingAgentsGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert (tmp_path / "complete_report.md").exists()
    assert out == tmp_path / "complete_report.md"


@pytest.mark.unit
def test_save_reports_defaults_under_results_dir(tmp_path):
    mock_self = SimpleNamespace(config={"results_dir": str(tmp_path)})
    out = TradingAgentsGraph.save_reports(mock_self, _state(), "AAPL")
    assert out.exists()
    assert out.parent.parent.name == "reports"  # results_dir/reports/AAPL_<stamp>/...
    assert out.parent.name.startswith("AAPL_")


@pytest.mark.unit
def test_streamlit_save_analysis_report_persists_session_log(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))

    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("handels_agenten_streamlit_app", app_path)
    streamlit_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streamlit_app)

    monkeypatch.setitem(streamlit_app.DEFAULT_CONFIG, "results_dir", str(tmp_path))

    out_dir = streamlit_app._save_analysis_report(_state(), "nvda", date(2026, 10, 2))

    assert out_dir.exists()
    assert out_dir.parent.name == "2026-10-02"
    assert out_dir.parent.parent.name == "NVDA"
    assert out_dir.parent.parent.parent.name == "streamlit_sessions"
    assert (out_dir / "complete_report.md").exists()
