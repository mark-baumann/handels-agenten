"""Tests for turning graph states into live dashboard events."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.live_progress import ProgressTracker, stage_progress

pytestmark = pytest.mark.unit


def test_tracker_emits_only_new_events():
    tracker = ProgressTracker()
    call = AIMessage(content="", tool_calls=[{"name": "get_stock_data", "args": {"symbol": "AAPL"}, "id": "c1"}])
    state = {"messages": [HumanMessage(content="AAPL"), call]}
    events = tracker.update(state)
    assert [e.kind for e in events] == ["tool"]
    assert "get_stock_data" in events[0].title and "AAPL" in events[0].body

    state["messages"].append(ToolMessage(content="x" * 1000, tool_call_id="c1", name="get_stock_data"))
    state["messages"].append(AIMessage(content="Der Trend ist positiv."))
    state["market_report"] = "# Markt"
    events = tracker.update(state)
    assert [e.kind for e in events] == ["data", "thought", "report"]
    assert events[0].body.endswith("…") and len(events[0].body) < 700
    assert tracker.update(state) == []


def test_tracker_reports_debate_turns_and_reasoning():
    tracker = ProgressTracker()
    reasoning = AIMessage(content="ok", additional_kwargs={"reasoning_content": "hmm"})
    events = tracker.update({
        "messages": [reasoning],
        "investment_debate_state": {"current_response": "Bull Analyst: kaufen"},
    })
    assert [e.title for e in events][:2] == ["🧠 Denkt nach", "💭 Agent"]
    assert events[-1].kind == "debate"
    events = tracker.update({"investment_debate_state": {"current_response": "Bear Analyst: nein"}})
    assert [e.body for e in events] == ["Bear Analyst: nein"]


def test_stage_progress():
    assert stage_progress({}) == (0.0, "Marktanalyse")
    fraction, upcoming = stage_progress({
        "market_report": "x",
        "sentiment_report": "x",
        "investment_debate_state": {"judge_decision": "x"},
    })
    assert fraction == pytest.approx(3 / 7)
    assert upcoming == "Nachrichten"


def test_propagate_streams_states_to_on_chunk(tmp_path):
    import functools
    from unittest.mock import MagicMock

    from tradingagents.agents.utils.memory import TradingMemoryLog
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    partial = {"market_report": "m"}
    final = {
        "final_trade_decision": "Rating: Buy", "company_of_interest": "NVDA",
        "trade_date": "2026-01-10", "market_report": "m", "sentiment_report": "",
        "news_report": "", "fundamentals_report": "",
        "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "",
                                    "current_response": "", "judge_decision": ""},
        "investment_plan": "", "trader_investment_plan": "",
        "risk_debate_state": {"aggressive_history": "", "conservative_history": "",
                              "neutral_history": "", "history": "", "judge_decision": ""},
    }
    graph = MagicMock()
    graph.memory_log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    graph.log_states_dict = {}
    graph.debug = False
    graph.config = {"results_dir": str(tmp_path)}
    graph.graph.stream.return_value = iter([partial, final])
    graph.propagator.get_graph_args.return_value = {}
    graph.signal_processor.process_signal.return_value = "Buy"
    graph._run_graph = functools.partial(TradingAgentsGraph._run_graph, graph)

    seen = []
    state, _ = TradingAgentsGraph.propagate(graph, "NVDA", "2026-01-10", on_chunk=seen.append)
    assert seen == [partial, final]
    assert state is final
    graph.graph.invoke.assert_not_called()
