"""Tests for the daily "Entdecken" list."""

from __future__ import annotations

import pytest

from tradingagents.discovery import discover, parse_picks, scout_prompt

pytestmark = pytest.mark.unit


def _cand(symbol, change=1.0):
    return {"symbol": symbol, "name": symbol + " Inc", "price": 10.0, "currency": "USD",
            "change_pct": change, "change_52w_pct": 5.0, "pe": 20.0, "market_cap": 2e9,
            "screens": ["most_actives"]}


def test_parse_picks_only_accepts_known_candidates():
    text = ('Hier: [{"symbol": "nvda", "reason": "KI-Boom"}, {"symbol": "FAKE"}, '
            '{"symbol": "NVDA"}, {"symbol": "T", "reason": "billig"}]')
    assert parse_picks(text, {"NVDA", "T"}, 5) == [
        {"symbol": "NVDA", "reason": "KI-Boom"}, {"symbol": "T", "reason": "billig"},
    ]
    assert parse_picks(text, {"NVDA", "T"}, 1) == [{"symbol": "NVDA", "reason": "KI-Boom"}]
    assert parse_picks("nope", {"NVDA"}, 3) == []
    assert parse_picks("[not json]", {"NVDA"}, 3) == []


def test_prompt_lists_candidates_and_excludes_tracked():
    prompt = scout_prompt([_cand("NVDA")], ["AAPL"], 10)
    assert "nicht wählen): AAPL" in prompt
    assert "NVDA | NVDA Inc | 10.00 USD" in prompt


def test_discover_skips_tracked_and_merges_screener_data():
    calls = []

    def scout(candidates, tracked, config, count):
        calls.append(([c["symbol"] for c in candidates], tracked, count))
        return [{"symbol": "T", "reason": "Dividende"}]

    result = discover(["aapl"], {"llm_provider": "x"}, count=2,
                      candidates_fn=lambda: [_cand("AAPL"), _cand("T", -2.0), _cand("NVDA")],
                      scout=scout)
    assert calls == [(["T", "NVDA"], ["AAPL"], 2)]
    assert result["error"] is None
    assert result["picks"] == [{**_cand("T", -2.0), "reason": "Dividende"}]


def test_discover_falls_back_without_llm_or_on_error():
    candidates = lambda: [_cand("A"), _cand("B"), _cand("C")]  # noqa: E731
    no_llm = discover([], None, count=2, candidates_fn=candidates)
    assert [p["symbol"] for p in no_llm["picks"]] == ["A", "B"]
    assert all(p["reason"] == "" for p in no_llm["picks"])

    def broken(*_):
        raise RuntimeError("429 rate limit")

    failed = discover([], {"llm_provider": "x"}, count=1, candidates_fn=candidates, scout=broken)
    assert [p["symbol"] for p in failed["picks"]] == ["A"]
    assert "429" in failed["error"]
