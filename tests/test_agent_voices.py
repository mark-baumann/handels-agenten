"""The dashboard shows each agent's statement parsed from a stored report."""

from __future__ import annotations

import pytest

from tradingagents.agent_voices import excerpt, parse_voices
from tradingagents.reporting import write_report_tree

pytestmark = pytest.mark.unit


def _state():
    return {
        "market_report": "# Technische Analyse\n\nDer Kurs liegt **über** dem 50-Tage-Schnitt.\nRSI 58.\n\n| a | b |",
        "news_report": "Apple kürzt iPhone-Bestellungen.",
        "investment_debate_state": {
            "bull_history": "Bull Analyst: Starke Services-Marge.",
            "bear_history": "Bear Analyst: Bewertung zu hoch.",
            "judge_decision": "Wir folgen dem Bull. Rating: Overweight",
        },
        "trader_investment_plan": "Kaufe 10 % Position. FINAL TRANSACTION PROPOSAL: **BUY**",
        "risk_debate_state": {
            "aggressive_history": "Aggressiv: mehr kaufen.",
            "conservative_history": "Konservativ: Stop-Loss setzen.",
            "neutral_history": "Neutral: halbe Position.",
            "judge_decision": "**Rating: Overweight**\n\nSchrittweise aufbauen.",
        },
    }


def test_parse_voices_from_real_report_tree(tmp_path):
    markdown = write_report_tree(_state(), "AAPL", tmp_path).read_text(encoding="utf-8")
    voices = parse_voices(markdown)
    assert [v.agent for v in voices] == [
        "Market Analyst", "News Analyst", "Bull Researcher", "Bear Researcher",
        "Research Manager", "Trader", "Aggressive Analyst", "Conservative Analyst",
        "Neutral Analyst", "Portfolio Manager",
    ]
    by_name = {v.agent: v for v in voices}
    assert by_name["Bear Researcher"].text == "Bear Analyst: Bewertung zu hoch."
    assert by_name["Portfolio Manager"].team == "decision"
    assert "Schrittweise aufbauen." in by_name["Portfolio Manager"].text
    # Section headings of the next team never leak into an agent's text.
    assert "## III" not in by_name["Research Manager"].text


def test_excerpt_skips_headings_and_markup():
    text = "# Technische Analyse\n\nDer Kurs liegt **über** dem Schnitt.\nRSI 58.\n\n| a | b |"
    assert excerpt(text) == "Der Kurs liegt über dem Schnitt. RSI 58."
    long = "Wort " * 200
    short = excerpt(long, limit=50)
    assert short.endswith(" …") and len(short) <= 52


def test_parse_voices_handles_empty_and_unknown_sections():
    assert parse_voices("") == []
    assert parse_voices("### Something Else\ntext") == []


def test_excerpt_drops_rating_line_and_speaker_tag():
    assert excerpt("**Rating: Overweight**\n\nSchrittweise aufbauen.") == "Schrittweise aufbauen."
    assert excerpt("Bull Analyst: Die Marge wächst.") == "Die Marge wächst."


def test_escape_markdown_dollars():
    from tradingagents.agent_voices import escape_markdown

    assert escape_markdown("bei 172 $ und 188 $") == "bei 172 \\$ und 188 \\$"
