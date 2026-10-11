"""Tests for the news direction ratings shown in the dashboard."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tradingagents import news_impact
from tradingagents.portfolio import PortfolioStore

pytestmark = pytest.mark.unit


def test_parse_ratings_tolerates_prose_and_bad_items():
    text = (
        'Hier ist meine Bewertung:\n[{"id": "a", "direction": "UP", "confidence": 140, '
        '"reason": "Starke Zahlen"}, {"id": "b", "direction": "sideways"}, "x", '
        '{"id": "c", "direction": "down", "confidence": "n/a"}]'
    )
    ratings = news_impact.parse_ratings(text)
    assert ratings == {
        "a": {"direction": "up", "confidence": 100, "reason": "Starke Zahlen"},
        "c": {"direction": "down", "confidence": 0, "reason": ""},
    }
    assert news_impact.parse_ratings("keine Ahnung") == {}
    assert news_impact.parse_ratings("[not json]") == {}


def test_price_reaction_since_publication():
    prices = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09"]),
    )
    published = datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc)
    assert news_impact.price_reaction(prices, published) == pytest.approx(-0.01)
    assert news_impact.price_reaction(prices, None) is None
    assert news_impact.price_reaction(None, published) is None
    early = datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert news_impact.price_reaction(prices, early) is None


def test_ratings_are_cached_per_article(tmp_path, monkeypatch):
    store = PortfolioStore(tmp_path / "p.sqlite3")
    article = {"title": "AAPL beats", "summary": "", "publisher": "X", "link": "https://a",
               "pub_date": None}
    article["key"] = news_impact.article_key(article)
    monkeypatch.setattr(news_impact, "fetch_news", lambda symbol, limit: [dict(article)])
    calls = []

    def fake_rate(symbol, articles, config):
        calls.append([a["key"] for a in articles])
        return {a["key"]: {"direction": "up", "confidence": 80, "reason": "gut"} for a in articles}

    monkeypatch.setattr(news_impact, "rate_articles", fake_rate)
    articles, error = news_impact.news_with_impact(store, "AAPL", {"llm_provider": "x"})
    assert error is None and articles[0]["rating"]["direction"] == "up"
    articles, _ = news_impact.news_with_impact(store, "AAPL", {"llm_provider": "x"})
    assert len(calls) == 1 and articles[0]["rating"]["confidence"] == 80


def test_unrated_without_config_and_errors_are_reported(tmp_path, monkeypatch):
    store = PortfolioStore(tmp_path / "p.sqlite3")
    article = {"title": "t", "summary": "", "publisher": "X", "link": "l", "pub_date": None, "key": "k"}
    monkeypatch.setattr(news_impact, "fetch_news", lambda symbol, limit: [dict(article)])

    def boom(*_):
        raise RuntimeError("429")

    monkeypatch.setattr(news_impact, "rate_articles", boom)
    articles, error = news_impact.news_with_impact(store, "AAPL", None)
    assert error is None and articles[0]["rating"] is None
    articles, error = news_impact.news_with_impact(store, "AAPL", {"llm_provider": "x"})
    assert error == "429" and articles[0]["rating"] is None
