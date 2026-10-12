"""Google News scraper must degrade gracefully and filter by date window.

The scraper is keyless and self-contained; these tests cover title cleaning,
date-window filtering (no look-ahead into a historical window), and the
graceful placeholder on fetch/parse failure.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingagents.dataflows import google_news

_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>Nvidia beats earnings - Reuters</title>
      <link>https://news.google.com/rss/articles/abc</link>
      <pubDate>Sat, 05 Sep 2026 22:45:04 GMT</pubDate>
      <source>Reuters</source>
    </item>
    <item>
      <title>Markets rally on Fed signal</title>
      <link>https://news.google.com/rss/articles/def</link>
      <pubDate>Fri, 04 Sep 2026 20:00:00 GMT</pubDate>
      <source>Bloomberg</source>
    </item>
  </channel>
</rss>
"""


@pytest.mark.unit
def test_clean_title_strips_source_suffix():
    assert google_news._clean_title("Nvidia beats earnings - Reuters") == "Nvidia beats earnings"
    assert google_news._clean_title("No suffix here") == "No suffix here"


@pytest.mark.unit
def test_parse_pub_date_returns_aware_datetime():
    dt = google_news._parse_pub_date("Sat, 05 Sep 2026 22:45:04 GMT")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt == datetime(2026, 9, 5, 22, 45, 4, tzinfo=timezone.utc)


@pytest.mark.unit
def test_parse_pub_date_invalid_returns_none():
    assert google_news._parse_pub_date("not a date") is None
    assert google_news._parse_pub_date("") is None


@pytest.mark.unit
def test_window_excludes_future_in_backtest():
    start = datetime(2026, 9, 1)
    end = datetime(2026, 9, 7)
    inside = datetime(2026, 9, 5, tzinfo=timezone.utc)
    future = datetime(2026, 9, 20, tzinfo=timezone.utc)
    assert google_news._in_window(inside, start, end) is True
    assert google_news._in_window(future, start, end) is False


@pytest.mark.unit
def test_fetch_rss_parses_items(monkeypatch):
    def _fake_urlopen(req, timeout):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return _SAMPLE_RSS.encode("utf-8")

        return _Resp()

    monkeypatch.setattr(google_news, "urlopen", _fake_urlopen)
    articles = google_news._fetch_rss("NVDA", "de", "DE", timeout=10.0)
    assert len(articles) == 2
    assert articles[0]["title"] == "Nvidia beats earnings"
    assert articles[0]["source"] == "Reuters"
    assert articles[0]["pub_date"] is not None


@pytest.mark.unit
def test_fetch_rss_returns_empty_on_http_error(monkeypatch):
    def _raise(req, timeout):
        raise OSError("network down")

    monkeypatch.setattr(google_news, "urlopen", _raise)
    assert google_news._fetch_rss("NVDA", "de", "DE", timeout=10.0) == []


@pytest.mark.unit
def test_get_news_google_returns_placeholder_on_failure(monkeypatch):
    monkeypatch.setattr(google_news, "_fetch_rss", lambda *a, **k: [])
    result = google_news.get_news_google("NVDA", "2026-09-01", "2026-09-07")
    assert "No news found" in result


def _art(title, day, source="Reuters"):
    return {"title": title, "source": source, "link": f"https://x/{title}",
            "pub_date": datetime(2026, 10, day, 12, tzinfo=timezone.utc)}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("title", "noise"),
    [
        ("AAPL Oct 2026 355.000 put (AAPL261012P00355000) Aktienkurs, Nachrichten, Kurs und Verlauf", True),
        ("AAPL Dec 2026 355.000 call (AAPL261218C00355000) Kursverlauf und Daten", True),
        ("SAP SE Aktie (716460) - Kurs Xetra", True),
        ("Apple stock falls on report of iPhone 18 Pro order cuts", False),
        ("Analysts call Nvidia a buy after earnings", False),
    ],
)
def test_noise_filter_drops_quote_pages_only(title, noise):
    assert google_news._is_noise(title) is noise


@pytest.mark.unit
def test_ticker_search_uses_company_name_market_language_and_date_window(monkeypatch):
    calls = []

    def fake_fetch(query, lang, region, timeout):
        calls.append((query, lang, region))
        return [_art("AAPL Oct 2026 355.000 put (AAPL261012P00355000)", 8),
                _art("Apple stock falls", 9), _art("Old Apple story", 1)]

    monkeypatch.setattr(google_news, "_fetch_rss", fake_fetch)
    monkeypatch.setattr(google_news, "_company_name", lambda s: "Apple")
    articles = google_news.ticker_news_articles("AAPL", "2026-10-02", "2026-10-09")
    assert calls == [('"Apple" stock after:2026-10-02 before:2026-10-10', "en-US", "US")]
    assert [a["title"] for a in articles] == ["Apple stock falls"]

    calls.clear()
    monkeypatch.setattr(google_news, "_company_name", lambda s: "SAP")
    google_news.ticker_news_articles("SAP.DE", "2026-10-02", "2026-10-09")
    assert calls[0] == ('"SAP" Aktie after:2026-10-02 before:2026-10-10', "de", "DE")


@pytest.mark.unit
def test_ticker_search_falls_back_to_symbol(monkeypatch):
    calls = []

    def fake_fetch(query, lang, region, timeout):
        calls.append(query)
        return [] if query.startswith('"') else [_art("ACME beats", 5)]

    monkeypatch.setattr(google_news, "_fetch_rss", fake_fetch)
    monkeypatch.setattr(google_news, "_company_name", lambda s: "Acme")
    articles = google_news.ticker_news_articles("ACME", "2026-10-02", "2026-10-09")
    assert calls == ['"Acme" stock after:2026-10-02 before:2026-10-10',
                     "ACME stock after:2026-10-02 before:2026-10-10"]
    assert articles[0]["title"] == "ACME beats"


@pytest.mark.unit
def test_company_name_strips_legal_suffix(monkeypatch):
    import tradingagents.agents.utils.agent_utils as agent_utils

    for full, short in [("Apple Inc.", "Apple"), ("SAP SE", "SAP"),
                        ("NVIDIA Corporation", "NVIDIA"), ("Alphabet Inc. Class A", "Alphabet")]:
        monkeypatch.setattr(agent_utils, "resolve_instrument_identity",
                            lambda s, full=full: {"company_name": full})
        assert google_news._company_name("X") == short


@pytest.mark.unit
def test_global_news_queries_each_topic_and_merges(monkeypatch):
    from tradingagents.dataflows import config as dataflow_config

    monkeypatch.setattr(dataflow_config, "get_config",
                        lambda: {"global_news_queries": ["Fed rates", "oil supply"]})
    by_topic = {
        "Fed rates": [_art("Fed holds", 8), _art("Fed minutes", 7), _art("Shared story", 6)],
        "oil supply": [_art("Oil jumps", 8), _art("Shared story", 6)],
    }
    seen = []

    def fake_fetch(query, lang, region, timeout):
        topic = query.split(" after:")[0]
        seen.append(query)
        return by_topic[topic]

    monkeypatch.setattr(google_news, "_fetch_rss", fake_fetch)
    articles = google_news.global_news_articles("2026-10-09", 7, 10)
    assert sorted(seen) == ["Fed rates after:2026-10-02 before:2026-10-10",
                            "oil supply after:2026-10-02 before:2026-10-10"]
    assert [a["title"] for a in articles] == ["Fed holds", "Oil jumps", "Fed minutes", "Shared story"]
    text = google_news.get_global_news_google("2026-10-09", 7, 10)
    assert "Fed holds" in text and "No global news" not in text
