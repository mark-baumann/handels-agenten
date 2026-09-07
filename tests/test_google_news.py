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
