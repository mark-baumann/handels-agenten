"""Keyless Google News scraper for ticker-specific and global headlines.

The trading agents already consume news through yfinance / Alpha Vantage, but
those vendors need a network round-trip per symbol and (for Alpha Vantage) an
API key. This module adds a lightweight, keyless source built on Google News'
public RSS search feed (``news.google.com/rss/search``), which returns up to 100
headlines per query with no registration and no key.

It is deliberately self-contained and mirrors the other dataflow fetchers
(``reddit.py``, ``stocktwits.py``): a short timeout, graceful degradation to a
placeholder string on any HTTP/parse failure, and a formatted plaintext return
so callers get a uniform interface regardless of whether the network call
succeeded. The Google redirect link is kept as-is — it resolves to the source
article in a browser and needs no scraping of the redirect target.

No API key required.
"""

from __future__ import annotations

import html
import http.client
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen

from .symbol_utils import crypto_base, normalize_symbol

logger = logging.getLogger(__name__)

_API = "https://news.google.com/rss/search?q={query}&hl={lang}&gl={region}&ceid={region}:{lang}"
_UA = "tradingagents/0.3 (+https://github.com/TauricResearch/TradingAgents)"

# Google News titles carry a trailing " - <source>" suffix; strip it so the
# source is reported separately instead of duplicated in the headline.
_TITLE_SOURCE_RE = re.compile(r"\s+-\s+[^-]+$")


def _clean_title(title: str) -> str:
    """Strip the trailing `` - <source>`` suffix Google News appends to titles."""
    return _TITLE_SOURCE_RE.sub("", html.unescape(title or "")).strip()


def _parse_pub_date(pub_date: str) -> datetime | None:
    """Parse an RFC-822 pubDate into a UTC-aware datetime, or None on failure."""
    if not pub_date:
        return None
    try:
        # email.utils.parsedate_to_datetime handles RFC-822/1123 and returns
        # an aware datetime; fall back to a naive parse for odd formats.
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(pub_date)
    except (TypeError, ValueError):
        return None


def _fetch_rss(query: str, lang: str, region: str, timeout: float) -> list[dict]:
    """Fetch and parse the Google News RSS feed for ``query``.

    Returns a list of article dicts (``title``, ``source``, ``link``,
    ``pub_date``). An empty list means no items or a fetch/parse failure.
    """
    url = _API.format(query=quote(query), lang=lang, region=region)
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/rss+xml"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (OSError, http.client.HTTPException) as exc:
        logger.warning("Google News fetch failed for %r: %s", query, exc)
        return []

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        logger.warning("Google News parse failed for %r: %s", query, exc)
        return []

    articles = []
    for item in root.findall(".//item"):
        title = _clean_title(item.findtext("title") or "")
        source = (item.findtext("source") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = _parse_pub_date(item.findtext("pubDate") or "")
        if not title:
            continue
        articles.append(
            {
                "title": title,
                "source": source or "Unknown",
                "link": link,
                "pub_date": pub_date,
            }
        )
    return articles


def _in_window(pub_date: datetime | None, start_dt: datetime, end_dt: datetime) -> bool:
    """Whether an article belongs in the half-open window ``[start, end + 1 day)``.

    Undated articles are kept only when the window reaches the present (live
    run), mirroring the yfinance news filter so a historical/backtest window
    cannot leak future headlines.
    """
    end = end_dt.replace(tzinfo=timezone.utc) if end_dt.tzinfo is None else end_dt.astimezone(timezone.utc)
    if pub_date is not None:
        start = start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt.astimezone(timezone.utc)
        return start <= pub_date < end + timedelta(days=1)
    return end >= datetime.now(timezone.utc) - timedelta(days=1)


def _format_articles(articles: list[dict], header: str) -> str:
    """Render a list of article dicts into a markdown-ish plaintext block."""
    lines = [header, ""]
    for a in articles:
        date_str = a["pub_date"].strftime("%Y-%m-%d") if a["pub_date"] else "?"
        lines.append(f"### {a['title']} (source: {a['source']})")
        lines.append(f"Published: {date_str}")
        if a["link"]:
            lines.append(f"Link: {a['link']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def get_news_google(
    ticker: str,
    start_date: str,
    end_date: str,
    lang: str = "de",
    region: str = "DE",
    limit: int = 20,
) -> str:
    """Retrieve recent Google News headlines for ``ticker``.

    Args:
        ticker: Ticker symbol (e.g. "AAPL", "BTC-USD").
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format.
        lang: Google News language code (default "de").
        region: Google News region code (default "DE").
        limit: Maximum number of articles to return.

    Returns:
        Formatted string containing news headlines, or a placeholder on failure.
    """
    # Crypto reaches us as a Yahoo pair (BTC-USD); search for the base so the
    # query matches headlines instead of near-nothing.
    query = crypto_base(ticker) or normalize_symbol(ticker)
    articles = _fetch_rss(query, lang, region, timeout=10.0)

    if not articles:
        return f"No news found for {ticker}"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    filtered = [a for a in articles if _in_window(a["pub_date"], start_dt, end_dt)][:limit]

    if not filtered:
        return f"No news found for {ticker} between {start_date} and {end_date}"

    return _format_articles(filtered, f"## {ticker} News (Google News), from {start_date} to {end_date}:")


def get_global_news_google(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
    lang: str = "de",
    region: str = "DE",
) -> str:
    """Retrieve global/macro economic headlines from Google News.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back.
        limit: Maximum number of articles to return.
        lang: Google News language code (default "de").
        region: Google News region code (default "DE").

    Returns:
        Formatted string containing global headlines, or a placeholder on failure.
    """
    query = "Finanzmärkte Wirtschaft Zentralbanken"
    articles = _fetch_rss(query, lang, region, timeout=10.0)

    if not articles:
        return f"No global news found for {curr_date}"

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    filtered = [a for a in articles if _in_window(a["pub_date"], start_dt, curr_dt)][:limit]

    if not filtered:
        return f"No global news found between {start_dt.strftime('%Y-%m-%d')} and {curr_date}"

    return _format_articles(
        filtered,
        f"## Global Market News (Google News), from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:",
    )
