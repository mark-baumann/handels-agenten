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


# Listing pages that Google News indexes as "news" for a ticker: option
# contract quote pages (``AAPL261012P00337500``) and generic quote/key-figure
# pages. They carry no information and crowd out real articles.
_NOISE_RES = (
    re.compile(r"\b[A-Z]{1,6}\d{6}[CP]\d{8}\b"),
    re.compile(r"\b(call|put)\s*\(", re.IGNORECASE),
    re.compile(
        r"Aktienkurs, Nachrichten, Kurs und Verlauf|Kursverlauf und Daten|Aktuelle Kennzahlen"
        r"|Stock Price, News, Quote|Stock Price Today|Kurs Xetra",
        re.IGNORECASE,
    ),
)

# Exchanges whose coverage is mostly German-language.
_GERMAN_SUFFIXES = (".DE", ".F", ".SG", ".MU", ".DU", ".HM", ".BE", ".HA", ".VI")

# Legal-form suffixes dropped from company names to get a searchable brand.
_LEGAL_SUFFIX_RE = re.compile(
    r"(,?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|ag|se|sa|nv|"
    r"n\.v|s\.a|holdings?|group|class [a-c])\.?)+$",
    re.IGNORECASE,
)


def _is_noise(title: str) -> bool:
    return any(pattern.search(title) for pattern in _NOISE_RES)


def _locale(symbol: str) -> tuple[str, str, str]:
    """(lang, region, stock word) for the market a symbol trades on."""
    if symbol.upper().endswith(_GERMAN_SUFFIXES):
        return "de", "DE", "Aktie"
    return "en-US", "US", "stock"


def _company_name(symbol: str) -> str | None:
    """Short, searchable company name (``Apple Inc.`` → ``Apple``), if known."""
    try:
        from tradingagents.agents.utils.agent_utils import resolve_instrument_identity

        name = resolve_instrument_identity(symbol).get("company_name")
    except Exception:  # noqa: BLE001 — identity is best effort
        return None
    if not name:
        return None
    short = _LEGAL_SUFFIX_RE.sub("", name.strip()).strip(" ,.")
    return short or None


def _date_filter(start_dt: datetime, end_dt: datetime) -> str:
    """Google search operators for the window; ``before:`` is exclusive."""
    return (
        f"after:{start_dt.strftime('%Y-%m-%d')} "
        f"before:{(end_dt + timedelta(days=1)).strftime('%Y-%m-%d')}"
    )


def ticker_news_articles(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int = 20,
) -> list[dict]:
    """Relevant, de-noised headlines about ``ticker`` within the window.

    Searches by company name (plus "stock"/"Aktie") in the market's language
    instead of the bare ticker, which mostly returns option quote pages. Falls
    back to the ticker when no name is known or the name search finds nothing.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    window = _date_filter(start_dt, end_dt)

    base = crypto_base(ticker)
    symbol = normalize_symbol(ticker)
    if base:
        queries = [f"{base} crypto {window}"]
        lang, region = "en-US", "US"
    else:
        lang, region, word = _locale(symbol)
        name = _company_name(symbol)
        plain = symbol.split(".")[0]
        queries = [f'"{name}" {word} {window}'] if name else []
        queries.append(f"{plain} {word} {window}")

    for query in queries:
        articles = [
            a for a in _fetch_rss(query, lang, region, timeout=10.0)
            if not _is_noise(a["title"]) and _in_window(a["pub_date"], start_dt, end_dt)
        ]
        if articles:
            articles.sort(key=lambda a: a["pub_date"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            return articles[:limit]
    return []


def get_news_google(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int = 20,
) -> str:
    """Retrieve recent Google News headlines for ``ticker``.

    Args:
        ticker: Ticker symbol (e.g. "AAPL", "SAP.DE", "BTC-USD").
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format.
        limit: Maximum number of articles to return.

    Returns:
        Formatted string containing news headlines, or a placeholder on failure.
    """
    articles = ticker_news_articles(ticker, start_date, end_date, limit)
    if not articles:
        return f"No news found for {ticker} between {start_date} and {end_date}"
    return _format_articles(articles, f"## {ticker} News (Google News), from {start_date} to {end_date}:")


def global_news_articles(curr_date: str, look_back_days: int = 7, limit: int = 10) -> list[dict]:
    """Macro headlines for the configured ``global_news_queries`` topics.

    One date-restricted search per topic (fetched in parallel), merged and
    de-duplicated, newest first, spread across topics.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .config import get_config

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    window = _date_filter(start_dt, curr_dt)
    topics = get_config().get("global_news_queries") or ["stock market economy"]

    def fetch(topic: str) -> list[dict]:
        return [
            a for a in _fetch_rss(f"{topic} {window}", "en-US", "US", timeout=10.0)
            if _in_window(a["pub_date"], start_dt, curr_dt)
        ]

    with ThreadPoolExecutor(max_workers=min(5, len(topics))) as pool:
        per_topic = list(pool.map(fetch, topics))

    # Round-robin over topics so one busy topic cannot crowd out the others.
    merged, seen = [], set()
    for rank in range(max((len(t) for t in per_topic), default=0)):
        for articles in per_topic:
            if rank < len(articles):
                article = articles[rank]
                key = article["title"].lower()
                if key not in seen:
                    seen.add(key)
                    merged.append(article)
    return merged[:limit]


def get_global_news_google(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Retrieve global/macro economic headlines from Google News.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back.
        limit: Maximum number of articles to return.

    Returns:
        Formatted string containing global headlines, or a placeholder on failure.
    """
    articles = global_news_articles(curr_date, look_back_days, limit)
    start = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    if not articles:
        return f"No global news found between {start} and {curr_date}"
    return _format_articles(
        articles, f"## Global Market News (Google News), from {start} to {curr_date}:"
    )
