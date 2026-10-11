"""Recent news per holding with an LLM estimate of the price direction.

Each article is rated once ("up", "down" or "neutral" plus a short German
reason) and cached in SQLite, so reloading the dashboard costs no LLM calls.
Next to the estimate the dashboard shows how the stock actually moved since the
article was published.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.yfinance_news import _extract_article_data
from tradingagents.portfolio import PortfolioStore

logger = logging.getLogger(__name__)

DIRECTIONS = ("up", "down", "neutral")


def article_key(article: dict) -> str:
    raw = article.get("link") or article.get("title") or ""
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def fetch_news(symbol: str, limit: int = 10) -> list[dict]:
    """Latest Yahoo Finance articles for ``symbol``, newest first."""
    articles = []
    for raw in (yf.Ticker(symbol).news or [])[: limit * 2]:
        data = _extract_article_data(raw)
        if not data.get("title") or data["title"] == "No title":
            continue
        data["key"] = article_key(data)
        articles.append(data)
    articles.sort(
        key=lambda a: a["pub_date"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True
    )
    return articles[:limit]


def _prompt(symbol: str, articles: list[dict]) -> str:
    lines = [
        f"Du bewertest Nachrichten zur Aktie {symbol}.",
        "Schätze für jede Meldung, ob sie den Aktienkurs eher steigen (up), fallen (down) "
        "lassen oder kaum beeinflussen (neutral) wird.",
        "Antworte NUR mit einem JSON-Array, ein Objekt je Meldung: "
        '{"id": "<id>", "direction": "up|down|neutral", "confidence": 0-100, '
        '"reason": "<ein kurzer Satz auf Deutsch>"}',
        "",
    ]
    for article in articles:
        summary = (article.get("summary") or "").strip().replace("\n", " ")[:500]
        lines.append(f"id={article['key']} | {article['title']} | {summary}")
    return "\n".join(lines)


def parse_ratings(text: str) -> dict[str, dict]:
    """Extract ``{id: rating}`` from an LLM answer; tolerant of prose around JSON."""
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return {}
    ratings = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        direction = str(item.get("direction", "")).lower().strip()
        if direction not in DIRECTIONS or not item.get("id"):
            continue
        try:
            confidence = max(0, min(100, int(item.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0
        ratings[str(item["id"])] = {
            "direction": direction,
            "confidence": confidence,
            "reason": str(item.get("reason", "")).strip(),
        }
    return ratings


def rate_articles(symbol: str, articles: list[dict], config: dict) -> dict[str, dict]:
    """Ask the quick-think model for a direction per article."""
    if not articles:
        return {}
    from tradingagents.llm_clients.factory import create_llm_client

    llm = create_llm_client(
        provider=config["llm_provider"],
        model=config["quick_think_llm"],
        base_url=config.get("backend_url"),
    ).get_llm()
    answer = llm.invoke(_prompt(symbol, articles))
    content = answer.content if hasattr(answer, "content") else answer
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return parse_ratings(str(content))


def price_reaction(prices: pd.Series | None, published: datetime | None) -> float | None:
    """Change from the last close before publication to the latest close."""
    if prices is None or published is None:
        return None
    prices = prices.dropna()
    if prices.empty:
        return None
    cutoff = pd.Timestamp(published)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    before = prices[prices.index < cutoff.normalize()]
    if before.empty or before.iloc[-1] == 0:
        return None
    return float(prices.iloc[-1] / before.iloc[-1] - 1)


def news_with_impact(
    store: PortfolioStore,
    symbol: str,
    config: dict | None,
    prices: pd.Series | None = None,
    limit: int = 10,
) -> tuple[list[dict], str | None]:
    """Articles enriched with rating and price reaction, plus an optional error.

    Without ``config`` (no API key) articles are returned unrated.
    """
    articles = fetch_news(symbol, limit)
    cached = store.get_news_impacts(symbol)
    error = None
    unrated = [a for a in articles if a["key"] not in cached]
    if unrated and config is not None:
        try:
            fresh = rate_articles(symbol, unrated, config)
        except Exception as exc:
            logger.warning("News rating for %s failed: %s", symbol, exc)
            error = str(exc)
        else:
            if fresh:
                store.save_news_impacts(symbol, fresh)
                cached.update(fresh)
    for article in articles:
        article["rating"] = cached.get(article["key"])
        article["reaction"] = price_reaction(prices, article["pub_date"])
    return articles, error
