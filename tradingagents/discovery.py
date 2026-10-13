""""Entdecken": the agents look at stocks beyond your favorites.

Once a day Yahoo Finance screeners (most active, gainers, losers, undervalued,
growth tech) produce candidates. The quick-think model picks the most
interesting ones and says in one or two sentences why. This is deliberately
cheap — a single LLM call; the full multi-agent analysis only runs once you
mark a candidate as favorite.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable

from tradingagents.dataflows.symbol_utils import is_yahoo_safe

logger = logging.getLogger(__name__)

SCREENERS = (
    "most_actives",
    "day_gainers",
    "day_losers",
    "undervalued_large_caps",
    "growth_technology_stocks",
)

SCREEN_LABELS = {
    "most_actives": "meistgehandelt",
    "day_gainers": "Tagesgewinner",
    "day_losers": "Tagesverlierer",
    "undervalued_large_caps": "unterbewertet",
    "growth_technology_stocks": "Tech-Wachstum",
}


def discovery_enabled() -> bool:
    return os.environ.get("TRADINGAGENTS_DISCOVERY", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def discovery_count() -> int:
    try:
        return max(1, int(os.environ.get("TRADINGAGENTS_DISCOVERY_COUNT", "10")))
    except ValueError:
        return 10


def screener_candidates(per_screen: int = 15) -> list[dict]:
    """Equity candidates from several Yahoo screeners, de-duplicated."""
    import yfinance as yf

    candidates: dict[str, dict] = {}
    for screen in SCREENERS:
        try:
            quotes = yf.screen(screen, count=per_screen).get("quotes", [])
        except Exception as exc:
            logger.warning("Screener %s failed: %s", screen, exc)
            continue
        for quote in quotes:
            symbol = quote.get("symbol")
            if quote.get("quoteType") not in (None, "EQUITY") or not symbol or not is_yahoo_safe(symbol):
                continue
            entry = candidates.setdefault(symbol, {
                "symbol": symbol,
                "name": quote.get("shortName") or quote.get("longName") or symbol,
                "price": quote.get("regularMarketPrice"),
                "currency": quote.get("currency"),
                "change_pct": quote.get("regularMarketChangePercent"),
                "change_52w_pct": quote.get("fiftyTwoWeekChangePercent"),
                "pe": quote.get("trailingPE"),
                "market_cap": quote.get("marketCap"),
                "screens": [],
            })
            entry["screens"].append(screen)
    return list(candidates.values())


def _fmt(value, pattern: str = "{:.1f}") -> str:
    return pattern.format(value) if isinstance(value, (int, float)) else "-"


def scout_prompt(candidates: list[dict], tracked: list[str], count: int) -> str:
    lines = [
        "Du bist Aktien-Scout eines Analystenteams. Wähle aus den Kandidaten die "
        f"{count} Aktien, die heute einen genaueren Blick wert sind – Chancen wie Risiken.",
        "Bereits beobachtet (nicht wählen): " + (", ".join(tracked) or "nichts") + ".",
        "Begründe jede Wahl in 1–2 Sätzen auf Deutsch, konkret (Bewegung, Bewertung, Branche).",
        'Antworte NUR mit einem JSON-Array: [{"symbol": "...", "reason": "..."}]',
        "",
        "symbol | name | kurs | heute % | 52W % | KGV | marktkap. | rangliste",
    ]
    for c in candidates:
        lines.append(
            f"{c['symbol']} | {c['name']} | {_fmt(c['price'], '{:.2f}')} {c.get('currency') or ''} | "
            f"{_fmt(c['change_pct'])} | {_fmt(c['change_52w_pct'])} | {_fmt(c['pe'])} | "
            f"{_fmt((c['market_cap'] or 0) / 1e9, '{:.1f}')} Mrd | {','.join(c['screens'])}"
        )
    return "\n".join(lines)


def parse_picks(text: str, allowed: set[str], limit: int) -> list[dict]:
    """``[{symbol, reason}]`` from an LLM answer, only for known candidates."""
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    picks: list[dict] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if symbol in allowed and symbol not in {p["symbol"] for p in picks}:
            picks.append({"symbol": symbol, "reason": str(item.get("reason", "")).strip()})
        if len(picks) >= limit:
            break
    return picks


def llm_scout(candidates: list[dict], tracked: list[str], config: dict, count: int) -> list[dict]:
    from tradingagents.llm_clients.factory import create_llm_client

    llm = create_llm_client(
        provider=config["llm_provider"],
        model=config["quick_think_llm"],
        base_url=config.get("backend_url"),
    ).get_llm()
    answer = llm.invoke(scout_prompt(candidates, tracked, count))
    content = answer.content if hasattr(answer, "content") else answer
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return parse_picks(str(content), {c["symbol"] for c in candidates}, count)


def discover(
    tracked: list[str],
    config: dict | None,
    count: int | None = None,
    candidates_fn: Callable[[], list[dict]] = screener_candidates,
    scout: Callable[[list[dict], list[str], dict, int], list[dict]] = llm_scout,
) -> dict:
    """Today's discoveries: ``{"picks": [...], "error": str | None}``.

    Each pick carries the screener data plus the scout's ``reason``. Without a
    usable LLM (no key, rate limit) the most active candidates are returned
    without a reason, so the page is never empty.
    """
    count = count or discovery_count()
    tracked_set = {s.upper() for s in tracked}
    candidates = [c for c in candidates_fn() if c["symbol"].upper() not in tracked_set]
    by_symbol = {c["symbol"]: c for c in candidates}
    error = None
    picks: list[dict] = []
    if candidates and config is not None:
        try:
            picks = scout(candidates, sorted(tracked_set), config, count)
        except Exception as exc:
            logger.warning("Discovery scout failed: %s", exc)
            error = str(exc)[:300]
    if not picks:
        picks = [{"symbol": c["symbol"], "reason": ""} for c in candidates[:count]]
    return {
        "picks": [{**by_symbol[p["symbol"]], "reason": p["reason"]} for p in picks if p["symbol"] in by_symbol],
        "error": error,
    }
