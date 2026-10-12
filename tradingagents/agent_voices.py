"""Split a stored analysis report into what each agent said.

``write_report_tree`` renders every run as one markdown document with a
``### <Agent>`` section per agent. The dashboard does not show that document
as a wall of text; it shows each agent as a voice: who it is, its stance in a
short excerpt, and the full text on demand. Works for every stored report,
old ones included, because it only relies on the section headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Agent heading → (team, icon, German name, German role), in pipeline order.
AGENTS: dict[str, tuple[str, str, str, str]] = {
    "Market Analyst": ("analysts", "📈", "Markt-Analyst", "Kurs & Charttechnik"),
    "Sentiment Analyst": ("analysts", "💬", "Sentiment-Analyst", "Stimmung in News, Reddit, StockTwits"),
    "News Analyst": ("analysts", "📰", "News-Analyst", "Firmen- & Makro-Nachrichten"),
    "Fundamentals Analyst": ("analysts", "🏦", "Fundamental-Analyst", "Bilanz & Bewertung"),
    "Bull Researcher": ("debate", "🐂", "Bull", "argumentiert für den Kauf"),
    "Bear Researcher": ("debate", "🐻", "Bear", "argumentiert dagegen"),
    "Research Manager": ("debate", "🧑‍⚖️", "Research-Manager", "entscheidet die Debatte"),
    "Trader": ("trader", "🧑‍💼", "Trader", "konkreter Handelsplan"),
    "Aggressive Analyst": ("risk", "🔥", "Aggressiv", "Risiko-Team"),
    "Conservative Analyst": ("risk", "🛡️", "Konservativ", "Risiko-Team"),
    "Neutral Analyst": ("risk", "⚖️", "Neutral", "Risiko-Team"),
    "Portfolio Manager": ("decision", "✅", "Portfolio-Manager", "Endentscheidung"),
}

_SECTION_RE = re.compile(r"^### (.+?)\s*$", re.MULTILINE)
_MARKUP_RE = re.compile(r"^\s*(#+|\||[-*]{3,}\s*$)")


@dataclass(frozen=True)
class Voice:
    agent: str
    team: str
    icon: str
    name: str
    role: str
    text: str

    @property
    def excerpt(self) -> str:
        return excerpt(self.text)


def parse_voices(report_markdown: str) -> list[Voice]:
    """Agent voices found in ``report_markdown``, in pipeline order."""
    matches = list(_SECTION_RE.finditer(report_markdown or ""))
    texts: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name not in AGENTS:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report_markdown)
        body = report_markdown[match.end():end]
        # A team section (``## II. ...``) may follow; it does not belong to the agent.
        body = re.split(r"^## ", body, maxsplit=1, flags=re.MULTILINE)[0].strip()
        if body:
            texts[name] = body
    return [
        Voice(name, *AGENTS[name], texts[name]) for name in AGENTS if name in texts
    ]


_SPEAKER_RE = re.compile(
    r"^\s*\**(Bull|Bear|Aggressive|Conservative|Neutral|Risky|Safe)( Analyst| Researcher)?\**:\s*",
    re.IGNORECASE,
)
_RATING_LINE_RE = re.compile(r"^\W*(final )?rating\W*:", re.IGNORECASE)


def escape_markdown(text: str) -> str:
    """Escape ``$`` so Streamlit does not render prices as LaTeX formulas."""
    return (text or "").replace("$", "\\$")


def excerpt(text: str, limit: int = 320) -> str:
    """First prose lines of ``text``, cut at ``limit``.

    Headings, tables, a bare "Rating: X" line and a leading "Bull Analyst:"
    speaker tag are skipped; the card already shows who speaks and the rating.
    """
    lines = []
    for line in (text or "").splitlines():
        if not line.strip() or _MARKUP_RE.match(line) or _RATING_LINE_RE.match(line):
            if lines:
                break
            continue
        lines.append(_SPEAKER_RE.sub("", line.strip()) if not lines else line.strip())
    short = " ".join(lines) or (text or "").strip()
    short = re.sub(r"\*\*|__", "", short)
    if len(short) <= limit:
        return short
    cut = short[:limit].rsplit(" ", 1)[0]
    return cut + " …"
