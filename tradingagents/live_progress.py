"""Turn successive graph states into readable live events for the dashboard.

``TradingAgentsGraph.propagate(on_chunk=...)`` hands over the full state after
every node. :class:`ProgressTracker` compares each state with what it has seen
before and yields only what is new: an agent's thoughts, its tool calls, the
data a tool returned, finished analyst reports and each debate turn.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

_TOOL_PREVIEW_CHARS = 600

_REPORTS = (
    ("market_report", "📈 Marktanalyse fertig"),
    ("sentiment_report", "💬 Sentiment-Analyse fertig"),
    ("news_report", "📰 Nachrichten-Analyse fertig"),
    ("fundamentals_report", "🏦 Fundamentalanalyse fertig"),
    ("trader_investment_plan", "🧑‍💼 Trader-Plan"),
    ("final_trade_decision", "✅ Finale Entscheidung"),
)

_DEBATES = (
    ("investment_debate_state", "current_response", "⚖️ Research-Debatte"),
    ("investment_debate_state", "judge_decision", "🧑‍⚖️ Research Manager"),
    ("risk_debate_state", "current_aggressive_response", "🔥 Risiko: Aggressiv"),
    ("risk_debate_state", "current_conservative_response", "🛡️ Risiko: Konservativ"),
    ("risk_debate_state", "current_neutral_response", "⚖️ Risiko: Neutral"),
)


@dataclass(frozen=True)
class ProgressEvent:
    title: str
    body: str
    kind: str  # "thought" | "tool" | "data" | "report" | "debate"


def _text(content) -> str:
    """Message content as plain text (content may be a list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "reasoning"):
                parts.append(str(block.get("text") or block.get("reasoning") or ""))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _format_args(args) -> str:
    if not isinstance(args, dict):
        return ""
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


class ProgressTracker:
    """Remembers what was already shown and emits only new events."""

    def __init__(self) -> None:
        self._seen_messages: set[tuple] = set()
        self._seen_values: dict[str, str] = {}

    def _changed(self, key: str, value) -> str | None:
        text = value.strip() if isinstance(value, str) else ""
        if not text or self._seen_values.get(key) == text:
            return None
        self._seen_values[key] = text
        return text

    def _message_events(self, message) -> list[ProgressEvent]:
        kind = getattr(message, "type", "")
        content = _text(getattr(message, "content", ""))
        tool_calls = getattr(message, "tool_calls", None) or []
        signature = (
            kind,
            getattr(message, "id", None),
            content,
            tuple(call.get("id") or call.get("name") for call in tool_calls),
        )
        if signature in self._seen_messages:
            return []
        self._seen_messages.add(signature)

        events = []
        if kind == "ai":
            reasoning = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                events.append(ProgressEvent("🧠 Denkt nach", reasoning.strip(), "thought"))
            if content:
                events.append(ProgressEvent("💭 Agent", content, "thought"))
            for call in tool_calls:
                events.append(
                    ProgressEvent(
                        f"🔧 Ruft `{call.get('name', 'tool')}` auf",
                        _format_args(call.get("args")),
                        "tool",
                    )
                )
        elif kind == "tool" and content:
            preview = content[:_TOOL_PREVIEW_CHARS]
            if len(content) > _TOOL_PREVIEW_CHARS:
                preview += " …"
            name = getattr(message, "name", None) or "Tool"
            events.append(ProgressEvent(f"📥 Daten von `{name}`", preview, "data"))
        return events

    def update(self, state: dict) -> list[ProgressEvent]:
        events: list[ProgressEvent] = []
        for message in state.get("messages") or []:
            events.extend(self._message_events(message))
        for key, title in _REPORTS:
            text = self._changed(key, state.get(key))
            if text:
                events.append(ProgressEvent(title, text, "report"))
        for parent, key, title in _DEBATES:
            nested = state.get(parent)
            if isinstance(nested, dict):
                text = self._changed(f"{parent}.{key}", nested.get(key))
                if text:
                    events.append(ProgressEvent(title, text, "debate"))
        return events


# Keys that mark the pipeline stages, in order, for a progress bar.
_STAGES = (
    ("market_report", "Marktanalyse"),
    ("sentiment_report", "Sentiment"),
    ("news_report", "Nachrichten"),
    ("fundamentals_report", "Fundamentaldaten"),
    ("investment_debate_state.judge_decision", "Research-Debatte"),
    ("trader_investment_plan", "Trader"),
    ("final_trade_decision", "Risikomanagement"),
)


def stage_progress(state: dict) -> tuple[float, str]:
    """Fraction of finished pipeline stages and the name of the next one."""
    def filled(path: str) -> bool:
        value = state
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return isinstance(value, str) and bool(value.strip())

    done = [filled(path) for path, _ in _STAGES]
    upcoming = next((label for (_, label), ok in zip(_STAGES, done, strict=True) if not ok), "Fertig")
    return sum(done) / len(_STAGES), upcoming


class AnalysisJob:
    """A manual analysis running in a background thread.

    The thread collects :class:`ProgressEvent` objects and saves the report
    itself, so the result is never lost — even if the browser tab is closed
    or the page reruns while the agents are still working.
    """

    def __init__(self, store, symbol: str, analysis_date: str, config: dict):
        self.symbol = symbol
        self.analysis_date = analysis_date
        self.started = time.monotonic()
        self.events: list[ProgressEvent] = []
        self.state: dict = {}
        self.result: tuple[int, str, str] | None = None
        self.error: str | None = None
        self._tracker = ProgressTracker()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, args=(store, config), name=f"analysis-{symbol}", daemon=True
        )
        self._thread.start()

    def _on_chunk(self, state: dict) -> None:
        events = self._tracker.update(state)
        with self._lock:
            self.state = state
            self.events.extend(events)

    def _run(self, store, config: dict) -> None:
        from tradingagents.daily_reports import run_and_store_analysis

        try:
            self.result = run_and_store_analysis(
                store, self.symbol, self.analysis_date, config, on_chunk=self._on_chunk
            )
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def snapshot(self) -> tuple[list[ProgressEvent], dict]:
        with self._lock:
            return list(self.events), self.state

    def elapsed(self) -> float:
        return time.monotonic() - self.started
