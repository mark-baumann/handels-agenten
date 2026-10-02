"""Quality checks for generated trading reports.

These utilities detect the common failure modes seen in multi-agent trading
reports: conflicting final actions, missing required analyst sections, and
formatting glitches that can silently poison downstream decisions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

ACTION_PRIORITY = {
    "SELL": 5,
    "UNDERWEIGHT": 4,
    "BUY": 3,
    "OVERWEIGHT": 2,
    "HOLD": 1,
}
ACTION_ALIASES = {
    "buy": "BUY",
    "bullish": "BUY",
    "long": "BUY",
    "overweight": "OVERWEIGHT",
    "sell": "SELL",
    "short": "SELL",
    "underweight": "UNDERWEIGHT",
    "hold": "HOLD",
    "neutral": "HOLD",
    "flat": "HOLD",
}

FINAL_ACTION_PATTERN = re.compile(
    r"FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*\*\*(BUY|HOLD|SELL)\*\*",
    re.IGNORECASE,
)
RATING_PATTERN = re.compile(
    r"\*\*(?:Rating|Recommendation)\*\*\s*:\s*(Buy|Overweight|Hold|Underweight|Sell)",
    re.IGNORECASE,
)
SECTION_PATTERNS = {
    "sentiment": [
        r"sentiment\s+analyst\s+report",
        r"💬\s*sentiment",
        r"sentiment analyst",
    ],
    "news": [
        r"news\s+analyst\s+report",
        r"📰\s*news",
        r"news analyst",
    ],
    "fundamentals": [
        r"fundamentals\s+analyst\s+report",
        r"📋\s*fundamentals",
        r"fundamentals analyst",
    ],
    "bull_bear_debate": [
        r"bull\s*vs\.?\s*bear\s*debate",
        r"🐂\s*🐻",
        r"bull.*bear.*debate",
    ],
}


def sanitize_report_text(report: Any) -> str:
    """Normalize common formatting glitches out of report text."""
    if report is None:
        return ""
    text = str(report)
    replacements = {
        "∗": "*",
        "＊": "*",
        "–": "-",
        "—": "-",
        "\u00a0": " ",
        "\u200b": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_final_actions(report: str) -> list[str]:
    actions = [match.group(1).upper() for match in FINAL_ACTION_PATTERN.finditer(report)]
    if not actions:
        actions = [match.group(1).upper() for match in RATING_PATTERN.finditer(report)]
    return actions


def normalize_action(value: Any) -> str:
    """Normalize a raw signal to one of the canonical actions."""
    if value is None:
        return "HOLD"
    text = str(value).strip().upper()
    if text in ACTION_ALIASES:
        return ACTION_ALIASES[text]
    for alias, canonical in ACTION_ALIASES.items():
        if text == alias.upper():
            return canonical
    if text.startswith("BUY"):
        return "BUY"
    if text.startswith("SELL"):
        return "SELL"
    if text.startswith("OVERWEIGHT"):
        return "OVERWEIGHT"
    if text.startswith("UNDERWEIGHT"):
        return "UNDERWEIGHT"
    if text.startswith("HOLD"):
        return "HOLD"
    return "HOLD"


def extract_agent_signals(report: Any) -> dict[str, str]:
    """Pull the represented final decisions from primary agent sections."""
    text = sanitize_report_text(report)
    signals: dict[str, str] = {}

    market_match = re.search(
        r"market\s+analyst.*?(?:FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*\*\*(BUY|HOLD|SELL)\*\*|\*\*(?:Rating|Recommendation)\*\*\s*:\s*(Buy|Overweight|Hold|Underweight|Sell))",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if market_match:
        signals["market"] = normalize_action(market_match.group(1) or market_match.group(2))

    trader_match = re.search(
        r"trader(?:\s+agent)?(?:.*?)(?:FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*\*\*(BUY|HOLD|SELL)\*\*|\*\*(?:Rating|Recommendation)\*\*\s*:\s*(Buy|Overweight|Hold|Underweight|Sell))",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if trader_match:
        signals["trader"] = normalize_action(trader_match.group(1) or trader_match.group(2))

    research_match = re.search(
        r"research\s+manager.*?(?:\*\*(?:Rating|Recommendation)\*\*\s*:\s*(Buy|Overweight|Hold|Underweight|Sell)|FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*\*\*(BUY|HOLD|SELL)\*\*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if research_match:
        signals["research"] = normalize_action(research_match.group(1) or research_match.group(2))

    if not signals:
        for action in _find_final_actions(text):
            signals.setdefault("fallback", normalize_action(action))

    return signals


def resolve_agent_conflict(report: Any) -> dict[str, Any]:
    """Resolve contradictory agent signals conservatively and reject unsafe final decisions."""
    text = sanitize_report_text(report)
    quality = analyze_report_quality(text)
    signals = extract_agent_signals(text)

    if not signals or quality["issues"]:
        return {
            "accepted": False,
            "final_action": "HOLD",
            "position_size": "0-5%",
            "reason": "; ".join(quality["issues"]) if quality["issues"] else "No valid agent consensus could be established.",
            "signals": signals,
        }

    distinct_actions = {normalize_action(action) for action in signals.values()}
    if len(distinct_actions) > 1:
        if any(action in {"SELL", "UNDERWEIGHT"} for action in distinct_actions):
            final_action = "HOLD"
        elif any(action in {"BUY", "OVERWEIGHT"} for action in distinct_actions) and "HOLD" in distinct_actions:
            final_action = "HOLD"
        else:
            final_action = max(distinct_actions, key=lambda value: ACTION_PRIORITY.get(value, 0))
        return {
            "accepted": False,
            "final_action": final_action,
            "position_size": "0-5%",
            "reason": "Conflicting signals across analyst, trader, and research agents were resolved conservatively.",
            "signals": signals,
        }

    final_action = max(signals.values(), key=lambda value: ACTION_PRIORITY.get(normalize_action(value), 0))
    return {
        "accepted": True,
        "final_action": normalize_action(final_action),
        "position_size": "5-10%" if normalize_action(final_action) in {"BUY", "OVERWEIGHT"} else "0-5%",
        "reason": "Signals are aligned; no conflict requiring a risk downgrade.",
        "signals": signals,
    }


def evaluate_final_decision(report: Any, base_action: Any) -> dict[str, Any]:
    """Apply a hard final decision gate before a report can be treated as actionable."""
    text = sanitize_report_text(report)
    quality = analyze_report_quality(text)
    normalized = normalize_action(base_action)

    if not quality["is_valid"] or normalized in {"BUY", "OVERWEIGHT"} and not quality["final_actions"]:
        return {
            "status": "rejected",
            "final_action": "HOLD",
            "position_size": "0-5%",
            "reason": "; ".join(quality["issues"]) if quality["issues"] else "Insufficiently validated report for a directional trade.",
        }

    if not quality["is_valid"]:
        return {
            "status": "rejected",
            "final_action": "HOLD",
            "position_size": "0-5%",
            "reason": "; ".join(quality["issues"]),
        }

    if normalized in {"BUY", "OVERWEIGHT"}:
        return {
            "status": "approved",
            "final_action": normalized,
            "position_size": "5% max" if normalized == "BUY" else "5-10%",
            "reason": "Validated report and aligned agent signals support a directional position.",
        }

    if normalized in {"SELL", "UNDERWEIGHT"}:
        return {
            "status": "approved",
            "final_action": normalized,
            "position_size": "10% max",
            "reason": "Validated report supports a defensive or de-risking move.",
        }

    return {
        "status": "hold",
        "final_action": "HOLD",
        "position_size": "0-5%",
        "reason": "Evidence is balanced; the system keeps risk controlled and avoids a speculative position.",
    }


def analyze_report_quality(report: Any) -> dict[str, Any]:
    """Return structured quality information for a generated trading report.

    A valid report should contain a single final action, all required sections,
    and no obvious contradictions.
    """
    text = sanitize_report_text(report)
    issues: list[str] = []

    actions = _find_final_actions(text)
    if not actions:
        issues.append("No final transaction proposal or rating could be detected.")
    else:
        action_counts = Counter(actions)
        if len(action_counts) > 1:
            issues.append(
                "Conflicting final signals detected: "
                + ", ".join(f"{name} ({count})" for name, count in sorted(action_counts.items()))
                + "."
            )

    missing_sections = []
    for label, patterns in SECTION_PATTERNS.items():
        if not any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns):
            missing_sections.append(label.replace("_", " "))
    if missing_sections:
        issues.append(
            "Missing required sections: " + ", ".join(missing_sections) + "."
        )

    if text and ("**o**" in text.lower() or "∗∗" in text):
        issues.append("Formatting glitches detected in markdown output; report should be sanitized before display.")

    return {
        "is_valid": not issues,
        "issues": issues,
        "final_actions": actions,
        "missing_sections": missing_sections,
        "sanitized_report": text,
    }
