"""
News / Event Risk Agent (Educational, Simulation-Only)

This agent flags *scheduled* high-impact events (e.g., CPI/FOMC/NFP) that can
increase volatility. In this repository it is intentionally conservative and
uses only:
- events provided in `market_data["calendar_events"]` (historical/delayed), or
- a small optional built-in example schedule (also historical/delayed)

It does NOT call external APIs and does NOT perform live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from .base_agent import AgentAssessment, BaseAgent


@dataclass(frozen=True)
class CalendarEvent:
    time: datetime
    event: str
    currency: str = ""
    impact: str = "high"  # "low" | "medium" | "high"


def _parse_event(raw: Dict[str, Any]) -> Optional[CalendarEvent]:
    try:
        t = raw.get("time") or raw.get("timestamp") or raw.get("date")
        if isinstance(t, datetime):
            dt = t
        elif isinstance(t, str) and t.strip():
            dt = datetime.fromisoformat(t.strip())
        else:
            return None
        return CalendarEvent(
            time=dt,
            event=str(raw.get("event") or raw.get("name") or "Scheduled event"),
            currency=str(raw.get("currency") or ""),
            impact=str(raw.get("impact") or "high").lower(),
        )
    except Exception:
        return None


class NewsFilterAgent(BaseAgent):
    """
    Educational event-risk filter.

    Output score meaning:
    - 0.0 = no nearby scheduled events (low news-driven risk)
    - 1.0 = high-impact event very soon (high volatility risk)
    """

    def __init__(self, initial_weight: float = 2.0):
        super().__init__(name="NewsFilterAgent", initial_weight=initial_weight)

        # Default buffers (minutes). Can be overridden via config by setting attrs.
        self.block_buffer_minutes = 60
        self.warning_buffer_minutes = 180

        # Small example schedule (optional). Kept as a fallback only.
        self.example_events: List[CalendarEvent] = []

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        ts = market_data.get("time")
        if not isinstance(ts, datetime):
            return AgentAssessment(
                score=0.50,
                explanation="No timestamp provided; cannot check for scheduled events.",
                details={"time": str(ts), "data_sufficiency": "insufficient"},
            )

        raw_events: Iterable[Dict[str, Any]] = market_data.get("calendar_events", []) or []
        events: List[CalendarEvent] = []
        for raw in raw_events:
            if isinstance(raw, dict):
                parsed = _parse_event(raw)
                if parsed:
                    events.append(parsed)

        # Fallback to example events if no events were provided.
        if not events and self.example_events:
            events = list(self.example_events)

        if not events:
            return AgentAssessment(
                score=0.50,
                explanation="No event calendar provided; skipping news risk check.",
                details={"events": 0, "data_sufficiency": "insufficient"},
            )

        upcoming: List[Dict[str, Any]] = []
        for e in events:
            minutes_until = (e.time - ts).total_seconds() / 60.0
            if 0 <= minutes_until <= self.warning_buffer_minutes:
                upcoming.append(
                    {
                        "event": e.event,
                        "currency": e.currency,
                        "impact": e.impact,
                        "minutes_until": int(minutes_until),
                        "time": e.time.isoformat(),
                    }
                )

        if not upcoming:
            return AgentAssessment(
                score=0.15,
                explanation="No high-impact scheduled events in the next few hours.",
                details={"events": len(events)},
            )

        upcoming.sort(key=lambda x: x["minutes_until"])
        next_event = upcoming[0]

        minutes_until = int(next_event["minutes_until"])
        impact = str(next_event.get("impact", "high"))

        # Score schedule: nearer + higher impact => higher risk.
        if minutes_until <= self.block_buffer_minutes and impact in {"high", "medium"}:
            score = 1.0
            explanation = f"High-impact event soon ({minutes_until} min): {next_event['event']}."
        elif minutes_until <= self.block_buffer_minutes:
            score = 0.75
            explanation = f"Event soon ({minutes_until} min): {next_event['event']}."
        else:
            # Within warning buffer, but not immediate.
            score = 0.55
            explanation = f"Scheduled event later ({minutes_until} min): {next_event['event']} — volatility may rise."

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={"next_event": next_event, "upcoming": upcoming[:5]},
        )
