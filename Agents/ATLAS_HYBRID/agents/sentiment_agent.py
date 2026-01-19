"""
Sentiment Agent (Educational, Simulation-Only)

This repository is a K–12 financial literacy project, so this sentiment agent is:
- offline (no API calls, no model downloads)
- simple and explainable (keyword-based)

Input:
- `market_data["headlines"]`: list of strings (historical/delayed headlines)

Output:
- normalized risk score (0..1) + short explanation
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class SentimentAgent(BaseAgent):
    def __init__(self, initial_weight: float = 0.8):
        super().__init__(name="SentimentAgent", initial_weight=initial_weight)

        # Small, student-friendly word lists (not exhaustive).
        self.positive_words = {
            "beat",
            "growth",
            "strong",
            "upgrade",
            "record",
            "profit",
            "surge",
            "bullish",
            "optimism",
        }
        self.negative_words = {
            "miss",
            "loss",
            "weak",
            "downgrade",
            "lawsuit",
            "crash",
            "bearish",
            "recession",
            "fear",
        }

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        headlines_raw: Iterable[Any] = market_data.get("headlines", []) or []
        headlines: List[str] = []
        for h in headlines_raw:
            if isinstance(h, str) and h.strip():
                headlines.append(h.strip())
            elif isinstance(h, dict) and isinstance(h.get("title"), str):
                headlines.append(h["title"].strip())

        if not headlines:
            return AgentAssessment(
                score=0.40,
                explanation="No headlines provided; sentiment risk is uncertain.",
                details={"headlines": 0},
            )

        pos = 0
        neg = 0
        for h in headlines[:20]:
            words = {w.strip(".,:;!?()[]\"'").lower() for w in h.split()}
            pos += sum(1 for w in words if w in self.positive_words)
            neg += sum(1 for w in words if w in self.negative_words)

        # Sentiment score in [-1, 1] (very rough).
        total = max(1, pos + neg)
        sentiment = (pos - neg) / total

        # Risk: extremes (very positive or very negative) can increase uncertainty.
        # Slightly weight negative extremes as higher risk.
        extreme = abs(sentiment)
        negativity = max(0.0, -sentiment)
        score = _clamp01(0.30 + (0.45 * extreme) + (0.15 * negativity))

        if sentiment <= -0.4:
            explanation = "News sentiment looks strongly negative; uncertainty/risk is higher."
        elif sentiment >= 0.4:
            explanation = "News sentiment looks strongly positive; hype can increase volatility/uncertainty."
        else:
            explanation = "News sentiment looks mixed/neutral; no strong sentiment warning."

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={"pos_hits": pos, "neg_hits": neg, "sentiment": round(sentiment, 2), "headline_count": len(headlines)},
        )

