from __future__ import annotations

from typing import Dict, List


def build_explanation(posture: str, flags: List[str], agent_signals: Dict[str, Dict[str, object]]) -> str:
    posture_text = {
        "GREENLIGHT": "Low risk/uncertainty based on the current signals.",
        "WATCH": "Caution: risk and uncertainty are elevated.",
        "STAND_DOWN": "High risk detected; pause and reduce exposure in learning scenarios.",
    }

    summary = posture_text.get(posture, "Risk posture computed.")
    if flags:
        summary = f"{summary} Flags: {', '.join(flags)}."

    top_agent = ""
    if agent_signals:
        top = sorted(
            agent_signals.items(),
            key=lambda item: float(item[1].get("score", 0.0)),
            reverse=True,
        )[0]
        top_agent = f" Top driver: {top[0]} - {top[1].get('reasoning', '')}"

    return f"{summary}{top_agent}".strip()
