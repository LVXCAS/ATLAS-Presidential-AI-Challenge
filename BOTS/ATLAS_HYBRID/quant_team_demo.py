#!/usr/bin/env python3

"""
AI Quant Team Demo (Track II)

Educational, simulation-only demo that produces:
- Baseline (simple rule) desk stance labels
- Multi-agent "quant desk" stance labels with interpretable rationale

This tool is NOT investment advice and does NOT execute trades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from quant_team_utils import (
    baseline_risk,
    generate_stress_windows,
    initialize_coordinator,
    load_config,
    make_market_data,
    quant_team_recommendation,
)


DISCLAIMER = (
    "DISCLAIMER: Educational use only. Not financial advice. "
    "This demo does not place trades or provide buy/sell signals."
)


def collect_agent_votes(coordinator, market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    votes: Dict[str, Dict[str, Any]] = {}
    for agent in coordinator.agents:
        vote, confidence, reasoning = agent.analyze(market_data)
        votes[agent.name] = {
            "vote": vote,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "weight": float(getattr(agent, "weight", 1.0)),
        }
    return votes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "track2_quant_team.json"),
        help="Path to Track II quant team config JSON",
    )
    parser.add_argument(
        "--window",
        default="all",
        choices=["all", "stable-range", "volatility-spike", "regime-shift"],
        help="Which stress window to run",
    )
    parser.add_argument("--json-out", default="", help="Optional path to write a JSON report")
    args = parser.parse_args()

    print(DISCLAIMER)
    config = load_config(args.config)
    coordinator = initialize_coordinator(config)

    windows = generate_stress_windows()
    if args.window != "all":
        windows = [w for w in windows if w.name == args.window]

    report: Dict[str, Any] = {"disclaimer": DISCLAIMER, "windows": []}

    for w in windows:
        print(f"\n=== Stress Window: {w.name} ===")
        print(w.description)
        window_rec: Dict[str, Any] = {"name": w.name, "description": w.description, "steps": []}

        for step in range(len(w.prices)):
            market_data = make_market_data(w.pair, w.prices, step)

            # Baseline (simple rules)
            indicators = dict(market_data["indicators"])
            indicators["_price"] = market_data["price"]
            baseline_label, baseline_meta = baseline_risk(indicators)

            # Multi-agent (quant team)
            votes = collect_agent_votes(coordinator, market_data)
            coach_label, coach_meta = quant_team_recommendation(votes)

            window_rec["steps"].append(
                {
                    "step": step,
                    "price": market_data["price"],
                    "is_stress": bool(w.is_stress_step[step]),
                    "baseline": {"label": baseline_label, "meta": baseline_meta},
                    "quant_team": {"label": coach_label, "meta": coach_meta},
                }
            )

        report["windows"].append(window_rec)

        # Print a compact summary for video/demo narration.
        stress_steps = sum(1 for s in window_rec["steps"] if s["is_stress"])
        baseline_false_safe = sum(
            1 for s in window_rec["steps"] if s["is_stress"] and s["baseline"]["label"] == "GREENLIGHT"
        )
        coach_false_safe = sum(
            1 for s in window_rec["steps"] if s["is_stress"] and s["quant_team"]["label"] == "GREENLIGHT"
        )
        print(f"Stress steps: {stress_steps}")
        print(f"Baseline GREENLIGHT-in-stress (lower is better): {baseline_false_safe}")
        print(f"Quant-team GREENLIGHT-in-stress (lower is better): {coach_false_safe}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
