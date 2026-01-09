#!/usr/bin/env python3

"""
Minimal evaluation artifact for Track II.

Compares:
1) A simple rule baseline
2) The multi-agent AI quant team

Across 3 synthetic stress windows, using a single primary metric:
GREENLIGHT-in-stress rate (lower is better).

This is intended to be understandable to non-technical reviewers and easy to reproduce.
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
    parser.add_argument("--json-out", default="submission/evaluation_results.json", help="Where to write JSON results")
    args = parser.parse_args()

    config = load_config(args.config)
    coordinator = initialize_coordinator(config)

    results: Dict[str, Any] = {
        "primary_metric": "greenlight_in_stress_rate",
        "windows": [],
        "summary": {},
    }

    total_stress = 0
    total_baseline_greenlight_in_stress = 0
    total_team_greenlight_in_stress = 0

    for w in generate_stress_windows():
        stress = 0
        baseline_greenlight_in_stress = 0
        team_greenlight_in_stress = 0

        for step in range(len(w.prices)):
            market_data = make_market_data(w.pair, w.prices, step)
            is_stress = bool(w.is_stress_step[step])

            indicators = dict(market_data["indicators"])
            indicators["_price"] = market_data["price"]
            baseline_label, _ = baseline_risk(indicators)

            votes = collect_agent_votes(coordinator, market_data)
            team_label, _ = quant_team_recommendation(votes)

            if is_stress:
                stress += 1
                if baseline_label == "GREENLIGHT":
                    baseline_greenlight_in_stress += 1
                if team_label == "GREENLIGHT":
                    team_greenlight_in_stress += 1

        total_stress += stress
        total_baseline_greenlight_in_stress += baseline_greenlight_in_stress
        total_team_greenlight_in_stress += team_greenlight_in_stress

        results["windows"].append(
            {
                "name": w.name,
                "description": w.description,
                "stress_steps": stress,
                "baseline_greenlight_in_stress": baseline_greenlight_in_stress,
                "quant_team_greenlight_in_stress": team_greenlight_in_stress,
                "baseline_greenlight_in_stress_rate": (baseline_greenlight_in_stress / stress) if stress else 0.0,
                "quant_team_greenlight_in_stress_rate": (team_greenlight_in_stress / stress) if stress else 0.0,
            }
        )

    results["summary"] = {
        "stress_steps_total": total_stress,
        "baseline_greenlight_in_stress_total": total_baseline_greenlight_in_stress,
        "quant_team_greenlight_in_stress_total": total_team_greenlight_in_stress,
        "baseline_greenlight_in_stress_rate": (total_baseline_greenlight_in_stress / total_stress) if total_stress else 0.0,
        "quant_team_greenlight_in_stress_rate": (total_team_greenlight_in_stress / total_stress) if total_stress else 0.0,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Human-readable printout (for screenshots/video).
    print("Track II Evaluation (simulation-only)")
    print(f"Primary metric: {results['primary_metric']} (lower is better)")
    print("")
    for w in results["windows"]:
        print(
            f"- {w['name']}: baseline={w['baseline_greenlight_in_stress_rate']:.2%}, "
            f"team={w['quant_team_greenlight_in_stress_rate']:.2%}"
        )
    print("")
    print(
        f"Overall: baseline={results['summary']['baseline_greenlight_in_stress_rate']:.2%}, "
        f"team={results['summary']['quant_team_greenlight_in_stress_rate']:.2%}"
    )
    print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
