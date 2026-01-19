#!/usr/bin/env python3

"""
Minimal evaluation artifact for Track II.

Compares:
1) A simple rule baseline
2) The multi-agent AI quant team

Across cached historical CSV data (or synthetic stress windows if cached data is missing),
using a single primary metric: GREENLIGHT-in-stress rate (lower is better).

This is intended to be understandable to non-technical reviewers and easy to reproduce.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from quant_team_utils import (
    baseline_assessment,
    derive_vote_confidence,
    get_stress_windows,
    initialize_coordinator,
    load_config,
    make_market_data,
    quant_team_assessment,
)


def collect_agent_votes(coordinator, market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    votes: Dict[str, Dict[str, Any]] = {}
    for agent in coordinator.agents:
        assessment = agent.assess(market_data)
        vote, confidence, reasoning = derive_vote_confidence(
            assessment.score,
            assessment.explanation,
            assessment.details,
        )
        votes[agent.name] = {
            "score": float(assessment.score),
            "explanation": assessment.explanation,
            "details": assessment.details,
            "vote": vote,
            "confidence": confidence,
            "reasoning": reasoning,
            "weight": float(getattr(agent, "weight", 1.0)),
            "is_veto": bool(agent in getattr(coordinator, "veto_agents", [])),
        }
    return votes


def _safe_iso(dt: Any) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "track2_quant_team.json"),
        help="Path to Track II quant team config JSON",
    )
    parser.add_argument(
        "--data-source",
        default="cached",
        choices=["synthetic", "cached"],
        help="Use synthetic stress windows or cached CSV data",
    )
    parser.add_argument("--asset-class", default="fx", choices=["fx", "equities"], help="Cached data asset class")
    parser.add_argument("--symbol", default="EURUSD", help="Cached data symbol (example: EURUSD, SPY)")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[2] / "data"),
        help="Root data directory for cached CSVs",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on cached rows")
    parser.add_argument("--json-out", default="submission/evaluation_results.json", help="Where to write JSON results")
    args = parser.parse_args()

    config = load_config(args.config)
    coordinator = initialize_coordinator(config)

    agent_roster = [
        {
            "name": a.name,
            "weight": float(getattr(a, "weight", 1.0)),
            "is_veto": bool(a in getattr(coordinator, "veto_agents", [])),
        }
        for a in coordinator.agents
    ]

    data_source = args.data_source
    windows = get_stress_windows(
        data_source=data_source,
        asset_class=args.asset_class,
        symbol=args.symbol,
        data_dir=args.data_dir,
        max_rows=args.max_rows,
    )
    actual_source = "cached" if windows and windows[0].data_source == "cached_csv" else "synthetic"
    results: Dict[str, Any] = {
        "schema_version": 2,
        "disclaimer": "Educational simulation only. No live trading. No guarantees.",
        "primary_metric": "greenlight_in_stress_rate",
        "method": (
            "Compares baseline rules vs. weighted multi-agent risk scoring on cached historical CSV data."
            if actual_source == "cached"
            else "Compares baseline rules vs. weighted multi-agent risk scoring on synthetic stress windows."
        ),
        "data_source": actual_source,
        "requested_data_source": data_source,
        "asset_class": args.asset_class,
        "symbol": args.symbol,
        "score_definition": {
            "agent_score": "0.0 = low risk/uncertainty, 1.0 = high risk/uncertainty",
            "aggregated_score": "Weighted average of agent scores (0..1)",
            "labels": {"GREENLIGHT": "< 0.25", "WATCH": "0.25-0.36", "STAND_DOWN": ">= 0.36"},
            "risk_posture": {"LOW": "GREENLIGHT", "ELEVATED": "WATCH", "HIGH": "STAND_DOWN"},
        },
        "agents": agent_roster,
        "windows": [],
        "summary": {},
    }

    total_stress = 0
    total_baseline_greenlight_in_stress = 0
    total_team_greenlight_in_stress = 0
    total_team_score_in_stress: List[float] = []

    for w in windows:
        stress = 0
        baseline_greenlight_in_stress = 0
        team_greenlight_in_stress = 0

        window_steps: List[Dict[str, Any]] = []
        team_scores_all: List[float] = []
        team_scores_stress: List[float] = []
        label_counts: Dict[str, int] = {"GREENLIGHT": 0, "WATCH": 0, "STAND_DOWN": 0}

        for step in range(len(w.prices)):
            market_data = make_market_data(
                w.pair,
                w.prices,
                step,
                timestamps=w.timestamps,
                volume_history=w.volumes,
                data_source=w.data_source,
            )
            is_stress = bool(w.is_stress_step[step])

            indicators = dict(market_data["indicators"])
            indicators["_price"] = market_data["price"]
            baseline = baseline_assessment(indicators)
            baseline_label = baseline["label"]

            votes = collect_agent_votes(coordinator, market_data)
            team_label, team_meta = quant_team_assessment(votes)
            team_score = float(team_meta.get("aggregated_score", team_meta.get("risk_score", 0.5)))

            label_counts[team_label] = label_counts.get(team_label, 0) + 1
            team_scores_all.append(team_score)
            if is_stress:
                team_scores_stress.append(team_score)
                total_team_score_in_stress.append(team_score)

            # Keep agent outputs compact for judging: score + explanation + vote/confidence.
            agents_out = {
                name: {
                    "score": round(float(v.get("score", 0.5)), 3),
                    "explanation": (str(v.get("explanation", "")).strip() or "No explanation provided."),
                    "vote": v.get("vote", ""),
                    "confidence": v.get("confidence", None),
                    "reasoning": v.get("reasoning", {}),
                    "weight": float(v.get("weight", 1.0)),
                    "is_veto": bool(v.get("is_veto", False)),
                }
                for name, v in votes.items()
            }
            agent_votes = {name: v.get("vote", "") for name, v in votes.items()}
            agent_confidence = {name: v.get("confidence", None) for name, v in votes.items()}

            window_steps.append(
                {
                    "step": step,
                    "time": _safe_iso(market_data.get("time")),
                    "price": float(market_data.get("price", 0.0)),
                    "is_stress": is_stress,
                    "baseline": baseline,
                    "quant_team": {
                        "label": team_label,
                        "aggregated_score": round(team_score, 3),
                        "market_condition": team_meta.get("market_condition", ""),
                        "risk_posture": team_meta.get("risk_posture", ""),
                        "confidence": team_meta.get("confidence", None),
                        "explanation": team_meta.get("explanation", ""),
                        "drivers": team_meta.get("drivers", []),
                        "risk_flags": team_meta.get("risk_flags", []),
                        "risk_flag_details": team_meta.get("risk_flag_details", []),
                        "insufficient_agents": team_meta.get("insufficient_agents", []),
                        "agent_votes": agent_votes,
                        "agent_confidence": agent_confidence,
                        "agents": agents_out,
                    },
                }
            )

            if is_stress:
                stress += 1
                if baseline_label == "GREENLIGHT":
                    baseline_greenlight_in_stress += 1
                if team_label == "GREENLIGHT":
                    team_greenlight_in_stress += 1

        total_stress += stress
        total_baseline_greenlight_in_stress += baseline_greenlight_in_stress
        total_team_greenlight_in_stress += team_greenlight_in_stress

        avg_team_score = (sum(team_scores_all) / len(team_scores_all)) if team_scores_all else 0.0
        avg_team_score_stress = (sum(team_scores_stress) / len(team_scores_stress)) if team_scores_stress else 0.0

        results["windows"].append(
            {
                "name": w.name,
                "description": w.description,
                "pair": w.pair,
                "data_source": w.data_source,
                "steps_total": len(w.prices),
                "stress_steps": stress,
                "baseline_greenlight_in_stress": baseline_greenlight_in_stress,
                "quant_team_greenlight_in_stress": team_greenlight_in_stress,
                "baseline_greenlight_in_stress_rate": (baseline_greenlight_in_stress / stress) if stress else 0.0,
                "quant_team_greenlight_in_stress_rate": (team_greenlight_in_stress / stress) if stress else 0.0,
                "avg_quant_team_score": round(avg_team_score, 3),
                "avg_quant_team_score_in_stress": round(avg_team_score_stress, 3),
                "quant_team_label_counts": label_counts,
                "steps": window_steps,
            }
        )

    results["summary"] = {
        "stress_steps_total": total_stress,
        "baseline_greenlight_in_stress_total": total_baseline_greenlight_in_stress,
        "quant_team_greenlight_in_stress_total": total_team_greenlight_in_stress,
        "baseline_greenlight_in_stress_rate": (total_baseline_greenlight_in_stress / total_stress) if total_stress else 0.0,
        "quant_team_greenlight_in_stress_rate": (total_team_greenlight_in_stress / total_stress) if total_stress else 0.0,
        "avg_quant_team_score_in_stress": round((sum(total_team_score_in_stress) / len(total_team_score_in_stress)) if total_team_score_in_stress else 0.0, 3),
    }

    if args.json_out in {"", "-"}:
        print(json.dumps(results, indent=2))
    else:
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
    if args.json_out not in {"", "-"}:
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
