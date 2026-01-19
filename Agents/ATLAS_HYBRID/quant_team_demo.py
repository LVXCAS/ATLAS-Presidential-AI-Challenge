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
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

from quant_team_utils import (
    baseline_assessment,
    derive_vote_confidence,
    get_stress_windows,
    initialize_coordinator,
    load_config,
    make_market_data,
    quant_team_assessment,
)


DISCLAIMER = (
    "DISCLAIMER: Educational simulation only - not financial advice. "
    "No live trading, no brokerage APIs, and no guarantees. "
    "Outputs explain risk and uncertainty; they are not buy/sell recommendations."
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


def _agent_role(agent_name: str) -> str:
    """
    Short, student-friendly descriptions of what each agent contributes.

    Keep these as concepts (how to think about risk/uncertainty), not instructions.
    """
    roles = {
        "TechnicalAgent": "Checks basic indicators (volatility, RSI, EMAs, Bollinger Bands) to flag unstable conditions.",
        "MarketRegimeAgent": "Estimates if the market is trending or choppy (how predictable the regime looks).",
        "CorrelationAgent": "Checks concentration risk: does this pair overlap with existing positions (shared currencies)?",
        "GSQuantAgent": "Computes a simple VaR-style risk proxy from historical volatility (institutional idea, local math).",
        "MonteCarloAgent": "Runs Monte Carlo simulations on historical returns to estimate the chance of big moves.",
        "RiskManagementAgent": "Applies safety rules (drawdown, losing streak, over-trading) using the provided account snapshot.",
        "NewsFilterAgent": "Flags scheduled high-impact events near the timestamp (calendar-based volatility risk).",
        "SessionTimingAgent": "Notes time-of-day effects: some hours are typically more liquid/steady than others.",
        "MultiTimeframeAgent": "Checks whether short/mid/long trends agree (mixed timeframes = higher uncertainty).",
        "VolumeLiquidityAgent": "Estimates liquidity risk from spread or volume (missing data = more uncertainty).",
        "SupportResistanceAgent": "Checks if price is near recent support/resistance levels (often a decision point).",
        "DivergenceAgent": "Looks for RSI divergence (momentum weakening vs price direction) which can raise uncertainty.",
        "VolumeAgent": "Looks for unusually low or spiking volume using provided volume history.",
    }
    return roles.get(agent_name, "Provides an additional risk/uncertainty perspective using the same delayed snapshot.")


def _print_section(title: str) -> None:
    print("")
    print("=" * 72)
    print(title)
    print("=" * 72)


def _to_float(value: Any, default: float) -> float:
    """
    Convert to float without treating 0/0.0 as missing.
    """
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _format_agent_line(
    name: str,
    weight: float,
    is_veto: bool,
    score: float | None = None,
    explanation: str | None = None,
) -> str:
    veto_tag = " (veto)" if is_veto else ""
    if score is None:
        return f"- {name}{veto_tag} (weight {weight:.2f}): {_agent_role(name)}"

    expl = (explanation or "").strip()
    expl_part = f" - {expl}" if expl else ""
    return f"- {name}{veto_tag}: score {score:.2f} (weight {weight:.2f}){expl_part}"


def _choose_example_step(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prefer a later step so agents that need more history (e.g., multi-timeframe,
    Monte Carlo) can run and contribute meaningful explanations.
    """
    preferred_min_step = 100
    steps_sorted = sorted(steps, key=lambda s: int(s.get("step", 0) or 0))

    stress_steps = [s for s in steps_sorted if s.get("is_stress")]
    if stress_steps:
        later_stress = [s for s in stress_steps if int(s.get("step", 0) or 0) >= preferred_min_step]
        return later_stress[0] if later_stress else stress_steps[-1]

    later_any = [s for s in steps_sorted if int(s.get("step", 0) or 0) >= preferred_min_step]
    return later_any[0] if later_any else steps_sorted[-1]


def _print_walkthrough_step(step_rec: Dict[str, Any]) -> None:
    price = float(step_rec.get("price", 0.0) or 0.0)
    timestamp = step_rec.get("time", "")

    baseline = step_rec.get("baseline", {}) or {}
    team = step_rec.get("quant_team", {}) or {}

    label = str(team.get("label", ""))
    market_condition = str(team.get("market_condition", ""))
    risk_posture = str(team.get("risk_posture", ""))
    agents = team.get("agents", {}) or {}

    # Recompute the aggregate from the displayed agent table so the printed
    # number matches the explanation (which is based on the unrounded score).
    if agents:
        veto_triggered = False
        weighted_sum = 0.0
        total_weight = 0.0
        for _name, v in agents.items():
            score = _to_float(v.get("score"), 0.5)
            score = max(0.0, min(1.0, score))
            weight = _to_float(v.get("weight"), 1.0)
            weight = max(0.0, weight)
            details = v.get("details", {}) if isinstance(v.get("details"), dict) else {}
            if details.get("data_sufficiency") == "insufficient":
                weight = 0.0
            is_veto = bool(v.get("is_veto", False))
            if is_veto and score >= 0.80 and details.get("data_sufficiency") != "insufficient":
                veto_triggered = True
            weighted_sum += score * weight
            total_weight += weight

        agg = 1.0 if veto_triggered else (weighted_sum / (total_weight or 1.0))
    else:
        agg = _to_float(team.get("aggregated_score", team.get("risk_score", 0.5)), 0.5)

    confidence = 1.0 - agg

    print(f"Step {step_rec.get('step')} snapshot: price={price:.5f}, time={timestamp}")
    print(f"- Baseline label: {baseline.get('label')} (score {float(baseline.get('score', 0.5)):.2f})")
    print(f"  Explanation: {baseline.get('explanation', '')}")
    posture_part = f" | posture: {risk_posture}" if risk_posture else ""
    print(f"- Quant-team label: {label}{posture_part} | market condition: {market_condition} | aggregated score: {agg:.2f}")
    print(f"  Confidence (1 - score): {confidence:.2f}")
    print(f"  Explanation: {team.get('explanation', '')}")

    drivers = team.get("drivers", []) or []
    if drivers:
        driver_summ = "; ".join(f"{d.get('agent')} ({float(d.get('score', 0.0)):.2f})" for d in drivers[:3])
        print(f"  Top drivers: {driver_summ}")

    risk_flags = team.get("risk_flags", []) or []
    if risk_flags:
        print(f"  Risk flags: {', '.join(str(f) for f in risk_flags)}")

    risk_flag_details = team.get("risk_flag_details", []) or []
    if risk_flag_details:
        details = "; ".join(
            f"{d.get('agent')}: {d.get('explanation')}" for d in risk_flag_details[:2] if d.get("agent")
        )
        if details:
            print(f"  Flag details: {details}")

    if not agents:
        return

    print("  Agent breakdown (higher score = higher uncertainty):")

    def _contrib(item: Tuple[str, Dict[str, Any]]) -> float:
        _, v = item
        score = _to_float(v.get("score"), 0.5)
        weight = _to_float(v.get("weight"), 1.0)
        details = v.get("details", {}) if isinstance(v.get("details"), dict) else {}
        if details.get("data_sufficiency") == "insufficient":
            weight = 0.0
        return score * weight

    for name, v in sorted(agents.items(), key=_contrib, reverse=True):
        details = v.get("details", {}) if isinstance(v.get("details"), dict) else {}
        data_sufficiency = details.get("data_sufficiency")
        effective_weight = _to_float(v.get("weight"), 1.0)
        explanation = str(v.get("explanation", "") or "")
        if data_sufficiency == "insufficient":
            effective_weight = 0.0
            explanation = (explanation + " (insufficient data)").strip()
        print(
            "  "
            + _format_agent_line(
                name=name,
                weight=effective_weight,
                is_veto=bool(v.get("is_veto", False)),
                score=_to_float(v.get("score"), 0.5),
                explanation=explanation,
            )
        )


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
        choices=["all", "stable", "stable-range", "volatility-spike", "regime-shift"],
        help="Which stress window to run",
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
    parser.add_argument("--json-out", default="", help="Optional path to write a JSON report")
    parser.add_argument(
        "--show-atlas-logs",
        action="store_true",
        help="Print internal coordinator init logs (off by default for a cleaner walkthrough)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    _print_section("ATLAS HYBRID - Multi-Agent Market Risk Walkthrough (Track II)")
    print(DISCLAIMER)
    print("")
    print("What this system analyzes:")
    print("- A delayed (historical) snapshot of a market scenario (price + recent history + simple indicators).")
    print("- Data can be cached CSVs stored in the local data folder (default) or synthetic fallback.")
    print(f"- Data source selected: {args.data_source}")
    print("")
    print("What the scores mean:")
    print("- Each agent outputs a score from 0.00 to 1.00.")
    print("  - 0.00 = calmer / more predictable conditions")
    print("  - 1.00 = higher uncertainty / higher risk of large moves")
    print("")
    print("How we combine agents:")
    print("- Aggregated score = weighted average of agent scores (0..1).")
    print("- Veto rule: any veto agent scoring >= 0.80 forces STAND_DOWN (safety-first).")
    print("- Agents with insufficient data return NEUTRAL and do not affect the aggregate.")
    print("- Confidence is reported as (1 - aggregated_score).")
    print("")
    print("Labels and market conditions:")
    print("- GREENLIGHT (< 0.25) -> CALM (LOW): overall uncertainty is low (for learning/demo purposes only).")
    print("- WATCH (0.25-0.36) -> ELEVATED (MODERATE): slow down and ask 'what changed?'")
    print("- STAND_DOWN (>= 0.36) -> STRESS (HIGH): uncertainty is high - pause and reduce risk")
    print("")
    print("Why this is educational (not financial advice):")
    print("- The goal is explainability: each agent gives a short, human-readable reason for its score.")
    print("- The system does not promise accuracy or profits, and it never executes trades.")

    atlas_logs = ""
    if args.show_atlas_logs:
        coordinator = initialize_coordinator(config)
    else:
        # Suppress internal coordinator prints so the demo reads like a walkthrough.
        buf = io.StringIO()
        with redirect_stdout(buf):
            coordinator = initialize_coordinator(config)
        atlas_logs = buf.getvalue().strip()

    roster: List[Tuple[str, float, bool]] = [
        (a.name, float(getattr(a, "weight", 1.0)), bool(a in getattr(coordinator, "veto_agents", [])))
        for a in coordinator.agents
    ]

    _print_section("Agent Roster (What Each Agent Contributes)")
    for name, weight, is_veto in roster:
        print(_format_agent_line(name=name, weight=weight, is_veto=is_veto))
    if atlas_logs and not args.show_atlas_logs:
        print("")
        print("(Internal init logs suppressed. Re-run with --show-atlas-logs to display them.)")

    windows = get_stress_windows(
        data_source=args.data_source,
        asset_class=args.asset_class,
        symbol=args.symbol,
        data_dir=args.data_dir,
        max_rows=args.max_rows,
    )
    actual_source = "cached" if windows and windows[0].data_source == "cached_csv" else "synthetic"
    if actual_source != args.data_source:
        print(f"Note: cached data unavailable; using {actual_source} data instead.")
    if args.data_source == "synthetic" and args.window != "all":
        selected = "stable" if args.window == "stable-range" else args.window
        windows = [w for w in windows if w.name == selected]
    elif args.data_source == "cached" and args.window != "all":
        print("Note: --window is ignored when using cached data.")

    report: Dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "data_source": actual_source,
        "requested_data_source": args.data_source,
        "asset_class": args.asset_class,
        "symbol": args.symbol,
        "windows": [],
    }

    for w in windows:
        _print_section(f"Scenario: {w.name}")
        print(w.description)
        print(f"Pair: {w.pair} | Steps: {len(w.prices)}")
        window_rec: Dict[str, Any] = {
            "name": w.name,
            "description": w.description,
            "data_source": w.data_source,
            "steps": [],
        }

        for step in range(len(w.prices)):
            market_data = make_market_data(
                w.pair,
                w.prices,
                step,
                timestamps=w.timestamps,
                volume_history=w.volumes,
                data_source=w.data_source,
            )

            # Baseline (simple rules)
            indicators = dict(market_data["indicators"])
            indicators["_price"] = market_data["price"]
            baseline = baseline_assessment(indicators)

            # Multi-agent (quant team)
            votes = collect_agent_votes(coordinator, market_data)
            coach_label, coach_meta = quant_team_assessment(votes)
            agent_votes = {name: v.get("vote", "") for name, v in votes.items()}
            agent_confidence = {name: v.get("confidence", None) for name, v in votes.items()}

            window_rec["steps"].append(
                {
                    "step": step,
                    "time": market_data.get("time").isoformat() if market_data.get("time") else "",
                    "price": market_data["price"],
                    "is_stress": bool(w.is_stress_step[step]),
                    "baseline": baseline,
                    "quant_team": {
                        "label": coach_label,
                        **coach_meta,
                        "agent_votes": agent_votes,
                        "agent_confidence": agent_confidence,
                        "agents": votes,
                    },
                }
            )

        report["windows"].append(window_rec)

        stress_steps = sum(1 for s in window_rec["steps"] if s["is_stress"])
        baseline_false_safe = sum(
            1 for s in window_rec["steps"] if s["is_stress"] and s["baseline"]["label"] == "GREENLIGHT"
        )
        coach_false_safe = sum(
            1 for s in window_rec["steps"] if s["is_stress"] and s["quant_team"]["label"] == "GREENLIGHT"
        )
        print("")
        print("Scenario summary (stress-aware metric):")
        print(f"- Stress steps: {stress_steps}/{len(window_rec['steps'])}")
        if stress_steps:
            print(f"- Baseline GREENLIGHT-in-stress: {baseline_false_safe} ({baseline_false_safe / stress_steps:.1%})")
            print(f"- Quant-team GREENLIGHT-in-stress: {coach_false_safe} ({coach_false_safe / stress_steps:.1%})")
        else:
            print("- No stress steps in this scenario.")

        print("")
        print("Walkthrough example (one snapshot):")
        example_step = _choose_example_step(window_rec["steps"])
        _print_walkthrough_step(example_step)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
