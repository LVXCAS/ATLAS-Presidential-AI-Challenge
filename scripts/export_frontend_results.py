#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export evaluation results for the frontend.")
    parser.add_argument(
        "--input",
        default=str(_repo_root() / "submission" / "evaluation_results.json"),
        help="Input evaluation JSON (default: submission/evaluation_results.json).",
    )
    parser.add_argument(
        "--output",
        default=str(_repo_root() / "frontend" / "src" / "data" / "results_cached.json"),
        help="Output JSON (default: frontend/src/data/results_cached.json).",
    )
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap on windows.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text())
    windows = payload.get("windows") or []
    if args.max_windows and len(windows) > args.max_windows:
        windows = windows[: args.max_windows]

    summary = payload.get("summary") or {}
    score_def = payload.get("score_definition") or {}

    output = {
        "data_source": payload.get("data_source", "cached"),
        "primary_metric": payload.get("primary_metric", "greenlight_in_stress_rate"),
        "summary": {
            "baseline_greenlight_in_stress_rate": summary.get("baseline_greenlight_in_stress_rate", 0.0),
            "quant_team_greenlight_in_stress_rate": summary.get("quant_team_greenlight_in_stress_rate", 0.0),
            "avg_quant_team_score_in_stress": summary.get("avg_quant_team_score_in_stress", 0.0),
        },
        "labels": score_def.get("labels", {}),
        "risk_posture_map": score_def.get("risk_posture", {}),
        "windows": [
            {
                "name": w.get("name", ""),
                "description": w.get("description", ""),
                "baseline_greenlight_in_stress_rate": w.get("baseline_greenlight_in_stress_rate", 0.0),
                "quant_team_greenlight_in_stress_rate": w.get("quant_team_greenlight_in_stress_rate", 0.0),
                "avg_quant_team_score_in_stress": w.get("avg_quant_team_score_in_stress", 0.0),
                "stress_steps": w.get("stress_steps", 0),
                "steps_total": w.get("steps_total", 0),
            }
            for w in windows
        ],
        "note": "Cached results are summarized from submission/evaluation_results.json to keep the website lightweight.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
